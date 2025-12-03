"""PDF Processing Pipeline with Local LLM preprocessing."""

import re
import operator
from io import BytesIO
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import pandas as pd

# ============ Configuration ============

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============ Extraction Schemas ============

class ExtractedMetric(BaseModel):
    """Metric found via pattern matching."""
    raw_text: str
    value: str
    value_type: Literal["currency", "percentage", "number", "date"]


class TextSection(BaseModel):
    """A section of text from the PDF."""
    heading: str | None = None
    content: str
    page_number: int
    has_metrics: bool = False


class PDFExtraction(BaseModel):
    """Complete extraction from PDF."""
    filename: str
    page_count: int
    tables: list[dict] = Field(default_factory=list)
    sections: list[TextSection] = Field(default_factory=list)
    raw_text: str = ""
    extracted_metrics: list[ExtractedMetric] = Field(default_factory=list)
    has_tables: bool = False


# ============ Pattern-Based Extraction (No LLM) ============

METRIC_PATTERNS = {
    "currency": r"\$\s*[\d,]+\.?\d*[KMB]?|\d+\.?\d*\s*(?:USD|EUR|INR)",
    "percentage": r"[\d.]+\s*%",
    "large_number": r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
    "decimal": r"\b\d+\.\d+\b",
}


def extract_metrics_from_text(text: str) -> list[ExtractedMetric]:
    """Extract metrics using regex patterns (no LLM cost)."""
    metrics = []
    
    for metric_type, pattern in METRIC_PATTERNS.items():
        for match in re.finditer(pattern, text):
            # Get surrounding context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 10)
            context = text[start:end].strip()
            
            metrics.append(ExtractedMetric(
                raw_text=context,
                value=match.group(),
                value_type=metric_type if metric_type != "large_number" else "number"
            ))
    
    return metrics


def extract_pdf_content(file_bytes: bytes, filename: str) -> PDFExtraction:
    """Extract all content from PDF without using LLM."""
    import pdfplumber
    
    extraction = PDFExtraction(filename=filename, page_count=0)
    all_text = []
    all_tables = []
    
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        extraction.page_count = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text() or ""
            if text.strip():
                all_text.append(text)
                extraction.sections.append(TextSection(
                    content=text,
                    page_number=i + 1,
                    has_metrics=bool(re.search(r'\d', text))
                ))
            
            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    # Convert to dict format
                    headers = [str(h).strip() if h else f"col_{j}" for j, h in enumerate(table[0])]
                    rows = []
                    for row in table[1:]:
                        row_dict = {}
                        for j, cell in enumerate(row):
                            if j < len(headers):
                                row_dict[headers[j]] = str(cell).strip() if cell else ""
                        rows.append(row_dict)
                    all_tables.append({"headers": headers, "rows": rows, "page": i + 1})
    
    extraction.raw_text = "\n\n".join(all_text)
    extraction.tables = all_tables
    extraction.has_tables = len(all_tables) > 0
    extraction.extracted_metrics = extract_metrics_from_text(extraction.raw_text)
    
    return extraction


# ============ State for PDF Processing Graph ============

class PDFProcessState(TypedDict):
    """State for PDF processing pipeline."""
    extraction: dict
    chunks: list[dict]
    chunk_summaries: list[str]
    combined_insights: dict
    status: str
    current_chunk: int


# ============ Local LLM Summarization ============

def get_local_llm():
    """Get Ollama LLM instance."""
    return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split text into chunks, trying to break at paragraphs."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


SUMMARIZE_PROMPT = """Extract key information from this text. Focus on:
- Numbers, metrics, KPIs
- Trends or comparisons
- Key findings or conclusions

Text:
{text}

Provide a concise summary with all important data points:"""


INSIGHT_PROMPT = """Based on these summaries from a document, generate structured insights.

Summaries:
{summaries}

Extracted Metrics (found in document):
{metrics}

Generate insights in this exact JSON format:
{{
    "summary_metrics": [
        {{"name": "metric name", "value": "value with units", "interpretation": "what it means"}}
    ],
    "trends": [
        {{"metric": "what metric", "direction": "increasing/decreasing/stable", "description": "explanation"}}
    ],
    "key_findings": [
        {{"title": "finding title", "detail": "explanation", "importance": "high/medium/low"}}
    ],
    "anomalies": [
        {{"title": "issue", "description": "what's unusual", "severity": "critical/warning/info"}}
    ]
}}

Return ONLY valid JSON:"""


# ============ Graph Nodes ============

def prepare_chunks(state: PDFProcessState) -> dict:
    """Prepare text chunks for processing."""
    extraction = state["extraction"]
    raw_text = extraction.get("raw_text", "")
    
    chunks = chunk_text(raw_text, max_chars=2000)
    
    return {
        "chunks": [{"text": c, "index": i} for i, c in enumerate(chunks)],
        "current_chunk": 0,
        "chunk_summaries": [],
        "status": "chunking_complete"
    }


async def summarize_chunk(state: PDFProcessState) -> dict:
    """Summarize current chunk using local LLM."""
    chunks = state["chunks"]
    current = state["current_chunk"]
    summaries = state.get("chunk_summaries", [])
    
    if current >= len(chunks):
        return {"status": "summarization_complete"}
    
    chunk_text = chunks[current]["text"]
    
    try:
        llm = get_local_llm()
        prompt = SUMMARIZE_PROMPT.format(text=chunk_text)
        summary = await llm.ainvoke(prompt)
        summaries.append(summary)
    except Exception as e:
        # Fallback: use first 500 chars as summary
        summaries.append(chunk_text[:500] + "...")
    
    return {
        "chunk_summaries": summaries,
        "current_chunk": current + 1,
        "status": "summarizing"
    }


def should_continue_summarizing(state: PDFProcessState) -> str:
    """Check if more chunks need summarization."""
    if state["current_chunk"] >= len(state["chunks"]):
        return "combine"
    return "summarize"


async def combine_insights(state: PDFProcessState) -> dict:
    """Combine chunk summaries into structured insights using local LLM."""
    summaries = state.get("chunk_summaries", [])
    extraction = state["extraction"]
    metrics = extraction.get("extracted_metrics", [])
    
    # Format metrics for prompt
    metrics_text = "\n".join([f"- {m['value']} ({m['value_type']}): {m['raw_text']}" for m in metrics[:20]])
    summaries_text = "\n\n---\n\n".join(summaries)
    
    try:
        llm = get_local_llm()
        prompt = INSIGHT_PROMPT.format(summaries=summaries_text, metrics=metrics_text)
        response = await llm.ainvoke(prompt)
        
        # Try to parse JSON from response
        import json
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            insights = json.loads(json_match.group())
        else:
            insights = {"key_findings": [{"title": "Document Summary", "detail": summaries_text[:1000], "importance": "medium"}]}
    except Exception as e:
        insights = {
            "summary_metrics": [],
            "trends": [],
            "key_findings": [{"title": "Extraction Complete", "detail": f"Extracted {len(metrics)} metrics from document", "importance": "medium"}],
            "anomalies": []
        }
    
    return {
        "combined_insights": insights,
        "status": "complete"
    }


def build_final_report(state: PDFProcessState) -> dict:
    """Build report in same format as Analyst Agent."""
    extraction = state["extraction"]
    insights = state.get("combined_insights", {})
    
    report = {
        "dataset_id": extraction.get("filename", "pdf_doc"),
        "source_type": "pdf",
        "dataset_meta": {
            "total_rows": 0,
            "total_columns": 0,
            "page_count": extraction.get("page_count", 0),
            "tables_found": len(extraction.get("tables", [])),
            "metrics_extracted": len(extraction.get("extracted_metrics", []))
        },
        "summary_metrics": insights.get("summary_metrics", []),
        "performance_metrics": [],
        "trends": insights.get("trends", []),
        "anomalies": insights.get("anomalies", []),
        "top_performers": [],
        "bottom_performers": [],
        "observations": insights.get("key_findings", []),
        "chart_data": {},
        "raw_metrics": extraction.get("extracted_metrics", []),
        "tables": extraction.get("tables", []),
        "analysis_complete": True
    }
    
    return {"combined_insights": report, "status": "complete"}


# ============ Build Graph ============

def create_pdf_processing_graph():
    """Create the PDF processing pipeline graph."""
    
    graph = StateGraph(PDFProcessState)
    
    # Add nodes
    graph.add_node("prepare_chunks", prepare_chunks)
    graph.add_node("summarize_chunk", summarize_chunk)
    graph.add_node("combine_insights", combine_insights)
    graph.add_node("build_report", build_final_report)
    
    # Set entry
    graph.set_entry_point("prepare_chunks")
    
    # Add edges
    graph.add_edge("prepare_chunks", "summarize_chunk")
    graph.add_conditional_edges(
        "summarize_chunk",
        should_continue_summarizing,
        {"summarize": "summarize_chunk", "combine": "combine_insights"}
    )
    graph.add_edge("combine_insights", "build_report")
    graph.add_edge("build_report", END)
    
    return graph.compile()


# ============ Main Entry Point ============

async def process_pdf(file_bytes: bytes, filename: str) -> dict:
    """Process a PDF and return analysis report."""
    
    # Stage 1: Extract content (no LLM)
    extraction = extract_pdf_content(file_bytes, filename)
    
    # Stage 2: If has tables, convert to DataFrame and use Analyst Agent
    if extraction.has_tables and extraction.tables:
        # Use first table as primary data
        table = extraction.tables[0]
        df = pd.DataFrame(table["rows"])
        
        # Import and run analyst agent
        from agents import run_analyst_agent
        report = await run_analyst_agent(df, f"pdf_{filename}")
        report["source_type"] = "pdf_table"
        report["pdf_meta"] = {
            "page_count": extraction.page_count,
            "tables_found": len(extraction.tables)
        }
        return report
    
    # Stage 3: Text-only PDF - use local LLM pipeline
    graph = create_pdf_processing_graph()
    
    initial_state = {
        "extraction": extraction.model_dump(),
        "chunks": [],
        "chunk_summaries": [],
        "combined_insights": {},
        "status": "starting",
        "current_chunk": 0
    }
    
    result = await graph.ainvoke(initial_state)
    
    return result["combined_insights"]

