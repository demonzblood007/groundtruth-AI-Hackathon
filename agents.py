"""Analyst Agent using LangGraph with ReAct pattern."""

import operator
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import pandas as pd

# ============ Configuration ============

MAX_ITERATIONS = 15


# ============ Detailed Schemas ============

class DatasetMeta(BaseModel):
    """Metadata about the dataset."""
    total_rows: int
    total_columns: int
    column_names: list[str]
    column_types: dict[str, str]
    date_column: str | None = None
    date_range: dict | None = None  # {"start": "2024-01-01", "end": "2024-12-01"}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []


class MetricInsight(BaseModel):
    """A computed metric with context."""
    name: str = Field(description="Metric name, e.g., 'Total Revenue', 'Average CTR'")
    value: str = Field(description="The computed value with units")
    category: str = Field(description="Category: 'summary', 'performance', 'efficiency', 'financial'")
    interpretation: str = Field(description="What this metric means in plain English")
    is_good: bool | None = Field(default=None, description="True if positive, False if concerning, None if neutral")


class TrendInsight(BaseModel):
    """A detected trend in the data."""
    metric: str = Field(description="Which metric is trending")
    direction: Literal["increasing", "decreasing", "stable", "volatile"]
    change_value: str = Field(description="e.g., '+15%', '-$2,500', 'no change'")
    period: str = Field(description="Time period of trend, e.g., 'daily', 'weekly', 'over date range'")
    significance: Literal["high", "medium", "low"]
    description: str = Field(description="Plain English description of the trend")


class AnomalyInsight(BaseModel):
    """An unusual pattern or outlier."""
    title: str = Field(description="Short title for the anomaly")
    description: str = Field(description="What is unusual")
    affected_rows: int | None = Field(default=None, description="How many rows affected")
    severity: Literal["critical", "warning", "info"]
    recommendation: str = Field(description="Suggested action")


class PerformerInsight(BaseModel):
    """Top or bottom performer in some category."""
    rank_type: Literal["top", "bottom"]
    entity_name: str = Field(description="Name of the entity, e.g., campaign name")
    entity_column: str = Field(description="Column used for grouping")
    metric_name: str = Field(description="Metric used for ranking")
    metric_value: str = Field(description="The value")
    comparison: str = Field(description="How it compares, e.g., '2x above average'")


class Observation(BaseModel):
    """General observation about the data."""
    title: str
    detail: str
    importance: Literal["high", "medium", "low"]


class AnalysisReport(BaseModel):
    """Complete analysis report from the Analyst Agent."""
    dataset_id: str
    dataset_meta: DatasetMeta
    
    # Insights organized by type
    summary_metrics: list[MetricInsight] = Field(default_factory=list)
    performance_metrics: list[MetricInsight] = Field(default_factory=list)
    trends: list[TrendInsight] = Field(default_factory=list)
    anomalies: list[AnomalyInsight] = Field(default_factory=list)
    top_performers: list[PerformerInsight] = Field(default_factory=list)
    bottom_performers: list[PerformerInsight] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    
    # Raw data for charts
    chart_data: dict = Field(default_factory=dict, description="Data formatted for charts")
    
    # Agent metadata
    iterations_used: int = 0
    analysis_complete: bool = False


# ============ State Definition ============

class AnalystState(TypedDict):
    """State for the Analyst Agent."""
    messages: Annotated[list[BaseMessage], operator.add]
    dataset_id: str
    report: dict  # Will hold AnalysisReport data
    iterations: int
    status: Literal["analyzing", "complete", "error"]


# ============ Tool Factory ============

def create_analyst_tools(df: pd.DataFrame, report: dict):
    """Create tools with access to the dataframe and report."""
    
    @tool
    def execute_code(code: str) -> str:
        """Execute Python/Pandas code to analyze the data.
        
        You have access to:
        - df: The pandas DataFrame
        - pd: pandas library
        - np: numpy library
        
        Use print() to see values or assign to 'result' variable.
        Run one operation at a time for clarity.
        """
        import numpy as np
        
        local_vars = {"df": df.copy(), "pd": pd, "np": np, "result": None}
        builtins = {
            "print": print, "len": len, "range": range, "list": list, 
            "dict": dict, "str": str, "int": int, "float": float, 
            "round": round, "sum": sum, "min": min, "max": max, 
            "abs": abs, "sorted": sorted, "zip": zip, "enumerate": enumerate,
            "True": True, "False": False, "None": None
        }
        
        try:
            result = eval(code, {"__builtins__": builtins}, local_vars)
            if result is not None:
                return str(result)[:2000]  # Limit output size
        except SyntaxError:
            pass
        
        try:
            exec(code, {"__builtins__": builtins}, local_vars)
            if local_vars.get("result") is not None:
                return str(local_vars["result"])[:2000]
            return "Executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def save_metric(name: str, value: str, category: str, interpretation: str, is_good: bool = None) -> str:
        """Save a computed metric to the report.
        
        Args:
            name: Metric name (e.g., 'Total Revenue', 'Average CTR')
            value: The value with units (e.g., '$45,000', '2.3%')
            category: One of 'summary', 'performance', 'efficiency', 'financial'
            interpretation: Plain English meaning
            is_good: True=positive, False=concerning, None=neutral
        """
        key = "summary_metrics" if category == "summary" else "performance_metrics"
        if key not in report:
            report[key] = []
        report[key].append({
            "name": name, "value": value, "category": category,
            "interpretation": interpretation, "is_good": is_good
        })
        return f"Saved metric: {name} = {value}"
    
    @tool
    def save_trend(metric: str, direction: str, change_value: str, period: str, 
                   significance: str, description: str) -> str:
        """Save a detected trend.
        
        Args:
            metric: Which metric is trending
            direction: 'increasing', 'decreasing', 'stable', or 'volatile'
            change_value: e.g., '+15%', '-$2,500'
            period: e.g., 'daily', 'weekly', 'over date range'
            significance: 'high', 'medium', or 'low'
            description: Plain English description
        """
        if "trends" not in report:
            report["trends"] = []
        report["trends"].append({
            "metric": metric, "direction": direction, "change_value": change_value,
            "period": period, "significance": significance, "description": description
        })
        return f"Saved trend: {metric} is {direction}"
    
    @tool
    def save_anomaly(title: str, description: str, severity: str, 
                     recommendation: str, affected_rows: int = None) -> str:
        """Save an anomaly or unusual pattern.
        
        Args:
            title: Short title
            description: What is unusual
            severity: 'critical', 'warning', or 'info'
            recommendation: Suggested action
            affected_rows: Number of rows affected (optional)
        """
        if "anomalies" not in report:
            report["anomalies"] = []
        report["anomalies"].append({
            "title": title, "description": description, "severity": severity,
            "recommendation": recommendation, "affected_rows": affected_rows
        })
        return f"Saved anomaly: {title} ({severity})"
    
    @tool
    def save_performer(rank_type: str, entity_name: str, entity_column: str,
                       metric_name: str, metric_value: str, comparison: str) -> str:
        """Save a top or bottom performer.
        
        Args:
            rank_type: 'top' or 'bottom'
            entity_name: Name of the entity (e.g., campaign name)
            entity_column: Column used for grouping
            metric_name: Metric used for ranking
            metric_value: The actual value
            comparison: How it compares (e.g., '2x above average')
        """
        key = "top_performers" if rank_type == "top" else "bottom_performers"
        if key not in report:
            report[key] = []
        report[key].append({
            "rank_type": rank_type, "entity_name": entity_name, 
            "entity_column": entity_column, "metric_name": metric_name,
            "metric_value": metric_value, "comparison": comparison
        })
        return f"Saved {rank_type} performer: {entity_name}"
    
    @tool
    def save_observation(title: str, detail: str, importance: str) -> str:
        """Save a general observation about the data.
        
        Args:
            title: Short title
            detail: The observation detail
            importance: 'high', 'medium', or 'low'
        """
        if "observations" not in report:
            report["observations"] = []
        report["observations"].append({
            "title": title, "detail": detail, "importance": importance
        })
        return f"Saved observation: {title}"
    
    @tool
    def save_chart_data(chart_name: str, chart_type: str, data_json: str) -> str:
        """Save data for a chart to be rendered in the report.
        
        Args:
            chart_name: Name/title for the chart
            chart_type: 'bar', 'line', 'pie', 'table'
            data_json: JSON string of the chart data
        """
        import json
        if "chart_data" not in report:
            report["chart_data"] = {}
        try:
            report["chart_data"][chart_name] = {
                "type": chart_type,
                "data": json.loads(data_json)
            }
            return f"Saved chart data: {chart_name}"
        except:
            return "Error: Invalid JSON for chart data"
    
    @tool
    def complete_analysis() -> str:
        """Call this when analysis is complete.
        
        Before calling, ensure you have saved:
        - At least 3 summary metrics
        - At least 1 trend (if time-based data)
        - Top and bottom performers
        - Any anomalies found
        """
        report["analysis_complete"] = True
        return "ANALYSIS_COMPLETE"
    
    return [execute_code, save_metric, save_trend, save_anomaly, 
            save_performer, save_observation, save_chart_data, complete_analysis]


# ============ System Prompt ============

ANALYST_PROMPT = """You are an expert data analyst. Your job is to thoroughly analyze a dataset and extract actionable insights.

You have access to a pandas DataFrame called 'df'. Use execute_code to run Python/Pandas analysis.

## ANALYSIS WORKFLOW

### Phase 1: Data Understanding
- Check shape, columns, dtypes: df.shape, df.columns, df.dtypes
- Preview data: df.head()
- Check for nulls: df.isnull().sum()
- Identify numeric vs categorical columns

### Phase 2: Summary Statistics
- Compute totals, means, medians for numeric columns
- Use save_metric() for each key metric
- Categories: 'summary' for totals, 'performance' for rates/ratios

### Phase 3: Grouping & Comparison
- Find natural grouping columns (names, categories, types)
- Group and aggregate to find top/bottom performers
- Use save_performer() for each

### Phase 4: Time Analysis (if date column exists)
- Check for date columns
- Analyze trends over time
- Use save_trend() for significant patterns

### Phase 5: Anomaly Detection
- Look for outliers (values > 2 std from mean)
- Check for missing data patterns
- Find unexpected values
- Use save_anomaly() for issues

### Phase 6: Chart Preparation
- Prepare top 5 by key metric for bar chart
- Prepare time series for line chart (if applicable)
- Use save_chart_data() with JSON

## RULES
1. Run ONE code block at a time
2. Check the output before proceeding
3. Save insights as you discover them
4. Be specific with numbers and percentages
5. Maximum {max_iter} iterations - be efficient
6. Call complete_analysis() when done

Start by exploring the data structure.""".format(max_iter=MAX_ITERATIONS)


# ============ Graph Construction ============

def create_analyst_graph(df: pd.DataFrame, dataset_id: str):
    """Create the Analyst Agent graph."""
    
    # Initialize report structure
    report = {
        "dataset_id": dataset_id,
        "dataset_meta": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        },
        "summary_metrics": [],
        "performance_metrics": [],
        "trends": [],
        "anomalies": [],
        "top_performers": [],
        "bottom_performers": [],
        "observations": [],
        "chart_data": {},
        "iterations_used": 0,
        "analysis_complete": False
    }
    
    tools = create_analyst_tools(df, report)
    tool_node = ToolNode(tools)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    def run_analyst(state: AnalystState) -> dict:
        """Run the analyst LLM."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response], 
            "iterations": state["iterations"] + 1,
            "report": report
        }
    
    def should_continue(state: AnalystState) -> str:
        """Determine if we should continue the loop."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if state["iterations"] >= MAX_ITERATIONS:
            return "end"
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc["name"] == "complete_analysis":
                    return "end"
            return "tools"
        
        return "end"
    
    def finalize(state: AnalystState) -> dict:
        """Finalize the analysis."""
        report["iterations_used"] = state["iterations"]
        return {"status": "complete", "report": report}
    
    # Build graph
    graph = StateGraph(AnalystState)
    graph.add_node("analyst", run_analyst)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)
    
    graph.set_entry_point("analyst")
    graph.add_conditional_edges("analyst", should_continue, {"tools": "tools", "end": "finalize"})
    graph.add_edge("tools", "analyst")
    graph.add_edge("finalize", END)
    
    return graph.compile(), report


# ============ Run Analysis ============

async def run_analyst_agent(df: pd.DataFrame, dataset_id: str) -> dict:
    """Run the analyst agent on a dataframe."""
    
    graph, report = create_analyst_graph(df, dataset_id)
    
    # Build context about the data
    sample_data = df.head(3).to_string()
    
    initial_state = {
        "messages": [
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"""Analyze this dataset (ID: {dataset_id})

Dataset Info:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column names: {df.columns.tolist()}
- Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}
- Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}

First 3 rows:
{sample_data}

Begin your analysis.""")
        ],
        "dataset_id": dataset_id,
        "report": report,
        "iterations": 0,
        "status": "analyzing"
    }
    
    result = await graph.ainvoke(initial_state)
    
    return result["report"]


# ============================================================================
#                           NARRATOR AGENT
# ============================================================================

class SlideContent(BaseModel):
    """Content for a single slide."""
    slide_number: int
    title: str
    bullet_points: list[str]
    speaker_notes: str | None = None
    has_chart: bool = False
    chart_reference: str | None = None


class NarrativeReport(BaseModel):
    """Complete narrative output for report generation."""
    executive_summary: str
    key_metrics_narrative: str
    performance_analysis: str
    trend_commentary: str
    risk_and_anomalies: str
    recommendations: list[str]
    slides: list[SlideContent]
    tone: str = "executive"


class NarratorState(TypedDict):
    """State for Narrator Agent."""
    analysis_report: dict
    executive_summary: str
    key_metrics_narrative: str
    performance_analysis: str
    trend_commentary: str
    risk_and_anomalies: str
    recommendations: list[str]
    slides: list[dict]
    current_step: str
    status: str


# ============ Narrator Prompts ============

EXECUTIVE_SUMMARY_PROMPT = """You are an executive report writer. Write a concise executive summary.

DATA ANALYSIS RESULTS:
{analysis_data}

Write 2-3 paragraphs (max 150 words) that:
1. Lead with the single most important finding
2. Highlight 2-3 key wins or achievements
3. Mention any critical concerns
4. Use specific numbers from the data

Tone: Confident, data-driven, suitable for C-level executives.
Do not use bullet points - write in flowing paragraphs."""


METRICS_NARRATIVE_PROMPT = """Transform these metrics into a clear narrative explanation.

METRICS:
{metrics}

Write 1-2 paragraphs explaining:
1. What these numbers mean in plain business terms
2. Which metrics are performing well (and why that matters)
3. Which metrics need attention

Use the actual values. Be specific, not vague."""


PERFORMANCE_PROMPT = """Analyze the performance data and write insights.

TOP PERFORMERS:
{top_performers}

BOTTOM PERFORMERS:
{bottom_performers}

Write 2 paragraphs:
1. First paragraph: Celebrate the top performers - what makes them successful
2. Second paragraph: Address the underperformers - what might be causing issues

Be specific with names and numbers."""


TRENDS_PROMPT = """Interpret these trends for business executives.

TRENDS DETECTED:
{trends}

Write 1-2 paragraphs explaining:
1. What direction things are moving
2. What's driving these changes (if apparent)
3. What this means for the business

If no significant trends, say so briefly."""


RISKS_PROMPT = """Summarize risks and anomalies for executive attention.

ANOMALIES DETECTED:
{anomalies}

Write 1 paragraph that:
1. Highlights the most critical issues first
2. Explains potential business impact
3. Suggests urgency level

If no anomalies, write a brief "all clear" statement."""


RECOMMENDATIONS_PROMPT = """Generate actionable recommendations based on this analysis.

FULL ANALYSIS:
{analysis}

Generate exactly 5 recommendations:
1. Each must start with an action verb (Increase, Reduce, Monitor, Investigate, etc.)
2. Each must reference specific data points
3. Order from highest to lowest priority

Format as a numbered list. Be specific and actionable."""


SLIDES_PROMPT = """Structure this content for a 5-slide executive presentation.

CONTENT:
- Executive Summary: {executive_summary}
- Metrics: {metrics}
- Performance: {performance}
- Recommendations: {recommendations}

For each slide, provide:
- slide_number (1-5)
- title (short, impactful)
- bullet_points (3-5 per slide, concise)
- speaker_notes (1-2 sentences for presenter)

Output as JSON array of slides."""


# ============ Narrator Graph Nodes ============

def create_narrator_graph():
    """Create the Narrator Agent graph."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    async def write_executive_summary(state: NarratorState) -> dict:
        """Generate executive summary."""
        analysis = state["analysis_report"]
        
        # Compile key data for the prompt
        analysis_data = f"""
Summary Metrics: {analysis.get('summary_metrics', [])}
Performance Metrics: {analysis.get('performance_metrics', [])}
Top Performers: {analysis.get('top_performers', [])}
Trends: {analysis.get('trends', [])}
Anomalies: {analysis.get('anomalies', [])}
"""
        
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(analysis_data=analysis_data)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "executive_summary": response.content,
            "current_step": "metrics"
        }
    
    async def write_metrics_narrative(state: NarratorState) -> dict:
        """Generate metrics explanation."""
        analysis = state["analysis_report"]
        
        metrics = analysis.get('summary_metrics', []) + analysis.get('performance_metrics', [])
        metrics_text = "\n".join([f"- {m.get('name')}: {m.get('value')} - {m.get('interpretation', '')}" for m in metrics])
        
        if not metrics_text.strip():
            return {"key_metrics_narrative": "No specific metrics were calculated for this dataset.", "current_step": "performance"}
        
        prompt = METRICS_NARRATIVE_PROMPT.format(metrics=metrics_text)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "key_metrics_narrative": response.content,
            "current_step": "performance"
        }
    
    async def write_performance_analysis(state: NarratorState) -> dict:
        """Generate performance analysis."""
        analysis = state["analysis_report"]
        
        top = analysis.get('top_performers', [])
        bottom = analysis.get('bottom_performers', [])
        
        top_text = "\n".join([f"- {p.get('entity_name')}: {p.get('metric_value')} ({p.get('comparison', '')})" for p in top]) or "None identified"
        bottom_text = "\n".join([f"- {p.get('entity_name')}: {p.get('metric_value')} ({p.get('comparison', '')})" for p in bottom]) or "None identified"
        
        prompt = PERFORMANCE_PROMPT.format(top_performers=top_text, bottom_performers=bottom_text)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "performance_analysis": response.content,
            "current_step": "trends"
        }
    
    async def write_trend_commentary(state: NarratorState) -> dict:
        """Generate trend analysis."""
        analysis = state["analysis_report"]
        
        trends = analysis.get('trends', [])
        trends_text = "\n".join([f"- {t.get('metric')}: {t.get('direction')} ({t.get('change_value')}) - {t.get('description', '')}" for t in trends])
        
        if not trends_text.strip():
            return {"trend_commentary": "No significant time-based trends were identified in this dataset.", "current_step": "risks"}
        
        prompt = TRENDS_PROMPT.format(trends=trends_text)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "trend_commentary": response.content,
            "current_step": "risks"
        }
    
    async def write_risks(state: NarratorState) -> dict:
        """Generate risk summary."""
        analysis = state["analysis_report"]
        
        anomalies = analysis.get('anomalies', [])
        anomalies_text = "\n".join([f"- [{a.get('severity', 'info').upper()}] {a.get('title')}: {a.get('description')}" for a in anomalies])
        
        if not anomalies_text.strip():
            return {"risk_and_anomalies": "No significant anomalies or risks were detected in this dataset.", "current_step": "recommendations"}
        
        prompt = RISKS_PROMPT.format(anomalies=anomalies_text)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "risk_and_anomalies": response.content,
            "current_step": "recommendations"
        }
    
    async def write_recommendations(state: NarratorState) -> dict:
        """Generate recommendations."""
        analysis = state["analysis_report"]
        
        # Compile full analysis context
        analysis_summary = f"""
Metrics: {analysis.get('summary_metrics', [])}
Top Performers: {analysis.get('top_performers', [])}
Bottom Performers: {analysis.get('bottom_performers', [])}
Trends: {analysis.get('trends', [])}
Anomalies: {analysis.get('anomalies', [])}
Observations: {analysis.get('observations', [])}
"""
        
        prompt = RECOMMENDATIONS_PROMPT.format(analysis=analysis_summary)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse recommendations from response
        lines = response.content.strip().split('\n')
        recommendations = [line.strip().lstrip('0123456789.-) ') for line in lines if line.strip()]
        
        return {
            "recommendations": recommendations[:5],
            "current_step": "slides"
        }
    
    async def build_slides(state: NarratorState) -> dict:
        """Generate slide content structure."""
        import json
        
        # Build slides from the generated content
        slides = [
            {
                "slide_number": 1,
                "title": "Executive Overview",
                "bullet_points": [
                    state.get("executive_summary", "")[:200] + "..." if len(state.get("executive_summary", "")) > 200 else state.get("executive_summary", "")
                ],
                "speaker_notes": "Open with the key finding to capture attention.",
                "has_chart": False
            },
            {
                "slide_number": 2,
                "title": "Key Metrics",
                "bullet_points": _extract_bullets(state.get("key_metrics_narrative", ""), max_bullets=5),
                "speaker_notes": "Walk through each metric and its business impact.",
                "has_chart": True,
                "chart_reference": "metrics_chart"
            },
            {
                "slide_number": 3,
                "title": "Performance Analysis",
                "bullet_points": _extract_bullets(state.get("performance_analysis", ""), max_bullets=5),
                "speaker_notes": "Highlight the gap between top and bottom performers.",
                "has_chart": True,
                "chart_reference": "performance_chart"
            },
            {
                "slide_number": 4,
                "title": "Trends & Risks",
                "bullet_points": _extract_bullets(state.get("trend_commentary", "") + " " + state.get("risk_and_anomalies", ""), max_bullets=5),
                "speaker_notes": "Address any concerns and their potential impact.",
                "has_chart": False
            },
            {
                "slide_number": 5,
                "title": "Recommendations",
                "bullet_points": state.get("recommendations", [])[:5],
                "speaker_notes": "End with clear action items and next steps.",
                "has_chart": False
            }
        ]
        
        return {
            "slides": slides,
            "current_step": "complete",
            "status": "complete"
        }
    
    # Build the graph
    graph = StateGraph(NarratorState)
    
    graph.add_node("write_summary", write_executive_summary)
    graph.add_node("write_metrics", write_metrics_narrative)
    graph.add_node("write_performance", write_performance_analysis)
    graph.add_node("write_trends", write_trend_commentary)
    graph.add_node("write_risks", write_risks)
    graph.add_node("write_recommendations", write_recommendations)
    graph.add_node("build_slides", build_slides)
    
    # Sequential flow
    graph.set_entry_point("write_summary")
    graph.add_edge("write_summary", "write_metrics")
    graph.add_edge("write_metrics", "write_performance")
    graph.add_edge("write_performance", "write_trends")
    graph.add_edge("write_trends", "write_risks")
    graph.add_edge("write_risks", "write_recommendations")
    graph.add_edge("write_recommendations", "build_slides")
    graph.add_edge("build_slides", END)
    
    return graph.compile()


def _extract_bullets(text: str, max_bullets: int = 5) -> list[str]:
    """Extract bullet points from narrative text."""
    if not text:
        return ["No data available"]
    
    # Split by sentences
    import re
    sentences = re.split(r'[.!?]+', text)
    bullets = []
    
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and len(bullets) < max_bullets:
            # Truncate long sentences
            if len(s) > 100:
                s = s[:97] + "..."
            bullets.append(s)
    
    return bullets if bullets else [text[:100] + "..." if len(text) > 100 else text]


async def run_narrator_agent(analysis_report: dict) -> dict:
    """Run the narrator agent to generate narrative content."""
    
    graph = create_narrator_graph()
    
    initial_state = {
        "analysis_report": analysis_report,
        "executive_summary": "",
        "key_metrics_narrative": "",
        "performance_analysis": "",
        "trend_commentary": "",
        "risk_and_anomalies": "",
        "recommendations": [],
        "slides": [],
        "current_step": "summary",
        "status": "generating"
    }
    
    result = await graph.ainvoke(initial_state)
    
    return {
        "executive_summary": result["executive_summary"],
        "key_metrics_narrative": result["key_metrics_narrative"],
        "performance_analysis": result["performance_analysis"],
        "trend_commentary": result["trend_commentary"],
        "risk_and_anomalies": result["risk_and_anomalies"],
        "recommendations": result["recommendations"],
        "slides": result["slides"],
        "status": "complete"
    }
