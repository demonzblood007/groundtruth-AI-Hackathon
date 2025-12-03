"""FastAPI with data ingestion endpoints."""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import pandas as pd
from io import StringIO, BytesIO
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl
from sqlalchemy import create_engine

app = FastAPI(title="H-OOI: Automated Insight Engine")

# In-memory storage for loaded dataframes
datasets: dict[str, pd.DataFrame] = {}


# ============ Request Models ============

class SQLSource(BaseModel):
    connection_string: str  # postgresql://user:pass@host:5432/db
    query: str              # SELECT * FROM table

class URLSource(BaseModel):
    url: HttpUrl

class MongoSource(BaseModel):
    uri: str          # mongodb://localhost:27017
    database: str
    collection: str


# ============ Data Loading ============

def save_dataset(df: pd.DataFrame) -> str:
    """Save dataframe and return dataset_id."""
    dataset_id = uuid4().hex[:8]
    datasets[dataset_id] = df
    return dataset_id

def get_df_info(df: pd.DataFrame) -> dict:
    """Get basic info about the dataframe."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


# ============ Endpoints ============

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    """Upload CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "File must be CSV")
    
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    dataset_id = save_dataset(df)
    
    return {"dataset_id": dataset_id, "info": get_df_info(df)}


@app.post("/ingest/excel")
async def ingest_excel(file: UploadFile = File(...)):
    """Upload Excel file."""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "File must be Excel (.xlsx or .xls)")
    
    content = await file.read()
    df = pd.read_excel(BytesIO(content))
    dataset_id = save_dataset(df)
    
    return {"dataset_id": dataset_id, "info": get_df_info(df)}


@app.post("/ingest/sql")
async def ingest_sql(source: SQLSource):
    """Load data from SQL database."""
    try:
        engine = create_engine(source.connection_string)
        df = pd.read_sql(source.query, engine)
        dataset_id = save_dataset(df)
        return {"dataset_id": dataset_id, "info": get_df_info(df)}
    except Exception as e:
        raise HTTPException(400, f"SQL error: {str(e)}")


@app.post("/ingest/url")
async def ingest_url(source: URLSource):
    """Load data from URL (CSV or JSON)."""
    url = str(source.url)
    try:
        if url.endswith('.json'):
            df = pd.read_json(url)
        else:
            df = pd.read_csv(url)
        dataset_id = save_dataset(df)
        return {"dataset_id": dataset_id, "info": get_df_info(df)}
    except Exception as e:
        raise HTTPException(400, f"URL load error: {str(e)}")


@app.post("/ingest/mongodb")
async def ingest_mongodb(source: MongoSource):
    """Load data from MongoDB collection."""
    try:
        from pymongo import MongoClient
        client = MongoClient(source.uri)
        db = client[source.database]
        cursor = db[source.collection].find()
        df = pd.DataFrame(list(cursor))
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)  # Convert ObjectId
        dataset_id = save_dataset(df)
        return {"dataset_id": dataset_id, "info": get_df_info(df)}
    except Exception as e:
        raise HTTPException(400, f"MongoDB error: {str(e)}")


@app.get("/datasets")
async def list_datasets():
    """List all loaded datasets."""
    return {
        did: {"rows": len(df), "columns": len(df.columns)} 
        for did, df in datasets.items()
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get info about a specific dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    return get_df_info(datasets[dataset_id])


@app.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 5):
    """Preview first N rows of a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    df = datasets[dataset_id]
    return df.head(rows).to_dict(orient="records")


# ============ PDF Ingestion ============

# Store for PDF analysis results
pdf_reports: dict[str, dict] = {}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file.
    
    - Extracts tables and text
    - If tables found: converts to DataFrame for analysis
    - If text only: uses local LLM for preprocessing
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "File must be PDF")
    
    from pdf_processor import process_pdf
    
    content = await file.read()
    report = await process_pdf(content, file.filename)
    
    # Store the report
    report_id = uuid4().hex[:8]
    pdf_reports[report_id] = report
    
    return {
        "report_id": report_id,
        "source_type": report.get("source_type", "pdf"),
        "pages": report.get("dataset_meta", {}).get("page_count", 0),
        "tables_found": report.get("dataset_meta", {}).get("tables_found", 0),
        "metrics_extracted": report.get("dataset_meta", {}).get("metrics_extracted", 0),
        "status": "complete"
    }


@app.get("/reports/{report_id}")
async def get_pdf_report(report_id: str):
    """Get the full analysis report for a PDF."""
    if report_id not in pdf_reports:
        raise HTTPException(404, "Report not found")
    return pdf_reports[report_id]


# ============ Analysis Endpoint ============

@app.post("/analyze/{dataset_id}")
async def analyze_dataset(dataset_id: str):
    """Run the Analyst Agent on a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    from agents import run_analyst_agent
    
    df = datasets[dataset_id]
    result = await run_analyst_agent(df, dataset_id)
    
    return result


@app.post("/narrate/{dataset_id}")
async def narrate_dataset(dataset_id: str):
    """Run Analyst + Narrator Agents to generate narrative report."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    from agents import run_analyst_agent, run_narrator_agent
    
    df = datasets[dataset_id]
    
    # Step 1: Run analysis
    analysis_report = await run_analyst_agent(df, dataset_id)
    
    # Step 2: Generate narratives
    narrative_report = await run_narrator_agent(analysis_report)
    
    return {
        "dataset_id": dataset_id,
        "analysis": analysis_report,
        "narrative": narrative_report
    }


@app.post("/narrate-report/{report_id}")
async def narrate_pdf_report(report_id: str):
    """Generate narrative from an existing PDF analysis report."""
    if report_id not in pdf_reports:
        raise HTTPException(404, "Report not found")
    
    from agents import run_narrator_agent
    
    analysis_report = pdf_reports[report_id]
    narrative_report = await run_narrator_agent(analysis_report)
    
    return {
        "report_id": report_id,
        "analysis": analysis_report,
        "narrative": narrative_report
    }


# ============ Full Pipeline: Generate PPTX ============

# Store generated reports
generated_reports: dict[str, dict] = {}


@app.post("/generate/{dataset_id}")
async def generate_report(dataset_id: str):
    """Full pipeline: Analyze → Narrate → Generate PPTX.
    
    Returns a job_id to download the generated presentation.
    """
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    from agents import run_analyst_agent, run_narrator_agent
    from report_builder import build_report
    
    df = datasets[dataset_id]
    
    # Step 1: Analyze
    analysis = await run_analyst_agent(df, dataset_id)
    
    # Step 2: Narrate
    narrative = await run_narrator_agent(analysis)
    
    # Step 3: Build PPTX (guaranteed 6 slides)
    report_result = await build_report(analysis, narrative)
    
    # Store for download
    job_id = report_result.get("job_id", uuid4().hex[:8])
    generated_reports[job_id] = {
        "dataset_id": dataset_id,
        "output_path": report_result["output_path"],
        "slides_created": report_result["slides_created"],
        "template": report_result.get("template", "corporate_blue"),
        "status": report_result["status"]
    }
    
    return {
        "job_id": job_id,
        "status": "complete",
        "template": report_result.get("template"),
        "slides_created": report_result["slides_created"],
        "download_url": f"/download/{job_id}"
    }


@app.get("/download/{job_id}")
async def download_report(job_id: str):
    """Download the generated PPTX report."""
    if job_id not in generated_reports:
        raise HTTPException(404, "Report not found")
    
    report = generated_reports[job_id]
    filepath = report["output_path"]
    
    return FileResponse(
        path=filepath,
        filename=f"report_{job_id}.pptx",
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a report generation job."""
    if job_id not in generated_reports:
        raise HTTPException(404, "Job not found")
    
    return generated_reports[job_id]
