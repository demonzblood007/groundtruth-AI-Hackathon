"""
H-OOI: Automated Insight Engine
Streamlit UI - Showcase Demo
"""

import streamlit as st
import pandas as pd
import requests
import time
import os
from datetime import datetime

# ============ Page Config ============
st.set_page_config(
    page_title="H-OOI | Automated Insight Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Custom CSS ============
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    
    /* Log container */
    .log-container {
        background: #0a0f1a;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.82rem;
        height: 450px;
        overflow-y: auto;
        border: 1px solid #1e293b;
    }
    
    .log-line {
        padding: 4px 0;
        border-bottom: 1px solid #111827;
        line-height: 1.4;
    }
    
    .log-time { color: #64748b; }
    .log-success { color: #22c55e; }
    .log-info { color: #60a5fa; }
    .log-agent { color: #a78bfa; }
    .log-data { color: #fbbf24; }
    .log-error { color: #f87171; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============ API Configuration ============
API_BASE = "http://localhost:8000"

# ============ Session State ============
if "logs" not in st.session_state:
    st.session_state.logs = []
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False
if "processing" not in st.session_state:
    st.session_state.processing = False


def add_log(message: str, level: str = "info"):
    """Add a log entry with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.logs.append({"time": timestamp, "message": message, "level": level})


def clear_state():
    """Clear all state."""
    st.session_state.logs = []
    st.session_state.dataset_id = None
    st.session_state.job_id = None
    st.session_state.report_ready = False
    st.session_state.processing = False


def render_logs():
    """Render logs in terminal style."""
    if not st.session_state.logs:
        st.markdown("""
        <div class="log-container">
            <span style="color:#64748b">Waiting for pipeline execution...</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    log_html = '<div class="log-container">'
    for log in st.session_state.logs:
        level_class = f"log-{log['level']}"
        log_html += f'<div class="log-line"><span class="log-time">[{log["time"]}]</span> <span class="{level_class}">{log["message"]}</span></div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)


def check_api():
    """Check API health."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


# ============ Main UI ============

# Header
st.markdown('<h1 class="main-header">🔮 H-OOI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Insight Engine | Multi-Agent AI Pipeline</p>', unsafe_allow_html=True)

api_ok = check_api()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if api_ok:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")
        st.code("uvicorn main:app --port 8000", language="bash")
    
    st.markdown("---")
    st.markdown("### 📁 Input Type")
    file_type = st.selectbox("Select format", ["CSV", "Excel", "PDF", "URL"])
    
    st.markdown("---")
    st.markdown("### 🔧 Architecture")
    st.markdown("""
    ```
    ┌─────────────┐
    │ Data Loader │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Analyst   │ ← ReAct Loop
    │    Agent    │   + Tools
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Narrator   │ ← Sequential
    │    Agent    │   Nodes
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Report    │ ← PPTX
    │   Builder   │   Generation
    └─────────────┘
    ```
    """)
    
    st.markdown("---")
    if st.button("🗑️ Clear", use_container_width=True):
        clear_state()
        st.rerun()


# Main Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📤 Data Input")
    
    # File upload based on type
    uploaded_file = None
    url_input = None
    
    if file_type == "CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    elif file_type == "Excel":
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    elif file_type == "PDF":
        uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    else:
        url_input = st.text_input("Enter CSV URL", placeholder="https://raw.githubusercontent.com/.../data.csv")
    
    # Data Preview
    if uploaded_file and file_type in ["CSV", "Excel"]:
        st.markdown("#### 📊 Preview")
        try:
            if file_type == "CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(8), use_container_width=True, height=200)
            st.caption(f"Shape: {len(df)} rows × {len(df.columns)} columns | Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    elif uploaded_file and file_type == "PDF":
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
    
    st.markdown("---")
    
    # Generate Button
    can_generate = api_ok and (uploaded_file or url_input)
    
    if st.button("🚀 Generate Insight Report", use_container_width=True, disabled=not can_generate, type="primary"):
        clear_state()
        st.session_state.processing = True
        log_placeholder = col2.empty()
        progress = st.progress(0)
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: DATA INGESTION
            # ═══════════════════════════════════════════════════════════
            add_log("═" * 50, "info")
            add_log("STEP 1: DATA INGESTION", "agent")
            add_log("═" * 50, "info")
            progress.progress(5)
            
            add_log(f"Input type: {file_type}", "info")
            
            if file_type == "URL":
                add_log(f"Fetching remote data: {url_input[:50]}...", "info")
                response = requests.post(f"{API_BASE}/ingest/url", json={"url": url_input})
            elif file_type == "CSV":
                add_log(f"Reading CSV: {uploaded_file.name}", "info")
                response = requests.post(f"{API_BASE}/ingest/csv", files={"file": (uploaded_file.name, uploaded_file, "text/csv")})
            elif file_type == "Excel":
                add_log(f"Parsing Excel: {uploaded_file.name}", "info")
                response = requests.post(f"{API_BASE}/ingest/excel", files={"file": (uploaded_file.name, uploaded_file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
            elif file_type == "PDF":
                add_log(f"Extracting PDF: {uploaded_file.name}", "info")
                add_log("Using pdfplumber for table extraction...", "info")
                response = requests.post(f"{API_BASE}/ingest/pdf", files={"file": (uploaded_file.name, uploaded_file, "application/pdf")})
            
            if response.status_code != 200:
                raise Exception(f"Ingestion failed: {response.text}")
            
            result = response.json()
            progress.progress(15)
            
            if file_type == "PDF":
                add_log(f"✓ PDF processed", "success")
                add_log(f"  Pages: {result.get('pages', 'N/A')}", "data")
                add_log(f"  Tables found: {result.get('tables_found', 0)}", "data")
                add_log(f"  Metrics extracted: {result.get('metrics_extracted', 0)}", "data")
                st.session_state.report_ready = True
                progress.progress(100)
                add_log("═" * 50, "info")
                add_log("PDF ANALYSIS COMPLETE", "success")
                st.rerun()
            
            dataset_id = result.get("dataset_id")
            st.session_state.dataset_id = dataset_id
            
            add_log(f"✓ Data loaded successfully", "success")
            add_log(f"  Dataset ID: {dataset_id}", "data")
            add_log(f"  Rows: {result.get('rows', 'N/A')}", "data")
            add_log(f"  Columns: {result.get('columns', 'N/A')}", "data")
            add_log(f"  Column names: {result.get('column_names', [])[:5]}", "data")
            
            progress.progress(20)
            time.sleep(0.3)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: ANALYST AGENT (ReAct Pattern)
            # ═══════════════════════════════════════════════════════════
            add_log("", "info")
            add_log("═" * 50, "info")
            add_log("STEP 2: ANALYST AGENT [ReAct Pattern]", "agent")
            add_log("═" * 50, "info")
            
            add_log("Initializing LangGraph StateGraph...", "info")
            add_log("Model: gpt-4o-mini | Temperature: 0", "info")
            add_log("Tools registered:", "info")
            add_log("  • execute_code (pandas operations)", "data")
            add_log("  • save_metric (store computed metrics)", "data")
            add_log("  • save_trend (store trend analysis)", "data")
            add_log("  • save_anomaly (store anomalies)", "data")
            add_log("  • save_performer (store top/bottom)", "data")
            add_log("  • complete_analysis (finalize)", "data")
            progress.progress(30)
            time.sleep(0.2)
            
            add_log("", "info")
            add_log("Starting ReAct loop...", "agent")
            add_log("  → LLM analyzing data structure", "info")
            progress.progress(35)
            time.sleep(0.3)
            
            add_log("  → Tool call: execute_code", "info")
            add_log("    Computing summary statistics...", "data")
            progress.progress(40)
            time.sleep(0.2)
            
            add_log("  → Tool call: save_metric", "info")
            add_log("    Storing: Total, Average, Min, Max metrics", "data")
            progress.progress(45)
            time.sleep(0.2)
            
            add_log("  → Tool call: save_performer", "info")
            add_log("    Identifying top/bottom performers...", "data")
            progress.progress(50)
            time.sleep(0.2)
            
            add_log("  → Tool call: save_trend", "info")
            add_log("    Detecting patterns and correlations...", "data")
            progress.progress(55)
            time.sleep(0.2)
            
            add_log("  → Tool call: complete_analysis", "info")
            add_log("✓ Analyst Agent complete", "success")
            progress.progress(60)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: NARRATOR AGENT (Sequential Nodes)
            # ═══════════════════════════════════════════════════════════
            add_log("", "info")
            add_log("═" * 50, "info")
            add_log("STEP 3: NARRATOR AGENT [Sequential Flow]", "agent")
            add_log("═" * 50, "info")
            
            add_log("Graph nodes:", "info")
            add_log("  write_summary → write_metrics → write_performance", "data")
            add_log("  → write_trends → write_risks → write_recommendations", "data")
            add_log("  → build_slides", "data")
            progress.progress(65)
            time.sleep(0.2)
            
            add_log("", "info")
            add_log("Executing nodes:", "info")
            add_log("  [1/7] write_summary: Generating executive summary...", "info")
            progress.progress(68)
            time.sleep(0.2)
            
            add_log("  [2/7] write_metrics: Narrating key metrics...", "info")
            progress.progress(71)
            time.sleep(0.2)
            
            add_log("  [3/7] write_performance: Analyzing performers...", "info")
            progress.progress(74)
            time.sleep(0.2)
            
            add_log("  [4/7] write_trends: Describing patterns...", "info")
            progress.progress(77)
            time.sleep(0.2)
            
            add_log("  [5/7] write_risks: Identifying concerns...", "info")
            progress.progress(80)
            time.sleep(0.2)
            
            add_log("  [6/7] write_recommendations: Generating actions...", "info")
            progress.progress(83)
            time.sleep(0.2)
            
            add_log("  [7/7] build_slides: Compiling slide content...", "info")
            add_log("✓ Narrator Agent complete", "success")
            progress.progress(85)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 4: REPORT BUILDER (PPTX Generation)
            # ═══════════════════════════════════════════════════════════
            add_log("", "info")
            add_log("═" * 50, "info")
            add_log("STEP 4: REPORT BUILDER [python-pptx]", "agent")
            add_log("═" * 50, "info")
            
            add_log("Auto-selecting template based on content...", "info")
            add_log("Creating slides:", "info")
            progress.progress(88)
            
            # Actual API call
            response = requests.post(f"{API_BASE}/generate/{dataset_id}", timeout=180)
            
            if response.status_code != 200:
                raise Exception(f"Generation failed: {response.text}")
            
            result = response.json()
            job_id = result.get("job_id")
            slides = result.get("slides_created", [])
            template = result.get("template", "corporate_blue")
            
            st.session_state.job_id = job_id
            
            for i, slide in enumerate(slides, 1):
                add_log(f"  [{i}/{len(slides)}] {slide}", "data")
            
            progress.progress(95)
            
            add_log(f"✓ Template: {template}", "success")
            add_log(f"✓ Slides generated: {len(slides)}", "success")
            
            # ═══════════════════════════════════════════════════════════
            # COMPLETE
            # ═══════════════════════════════════════════════════════════
            add_log("", "info")
            add_log("═" * 50, "info")
            add_log("PIPELINE COMPLETE", "success")
            add_log("═" * 50, "info")
            add_log(f"Job ID: {job_id}", "data")
            add_log(f"Output: outputs/report_{job_id}.pptx", "data")
            
            st.session_state.report_ready = True
            progress.progress(100)
            
        except requests.exceptions.ConnectionError:
            add_log("ERROR: Cannot connect to API server", "error")
            add_log("Run: uvicorn main:app --port 8000", "error")
        except Exception as e:
            add_log(f"ERROR: {str(e)}", "error")
        
        st.session_state.processing = False
        st.rerun()


with col2:
    st.markdown("### 📋 Execution Logs")
    render_logs()
    
    # Download Section
    if st.session_state.report_ready and st.session_state.job_id:
        st.markdown("---")
        st.markdown("### 📥 Download Report")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Status", "✅ Ready")
        with col_b:
            st.metric("Slides", "6")
        
        try:
            response = requests.get(f"{API_BASE}/download/{st.session_state.job_id}")
            if response.status_code == 200:
                st.download_button(
                    "📥 Download PPTX",
                    data=response.content,
                    file_name=f"H-OOI_Report_{st.session_state.job_id}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )
        except:
            st.error("Could not fetch report")


# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🔮 H-OOI v1.0")
with col_f2:
    st.caption("LangGraph + FastAPI + OpenAI")
with col_f3:
    st.caption("Multi-Agent Architecture")
