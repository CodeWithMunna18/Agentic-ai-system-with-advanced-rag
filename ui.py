# ui.py
# ─────────────────────────────────────────────────────────────
# Streamlit UI — Document Intelligence Platform
# Phase 6 (improved): inline citation hyperlinks
# ─────────────────────────────────────────────────────────────

import re
import streamlit as st
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
.citation {
    display: inline-block;
    background: #1a73e8;
    color: white !important;
    font-size: 0.65em;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    vertical-align: super;
    line-height: 1;
    margin: 0 1px;
    text-decoration: none !important;
    cursor: pointer;
}
.citation:hover { background: #1558b0; }

.source-card {
    border: 1px solid #e0e0e0;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    background: #f8f9ff;
}
.source-card.high   { border-left-color: #34a853; background: #f6fff8; }
.source-card.medium { border-left-color: #fbbc04; background: #fffdf0; }
.source-card.low    { border-left-color: #ea4335; background: #fff8f7; }

.source-num  { font-weight: 700; color: #1a73e8; font-size: 0.9em; }
.source-file { font-weight: 600; font-size: 0.95em; color: #202124; }
.source-score {
    float: right;
    font-size: 0.8em;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
}
.score-high   { background: #e6f4ea; color: #137333; }
.score-medium { background: #fef7e0; color: #a56800; }
.score-low    { background: #fce8e6; color: #c5221f; }
.source-preview {
    font-size: 0.82em;
    color: #5f6368;
    margin-top: 4px;
    line-height: 1.4;
    font-style: italic;
}
.answer-text { font-size: 1.0em; line-height: 1.75; color: #202124; }
.meta-row    { font-size: 0.78em; color: #9aa0a6; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ══════════════════════════════════════════════════════════════
# CITATION HELPERS
# ══════════════════════════════════════════════════════════════

def _score_class(score: float) -> str:
    if score > 0.7:  return "high"
    if score > 0.4:  return "medium"
    return "low"

def _anchor(filename: str) -> str:
    return re.sub(r'[^a-z0-9]', '-', filename.lower())

def render_answer_with_citations(answer: str, sources: list) -> str:
    """
    Replace Gemini's [Source N] / [Source N, M] / [Doc: file] markers
    with inline HTML superscript badge links that jump to source cards.
    """
    if not sources:
        return answer

    file_to_idx = {s["source_file"]: i + 1 for i, s in enumerate(sources)}

    # Pattern 1: [Source 1] or [Source 1, 2] or [Source 1,2,3]
    def replace_source_refs(match):
        nums = re.findall(r'\d+', match.group(1))
        badges = ""
        for n in nums:
            idx = int(n)
            if 1 <= idx <= len(sources):
                fname = sources[idx - 1]["source_file"]
                badges += (
                    f'<a class="citation" href="#{_anchor(fname)}" '
                    f'title="{fname}">{idx}</a>'
                )
        return badges

    answer = re.sub(r'\[Source\s+([\d,\s]+)\]', replace_source_refs, answer, flags=re.IGNORECASE)

    # Pattern 2: [Doc: filename.ext]
    def replace_doc_refs(match):
        fname = match.group(1).strip()
        idx   = file_to_idx.get(fname)
        if idx:
            return (f'<a class="citation" href="#{_anchor(fname)}" '
                    f'title="{fname}">{idx}</a>')
        return f'<sup title="{fname}">[{fname}]</sup>'

    answer = re.sub(r'\[Doc:\s*([^\]]+)\]', replace_doc_refs, answer, flags=re.IGNORECASE)
    return answer


def render_source_cards(sources: list) -> str:
    """Render sources as styled cards below the answer."""
    if not sources:
        return ""

    html = "<div style='margin-top:16px'><strong style='font-size:0.85em;color:#5f6368'>📚 SOURCES</strong>"
    for i, src in enumerate(sources):
        fname   = src["source_file"]
        score   = src.get("score", 0)
        preview = src.get("preview", "").replace("<", "&lt;").replace(">", "&gt;")
        sc      = _score_class(score)
        anchor  = _anchor(fname)
        # Score display: if ≤1 treat as ratio, else already a percentage
        score_pct = int(score * 100) if score <= 1.0 else int(score)

        html += f"""
        <div class="source-card {sc}" id="{anchor}">
            <span class="source-num">[{i+1}]</span>
            <span class="source-file">&nbsp;{fname}</span>
            <span class="source-score score-{sc}">{score_pct}% match</span>
            <div class="source-preview">"{preview}"</div>
        </div>"""

    html += "</div>"
    return html


def render_response(entry: dict):
    """Render assistant answer with inline citation badges + source cards."""
    sources     = entry.get("sources", [])
    answer_html = render_answer_with_citations(entry.get("answer", ""), sources)
    source_html = render_source_cards(sources)

    st.markdown(f"""
    <div class="answer-text">{answer_html}</div>
    {source_html}
    <div class="meta-row">
        Mode: <code>{entry.get('mode','hybrid')}</code> &nbsp;·&nbsp;
        {entry.get('chunks_retrieved', 0)} chunks &nbsp;·&nbsp;
        {entry.get('timestamp', '')}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════

def check_api_status():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def get_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def ask_question(question, mode, source_file=None):
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "mode": mode, "source_file": source_file},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is it running? (uvicorn app:app --reload)"}
    except Exception as e:
        return {"error": str(e)}

def trigger_ingest(docs_dir):
    try:
        r = requests.post(f"{API_BASE}/ingest", json={"documents_dir": docs_dir}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def upload_file(file_bytes: bytes, filename: str) -> dict:
    """
    POST a file to /upload as multipart form data.
    This is how browsers send file uploads — the API receives
    it as an UploadFile object via FastAPI's File() dependency.
    """
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API."}
    except requests.exceptions.HTTPError as e:
        # Parse FastAPI's error detail from the response body
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 Doc Intelligence")
    st.caption("Powered by Gemini + ChromaDB")
    st.divider()

    st.subheader("⚡ API Status")
    if st.button("Refresh Status", use_container_width=True):
        st.session_state.api_status = check_api_status()

    if st.session_state.api_status is None:
        st.session_state.api_status = check_api_status()

    status = st.session_state.api_status
    if status:
        st.success("API Online")
        st.metric("Documents", status.get("num_documents", 0))
        st.metric("Total Chunks", status.get("total_chunks", 0))
    else:
        st.error("API Offline")
        st.info("Start with:\n```\nuvicorn app:app --reload\n```")

    st.divider()

    st.subheader("📄 Documents")
    docs_info = get_documents()
    doc_list  = docs_info.get("documents", [])
    if doc_list:
        for doc in doc_list:
            st.text(f"• {doc}")
    else:
        st.caption("No documents loaded")

    st.divider()

    # ── Upload new document ────────────────────────────────────
    st.subheader("📤 Upload Document")
    st.caption("PDF, TXT, or DOCX — instantly searchable after upload")

    uploaded = st.file_uploader(
        label="Choose a file",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Show file info before uploading
        size_kb = len(uploaded.getvalue()) // 1024
        st.info(f"📄 **{uploaded.name}** ({size_kb} KB)")

        if st.button("Upload & Ingest", type="primary", use_container_width=True):
            with st.spinner(f"Processing '{uploaded.name}'..."):
                result = upload_file(uploaded.getvalue(), uploaded.name)

            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(
                    f"✅ **{result['file_name']}** ingested!\n\n"
                    f"{result['chunks_created']} chunks added."
                )
                # Refresh sidebar status + doc list
                st.session_state.api_status = check_api_status()
                st.rerun()

    st.divider()
    st.subheader("🔄 Re-ingest Folder")
    docs_dir = st.text_input("Documents folder", value="documents")
    if st.button("Re-ingest", use_container_width=True, type="secondary"):
        with st.spinner("Ingesting..."):
            result = trigger_ingest(docs_dir)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(result.get("message", "Done!"))
            st.session_state.api_status = check_api_status()

    st.divider()
    st.subheader("⚙️ Search Settings")
    search_mode = st.selectbox(
        "Mode",
        options=["hybrid", "dense", "multi_doc"],
        help="hybrid: semantic+keyword | dense: semantic only | multi_doc: cross-document",
    )
    filter_doc  = st.selectbox("Scope to document", ["All documents"] + doc_list)
    source_file = None if filter_doc == "All documents" else filter_doc

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════
st.title("📚 Document Intelligence Platform")
st.caption("Ask questions — get grounded answers with inline source citations")

# ── Chat history ───────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        render_response(entry)

# ── Chat input ─────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=(status is None),
)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            result = ask_question(question, mode=search_mode, source_file=source_file)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            entry = {
                "question":         question,
                "answer":           result["answer"],
                "sources":          result.get("sources", []),
                "mode":             result.get("mode", search_mode),
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "timestamp":        datetime.now().strftime("%H:%M:%S"),
            }
            render_response(entry)
            st.session_state.chat_history.append(entry)