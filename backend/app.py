import streamlit as st
from agent.graph import build_graph
from rag.ingest import IngestDocuments
import uuid
import os
from pathlib import Path

st.set_page_config(page_title="Contract Intelligence Agent", layout="wide")
graph = build_graph()

# ── Ensure data directory exists at startup ──────────────────────────
BASE_DIR       = Path(__file__).parent          # directory  where app.py lives
CONTRACTS_DIR  = BASE_DIR / "data" / "contracts"
CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Sidebar: upload + contract selector ─────────────────────────────
with st.sidebar:
    st.header("Contracts")

    # Contract name input ABOVE the uploader
    contract_name = st.text_input(
        "Contract name (optional)",
        placeholder="e.g. Acme SaaS Agreement 2025"
    )

    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if uploaded:
        already_ingested = st.session_state.get("ingested_file") == uploaded.name

        if not already_ingested:
            cid          = str(uuid.uuid4())[:8]
            display_name = contract_name.strip() if contract_name.strip() else uploaded.name
            safe_fname   = f"{cid}_{uploaded.name}"                  # uuid prefix avoids collisions
            path         = CONTRACTS_DIR / safe_fname                 # absolute path — no relative issues

            with open(path, "wb") as f:
                f.write(uploaded.read())

            with st.spinner(f"Ingesting {display_name}..."):
                ingest = IngestDocuments()
                ingest.ingest_pdfs(str(path), cid)

            # Persist both the id and the human-readable name
            st.session_state["contract_id"]   = cid
            st.session_state["contract_name"] = display_name
            st.session_state["messages"]      = []   # reset chat on new upload
            st.session_state["retrieved"]     = []
            st.session_state["risk_flags"]    = []
            st.success(f"Ingested: {display_name}")

    # Show active contract name if one is loaded
    if "contract_name" in st.session_state:
        st.info(f"Active: **{st.session_state['contract_name']}**")

    # Risk flags panel
    if st.session_state.get("risk_flags"):
        st.divider()
        st.subheader("Risk flags")
        for flag in st.session_state.risk_flags:
            color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[flag.severity]
            with st.expander(
                f"{color} {flag.category.replace('_', ' ').title()} — {flag.section_reference}"
            ):
                st.caption(f"**Clause:** _{flag.clause_text}_")
                st.write(flag.plain_english)

# ── Main: chat + source cards ────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Ask your contract")

    if "contract_id" not in st.session_state:
        st.info("Upload a contract PDF in the sidebar to get started.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("confidence"):
                st.caption(f"Confidence: {msg['confidence']}")

    query = st.chat_input("Ask a question or type 'flag all risks'")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            result = graph.invoke({
                "query":       query,
                "contract_id": st.session_state["contract_id"],
                "messages":    st.session_state.messages,
                "retrieved":   [],
                "risk_flags":  [],
                "intent":      "",
                "answer":      None,
                "confidence":  None
            })

        answer = result.get("answer", "")
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    answer,
            "confidence": result.get("confidence")
        })
        st.session_state["retrieved"]  = result.get("retrieved", [])
        st.session_state["risk_flags"] = result.get("risk_flags", [])
        st.rerun()

with col2:
    st.header("Sources")
    for chunk in st.session_state.get("retrieved", []):
        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[chunk.confidence]
        with st.expander(f"{badge} p.{chunk.page} — score {chunk.score:.2f}"):
            st.caption(chunk.source_file.split("/")[-1])
            st.write(chunk.text[:300] + "...")