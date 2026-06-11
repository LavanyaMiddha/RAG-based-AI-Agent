import streamlit as st
from rag.ingest import IngestDocuments
from pathlib import Path


CONTRACTS_DIR  = "C:/RAG-based-AI-Agent/data/contract_files"
col1, col2= st.columns(2)

col1.header("Upload Documents")
form1 = col1.form(key="ingest_documents")
contract_name = form1.text_input(
        "Contract name (required)",
        placeholder="e.g. Acme SaaS Agreement 2025"
    )

uploaded = form1.file_uploader("Upload PDF", type="pdf")

submit_button = form1.form_submit_button("Ingest Document")

if submit_button:
    if not uploaded or not contract_name:
        form1.error("Please fill out all required fields marked with an asterisk (*).")
        form1.stop()
    if uploaded:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / uploaded.name
        form1.write(file_path)

    with form1.spinner(f"Ingesting Document with name {contract_name}..."):
        path = CONTRACTS_DIR  +f"/{uploaded.name}"
        ingest = IngestDocuments()
        ingest.ingest_pdfs(str(path), contract_name)
        form1.success(f"Ingested document {contract_name}")


col2.header("Chat with AI")
col2.selectbox("List of Available Contracts", ('docA', 'docB'))
