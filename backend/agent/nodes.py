import json
import re
from config import GEMINI
from rag.retriever import Retriever
from models.schema import RiskFlag
from agent.custom_agent_state import AgentState


def router_node(state: AgentState) -> AgentState:
    """Single LLM call — classify intent. No retrieval here."""
    prompt = f"""Classify the user's intent for a contract review tool.
Query: {state['query']}
Return ONLY one of: qa | risk_flag
- qa: user asks a question about contract terms
- risk_flag: user wants risks, red flags, or dangerous clauses identified"""

    resp = GEMINI.generate_content(prompt)
    state["intent"] = resp.text.strip().lower()
    return state


def rag_qa_node(state: AgentState) -> AgentState:
    query       = state["query"]
    contract_id = state["contract_id"]

    retriever = Retriever()
    chunks    = retriever.retrieve_documents(
        query, {"contract_id": {"$eq": contract_id}}
    )
    state["retrieved"] = chunks

    if not chunks:
        state["answer"]     = "No relevant clauses found in this contract."
        state["confidence"] = "LOW"
        return state

    context = "\n\n---\n\n".join([c.text for c in chunks])
    prompt  = f"""You are a contract analysis assistant. Answer ONLY using the provided clauses.
If the answer is not in the clauses, say "Not found in this contract."

CONTRACT CLAUSES:
{context}

QUESTION: {query}

Answer concisely. Quote the relevant clause text in your answer."""

    state["answer"]     = GEMINI.generate_content(prompt).text
    state["confidence"] = max(
        (c.confidence for c in chunks),
        key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x)
    )
    return state


def risk_flag_node(state: AgentState) -> AgentState:
    contract_id = state["contract_id"]

    retriever = Retriever()
    chunks    = retriever.retrieve_documents(
        "liability termination IP data privacy indemnification renewal",
        {"contract_id": {"$eq": contract_id}}
    )
    state["retrieved"] = chunks

    if not chunks:
        state["risk_flags"] = []
        state["answer"]     = "No clauses found to analyze for this contract."
        return state

    context = "\n\n---\n\n".join([c.text for c in chunks])
    prompt  = f"""You are a contract risk analyst. Review these contract clauses and identify all risky terms.

For each risk, output a JSON object with:
- clause_text: exact quoted text from the contract
- category: one of liability|ip_ownership|termination|data_privacy|indemnification|auto_renewal|jurisdiction|confidentiality
- severity: HIGH|MEDIUM|LOW
- plain_english: 1-2 sentence explanation a non-lawyer can understand
- section_reference: the section number if visible

CLAUSES:
{context}

Return a JSON array of risk objects. Only return JSON, no preamble."""

    raw      = GEMINI.generate_content(prompt).text
    json_str = re.search(r'\[.*\]', raw, re.DOTALL)

    if json_str:
        try:
            flags_data          = json.loads(json_str.group())
            state["risk_flags"] = [RiskFlag(**f) for f in flags_data]
        except (json.JSONDecodeError, TypeError):
            state["risk_flags"] = []
    else:
        state["risk_flags"] = []

    # ── Build a chat-facing summary ──────────────────────────────────
    n    = len(state["risk_flags"])
    high = sum(1 for f in state["risk_flags"] if f.severity == "HIGH")
    med  = sum(1 for f in state["risk_flags"] if f.severity == "MEDIUM")

    if n == 0:
        state["answer"] = "No significant risks identified in this contract."
    else:
        state["answer"] = (
            f"Found **{n} risk flags** — {high} high severity, {med} medium severity. "
            f"See the Risk flags panel in the sidebar for the full breakdown."
        )

    return state