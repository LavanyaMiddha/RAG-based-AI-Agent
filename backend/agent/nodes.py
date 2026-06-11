import google.generativeai as genai
from rag.retriever import Retriever
# from cache.semantic_cache import check_cache, write_cache
from models.schema import RiskFlag
from agent.custom_agent_state import AgentState
from dotenv import load_dotenv

load_dotenv()

GEMINI = genai.GenerativeModel("gemini-2.0-flash")

# ── Router ──────────────────────────────────────────────────────────
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

# ── RAG Q&A ─────────────────────────────────────────────────────────
def rag_qa_node(state: AgentState) -> AgentState:
    query       = state["query"]
    contract_id = state["contract_id"]

    # 1. Semantic cache check
    # cached = check_cache(query)
    # if cached:
        # state["answer"] = cached["answer"]
    # state["retrieved"] = []
    # return state

    # 2. Retrieve
    retriever = Retriever()
    chunks = retriever.retrieve_documents(query, {"contract_id": {"$eq":contract_id}})
    state["retrieved"] = chunks

    # 3. Build grounded prompt — no hallucination
    context = "\n\n---\n\n".join([c.text for c in chunks])
    prompt = f"""You are a contract analysis assistant. Answer ONLY using the provided clauses.
    If the answer is not in the clauses, say "Not found in this contract."

    CONTRACT CLAUSES:
    {context}

    QUESTION: {query}

    Answer concisely. Quote the relevant clause text in your answer."""

    # 4. Structured output on a separate model instance (avoids Gemini conflict)
    raw = GEMINI.generate_content(prompt).text

    # 5. Cache and return
    # write_cache(query, raw, [c.dict() for c in chunks])
    state["answer"] = raw
    state["confidence"] = max((c.confidence for c in chunks),
                               key=lambda x: ["LOW","MEDIUM","HIGH"].index(x))
    return state

# ── Risk Flag ────────────────────────────────────────────────────────
def risk_flag_node(state: AgentState) -> AgentState:
    contract_id = state["contract_id"]

    # Retrieve broad set of chunks — want full contract coverage
    retriever = Retriever()
    chunks = retriever.retrieve_documents("liability termination IP data privacy indemnification renewal",
                      {"contract_id": {"$eq":contract_id}})
    state["retrieved"] = chunks
    context = "\n\n---\n\n".join([c.text for c in chunks])

    prompt = f"""You are a contract risk analyst. Review these contract clauses and identify all risky terms.

        For each risk, output a JSON object with:
        - clause_text: exact quoted text from the contract
        - category: one of liability|ip_ownership|termination|data_privacy|indemnification|auto_renewal|jurisdiction|confidentiality
        - severity: HIGH|MEDIUM|LOW
        - plain_english: 1-2 sentence explanation a non-lawyer can understand
        - section_reference: the section number if visible

        CLAUSES:
        {context}

        Return a JSON array of risk objects. Only return JSON, no preamble."""

    raw = GEMINI.generate_content(prompt).text

    import json, re
    json_str = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_str:
        flags_data = json.loads(json_str.group())
        state["risk_flags"] = [RiskFlag(**f) for f in flags_data]

    return state