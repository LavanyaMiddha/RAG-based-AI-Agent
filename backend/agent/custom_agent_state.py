from typing import TypedDict, List, Optional
from models.schema import RetrievedChunk, RiskFlag

class AgentState(TypedDict):
    messages:      list        # full conversation history
    query:         str         # current user query
    contract_id:   str         # which contract is active
    intent:        str         # "qa" | "risk_flag" | "extract_clause"
    retrieved:     List[RetrievedChunk]
    answer:        Optional[str]
    risk_flags:    List[RiskFlag]
    confidence:    Optional[str]