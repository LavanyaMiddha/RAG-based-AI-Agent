from pydantic import BaseModel, Field
from typing import List, Literal

class RetrievedChunk(BaseModel):
    text: str
    source_file: str
    score: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

class ContractAnswer(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]
    grounded: bool          # True if answer is fully supported by chunks

class RiskFlag(BaseModel):
    clause_text: str        # exact quoted language from contract
    category: Literal[
        "liability", "ip_ownership", "termination",
        "data_privacy", "indemnification", "auto_renewal",
        "jurisdiction", "confidentiality"
    ]
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    plain_english: str      # 1–2 sentence explanation for non-lawyers
    section_reference: str  # e.g. "Section 4.2"

class RiskFlagResponse(BaseModel):
    flags: List[RiskFlag]
    overall_risk: Literal["HIGH", "MEDIUM", "LOW"]
    summary: str