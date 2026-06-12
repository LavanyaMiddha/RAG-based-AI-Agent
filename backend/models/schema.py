from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class RetrievedChunk(BaseModel):
    text: str
    source_file: str
    section: Optional[str] = None   # from h1/h2/h3 markdown headers
    score: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

class ContractAnswer(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]
    grounded: bool          # True if answer is fully supported by chunks

class RiskFlag(BaseModel):
    clause_text: str
    category: Literal[
        "liability", "ip_ownership", "termination",
        "data_privacy", "indemnification", "auto_renewal",
        "jurisdiction", "confidentiality"
    ]
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    plain_english: str
    section_reference: str

class RiskFlagResponse(BaseModel):
    flags: List[RiskFlag]
    overall_risk: Literal["HIGH", "MEDIUM", "LOW"]
    summary: str