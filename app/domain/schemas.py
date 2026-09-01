from typing import Literal

from pydantic import BaseModel, Field


class CaseInput(BaseModel):
    case_id: str | None = Field(default=None, max_length=128)
    text: str = Field(min_length=3, max_length=20_000)


class Candidate(BaseModel):
    category: str
    score: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    case_id: str | None
    category: str
    candidates: list[Candidate]
    stage: Literal["tfidf_xgboost", "deberta", "abstained"]
    confidence: float = Field(ge=0, le=1)
    confidence_semantics: str
    routing_reason: str | None
    taxonomy_version: str
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    cases: list[CaseInput] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    taxonomy_version: str
    model_version: str
    first_stage: str
    ambiguity_path: str
    artifacts_loaded: bool
