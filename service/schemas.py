# service/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class TransactionInput(BaseModel):
    description: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    top_features: List[str]

class FeedbackInput(BaseModel):
    description: str
    predicted_category: str
    predicted_confidence: float
    corrected_category: str
    user_comment: Optional[str] = None
