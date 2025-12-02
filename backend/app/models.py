from typing import Literal, Optional

from pydantic import BaseModel, Field

from .prompts import TaskType
from .safety import SafetyLevel


class AskRequest(BaseModel):
    """
    Request body for the /api/ask endpoint.

    task_type:
        - "medical_qa": general medical Q&A
        - "diagnosis": diagnostic reasoning (CoT)
        - "drug": medication-related questions
        - "lab": lab / imaging explanation
        - "education": health education and lifestyle advice

    temperature:
        Optional override for the model sampling temperature.
    max_tokens:
        Optional override for the maximum number of tokens to generate.
    """
    question: str = Field(..., description="User's medical question or symptom description.")
    task_type: TaskType = Field(
        "medical_qa",
        description="Type of task: medical_qa, diagnosis, drug, lab, education.",
    )
    language: Literal["zh", "en"] = Field(
        "zh",
        description="Answer language: 'zh' for Chinese, 'en' for English.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Optional sampling temperature for the LLM.",
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Optional maximum number of tokens to generate.",
    )


class SafetyResult(BaseModel):
    """
    Result of the keyword-based safety layer.
    """
    level: SafetyLevel
    message: str


class SeverityInfo(BaseModel):
    """
    Structured severity information parsed from the model's JSON block.

    severity:
        - "low": mild / self-care level
        - "moderate": should see a doctor soon
        - "high": potential emergency
    """
    severity: Optional[Literal["low", "moderate", "high"]] = None
    recommended_action: Optional[str] = None
    time_window: Optional[str] = None
    risk_notes: Optional[str] = None


class AskResponse(BaseModel):
    """
    Response body for the /api/ask endpoint.
    """
    answer: str
    safety: SafetyResult
    severity: Optional[SeverityInfo] = Field(
        None,
        description="Model-based structured risk assessment (parsed from JSON).",
    )
    # Optional: for debugging or report-writing, you can inspect the prompt used.
    used_prompt: Optional[str] = Field(
        None,
        description="The full prompt sent to the LLM (for debugging / analysis).",
    )
