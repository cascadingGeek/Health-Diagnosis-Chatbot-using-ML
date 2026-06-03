"""Pydantic schemas for direct (stateless) diagnosis requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


class DiagnosisRequest(BaseModel):
    """Direct inference request — bypasses the dialogue state machine.

    Attributes:
        symptoms: List of confirmed symptom strings.
    """

    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="List of symptom names (must match training vocabulary).",
    )


class DiagnosisResult(BaseModel):
    """Structured result returned by the ML inference pipeline (direct endpoint).

    Attributes:
        disease:      Predicted disease name.
        confidence:   Model confidence in [0, 1].
        description:  Plain-English description of the disease.
        precautions:  Recommended precautionary actions.
    """

    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    precautions: list[str]


class LLMDiagnosisResult(BaseModel):
    """Full diagnosis result returned by the three-layer hybrid pipeline.

    Field names use the same names as DiagnosisResult where possible so the
    frontend can access disease/confidence/precautions without changes.

    Attributes:
        disease:            Predicted disease name (None if inconclusive).
        confidence:         Dampened display confidence <= 0.89 (None if inconclusive).
        description:        Plain-English explanation from Layer 3.
        precautions:        Actionable steps recommended by Layer 3.
        is_plausible:       Whether Layer 3 accepted the Decision Tree prediction.
        urgency:            Severity tier: "mild" | "moderate" | "urgent".
        when_to_see_doctor: Specific guidance on when to seek care.
        disclaimer:         Mandatory medical disclaimer text.
    """

    disease: Optional[str] = None
    confidence: Optional[float] = None
    description: str = ""
    precautions: list[str] = []
    is_plausible: bool = False
    urgency: str = "moderate"
    when_to_see_doctor: str = ""
    disclaimer: str = ""


class InconclusiveResponse(BaseModel):
    """Returned when confidence is below the configured threshold.

    Attributes:
        error:   Fixed string ``"inconclusive"``.
        message: Human-readable explanation.
    """

    error: str = "inconclusive"
    message: str = (
        "Symptoms are inconclusive. Please consult a qualified medical professional."
    )
