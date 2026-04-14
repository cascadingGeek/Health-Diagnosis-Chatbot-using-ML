"""Pydantic schemas for direct (stateless) diagnosis requests and responses."""

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
    """Structured result returned by the ML inference pipeline.

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
