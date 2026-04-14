"""HTTP route for stateless, direct diagnosis.

POST /api/v1/diagnosis  — bypass the dialogue; infer directly from a symptom list.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import FeatureContractError, InconclusiveDiagnosisError
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResult, InconclusiveResponse
from app.services.diagnosis_service import diagnose

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])


def _get_registry(request: Request):
    """FastAPI dependency: retrieve the loaded ModelRegistry from app state."""
    return request.app.state.model_registry


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Stateless direct inference from a symptom list",
    responses={
        200: {"model": DiagnosisResult},
        422: {"description": "Unrecognised symptom name"},
        200: {"model": InconclusiveResponse, "description": "Inconclusive result"},
    },
)
async def post_diagnosis(
    body: DiagnosisRequest,
    registry=Depends(_get_registry),
) -> JSONResponse:
    """Run the ML pipeline directly on the provided symptom list.

    Args:
        body:     Validated request containing a list of symptom strings.
        registry: Injected model registry.

    Returns:
        ``DiagnosisResult`` on success, or ``InconclusiveResponse`` when
        confidence is below the configured threshold.

    Raises:
        422: If a symptom name cannot be matched in the training vocabulary.
    """
    try:
        result = diagnose(body.symptoms, registry)
        return JSONResponse(
            content=result.model_dump(),
            status_code=status.HTTP_200_OK,
        )
    except FeatureContractError as exc:
        return JSONResponse(
            content={"error": "unknown_symptom", "message": str(exc)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    except InconclusiveDiagnosisError:
        return JSONResponse(
            content=InconclusiveResponse().model_dump(),
            status_code=status.HTTP_200_OK,
        )
