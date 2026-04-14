"""Typed exception hierarchy for the application.

Raise these instead of plain Exception so callers can distinguish error kinds
without parsing message strings.
"""


class AppError(Exception):
    """Base class for all application errors."""


# ── ML / artifact errors ──────────────────────────────────────────────────────


class ArtifactLoadError(AppError):
    """Raised when a required ML artifact cannot be loaded from disk."""


class FeatureContractError(AppError):
    """Raised when a symptom name is not in the known symptom list.

    Args:
        symptom: The unrecognised symptom string.
    """

    def __init__(self, symptom: str) -> None:
        self.symptom = symptom
        super().__init__(f"Unknown symptom: '{symptom}'")


class ModelNotInitialisedError(AppError):
    """Raised when the model registry is accessed before it has been loaded."""


# ── Dialogue / session errors ─────────────────────────────────────────────────


class SessionNotFoundError(AppError):
    """Raised when a chat session UUID cannot be found in the database.

    Args:
        session_id: String representation of the missing UUID.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Chat session not found: '{session_id}'")


class InvalidStateTransitionError(AppError):
    """Raised when the dialogue state machine receives an unexpected transition.

    Args:
        current: Current state string.
        attempted: Attempted target state string.
    """

    def __init__(self, current: str, attempted: str) -> None:
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Cannot transition from state '{current}' to '{attempted}'"
        )


# ── Inference errors ──────────────────────────────────────────────────────────


class InconclusiveDiagnosisError(AppError):
    """Raised when model confidence is below the configured threshold."""
