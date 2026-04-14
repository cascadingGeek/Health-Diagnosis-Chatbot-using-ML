"""Integration tests for the chat API.

These tests use an in-memory SQLite database (via aiosqlite) and a stub
ModelRegistry so no real PostgreSQL or trained model is required.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.base import Base
from app.database.session import get_async_session
from app.main import app
from app.ml.model_registry import ModelRegistry

# ── SQLite in-memory engine for tests ─────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite://"


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with factory() as session:
        yield session


# ── Stub ModelRegistry ────────────────────────────────────────────────────────

SYMPTOM_LIST = [f"symptom_{i}" for i in range(132)]


def _make_stub_registry() -> ModelRegistry:
    """Return a ModelRegistry with a mock sklearn model."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get_symptom_list.return_value = SYMPTOM_LIST

    proba = np.zeros((1, 41))
    proba[0, 0] = 0.95  # high confidence on first class

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = proba
    mock_model.n_features_in_ = 132

    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Fungal infection"]

    registry.get_model.return_value = mock_model
    registry.get_label_encoder.return_value = mock_le
    registry.get_disease_info.return_value = {
        "Fungal infection": {
            "description": "A fungal skin condition.",
            "precautions": ["keep skin dry"],
        }
    }
    registry.get_metadata.return_value = {
        "symptom_list": SYMPTOM_LIST,
        "class_names": ["Fungal infection"] + [f"Disease_{i}" for i in range(40)],
        "model_version": "test",
        "accuracy": 1.0,
    }
    return registry


# ── App fixture with overridden dependencies ──────────────────────────────────


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession):
    stub_registry = _make_stub_registry()
    app.state.model_registry = stub_registry

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_async_session] = _override_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestChatMessage:
    async def test_new_session_greeting(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["state"] == "COLLECTING"
        assert "symptom" in data["bot_message"].lower() or "hello" in data["bot_message"].lower()

    async def test_session_id_persists_across_turns(self, client: AsyncClient):
        resp1 = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        session_id = resp1.json()["session_id"]

        resp2 = await client.post(
            "/api/v1/chat/message",
            json={"session_id": session_id, "message": "fever"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

    async def test_unknown_session_returns_404(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": str(uuid.uuid4()), "message": "hello"},
        )
        assert resp.status_code == 404

    async def test_response_schema_fields(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        data = resp.json()
        required_keys = {"session_id", "state", "bot_message", "confirmed_symptoms"}
        assert required_keys.issubset(data.keys())


class TestGetSession:
    async def test_get_existing_session(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        session_id = resp.json()["session_id"]

        get_resp = await client.get(f"/api/v1/chat/session/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert "state" in data
        assert "confirmed_symptoms" in data

    async def test_get_nonexistent_session_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/api/v1/chat/session/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestDiagnosisEndpoint:
    async def test_direct_diagnosis_with_known_symptoms(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/diagnosis",
            json={"symptoms": ["symptom_0", "symptom_1", "symptom_2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "disease" in data
        assert "confidence" in data

    async def test_unknown_symptom_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/diagnosis",
            json={"symptoms": ["completely_made_up_xyz_symptom_9999"]},
        )
        assert resp.status_code == 422


class TestFeedbackEndpoint:
    async def test_submit_feedback_for_existing_session(self, client: AsyncClient):
        # Create a session first
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        session_id = resp.json()["session_id"]

        fb_resp = await client.post(
            "/api/v1/feedback",
            json={"session_id": session_id, "rating": 4, "comment": "Great!"},
        )
        assert fb_resp.status_code == 201
        assert fb_resp.json()["success"] is True

    async def test_feedback_for_nonexistent_session_returns_404(
        self, client: AsyncClient
    ):
        resp = await client.post(
            "/api/v1/feedback",
            json={"session_id": str(uuid.uuid4()), "rating": 3},
        )
        assert resp.status_code == 404

    async def test_rating_out_of_range_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat/message",
            json={"session_id": None, "message": ""},
        )
        session_id = resp.json()["session_id"]

        fb_resp = await client.post(
            "/api/v1/feedback",
            json={"session_id": session_id, "rating": 6},
        )
        assert fb_resp.status_code == 422
