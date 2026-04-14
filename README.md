# Health Symptom Diagnosis Chatbot

**Development and Evaluation of a Machine Learning Based Health Symptom Diagnosis Chatbot Using Decision Tree Algorithm**

A multi-turn conversational chatbot that collects symptoms from a user through structured yes/no dialogue, builds a 132-feature binary vector, and predicts a likely medical condition using a trained `DecisionTreeClassifier`. Built as an academic project to demonstrate applied machine learning, REST API design, and modern Python engineering practices.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Project Architecture](#project-architecture)
6. [Directory Structure](#directory-structure)
7. [Dialogue State Machine](#dialogue-state-machine)
8. [API Reference](#api-reference)
9. [Database Schema](#database-schema)
10. [ML Pipeline](#ml-pipeline)
11. [Getting Started](#getting-started)
12. [Running the Tests](#running-the-tests)
13. [Training the Model](#training-the-model)
14. [Environment Variables](#environment-variables)
15. [Academic Evaluation](#academic-evaluation)

---

## Problem Statement

Access to preliminary medical guidance is a challenge in many settings — long wait times, limited availability of physicians, and the difficulty of self-assessing vague or overlapping symptoms. Most people resort to unstructured internet searches, which are unreliable and often anxiety-inducing.

This project addresses that gap with a structured, explainable AI system that:

- Guides users through a symptom assessment via natural dialogue.
- Uses a machine learning model trained on a curated 132-symptom, 41-disease dataset to predict a likely condition.
- Returns a diagnosis with a confidence score, a disease description, and recommended precautions.
- Falls back gracefully when confidence is below a configurable threshold, directing the user to seek professional medical advice.

> **Disclaimer:** This system is for academic purposes only. It is not a substitute for professional medical diagnosis or treatment.

---

## Solution Overview

The chatbot exposes a RESTful JSON API consumed by a Next.js frontend. The core workflow is:

1. The user opens a chat session.
2. The bot asks the user to name their primary symptom.
3. The bot asks up to 10 targeted yes/no questions about related symptoms.
4. Once at least 3 symptoms are confirmed, the bot summarises them and asks for confirmation.
5. On confirmation, the ML model predicts the most likely disease.
6. The result is shown with confidence percentage, a description, and precautionary advice.
7. The user can submit a 1–5 star rating and a comment as feedback.

Fuzzy string matching (`rapidfuzz`) is used to normalise user-entered symptom names against the training vocabulary, so minor spelling variations are handled gracefully without any LLM dependency.

---

## Features

| Feature | Detail |
|---|---|
| Multi-turn dialogue | Stateful yes/no symptom collection across multiple HTTP requests |
| ML prediction | `DecisionTreeClassifier` trained on 132 symptoms → 41 diseases |
| Confidence gating | Diagnosis only shown when model confidence ≥ configurable threshold (default 60%) |
| Fuzzy symptom matching | `rapidfuzz` normalises user input; no exact-match requirement |
| Direct inference endpoint | Stateless `POST /api/v1/diagnosis` for programmatic use |
| Session persistence | Full session state (symptoms, state, result) stored in PostgreSQL |
| Feedback collection | Per-session 1–5 rating + comment stored to DB |
| Structured logging | JSON log output on stdout — compatible with any log aggregator |
| Rate limiting | In-memory sliding-window limiter (120 req/min per IP) |
| Academic evaluation notebook | Cross-validation, confusion matrix, feature importance, DT vs RF comparison |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend language** | Python 3.12+ |
| **API framework** | FastAPI 0.115+ |
| **ASGI server** | Uvicorn |
| **ML** | scikit-learn `DecisionTreeClassifier`, joblib |
| **Feature engineering** | NumPy, rapidfuzz |
| **Database** | PostgreSQL |
| **ORM** | SQLAlchemy 2.0 (async, asyncpg driver) |
| **Migrations** | Alembic |
| **Data validation** | Pydantic v2, pydantic-settings |
| **Package manager** | uv (pyproject.toml + uv.lock) |
| **Frontend** | Next.js 14 (App Router), TypeScript, Tailwind CSS |
| **Testing** | pytest, pytest-asyncio, httpx `AsyncClient` |
| **Notebooks** | Jupyter, pandas, seaborn, matplotlib |

---

## Project Architecture

The backend follows a strict **4-layer architecture**. No layer may import from a layer above it.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Routes  (HTTP only; imports controllers via Depends) │
│  app/routes/v1/chat.py  diagnosis.py  feedback.py               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Layer 3 — Controllers  (orchestration; imports services only)  │
│  app/controllers/v1/chat_controller.py  feedback_controller.py  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Layer 2 — Services  (business logic; imports ML layer only)    │
│  app/services/dialogue_service.py  (state machine)              │
│  app/services/diagnosis_service.py (confidence thresholding)    │
│  app/services/session_service.py   (DB CRUD)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Layer 1 — ML  (no FastAPI, no DB, no HTTP)                     │
│  app/ml/model_registry.py  (singleton artifact loader)          │
│  app/ml/predictor.py       (pure predict() function)            │
│  app/ml/feature_builder.py (symptoms → binary numpy array)      │
└─────────────────────────────────────────────────────────────────┘
```

**Cross-cutting concerns** (imported by any layer):

- `app/core/config.py` — Pydantic `BaseSettings`; all env vars
- `app/core/exceptions.py` — typed exception hierarchy
- `app/core/logging.py` — structured JSON logging
- `app/core/lifespan.py` — startup (artifact loading, DB init, migrations) and shutdown hooks
- `app/database/` — SQLAlchemy base, async session factory, ORM models
- `app/schemas/` — Pydantic request/response models

---

## Directory Structure

```
project-root/
├── pyproject.toml              ← uv-managed; all dependencies declared here
├── uv.lock
├── .env.example                ← copy to .env and fill in values
├── alembic.ini
│
├── alembic/
│   ├── env.py                  ← async Alembic environment
│   ├── script.py.mako
│   └── versions/               ← generated migration files go here
│
├── app/
│   ├── main.py                 ← app factory: lifespan, middleware, routers
│   │
│   ├── core/
│   │   ├── config.py           ← Settings (Pydantic BaseSettings)
│   │   ├── exceptions.py       ← ArtifactLoadError, FeatureContractError, etc.
│   │   ├── logging.py          ← JSON formatter + configure_logging()
│   │   └── lifespan.py         ← load artifacts + DB pool on startup
│   │
│   ├── database/
│   │   ├── base.py             ← SQLAlchemy DeclarativeBase
│   │   ├── session.py          ← async engine, session factory, get_async_session()
│   │   └── models/
│   │       ├── session_log.py  ← ChatSession ORM model
│   │       └── feedback.py     ← Feedback ORM model
│   │
│   ├── schemas/
│   │   ├── chat.py             ← ChatMessageRequest/Response, ChatSessionResponse
│   │   ├── diagnosis.py        ← DiagnosisRequest, DiagnosisResult, InconclusiveResponse
│   │   └── feedback.py         ← FeedbackRequest, FeedbackResponse
│   │
│   ├── ml/
│   │   ├── model_registry.py   ← singleton; loaded once at startup via lifespan
│   │   ├── predictor.py        ← predict() + build_diagnosis_result()
│   │   └── feature_builder.py  ← build_vector() with rapidfuzz normalisation
│   │
│   ├── services/
│   │   ├── dialogue_service.py ← GREETING→COLLECTING→CONFIRMING→PREDICTING→DONE
│   │   ├── diagnosis_service.py← diagnose() with confidence threshold
│   │   └── session_service.py  ← create/get/save ChatSession, create Feedback
│   │
│   ├── controllers/v1/
│   │   ├── chat_controller.py
│   │   └── feedback_controller.py
│   │
│   ├── routes/v1/
│   │   ├── chat.py             ← POST /api/v1/chat/message, GET /api/v1/chat/session/{id}
│   │   ├── diagnosis.py        ← POST /api/v1/diagnosis
│   │   └── feedback.py         ← POST /api/v1/feedback
│   │
│   ├── middleware/
│   │   ├── cors.py             ← CORSMiddleware (origins from settings)
│   │   └── rate_limit.py       ← in-memory sliding-window limiter
│   │
│   └── artifacts/v1/           ← generated by train script; not committed
│       ├── model.joblib
│       ├── label_encoder.joblib
│       ├── metadata.json
│       └── disease_info.json
│
├── scripts/
│   ├── train_model.py          ← CLI: train + cross-validate + export artifacts
│   └── migrate_db.py           ← Alembic CLI wrapper
│
├── data/
│   ├── Training.csv            ← 132 symptoms + prognosis (you supply this)
│   ├── Testing.csv
│   ├── symptom_Description.csv ← optional: disease descriptions
│   └── symptom_precaution.csv  ← optional: precautionary measures
│
├── notebooks/
│   └── train_and_evaluate.ipynb← full academic evaluation notebook
│
└── tests/
    ├── unit/
    │   ├── test_feature_builder.py
    │   ├── test_predictor.py
    │   └── test_dialogue_service.py
    └── integration/
        └── test_chat_api.py    ← uses SQLite in-memory + stub registry
```

---

## Dialogue State Machine

Each call to `POST /api/v1/chat/message` advances the state machine by exactly one turn. All state is persisted in the `chat_sessions` database table between requests.

```
  ┌──────────┐   any message    ┌────────────┐
  │ GREETING │ ───────────────► │ COLLECTING │◄──────────────────┐
  └──────────┘                  └─────┬──────┘                   │
                                      │                           │ "no"
                        ≥3 confirmed  │                           │
                        AND (0 remain │                     ┌─────┴──────┐
                        OR 10 asked)  │                     │ CONFIRMING │
                                      ▼                     └─────┬──────┘
                               ┌────────────┐                     │ "yes"
                               │ CONFIRMING │─────────────────────┘
                               └─────┬──────┘
                                     │ "yes"
                                     ▼
                               ┌────────────┐
                               │ PREDICTING │  ← runs ML inference
                               └─────┬──────┘
                                     │
                                     ▼
                               ┌──────────┐
                               │   DONE   │  ← awaits feedback / restart
                               └──────────┘
```

| State | What the bot does | Transition |
|---|---|---|
| `GREETING` | Sends welcome message; asks user for their primary symptom | → `COLLECTING` |
| `COLLECTING` | Asks yes/no questions one at a time (max 10); tracks confirmed symptoms | → `CONFIRMING` when ≥ 3 confirmed |
| `CONFIRMING` | Lists confirmed symptoms; asks "Is this correct?" | → `PREDICTING` on yes; → `COLLECTING` (reset) on no |
| `PREDICTING` | Runs `DecisionTreeClassifier`; returns disease + confidence | → `DONE` |
| `DONE` | Offers restart or feedback submission | Terminal |

---

## API Reference

Interactive docs are available at `/api/docs` (Swagger UI) and `/api/redoc` once the server is running.

### `POST /api/v1/chat/message`

Send a user message and advance the dialogue.

**Request**
```json
{
  "session_id": null,
  "message": ""
}
```
Set `session_id` to `null` to start a new session. Pass the UUID returned in the response to continue an existing one.

**Response**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "state": "COLLECTING",
  "bot_message": "Hello! I'm Dr. Melvis...",
  "confirmed_symptoms": [],
  "diagnosis": null
}
```

When the state reaches `PREDICTING`, `diagnosis` is populated:
```json
{
  "diagnosis": {
    "disease": "Fungal infection",
    "confidence": 0.95,
    "description": "A fungal skin condition characterised by...",
    "precautions": ["keep skin dry", "use antifungal cream"]
  }
}
```

---

### `GET /api/v1/chat/session/{session_id}`

Retrieve the full stored state of a session.

**Response**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "state": "DONE",
  "confirmed_symptoms": ["itching", "skin_rash", "chills"],
  "asked_symptoms": ["itching", "skin_rash", "chills", "fever", "headache"],
  "predicted_disease": "Fungal infection",
  "confidence": 0.95,
  "completed": true
}
```

---

### `POST /api/v1/diagnosis`

Stateless, direct inference. Bypasses the dialogue state machine entirely. Useful for programmatic access or testing.

**Request**
```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

**Response — success**
```json
{
  "disease": "Fungal infection",
  "confidence": 0.95,
  "description": "...",
  "precautions": ["keep skin dry"]
}
```

**Response — inconclusive** (confidence < threshold)
```json
{
  "error": "inconclusive",
  "message": "Symptoms are inconclusive. Please consult a qualified medical professional."
}
```

**Response — unknown symptom** (`422`)
```json
{
  "error": "unknown_symptom",
  "message": "Unknown symptom: 'xyz'"
}
```

---

### `POST /api/v1/feedback`

Submit a rating and optional comment for a completed session.

**Request**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "rating": 4,
  "comment": "Very accurate!"
}
```

**Response** (`201 Created`)
```json
{ "success": true }
```

---

### `GET /health`

Liveness probe. Returns `200 {"status": "ok"}` when the application is running.

---

## Database Schema

### `chat_sessions`

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Auto-generated session identifier |
| `created_at` | timestamptz | UTC creation time |
| `state` | varchar(32) | Current dialogue state |
| `confirmed_symptoms` | JSON | List of confirmed symptom strings |
| `asked_symptoms` | JSON | All symptoms presented to the user |
| `predicted_disease` | varchar(256) | Disease name after prediction (nullable) |
| `confidence` | float | Model confidence score (nullable) |
| `completed` | boolean | `true` once session reaches DONE |

### `feedback`

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Auto-generated |
| `session_id` | UUID (FK) | References `chat_sessions.id` |
| `rating` | integer | 1–5 star rating |
| `comment` | varchar(2048) | Optional text comment |
| `created_at` | timestamptz | UTC submission time |

---

## ML Pipeline

### Dataset

The model is trained on a publicly available medical symptom dataset:

- **132 binary features** — each representing the presence/absence of a symptom
- **41 disease classes** — e.g. Fungal infection, Malaria, Dengue, Diabetes, Hypertension
- **Training set** — `data/Training.csv`
- **Test set** — `data/Testing.csv`

### Model

```python
DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
```

**Why a Decision Tree?**

- The symptom–disease relationship in this dataset is deterministic and non-overlapping. A full-depth tree perfectly separates the classes without overfitting.
- Decision Trees are fully interpretable: the decision path for any prediction can be read directly from the tree structure, which is valuable for academic transparency and for building user trust in a medical context.
- Evaluation shows near-identical accuracy to Random Forest on this dataset, so the added complexity and loss of interpretability of an ensemble is not justified.

### Feature Vector Contract

The `symptom_list` array in `metadata.json` defines the exact order of all 132 binary features used during training. The `feature_builder` module maps any user-provided symptom string to its position in this list using exact matching first, then fuzzy matching via `rapidfuzz` with an 80% similarity threshold.

### Artifacts

All four files are generated by `scripts/train_model.py` and stored in `app/artifacts/v1/`:

| File | Contents |
|---|---|
| `model.joblib` | Fitted `DecisionTreeClassifier` |
| `label_encoder.joblib` | Fitted `LabelEncoder` for the prognosis column |
| `metadata.json` | `symptom_list`, `class_names`, `model_version`, `accuracy`, `cv_mean_accuracy` |
| `disease_info.json` | Per-disease `description` and `precautions` list |

Artifacts are **not committed** to version control. They must be generated locally by running the training script.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 14+ (running locally or via Docker)
- `data/Training.csv` and `data/Testing.csv` in the `data/` directory

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/health_chatbot
CONFIDENCE_THRESHOLD=0.60
ALLOWED_ORIGINS=http://localhost:3000
```

### 4. Train the model

Place `Training.csv` and `Testing.csv` in `data/`. Optionally add `symptom_Description.csv` and `symptom_precaution.csv` for richer diagnosis output.

```bash
uv run python scripts/train_model.py
```

This generates `model.joblib`, `label_encoder.joblib`, `metadata.json`, and `disease_info.json` in `app/artifacts/v1/`.

### 5. Start the API

```bash
uv run uvicorn app.main:app --reload
```

The API is now live at `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- Health check: `http://localhost:8000/health`

> **Note:** The application runs Alembic migrations automatically on startup. No manual migration step is required on first run.

---

## Running the Tests

The test suite uses an in-memory SQLite database and a stub model registry, so no PostgreSQL connection or trained artifacts are required.

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov=app --cov-report=term-missing

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/
```

### Test coverage

| Suite | What it tests |
|---|---|
| `test_feature_builder.py` | `_normalise()`, `build_vector()`, fuzzy matching, `FeatureContractError` |
| `test_predictor.py` | `predict()`, `build_diagnosis_result()`, argmax selection |
| `test_dialogue_service.py` | All state transitions, yes/no parsing, reset flow, inconclusive handling |
| `test_chat_api.py` | Full HTTP round-trips via `httpx.AsyncClient` with mocked model and SQLite DB |

---

## Training the Model

The training script handles the complete pipeline:

```bash
uv run python scripts/train_model.py
```

Output:
```
INFO  Training set shape: (4920, 133)
INFO  Symptom count: 132
INFO  Disease classes: 41
INFO  Model trained
INFO  5-fold CV accuracy: ['1.0000', '1.0000', '1.0000', '1.0000', '1.0000'] — mean 1.0000 ± 0.0000
INFO  Test accuracy: 1.0000
INFO  Saved model.joblib
INFO  Saved label_encoder.joblib
INFO  Saved metadata.json
INFO  Saved disease_info.json
```

For the full academic evaluation (confusion matrix, feature importance chart, tree visualisation, Decision Tree vs Random Forest comparison) open the notebook:

```bash
uv run jupyter notebook notebooks/train_and_evaluate.ipynb
```

---

## Running Database Migrations

Migrations run automatically at startup. To run them manually:

```bash
# Apply all pending migrations
uv run python scripts/migrate_db.py upgrade head

# Roll back one step
uv run python scripts/migrate_db.py downgrade -1

# Auto-generate a migration after modifying ORM models
uv run python scripts/migrate_db.py revision --autogenerate -m "describe change"
```

---

## Environment Variables

All configuration is read from a `.env` file (or actual environment variables). There are no hardcoded values anywhere in the application.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full asyncpg connection string |
| `ARTIFACTS_DIR` | No | `app/artifacts/v1` | Path to trained artifact directory |
| `CONFIDENCE_THRESHOLD` | No | `0.60` | Minimum confidence to display a diagnosis (0–1) |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000` | Comma-separated CORS allowed origins |

---

## Academic Evaluation

The training notebook (`notebooks/train_and_evaluate.ipynb`) fulfils the following evaluation requirements:

1. **Dataset description** — shape, class distribution, imbalance analysis
2. **Preprocessing steps** — label encoding, feature/target split, validation checks
3. **Model training** — `DecisionTreeClassifier` with hyperparameter rationale
4. **5-fold stratified cross-validation** — accuracy per fold, mean ± std
5. **Confusion matrix** — seaborn heatmap on the test set
6. **Classification report** — precision, recall, F1-score per class
7. **Feature importance bar chart** — top 20 symptoms by Gini importance
8. **Decision tree visualisation** — `plot_tree` at max depth 3 for readability
9. **Comparison with Random Forest** — accuracy table and bar chart justifying the DT choice

---

## Contributing

This is an academic project and is not open for external contributions. To adapt it for your own use, fork the repository and follow the Getting Started guide above.

---

## License

This project is submitted as an academic dissertation. All rights reserved by the author.
