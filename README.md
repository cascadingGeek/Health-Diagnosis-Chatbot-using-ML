# Health Symptom Diagnosis Chatbot

**Development and Evaluation of a Machine Learning Based Health Symptom Diagnosis Chatbot Using Decision Tree Algorithm — with LLM-Augmented Natural Language Interface**

A multi-turn conversational chatbot that accepts free-form natural language from a user, extracts symptoms using Claude claude-sonnet-4-5, classifies the symptom set using a trained `DecisionTreeClassifier` (132 features, 41 disease classes), and then returns a clinically validated, human-readable diagnosis explanation — again via Claude. Built as a final-year Computer Science project at Federal University Oye-Ekiti to demonstrate applied machine learning, large language model integration, REST API design, and modern Python engineering.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Why an LLM Was Added](#why-an-llm-was-added)
3. [Solution Overview](#solution-overview)
4. [LLM Hybrid Architecture](#llm-hybrid-architecture)
5. [Features](#features)
6. [Tech Stack](#tech-stack)
7. [Project Architecture](#project-architecture)
8. [Directory Structure](#directory-structure)
9. [Dialogue State Machine](#dialogue-state-machine)
10. [API Reference](#api-reference)
11. [Database Schema](#database-schema)
12. [ML Pipeline](#ml-pipeline)
13. [Getting Started](#getting-started)
14. [Running the Tests](#running-the-tests)
15. [Training the Model](#training-the-model)
16. [Environment Variables](#environment-variables)
17. [Academic Evaluation](#academic-evaluation)

---

## Problem Statement

Access to preliminary medical guidance is a challenge in many settings — long wait times, limited availability of physicians, and the difficulty of self-assessing vague or overlapping symptoms. Most people resort to unstructured internet searches, which are unreliable and often anxiety-inducing.

This project addresses that gap with a structured, explainable AI system that:

- Accepts free-form natural language symptom descriptions from the user.
- Uses Claude claude-sonnet-4-5 to extract structured symptom tokens from that natural language.
- Uses a Decision Tree classifier trained on a curated 132-symptom, 41-disease dataset to predict the most likely condition.
- Uses Claude again to validate the prediction clinically and generate an empathetic, plain-English explanation with precautions and urgency guidance.
- Returns a confidence score, a disease description, recommended precautions, and a mandatory medical disclaimer.
- Falls back gracefully when confidence is low or the prediction is clinically implausible, directing the user to seek professional advice.

> **Disclaimer:** This system is for academic purposes only. It is not a substitute for professional medical diagnosis or treatment.

---

## Why an LLM Was Added

The first version of this chatbot used a fully deterministic pipeline: a 5-state FSM collected symptoms through rigid yes/no questions, and `rapidfuzz` fuzzy-matched the user's one-word answers against the 132-symptom training vocabulary. That approach had three hard limitations that motivated adding a large language model.

### 1. Users Do Not Speak in Symptom Tokens

The Decision Tree's training vocabulary is `snake_case` dataset identifiers — `pain_in_anal_region`, `burning_micturition`, `blurred_and_distorted_vision`. Real users say things like:

> *"I've had a really bad tummy ache since this morning and I'm feeling nauseous"*
> *"My legs have been cramping up at night"*
> *"There's a burning sensation when I use the toilet"*

The fuzzy matcher could handle minor spelling variations of single words but could not parse a full sentence, resolve pronouns, understand negations ("I do NOT have chest pain"), or map colloquial phrases like "tummy ache" or "can't keep food down" to the correct tokens. Every symptom had to be entered as a single keyword. This made the chatbot feel robotic and was a known source of incorrect feature vectors.

### 2. The Symptom-Collection Flow Was Rigid and Clinically Naive

The original FSM asked questions from a predefined template list in a fixed order, regardless of what the user had already told it. If a user mentioned "high fever and shivering", the bot would still ask about itching or skin rash next because that was next in the rotation. This produced irrelevant follow-up questions that broke the conversational flow and sometimes caused users to abandon the session.

### 3. The Output Was Static

The diagnosis response was assembled by looking up a disease name in a static `disease_info.json` file and returning its pre-written description and precautions. There was no:

- Validation that the predicted disease actually matched the user's reported symptoms.
- Adjustment of tone based on urgency (a "Heart Attack" prediction got the same calm description template as a "Common Cold").
- Handling of edge cases where the Decision Tree produced a clinically implausible result.
- Personalised explanation connecting the user's specific symptoms to the predicted condition.

### The Decision: Claude claude-sonnet-4-5

Claude claude-sonnet-4-5 was chosen as the language model for both ends of the pipeline because:

- **Structured JSON output reliability.** Both layers issue instructions to respond with JSON only. Claude claude-sonnet-4-5 follows structured output instructions consistently, which is critical because its JSON is parsed directly and passed to the Decision Tree or returned to the frontend.
- **Medical context understanding.** The model correctly maps colloquial symptom descriptions to clinical terms, understands negations and qualifiers, and can assess clinical plausibility of a diagnosis against a symptom set.
- **Cost efficiency.** At the per-token pricing of claude-sonnet-4-5, a complete end-to-end session (Layer 1 + Layer 3) consumes approximately 3,000–6,000 input tokens and 400–800 output tokens — affordable for an academic prototype.
- **Clear academic boundary.** Claude does not make diagnostic predictions. It extracts symptoms and explains the Decision Tree's output. This preserves the academic integrity of the project, which is explicitly evaluated on the Decision Tree's performance.

---

## Solution Overview

The chatbot exposes a RESTful JSON API consumed by a Next.js frontend. The core workflow is:

1. The user opens a chat session and describes their symptoms in free natural language.
2. **Layer 1 — Claude (Input):** Extracts confirmed and denied symptoms from the user's message and conversation history. Maps colloquial language to the 132-symptom dataset vocabulary. Generates one targeted follow-up question if more information is needed.
3. Steps 2–3 repeat (up to 8 questions) until at least 3 symptoms are confirmed.
4. **Layer 2 — Decision Tree:** Receives the binary symptom vector. Predicts the most likely disease from 41 classes and returns a raw confidence score.
5. **Layer 3 — Claude (Output):** Validates that the prediction is clinically plausible given the confirmed symptoms. Generates a plain-English, tone-appropriate explanation with 3–5 precautions, urgency level, and when-to-see-a-doctor guidance. Appends a mandatory medical disclaimer.
6. The result is shown to the user with a dampened confidence score (never displayed above 87%, to be honest about a synthetic dataset) and the full disclaimer.
7. The user can submit a 1–5 star rating and comment as feedback.

---

## LLM Hybrid Architecture

```
User free-text message
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 1 — Claude claude-sonnet-4-5 (Input)        │
│                                       │
│  Role: Natural language understanding │
│  - Parse free-form symptom text       │
│  - Map to 132-token dataset vocab     │
│  - Track confirmed / denied symptoms  │
│  - Generate targeted follow-up Qs     │
│  - Return structured JSON action      │
│                                       │
│  Does NOT make diagnostic predictions │
└───────────────────┬───────────────────┘
                    │  confirmed_symptoms: [token, token, ...]
                    ▼
┌───────────────────────────────────────┐
│  Layer 2 — DecisionTreeClassifier     │
│            (scikit-learn)             │
│                                       │
│  Role: Classification only            │
│  - Build 132-bit binary feature vector│
│  - Predict disease class              │
│  - Return raw predict_proba score     │
│                                       │
│  This is the ONLY diagnostic step.   │
│  Claude never overrides this result.  │
└───────────────────┬───────────────────┘
                    │  predicted_disease + raw_confidence
                    ▼
┌───────────────────────────────────────┐
│  Layer 3 — Claude claude-sonnet-4-5 (Output)       │
│                                       │
│  Role: Validation + response gen      │
│  - Check clinical plausibility        │
│  - Generate empathetic explanation    │
│  - Set urgency: mild / moderate /     │
│    urgent                             │
│  - Provide 3–5 actionable precautions │
│  - Dampen confidence (cap at 87%)     │
│  - Append mandatory disclaimer        │
│                                       │
│  Returns inconclusive if implausible  │
└───────────────────────────────────────┘
        │
        ▼
User sees final diagnosis response
```

### The Academic Boundary

The Decision Tree makes every diagnostic prediction. Claude's role is language understanding (Layer 1) and response generation (Layer 3) only. Claude cannot and does not change which disease is predicted. If the Decision Tree's prediction is clinically implausible given the confirmed symptoms, Layer 3 returns an inconclusive response — it does not substitute a different disease. This boundary is enforced in code and documented throughout the codebase.

### Confidence Dampening

The Kaggle synthetic dataset produces Decision Trees that memorise training patterns and return raw `predict_proba` values of 1.0 very frequently. Displaying "100% confidence" to a user seeking medical guidance would be misleading. The `_dampen_confidence()` function applies the following transform before any score reaches the frontend:

| Raw score | Displayed range | Notes |
|-----------|----------------|-------|
| `≥ 0.90` | `0.75 – 0.87` | Linear map + ±0.02 random jitter so repeated calls don't display the same round number |
| `0.60 – 0.89` | `raw × 0.92` | Light dampening only |
| `< 0.60` | Unchanged | Model is already signalling low confidence |

The hard ceiling is **0.87**. A raw score of 1.0 will never appear in any response sent to the frontend.

### Startup Validation

On every server start, before accepting any HTTP requests, the application:

1. Makes a real Anthropic API call to confirm the key is valid and the account has credits. Startup aborts with a clear error if the key returns 401 (invalid) or a zero-credit error.
2. Runs `SELECT 1` against the configured PostgreSQL instance to confirm the database is reachable. Startup aborts if the database cannot be contacted.
3. Validates the ML artifact feature contract (metadata symptom count must match the trained model's `n_features_in_`).

---

## Features

| Feature | Detail |
|---|---|
| Free-text input | Users describe symptoms in natural sentences — no keyword formatting required |
| LLM symptom extraction | Claude claude-sonnet-4-5 (Layer 1) maps natural language to 132 dataset tokens |
| Intelligent follow-up questions | Layer 1 generates clinically relevant questions based on what has been confirmed/denied so far |
| ML prediction | `DecisionTreeClassifier` trained on 132 symptoms → 41 diseases (Layer 2) |
| LLM output validation | Claude claude-sonnet-4-5 (Layer 3) checks clinical plausibility before showing the result |
| Tone-aware responses | Layer 3 adjusts urgency and tone: mild (colds), moderate (malaria/typhoid), urgent (cardiac/stroke) |
| Confidence dampening | Raw DT probability capped and jittered to max 87% — honest for a synthetic dataset |
| Medical disclaimer | Present in every diagnosis response — hardcoded safety guarantee, never optional |
| Confidence gating | Prediction flagged inconclusive if raw confidence < 60% or prediction is clinically implausible |
| Session persistence | Full conversation history, confirmed symptoms, and final diagnosis stored in PostgreSQL |
| Startup safety checks | Anthropic key + DB reachability verified before server accepts traffic |
| Fuzzy symptom matching | `rapidfuzz` as offline fallback for the direct `/diagnosis` endpoint |
| Direct inference endpoint | Stateless `POST /api/v1/diagnosis` bypasses dialogue for programmatic access |
| Feedback collection | Per-session 1–5 rating + comment stored to DB |
| Structured logging | JSON log output on stdout — includes per-call Claude token usage |
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
| **LLM** | Anthropic Claude claude-sonnet-4-5 via `anthropic==0.40.0` SDK |
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
│  Layer 2 — Services  (business logic)                           │
│  app/services/llm_orchestrator.py  ← hybrid pipeline entry point│
│  app/services/predictor_service.py ← Decision Tree wrapper      │
│  app/services/dialogue_service.py  ← legacy FSM (kept intact)   │
│  app/services/diagnosis_service.py ← confidence thresholding    │
│  app/services/session_service.py   ← DB CRUD                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Layer 1 — ML + LLM core  (no FastAPI, no DB, no HTTP)          │
│  app/ml/model_registry.py        — singleton artifact loader    │
│  app/ml/predictor.py             — pure predict() function      │
│  app/ml/feature_builder.py       — symptoms → binary numpy array│
│  app/core/llm_input_processor.py — Layer 1: Claude extracts syms│
│  app/core/llm_output_validator.py— Layer 3: Claude validates+gen│
└─────────────────────────────────────────────────────────────────┘
```

**Cross-cutting concerns** (imported by any layer):

- `app/core/config.py` — Pydantic `BaseSettings`; all env vars including `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`
- `app/core/exceptions.py` — typed exception hierarchy
- `app/core/logging.py` — structured JSON logging
- `app/core/lifespan.py` — startup checks (Anthropic key, DB, artifact contract, migrations) and shutdown
- `app/core/symptom_synonyms.py` — 149-entry colloquial → canonical synonym map
- `app/core/nlp_parser.py` — deterministic offline fallback parser (n-gram + fuzzy)
- `app/core/symptom_router.py` — symptom family routing for follow-up question selection
- `app/core/question_templates.py` — natural language question templates per symptom
- `app/core/diagnosis_guard.py` — anchor symptom validation for high-stakes diseases
- `app/core/disease_overrides.py` — documented wrong-prediction corrections
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
│   └── versions/               ← generated migration files
│
├── app/
│   ├── main.py                 ← app factory: lifespan, middleware, routers
│   │
│   ├── core/
│   │   ├── config.py           ← Settings (Pydantic BaseSettings)
│   │   ├── exceptions.py       ← ArtifactLoadError, FeatureContractError, etc.
│   │   ├── logging.py          ← JSON formatter + configure_logging()
│   │   ├── lifespan.py         ← startup checks + artifact/DB init
│   │   ├── llm_input_processor.py   ← Layer 1: Claude extracts symptoms
│   │   ├── llm_output_validator.py  ← Layer 3: Claude validates + explains
│   │   ├── symptom_synonyms.py ← 149-entry colloquial→canonical map
│   │   ├── nlp_parser.py       ← deterministic offline fallback parser
│   │   ├── symptom_router.py   ← symptom family → follow-up routing
│   │   ├── question_templates.py    ← per-symptom question templates
│   │   ├── diagnosis_guard.py  ← anchor symptom rules (9 high-stakes diseases)
│   │   └── disease_overrides.py     ← documented DT correction overrides
│   │
│   ├── database/
│   │   ├── base.py             ← SQLAlchemy DeclarativeBase
│   │   ├── session.py          ← async engine, session factory, get_async_session()
│   │   └── models/
│   │       ├── session_log.py  ← ChatSession ORM (incl. LLM fields)
│   │       └── feedback.py     ← Feedback ORM model
│   │
│   ├── schemas/
│   │   ├── chat.py             ← ChatMessageRequest/Response, ChatSessionResponse
│   │   ├── diagnosis.py        ← DiagnosisResult, LLMDiagnosisResult, InconclusiveResponse
│   │   └── feedback.py         ← FeedbackRequest, FeedbackResponse
│   │
│   ├── ml/
│   │   ├── model_registry.py   ← singleton; loaded once at startup
│   │   ├── predictor.py        ← predict() + build_diagnosis_result()
│   │   └── feature_builder.py  ← build_vector() with rapidfuzz normalisation
│   │
│   ├── services/
│   │   ├── llm_orchestrator.py ← hybrid pipeline: Layer 1 → 2 → 3
│   │   ├── predictor_service.py← Decision Tree wrapper (Layer 2)
│   │   ├── dialogue_service.py ← legacy FSM (GREETING→DONE)
│   │   ├── diagnosis_service.py← stateless diagnose() with confidence gate
│   │   └── session_service.py  ← create/get/save ChatSession + Feedback
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
│   ├── migrate_db.py           ← Alembic CLI wrapper
│   └── run_regression_tests.py ← 8-test live regression suite against running server
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

Each call to `POST /api/v1/chat/message` advances the state machine by exactly one turn. All state is persisted in the `chat_sessions` database table between requests. The LLM orchestrator handles the `COLLECTING` state; the FSM handles transitions and the outer session lifecycle.

```
  ┌──────────┐   any message    ┌────────────┐
  │ GREETING │ ───────────────► │ COLLECTING │◄──────────────────┐
  └──────────┘                  └─────┬──────┘                   │
                                      │                           │ "no"
                                      │ Layer 1 extracts syms     │
                                      │ Layer 1 asks follow-ups   │
                        ≥3 confirmed  │                     ┌─────┴──────┐
                        confirmed     │                     │ CONFIRMING │
                                      ▼                     └─────┬──────┘
                               ┌────────────┐                     │ "yes"
                               │ PREDICTING │─────────────────────┘
                               │ (Layer 2 + │
                               │  Layer 3)  │
                               └─────┬──────┘
                                     │
                                     ▼
                               ┌──────────┐
                               │   DONE   │  ← awaits feedback / restart
                               └──────────┘
```

| State | What happens | Transition |
|---|---|---|
| `GREETING` | Sends welcome message; transitions immediately | → `COLLECTING` |
| `COLLECTING` | Layer 1 (Claude) extracts symptoms from free text; asks follow-up questions | → `PREDICTING` once ≥ 3 symptoms confirmed |
| `CONFIRMING` | Listed confirmed symptoms; asks "Is this correct?" | → `PREDICTING` on yes; → `COLLECTING` on no |
| `PREDICTING` | Layer 2 (DT) classifies; Layer 3 (Claude) validates and explains | → `DONE` |
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
  "message": "I've been having a bad headache and I feel nauseous"
}
```

Set `session_id` to `null` to start a new session. Pass the UUID returned in the response to continue an existing one.

**Response — collecting symptoms**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "state": "COLLECTING",
  "bot_message": "Do you also have a high fever or chills along with the headache?",
  "confirmed_symptoms": ["headache", "nausea"],
  "diagnosis": null,
  "response_type": "question"
}
```

**Response — diagnosis**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "state": "DONE",
  "bot_message": "Typhoid is a bacterial infection caused by Salmonella typhi...",
  "confirmed_symptoms": ["high_fever", "headache", "nausea", "chills", "sweating"],
  "diagnosis": {
    "disease": "Typhoid",
    "confidence": 0.86,
    "description": "Typhoid is a bacterial infection caused by Salmonella typhi...",
    "precautions": [
      "Visit a doctor within 24 hours for blood tests",
      "Stay well-hydrated with clean boiled water",
      "Rest and avoid physical exertion until fever subsides"
    ],
    "is_plausible": true,
    "urgency": "moderate",
    "when_to_see_doctor": "Within 24 hours. Seek immediate emergency care if...",
    "disclaimer": "⚠️ Important Disclaimer: This is a preliminary AI-assisted assessment only..."
  },
  "response_type": "diagnosis"
}
```

**Response — inconclusive** (low confidence or clinically implausible prediction)
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "state": "DONE",
  "bot_message": "Your symptoms do not clearly match a single condition in our system.",
  "confirmed_symptoms": ["cough", "fatigue"],
  "diagnosis": {
    "disease": null,
    "confidence": null,
    "is_plausible": false,
    "urgency": "moderate",
    "precautions": ["Please visit the nearest clinic or hospital for a proper evaluation"],
    "when_to_see_doctor": "As soon as possible",
    "disclaimer": "⚠️ Important Disclaimer: ..."
  },
  "response_type": "inconclusive"
}
```

---

### `GET /api/v1/chat/session/{session_id}`

Retrieve the full stored state of a session.

---

### `POST /api/v1/diagnosis`

Stateless, direct inference. Bypasses the dialogue state machine and LLM layers entirely. Accepts a list of symptom tokens and returns a raw Decision Tree result. Useful for programmatic access or testing the ML layer in isolation.

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
| `state` | varchar(32) | Current FSM state |
| `fsm_state` | varchar(32) | LLM pipeline FSM state (GREETING/COLLECTING/DONE) |
| `confirmed_symptoms` | JSONB | Symptom tokens confirmed in this session |
| `denied_symptoms` | JSONB | Symptom tokens denied in this session |
| `conversation_history` | JSONB | Full `{role, content}` turn list |
| `questions_asked` | integer | Number of Layer 1 follow-up questions issued |
| `final_diagnosis` | JSONB | Complete Layer 3 diagnosis dict (nullable) |
| `predicted_disease` | varchar(256) | Disease name after prediction (nullable) |
| `confidence` | float | Dampened display confidence (nullable) |
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

The model is trained on a publicly available medical symptom dataset (Kaggle):

- **132 binary features** — each representing the presence/absence of a symptom
- **41 disease classes** — e.g. Fungal infection, Malaria, Dengue, Diabetes, Hypertension, Typhoid
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

**Note on raw confidence scores.** The synthetic Kaggle dataset causes the trained Decision Tree to return `predict_proba` values of exactly 1.0 on many inputs because it has memorised the patterns perfectly. These 1.0 scores are transformed by `_dampen_confidence()` before display — see [Confidence Dampening](#llm-hybrid-architecture) above.

### Feature Vector Contract

The `symptom_list` array in `metadata.json` defines the exact order of all 132 binary features used during training. The `feature_builder` module maps any user-provided symptom token to its position in this list. In the LLM pipeline, Layer 1 performs this mapping from natural language; the `feature_builder` is the fallback for the direct `/diagnosis` endpoint.

### Artifacts

All four files are generated by `scripts/train_model.py` and stored in `app/artifacts/v1/`:

| File | Contents |
|---|---|
| `model.joblib` | Fitted `DecisionTreeClassifier` |
| `label_encoder.joblib` | Fitted `LabelEncoder` for the prognosis column |
| `metadata.json` | `symptom_list`, `class_names`, `model_version`, `accuracy`, `cv_mean_accuracy` |
| `disease_info.json` | Per-disease `description` and `precautions` list |

Artifacts are **not committed** to version control. Generate them locally by running the training script.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 14+ (running locally or via Docker / Supabase)
- An [Anthropic API key](https://console.anthropic.com) with available credits
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

Edit `.env` — the two Anthropic values and the database URL are required:

```dotenv
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/health_chatbot

# Anthropic — required for Layer 1 (input) and Layer 3 (output) of the hybrid pipeline
# Get your key from console.anthropic.com → API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
ANTHROPIC_MODEL=claude-sonnet-4-5

# Inference
CONFIDENCE_THRESHOLD=0.60
```

> **Important:** `ANTHROPIC_API_KEY` is the single source of truth for the API key. It is read from `.env` only — it never appears in source code. To rotate the key, update `.env` and restart the server.

### 4. Train the model

Place `Training.csv` and `Testing.csv` in `data/`. Optionally add `symptom_Description.csv` and `symptom_precaution.csv` for richer static diagnosis output.

```bash
uv run python scripts/train_model.py
```

This generates `model.joblib`, `label_encoder.joblib`, `metadata.json`, and `disease_info.json` in `app/artifacts/v1/`.

### 5. Start the API

```bash
uv run uvicorn app.main:app --reload
```

On startup the server will:
1. Validate the Anthropic API key (makes a real call — will abort if invalid or zero credits)
2. Load ML artifacts
3. Verify database connectivity (`SELECT 1`)
4. Run any pending Alembic migrations

The API is then live at `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- Health check: `http://localhost:8000/health`

---

## Running the Tests

The unit/integration test suite uses an in-memory SQLite database and a stub model registry, so no PostgreSQL connection or trained artifacts are required. The LLM layers are not called in these tests.

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

### Running the Live Regression Suite

Eight end-to-end tests covering the full hybrid pipeline (Layer 1 + 2 + 3) can be run against a live server. These make real Anthropic API calls and require a valid `ANTHROPIC_API_KEY` with credits.

```bash
# Start the server first
uv run uvicorn app.main:app --port 8001

# In another terminal
uv run python scripts/run_regression_tests.py
```

| Test | Verifies |
|---|---|
| 1 | Free-text sentence → Layer 1 extracts headache + nausea |
| 2 | Cough path never predicts Heart Attack |
| 3 | Anal pain maps to `pain_in_anal_region`, not respiratory symptoms |
| 4 | Leg cramps maps to `cramps`/`muscle_pain`, not `blood_in_sputum` |
| 5 | Fever + chills → Malaria or Typhoid |
| 6 | Eye symptoms → clinically plausible result |
| 7 | Raw DT confidence of 1.0 dampened to ≤ 0.87 before display |
| 8 | Disclaimer field present and correct in every diagnosis response |

### Unit test coverage

| Suite | What it tests |
|---|---|
| `test_feature_builder.py` | `_normalise()`, `build_vector()`, fuzzy matching, `FeatureContractError` |
| `test_predictor.py` | `predict()`, `build_diagnosis_result()`, argmax selection |
| `test_dialogue_service.py` | All FSM state transitions, yes/no parsing, reset flow |
| `test_chat_api.py` | Full HTTP round-trips via `httpx.AsyncClient` with mocked model and SQLite |

---

## Training the Model

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

Note: 1.0 cross-validation accuracy is expected on this synthetic dataset and reflects the dataset's structure, not overfitting. See the [note on confidence dampening](#llm-hybrid-architecture) for how this is handled at display time.

For the full academic evaluation (confusion matrix, feature importance chart, tree visualisation, Decision Tree vs Random Forest comparison):

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

All configuration is read from `.env` (or actual environment variables in Railway/Render). There are no hardcoded values anywhere in the application.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full asyncpg connection string |
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key — Layer 1 and Layer 3 of the hybrid pipeline |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5` | Anthropic model ID — change here only, never in source code |
| `ARTIFACTS_DIR` | No | `app/artifacts/v1` | Path to trained artifact directory |
| `CONFIDENCE_THRESHOLD` | No | `0.60` | Minimum raw DT confidence to attempt a diagnosis |
| `CORS_ORIGINS` | No | `http://localhost:3000` | JSON array or comma-separated list of allowed CORS origins |

> **Rotating the API key:** Update `ANTHROPIC_API_KEY` in `.env` (locally) or in Railway/Render's environment variables panel, then restart the server. No code change required.

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

**On the LLM integration as a contribution.** The Decision Tree remains the sole classification engine — its performance metrics are unchanged. The LLM integration addresses a separate research question: how can a structured ML model trained on a constrained vocabulary be made accessible to users who communicate in natural language? Layer 1 (Claude) acts as a natural language interface that translates free-form input into the token vocabulary the Decision Tree was trained on. Layer 3 (Claude) acts as a clinical reasoning layer that assesses the model's output and communicates it in a form appropriate for a non-clinical user. This separation of concerns — classifier vs. communicator — is the architectural contribution of this project.

---

## Contributing

This is an academic project and is not open for external contributions. To adapt it for your own use, fork the repository and follow the Getting Started guide above.

---

## License

This project is submitted as an academic dissertation at Federal University Oye-Ekiti. All rights reserved by the author.
