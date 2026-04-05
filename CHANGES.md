# NexResolve — Phase 3 Changes (React Frontend)

## Phase 3 — Files Created

| File | Purpose |
|---|---|
| `frontend/package.json` | React 18 + Vite + Tailwind + React Router + Axios |
| `frontend/vite.config.js` | Vite dev server with proxy for `/auth`, `/ticket`, `/history` |
| `frontend/tailwind.config.js` | Tailwind content paths |
| `frontend/postcss.config.js` | PostCSS with tailwind + autoprefixer |
| `frontend/index.html` | HTML shell mounting `#root` |
| `frontend/src/main.jsx` | React entry — wraps app in `BrowserRouter` + `AuthProvider` |
| `frontend/src/index.css` | Tailwind directives |
| `frontend/src/App.jsx` | Routes + `ProtectedRoute` guard |
| `frontend/src/context/AuthContext.jsx` | JWT auth context — login, logout, token verification on mount |
| `frontend/src/services/api.js` | Axios instance — auth interceptor, all API call functions |
| `frontend/src/components/Navbar.jsx` | Fixed top navbar — username + logout |
| `frontend/src/components/TicketCard.jsx` | History card — outcome badge, strategy badge, timestamp |
| `frontend/src/components/NLPPanel.jsx` | Dark sidebar — intent, urgency, entities, sentiment, RAG sim |
| `frontend/src/components/RLDecisionPanel.jsx` | Dark sidebar — strategy badge, action, outcome |
| `frontend/src/pages/LoginPage.jsx` | Email + password form, inline error |
| `frontend/src/pages/RegisterPage.jsx` | Username + email + password form, inline error |
| `frontend/src/pages/DashboardPage.jsx` | Ticket history list with skeleton loading and empty state |
| `frontend/src/pages/NewTicketPage.jsx` | Two-panel chat UI — textarea submit, bot bubbles, typing indicator |
| `frontend/src/pages/TicketDetailPage.jsx` | Two-panel read-only view of a stored ticket |

## Phase 3 — How to run

```bash
# Terminal 1 — Backend (from project root)
venv/Scripts/uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install        # first time only
npm run dev
# Open http://localhost:5173
```

MongoDB must be running (`mongod --dbpath C:\data\db`) before starting the backend.

---

# NexResolve — Phase 2 Changes

## Files Created / Modified

### New files

| File | Purpose |
|---|---|
| `configs/auth_config.yaml` | JWT secret, expiry hours, bcrypt rounds |
| `configs/db_config.yaml` | MongoDB URI and database name |
| `db/__init__.py` | Package marker |
| `db/connection.py` | Motor async MongoDB client (single instance) |
| `db/models.py` | `build_user_doc` and `build_ticket_doc` dict builders |
| `api/__init__.py` | Package marker |
| `api/auth_utils.py` | `create_token`, `verify_token`, `get_current_user` dependency |
| `api/routes/__init__.py` | Package marker |
| `api/routes/auth.py` | `POST /auth/register`, `POST /auth/login` |
| `api/routes/tickets.py` | `POST /ticket/new`, `GET /history/me`, `GET /ticket/{id}` |
| `api/pipeline.py` | Global model singletons + `load_all_models()` + `run_pipeline()` |
| `api/main.py` | FastAPI app, lifespan hook, CORS, router registration |

### Modified files

| File | Change |
|---|---|
| `nlp/nlp_pipeline.py` | Appended `run_nlp_pipeline(ticket_text)` at the bottom — **no existing code touched** |

---

## Packages Installed (into venv)

```
fastapi         uvicorn         motor
python-jose     passlib[bcrypt] pyyaml
python-multipart
```

---

## How to start the server

> **Run every command from the project root** (`c:\Users\Samridhaa\OneDrive\Desktop\nexresolve`)

```bash
# Activate the virtual environment (if not already active)
venv\Scripts\activate

# Start MongoDB (in a separate terminal, if not running as a service)
mongod --dbpath C:\data\db

# Start the API server
venv\Scripts\uvicorn api.main:app --reload --port 8000
```

You will see startup output like:

```
[startup] State scaler loaded
[startup] L1 agent loaded on cpu
[startup] L2 agent loaded
[startup] Generator loaded on cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## How to test each endpoint

### Option 1 — Swagger UI (easiest)

Open your browser and go to:

```
http://localhost:8000/docs
```

The interactive Swagger UI lets you call every endpoint with a form. Click
**Authorize** (top-right) and paste your JWT token after registering.

---

### Option 2 — curl commands

#### Health check
```bash
curl http://localhost:8000/
# Expected: {"status":"ok"}
```

#### Register a new user
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"alice\", \"email\": \"alice@test.com\", \"password\": \"pass123\"}"
# Expected: {"access_token": "...", "token_type": "bearer", "username": "alice"}
```

#### Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"alice@test.com\", \"password\": \"pass123\"}"
# Expected: {"access_token": "...", "token_type": "bearer", "username": "alice"}
```

Save the token from either response:
```bash
TOKEN="<paste access_token value here>"
```

#### Submit a ticket (runs full ML pipeline)
```bash
curl -X POST http://localhost:8000/ticket/new \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"text\": \"My Jupyter notebook crashes with a CUDA out of memory error on Windows 11 after updating PyTorch to 2.1.\"}"
# Expected: full pipeline result + ticket_id
```

#### View ticket history
```bash
curl http://localhost:8000/history/me \
  -H "Authorization: Bearer $TOKEN"
# Expected: list of past tickets (newest first)
```

#### Retrieve a single ticket
```bash
curl http://localhost:8000/ticket/<ticket_id> \
  -H "Authorization: Bearer $TOKEN"
# Expected: full stored ticket document
```

---

### Option 3 — Python httpx (for scripted testing)

```python
import httpx

BASE = "http://localhost:8000"

# Register
r = httpx.post(f"{BASE}/auth/register", json={
    "username": "bob", "email": "bob@test.com", "password": "secret"
})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Submit ticket
r = httpx.post(f"{BASE}/ticket/new", headers=headers, json={
    "text": "Getting AttributeError on model.fit() after pip upgrade on Ubuntu 20.04"
})
print(r.json())

# History
r = httpx.get(f"{BASE}/history/me", headers=headers)
print(r.json())
```

---

## Architecture overview

```
POST /ticket/new
       │
       ▼
run_pipeline(text, user_id)
       │
       ├─ run_nlp_pipeline(text)        ← intent + entity + missing + urgency
       │                                    + sentiment + clarification
       ├─ retrieve(text, top_k=5)       ← FAISS nearest-neighbour search
       ├─ build_single_state(row)       ← 37-dim state vector
       ├─ scaler.transform(state)       ← MinMaxScaler normalisation
       ├─ get_action_mask(state_norm)   ← valid-action binary mask
       ├─ L1.select_action(...)         ← strategy: ROUTE/CLARIFY/SUGGEST/ESCALATE
       ├─ L2.select_action(...)         ← specific action (17-way)
       └─ generator.generate(...)       ← FLAN-T5 response (SUGGEST only)
               │
               ▼
       build_ticket_doc(...)
               │
               ▼
       MongoDB  tickets  collection
```

---

## GPU safety

- Generator loads on CUDA if available; falls back to CPU automatically on OOM.
- If a GPU OOM occurs **during generation** at inference time, the generator's
  `.device` and `.model` are hot-swapped to CPU and the call is retried once.
- Both the load-time and inference-time fallbacks are handled in `api/pipeline.py`
  (`load_all_models` and `_safe_generate`).

---

## Notes

- MongoDB must be running before starting the server.
- Default connection: `mongodb://localhost:27017`, database `nexresolve`.
- JWT tokens expire after **24 hours** (configurable in `configs/auth_config.yaml`).
- The FLAN-T5 generator only fires when `strategy == "suggest"` AND
  `top_similarity >= 0.50` — for all other strategies `response` is `null`.
