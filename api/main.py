"""
NexResolve FastAPI application entry point.

Run from the project root with:
    venv/Scripts/uvicorn api.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.pipeline import load_all_models
from api.routes import auth, tickets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models once before the server starts accepting requests."""
    load_all_models()
    yield
    # (teardown logic can go here if needed)


app = FastAPI(
    title="NexResolve API",
    description="Intelligent ticket resolution powered by hierarchical DQN + RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(tickets.router)


@app.get("/")
def health():
    return {"status": "ok"}
