"""FastAPI server for the Text-to-SQL Transformer front end.

Loads the checkpoint produced by Text_to_SQL_Transformer.ipynb (expects
../results/best.pt and ../results/sql_sp.model) once at startup, then serves
POST /api/query.

Run with:
    .venv\\Scripts\\uvicorn server:app --reload --port 8000   (Windows)
    .venv/bin/uvicorn server:app --reload --port 8000          (macOS/Linux)
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import TextToSQLEngine

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best.pt"
TOKENIZER_PATH = RESULTS_DIR / "sql_sp.model"

app = FastAPI(title="Text-to-SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[TextToSQLEngine] = None
load_error: Optional[str] = None


@app.on_event("startup")
def load_engine():
    global engine, load_error
    if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
        load_error = (
            f"Missing model files. Expected:\n"
            f"  {CHECKPOINT_PATH}\n  {TOKENIZER_PATH}\n"
            f"Unzip results_bundle.zip from the Kaggle run into {RESULTS_DIR} and restart."
        )
        return
    try:
        engine = TextToSQLEngine(CHECKPOINT_PATH, TOKENIZER_PATH)
    except Exception as e:  # noqa: BLE001 - surface any load failure to the API
        load_error = f"Failed to load model: {e}"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    columns: List[str] = Field(..., min_length=1)
    method: str = Field("beam", pattern="^(greedy|beam)$")


class QueryResponse(BaseModel):
    ok: bool
    sql: Optional[str] = None
    tokenized: Optional[str] = None
    error: Optional[str] = None


@app.get("/api/status")
def status():
    if engine is None:
        return {"ready": False, "error": load_error}
    return {
        "ready": True,
        "checkpoint_epoch": engine.epoch,
        "checkpoint_dev_loss": engine.dev_loss,
        "vocab_size": engine.sp.get_piece_size(),
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail=load_error or "Model not loaded.")

    columns = [c.strip() for c in req.columns if c.strip()]
    if not columns:
        raise HTTPException(status_code=400, detail="At least one column name is required.")

    try:
        result = engine.query(req.question, columns, method=req.method)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        ok=result["ok"],
        sql=result.get("sql"),
        tokenized=result.get("tokenized"),
        error=result.get("error"),
    )
