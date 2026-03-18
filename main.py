# --- MUST BE AT THE VERY TOP ---
import os

# MUST be the first lines in the file
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from eda_utils import run_eda, generate_pdf_report
from ml_utils import run_ml
from survival_utils import run_survival
from causal_utils import run_causal
from rl_utils import train_agent, get_recommendation

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOADS    = BASE_DIR / "uploads"
PLOTS_DIR  = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
for d in (UPLOADS, PLOTS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

CURRENT_FILE: Optional[Path] = None
CURRENT_DF:   Optional[pd.DataFrame] = None

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ChurnSense AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_df() -> pd.DataFrame:
    if CURRENT_DF is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    return CURRENT_DF


# ── upload ─────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_FILE, CURRENT_DF
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    dest = UPLOADS / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    CURRENT_FILE = dest
    CURRENT_DF   = pd.read_csv(dest)
    return {
        "filename": file.filename,
        "rows": int(CURRENT_DF.shape[0]),
        "columns": int(CURRENT_DF.shape[1]),
        "column_names": list(CURRENT_DF.columns),
    }


# ── EDA ────────────────────────────────────────────────────────────────────────
@app.get("/eda")
async def eda():
    df = _get_df()
    result = run_eda(df)
    return result


@app.get("/eda/report")
async def eda_report():
    df = _get_df()
    result = run_eda(df)
    report_path = str(PLOTS_DIR / "eda_report.pdf")
    generate_pdf_report(df, result, report_path)
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename="eda_report.pdf",
    )


# ── ML ─────────────────────────────────────────────────────────────────────────
@app.get("/ml")
async def ml():
    df = _get_df()
    return run_ml(df)


# ── Survival ───────────────────────────────────────────────────────────────────
@app.get("/survival")
async def survival():
    df = _get_df()
    return run_survival(df)


# ── Causal ─────────────────────────────────────────────────────────────────────
@app.get("/causal")
async def causal():
    df = _get_df()
    return run_causal(df)


# ── RL: train ─────────────────────────────────────────────────────────────────
@app.post("/rl/train")
async def rl_train(episodes: int = 300):
    logs = train_agent(n_episodes=episodes)
    return {"logs": logs}


# ── RL: recommend ─────────────────────────────────────────────────────────────
class CustomerProfile(BaseModel):
    age: float
    fee: float
    activity: float


@app.post("/rl/recommend")
async def rl_recommend(profile: CustomerProfile):
    recommendation = get_recommendation(profile.age, profile.fee, profile.activity)
    return {"recommendation": recommendation}


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

# Add this model in main.py
class TrainRequest(BaseModel):
    episodes: int = 100

@app.post("/rl/train")
async def rl_train(request: TrainRequest):
    # This now accepts JSON: {"episodes": 50}
    logs = train_agent(n_episodes=request.episodes)
    return {"status": "success", "logs": logs}
