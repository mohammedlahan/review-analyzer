"""
main.py – FastAPI backend for Customer Review Analyzer
=======================================================
Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import pipeline

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = Path(__file__).resolve().parent / "model"
FRONTEND_HTML = PROJECT_ROOT / "frontend" / "index.html"

# ---------------------------------------------------------------------------
# LOAD MODEL ONCE AT STARTUP
# ---------------------------------------------------------------------------
print(f"[startup] Loading model from: {MODEL_DIR}")
classifier = pipeline(
    "text-classification",
    model=str(MODEL_DIR),
    top_k=None,   # return all label scores, not just the top one
)
print("[startup] Model loaded successfully ✓")

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(title="Customer Review Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def run_classifier(text: str) -> dict:
    """Run the model and return {sentiment, confidence, scores}."""
    raw = classifier(text, truncation=True, max_length=128)[0]
    scores    = {item["label"]: round(item["score"], 4) for item in raw}
    best      = max(scores, key=scores.get)
    return {"sentiment": best, "confidence": scores[best], "scores": scores}

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "model": str(MODEL_DIR)}


class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    return run_classifier(req.text)


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    if "review" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must have a column named 'review'.")
    if len(df) > 500:
        raise HTTPException(status_code=422, detail=f"CSV has {len(df)} rows; max is 500.")
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV has no rows.")

    results = []
    for idx, row in df.iterrows():
        text = str(row["review"]).strip()
        if not text or text.lower() == "nan":
            results.append({"row": idx+1, "review": text, "sentiment": "unknown",
                            "confidence": 0.0, "scores": {"negative":0.0,"neutral":0.0,"positive":0.0}})
            continue
        prediction = run_classifier(text)
        results.append({"row": idx+1, "review": text, **prediction})

    valid   = [r for r in results if r["sentiment"] != "unknown"]
    total   = len(valid)
    counts  = {"negative": 0, "neutral": 0, "positive": 0}
    for r in valid:
        if r["sentiment"] in counts:
            counts[r["sentiment"]] += 1

    pct_neg  = round(counts["negative"] / total * 100, 1) if total > 0 else 0.0
    negatives = sorted([r for r in valid if r["sentiment"] == "negative"],
                       key=lambda r: r["confidence"], reverse=True)

    return {
        "results": results,
        "summary": {
            "total": total,
            "counts": counts,
            "pct_negative": pct_neg,
            "top_negatives": negatives[:5],
        },
    }


# ---------------------------------------------------------------------------
# SERVE FRONTEND
# ---------------------------------------------------------------------------
@app.get("/")
@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""):
    if FRONTEND_HTML.exists():
        return FileResponse(str(FRONTEND_HTML))
    return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)
