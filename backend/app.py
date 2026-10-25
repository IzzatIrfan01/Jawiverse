import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.llm_inference import inferencer

LOG_PATH = BASE_DIR / "jawiverse.log"
# React production build (npm run build); served at / when present, e.g. in the Docker image
FRONTEND_DIR = Path(os.environ.get("JAWIVERSE_FRONTEND_DIR", BASE_DIR.parent / "react-frontend" / "build"))

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8")
    ]
)
logger = logging.getLogger("jawiverse-api")

# Initialize app
app = FastAPI(title="Jawi-Rumi Transliteration API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "JAWIVERSE_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://127.0.0.1:5500",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class TransliterationRequest(BaseModel):
    model: str = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "jawiverse-api",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": getattr(inferencer, "device", "unknown"),
        "model_path": getattr(inferencer, "model_path", None),
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"Response status: {response.status_code} | elapsed_ms={elapsed_ms}")
    return response


@app.post("/transliterate")
def transliterate(request: TransliterationRequest):
    model = request.model.lower()
    text = request.text

    logger.info("Transliteration requested: model=%s, input_length=%d", model, len(text))

    # Use LLM for jawi2rumi and rumi2jawi only
    if model == "jawi2rumi":
        start = time.perf_counter()
        result = inferencer.transliterate(text, mode="jawi2rumi")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    elif model == "rumi2jawi":
        start = time.perf_counter()
        result = inferencer.transliterate(text, mode="rumi2jawi")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    else:
        logger.error(f"Invalid model selected: {model}")
        raise HTTPException(status_code=400, detail="Invalid model selected. Use 'jawi2rumi' or 'rumi2jawi'")

    logger.info("Transliteration completed: output_length=%d | elapsed_ms=%s", len(result), elapsed_ms)
    return {"model": model, "input": text, "output": result, "elapsed_ms": elapsed_ms}


# Mounted last so /health, /transliterate and /docs keep priority over the static files
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIR)