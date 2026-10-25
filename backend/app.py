import logging
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.llm_inference import inferencer

LOG_PATH = BASE_DIR / "jawiverse.log"

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
    allow_origins=["*"],   # or ["http://127.0.0.1:5500"] if using VSCode Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class TransliterationRequest(BaseModel):
    model: str   # classical2rumi, rumi2classical, modern2rumi, rumi2modern
    text: str


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
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"Response status: {response.status_code} | elapsed_ms={elapsed_ms}")
    return response


@app.post("/transliterate")
def transliterate(request: TransliterationRequest):
    model = request.model.lower()
    text = request.text

    logger.info(f"Transliteration requested: model={model}, text={text}")

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

    logger.info(f"Transliteration result: {result} | elapsed_ms={elapsed_ms}")
    return {"model": model, "input": text, "output": result, "elapsed_ms": elapsed_ms}