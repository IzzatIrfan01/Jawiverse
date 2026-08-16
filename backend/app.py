import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from models.llm_inference import inferencer

from fastapi.middleware.cors import CORSMiddleware

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("jawiverse.log", encoding="utf-8")  # File
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/transliterate")
def transliterate(request: TransliterationRequest):
    model = request.model.lower()
    text = request.text

    logger.info(f"Transliteration requested: model={model}, text={text}")

    # Use LLM for jawi2rumi and rumi2jawi only
    if model == "jawi2rumi":
        result = inferencer.transliterate(text, mode="jawi2rumi")
    elif model == "rumi2jawi":
        result = inferencer.transliterate(text, mode="rumi2jawi")
    else:
        logger.error(f"Invalid model selected: {model}")
        raise HTTPException(status_code=400, detail="Invalid model selected. Use 'jawi2rumi' or 'rumi2jawi'")

    logger.info(f"Transliteration result: {result}")
    return {"model": model, "input": text, "output": result}