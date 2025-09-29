import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Import your functions
from models.classical import classical_jawi_to_rumi, rumi_to_classical_jawi
from models.modern import modern_jawi_to_rumi, rumi_to_modern_jawi

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

    if model == "classical2rumi":
        result = classical_jawi_to_rumi(text)
    elif model == "rumi2classical":
        result = rumi_to_classical_jawi(text)
    elif model == "modern2rumi":
        result = modern_jawi_to_rumi(text)
    elif model == "rumi2modern":
        result = rumi_to_modern_jawi(text)
    else:
        logger.error(f"Invalid model selected: {model}")
        raise HTTPException(status_code=400, detail="Invalid model selected")

    logger.info(f"Transliteration result: {result}")
    return {"model": model, "input": text, "output": result}