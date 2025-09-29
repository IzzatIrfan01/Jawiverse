from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your functions
from models.classical import classical_jawi_to_rumi, rumi_to_classical_jawi
from models.modern import modern_jawi_to_rumi, rumi_to_modern_jawi

# Initialize app
app = FastAPI(title="Jawi-Rumi Transliteration API")

# Request schema
class TransliterationRequest(BaseModel):
    model: str   # classical2rumi, rumi2classical, modern2rumi, rumi2modern
    text: str

@app.post("/transliterate")
def transliterate(request: TransliterationRequest):
    model = request.model.lower()
    text = request.text

    if model == "classical2rumi":
        result = classical_jawi_to_rumi(text)
    elif model == "rumi2classical":
        result = rumi_to_classical_jawi(text)
    elif model == "modern2rumi":
        result = modern_jawi_to_rumi(text)
    elif model == "rumi2modern":
        result = rumi_to_modern_jawi(text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    return {"model": model, "input": text, "output": result}
