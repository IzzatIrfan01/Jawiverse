Back-end for jawiverse

Serves the trained model in `../jawiverse_server/nirnama/v1/final` (loader shared with
`jawiverse_server/infer_jawi_rumi.py`). Works on GPU or CPU.

Setup (from the repo root)

GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124   (match your CUDA)
pip install -r jawiverse_server/requirements.txt fastapi uvicorn

CPU only (no Unsloth/bitsandbytes needed; first run downloads the ~6.4 GB base model,
needs ~13 GB RAM, a few seconds per request):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements-cpu.txt

Run
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000

Optional environment variables
JAWIVERSE_DEVICE=auto|cuda|cpu      (default auto)
JAWIVERSE_GPU_ID=1                  (physical GPU index from nvidia-smi)
JAWIVERSE_MODEL_PATH=...            (default ../jawiverse_server/nirnama/v1/final)
JAWIVERSE_MAX_NEW_TOKENS=512
JAWIVERSE_CORS_ORIGINS=...          (comma-separated frontend origins; defaults to local dev origins)
JAWIVERSE_FRONTEND_DIR=...          (default ../react-frontend/build; served at / if it exists)

Docker (frontend + backend in one container)
The root Dockerfile builds react-frontend and serves it from this API, so the website,
/transliterate, /health and /docs all share one port and no CORS setup is needed.
docker build -t jawiverse-app:local .
docker run -d --name jawiverse-app -p 7860:7860 -v jawiverse-hf-cache:/data jawiverse-app:local
Open http://localhost:7860. The first start downloads the ~6.4 GB base model into the
jawiverse-hf-cache volume (several minutes); later starts reuse it. Requests fail until
the log shows "Application startup complete".

Hugging Face Docker Space
The repository root contains the Space Dockerfile and metadata. Select Docker when
creating the Space, then deploy the repository. The container runs on port 7860,
uses CPU inference, and downloads the base model into the Space cache at startup.
It serves the website itself, so the Space URL is the whole app.
CPU Basic provides 16 GB RAM; model loading may be close to that limit and CPU
inference is slow. Check the Space logs and `/health` before using it.

Frontend API URL: a production build with `REACT_APP_API_URL` unset calls the API on
its own origin (the combined container). Set it only to host the frontend separately,
to the API's HTTPS origin (without `/transliterate`), and add that frontend origin to
`JAWIVERSE_CORS_ORIGINS`. `npm start` (development) talks to http://localhost:8000.

Endpoint: POST /transliterate  ->  {"model": ..., "input": ..., "output": ...}

1. Jawi → Rumi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"jawi2rumi\",\"text\":\"ڤڠاجارن\"}"

2. Rumi → Jawi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"rumi2jawi\",\"text\":\"pengajaran\"}"

Jawi output is right-to-left; terminals may show it reversed, but the text itself is correct.
Show it in the frontend with dir="rtl" (or dir="auto").
