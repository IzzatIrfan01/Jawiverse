# Stage 1: build the React frontend into static files
FROM node:20-alpine AS frontend

WORKDIR /frontend
COPY react-frontend/package.json react-frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY react-frontend/ ./
RUN npm run build

# Stage 2: API + model, also serving the built frontend at /
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    JAWIVERSE_DEVICE=cpu \
    JAWIVERSE_MAX_NEW_TOKENS=128 \
    JAWIVERSE_MODEL_PATH=/app/jawiverse_server/nirnama/v1/final \
    HF_HOME=/data/.cache/huggingface

WORKDIR /app

COPY backend/requirements-cpu.txt /tmp/requirements-cpu.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements-cpu.txt

COPY backend /app/backend
COPY jawiverse_server/common.py /app/jawiverse_server/common.py
COPY jawiverse_server/infer_jawi_rumi.py /app/jawiverse_server/infer_jawi_rumi.py
COPY jawiverse_server/nirnama/v1/final /app/jawiverse_server/nirnama/v1/final
COPY --from=frontend /frontend/build /app/react-frontend/build

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]