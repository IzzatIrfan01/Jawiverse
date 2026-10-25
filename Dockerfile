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

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]