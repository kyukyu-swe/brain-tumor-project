# Container for Google Cloud Run (the AI backend: app.py + core/ + model).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

# OpenCV (used by grad-cam) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (much smaller than the default GPU build; Cloud Run has no GPU)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
COPY deploy/cloudrun/requirements.txt .
RUN pip install -r requirements.txt

COPY app.py ./
COPY core ./core
COPY model/best_model.pth ./model/best_model.pth
COPY data/samples ./data/samples

# Cloud Run tells the app which port to use in $PORT (usually 8080)
EXPOSE 8080
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} exec python app.py"]
