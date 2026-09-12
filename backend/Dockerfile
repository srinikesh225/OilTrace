# OILTRACE backend — FastAPI on Python 3.11.
# The container bundles the generated sample_data so it runs fully offline.
FROM python:3.11-slim

# rasterio/opencv need a few system libs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY sample_data ./sample_data

# Regenerate sample data at build time if it is missing (it is committed, so
# this is a safety net, not a network call).
RUN test -f sample_data/scene_normal/scene.tif || python -m scripts.make_sample_data

# Bind to the platform-provided $PORT (Render, Cloud Run, etc.), defaulting to
# 8000 for local `docker run`. Shell form so the variable is expanded.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
