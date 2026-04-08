FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create an isolated virtual environment inside the image and install Python tooling there.
RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies first so Docker can reuse this layer when only app code changes.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY graders/ ./graders/
COPY data/ ./data/
COPY server/ ./server/

# Expose the OpenEnv server port.
EXPOSE 7860

# Health check uses the virtualenv Python because PATH already points at /opt/venv/bin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run the app with the virtualenv interpreter.
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
