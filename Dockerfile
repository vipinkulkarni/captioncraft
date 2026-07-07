FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY prompts/ prompts/

# Hackathon injects nothing at runtime — bake in your API key at build time
ARG FIREWORKS_API_KEY
ARG FIREWORKS_MODEL
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ENV FIREWORKS_MODEL=${FIREWORKS_MODEL}
ENV PYTHONUNBUFFERED=1

# Container reads /input/tasks.json and writes /output/results.json
ENTRYPOINT ["python", "-m", "src.main"]