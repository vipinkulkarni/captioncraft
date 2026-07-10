FROM python:3.11-slim

WORKDIR /app

COPY requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt

COPY src/ src/
COPY prompts/ prompts/

# Hackathon injects nothing at runtime — bake credentials and models at build time
# (Track 2: "use your own credentials inside the container")
ARG FIREWORKS_API_KEY
ARG VISION_MODEL=accounts/fireworks/models/minimax-m3
ARG CAPTION_MODEL=accounts/fireworks/models/deepseek-v4-flash

ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ENV VISION_MODEL=${VISION_MODEL}
ENV CAPTION_MODEL=${CAPTION_MODEL}
ENV PYTHONUNBUFFERED=1
ENV PARALLEL_STYLES=1
ENV DOWNLOAD_READ_TIMEOUT=180
ENV DESCRIBE_MAX_TOKENS=1200
ENV DESCRIBE_TEMPERATURE=0.2
ENV STYLE_MAX_TOKENS=400
ENV FRAME_INTERVAL_S=4
ENV FRAME_COUNT_MIN=8
ENV FRAME_COUNT_MAX=24
ENV FRAME_WIDTH=384
ENV API_TIMEOUT_S=45
ENV DESCRIBE_MAX_ATTEMPTS=2
ENV STYLE_MAX_ATTEMPTS=3
ENV FRIENDLY_FAILURES=1

# Container reads /input/tasks.json and writes /output/results.json
ENTRYPOINT ["python", "-m", "src.main"]
