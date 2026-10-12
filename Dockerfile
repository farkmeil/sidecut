FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr tesseract-ocr-eng mkvtoolnix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[all]'

ENV SIDECUT_CONFIG=/config/sidecut.yaml \
    SIDECUT_MODEL_DIR=/data/models \
    PYTHONUNBUFFERED=1

VOLUME ["/config", "/data"]
ENTRYPOINT ["sidecut"]
CMD ["watch"]
