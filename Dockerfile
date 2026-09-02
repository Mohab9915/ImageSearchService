FROM python:3.11-slim

WORKDIR /app

# libgomp1 is torch's OpenMP runtime; libpq for psycopg2. Nothing else is
# needed -- there is no OpenCV, no zbar and no ffmpeg here, which is most of
# why this image is a fraction of the chatbot's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image. Downloading them on first request means
# the first customer to send a photograph waits for a 400MB download; baking
# them also means a Hugging Face outage cannot stop a deploy from serving.
ENV HF_HOME=/opt/hf
RUN python -c "\
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES as T;\
T.setdefault('siglip', 'SiglipTokenizer');\
from transformers import SiglipModel, SiglipProcessor;\
m='google/siglip-base-patch16-224';\
SiglipModel.from_pretrained(m); SiglipProcessor.from_pretrained(m);\
print('siglip cached')"

COPY . .

RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app /opt/hf
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["gunicorn", "main:app", "-c", "gunicorn_conf.py"]
