# Cross-machine reuse experiment sandbox image. Build context = khala repo root.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY khala/ /app/khala/
COPY pyproject.toml /app/
COPY scripts/reuse-sandbox/box.py /app/box.py
RUN pip install --no-cache-dir --quiet -e /app psycopg[binary]
CMD ["sleep", "infinity"]