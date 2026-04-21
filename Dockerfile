FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src
COPY config /app/config

RUN pip install -e .

EXPOSE 8002
CMD ["uvicorn", "cozyvoice.main:app", "--host", "0.0.0.0", "--port", "8002"]
