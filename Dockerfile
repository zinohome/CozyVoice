FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 创建非 root 用户
RUN groupadd --system cozy \
    && useradd --system --gid cozy --home /app --shell /usr/sbin/nologin cozy

COPY pyproject.toml /app/
COPY src /app/src
COPY config /app/config

RUN pip install -e . \
    && chown -R cozy:cozy /app

USER cozy

EXPOSE 8002
CMD ["uvicorn", "cozyvoice.main:app", "--host", "0.0.0.0", "--port", "8002"]
