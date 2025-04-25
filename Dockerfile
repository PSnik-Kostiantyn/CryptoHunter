FROM python:3.12.4-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends

RUN pip install poetry

WORKDIR /app
COPY . .
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

EXPOSE 8000