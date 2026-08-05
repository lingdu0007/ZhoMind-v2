# Deterministic production image: installs strictly from the committed uv.lock.
# `uv sync --frozen` fails the build whenever pyproject.toml and uv.lock drift,
# so the image never silently resolves dependencies outside the committed lock.
# The dev extras (pytest, ruff, pyright) are intentionally excluded; the project
# itself is installed from the lock, matching the previous `pip install .`.
FROM ghcr.io/astral-sh/uv:0.11.15-python3.11-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev

COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
