# Stage 1: Install the application and its runtime dependencies.
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ ./src/

# Install only what the prediction API needs.
RUN pip install --no-cache-dir --prefix=/install \
        "fastapi" \
        "uvicorn" \
        "pydantic-settings" \
        "scikit-learn>=1.9.1" \
    && pip install --no-cache-dir --no-deps --prefix=/install .


# Stage 2: Minimal runtime image.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PRODML_PROJECT_ROOT=/app

# Create a non-root user and application directory.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

# Copy installed packages from the builder.
COPY --from=builder /install /usr/local

# Include the model required for standalone inference.
COPY --chown=appuser:appuser models/model.pkl /app/models/model.pkl

WORKDIR /app

USER appuser

EXPOSE 8000

# Check the API and verify that the model has loaded.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
