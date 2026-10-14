## Development workflow

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Lint and format

```bash
uv run ruff check src tests
uv run black --check src tests
```

### 3. Run tests with coverage

```bash
uv run pytest -v --cov=src/prodml --cov-report=term-missing
```

### 4. Train the model

```bash
uv run python -m prodml.train
```

This trains the model, evaluates validation MAE and RMSE, and saves the artifact to `models/model.pkl`.

### 5. Serve the API

```bash
uv run uvicorn prodml.api.main:app --reload --port 8000
```

The API command becomes runnable after Module 05, when `src/prodml/api/main.py` is implemented.
