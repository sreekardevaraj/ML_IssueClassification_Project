# Support Case Classification Engine

Production-oriented FastAPI service for IT support case triage across 17 categories.

## Current state

The repository serves only versioned TF-IDF/XGBoost artifacts. Train the bundle before starting production inference. TF-IDF, XGBoost, and optional DeBERTa integration points are isolated behind adapters.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for OpenAPI. Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/predict `
  -ContentType 'application/json' `
  -Body '{"case_id":"C-1001","text":"My VPN is not connecting"}'
```

## Streamlit console

Install the UI extra and start it in a second terminal while the API is running:

```powershell
pip install -e ".[ui]"
streamlit run app/ui/streamlit_app.py
```

Open `http://127.0.0.1:8501`. The console supports single-case issue detection, ranked alternatives, confidence/routing state, and request tracking. It calls FastAPI rather than duplicating inference logic.

## API

- `GET /health/live` process liveness
- `GET /health/ready` artifact readiness
- `POST /v1/predict` single case classification
- `POST /v1/predict/batch` up to 100 cases
- `GET /v1/model-info` non-secret model metadata

The response includes ranked candidates, stage, confidence semantics, taxonomy version, model version, routing reason, and request ID. Low-confidence cases abstain unless LLM escalation is explicitly enabled.

## Automatic result storage

Every successful single or batch prediction is automatically appended to:

```text
results/predictions.jsonl
```

Each line contains prediction metadata, ranked candidates, stage, confidence, model version, taxonomy version, and request ID. Case text is excluded by default. Set `SUPPORT_PERSIST_CASE_TEXT=true` only when your data-retention policy permits storing the original ticket text.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

The API is available on port 8000 and the Streamlit console on port 8501.

The image runs as a non-root user and exposes port 8000. Do not place provider credentials in `.env` committed to source control.

## Production model path

1. Prepare a labeled dataset with `text` and `label` columns. Deduplicate and create leakage-safe stratified splits. For a local end-to-end demonstration, generate exactly 5,000 synthetic records:

```powershell
python scripts/generate_dataset.py --rows 5000 --output data/cases_5000.csv
```
2. Fit TF-IDF features and train XGBoost, then persist a manifest containing taxonomy, preprocessing, feature schema, and model versions:

```powershell
python scripts/train.py --data data/cases_5000.csv --output-dir models/production
python scripts/evaluate.py --data data/cases_5000.csv --artifact-dir models/production
```

3. Set `SUPPORT_ARTIFACT_DIR=models/production` to serve the trained bundle.
4. Add the optional DeBERTa artifact and configure its device/model identifier.
5. Evaluate Top-1/Top-2/Top-3 accuracy, macro/weighted F1, per-class metrics, confusion matrix, calibration, escalation coverage, latency, cost, and abstention rate.

The training job fits TF-IDF on the training records and saves `tfidf.joblib`. Synthetic metrics are pipeline-validation results, not customer-performance results.

The generated 5K benchmark reports its metrics in `models/production/evaluation.json`. Because those records use controlled category templates, the metrics validate pipeline mechanics only; they do not establish production generalization. Replace the synthetic CSV with real support narratives before making accuracy claims.

## Checks

```powershell
pytest
python -m compileall app
```

## CI/CD

The GitHub Actions workflow at `.github/workflows/ci-cd.yml` runs on pull requests and pushes to `main`. It compiles the application, runs tests, and builds both the API and Streamlit Docker images. Pushes to `main` and version tags publish images to GitHub Container Registry:

```text
ghcr.io/<owner>/support-case-classification-engine
ghcr.io/<owner>/support-case-classification-engine-ui
```

In the GitHub repository settings, enable Actions and ensure the workflow has package write permission under **Settings > Actions > General > Workflow permissions**. Model artifacts and customer data remain excluded from the repository; publish them through your approved artifact storage or deployment process.
