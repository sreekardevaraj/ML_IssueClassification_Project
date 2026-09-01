---
name: support-case-classification-engine
description: 'Build, extend, or productionize a Python IT support case classification engine using Pandas, FastText, XGBoost, DeBERTa, Llama 3.3 70B, FastAPI, and Docker. Use for two-stage ticket triage, 17-category taxonomy design, confidence-based LLM escalation, model evaluation, error analysis, and deployable ML services.'
argument-hint: '[dataset path, label schema, or implementation request]'
user-invocable: true
disable-model-invocation: false
---

# Support Case Classification Engine

## Outcome

Deliver a reproducible, production-ready text classification service for IT support cases. The default design is:

1. A fast first stage that creates FastText text embeddings and predicts one of 17 issue categories with XGBoost.
2. A confidence-aware ambiguity path that enriches the representation with DeBERTa embeddings and sends only eligible low-confidence cases to a Llama 3.3 70B decision layer.
3. A versioned FastAPI service, Docker image, tests, observability, and documented training/inference operations.

Treat `XX%` values in a product description as placeholders. Never report them as achieved metrics. Compute metrics from the supplied data and write the real values into generated reports.

## When to Use

- Create a new support-ticket classification project from an empty workspace.
- Add or repair the FastText/XGBoost baseline, DeBERTa fallback, or Llama decision layer.
- Optimize a 17-class support taxonomy using confusion matrices and error analysis.
- Expose trained classification artifacts through FastAPI and Docker.
- Improve low-confidence routing, batch inference, evaluation, or production readiness.

## Required Inputs and Assumptions

Before implementation, identify or ask for only missing information that blocks execution:

- Source data location and format, with the text column and target label column.
- Whether labels are already fixed at 17 categories or require taxonomy discovery.
- Runtime constraints: CPU/GPU, memory, latency target, throughput target, and deployment platform.
- Model access details for Llama 3.3 70B, including provider, endpoint contract, timeout, and privacy requirements. Never put credentials in source, Dockerfiles, or committed config.
- Whether training data contains PII, and the required redaction or retention policy.

If data or model credentials are unavailable, build the complete code path with explicit configuration and deterministic mocks/fixtures for tests. Do not invent training results.

## Procedure

### 1. Establish the repository

Create a structured Python project with separation between domain logic, model adapters, training jobs, API delivery, and tests. Prefer this layout unless existing project conventions require an equivalent:

```text
.
├── app/
│   ├── api/                 # FastAPI routes, schemas, exception handlers
│   ├── core/                # settings, logging, lifecycle, security
│   ├── domain/              # typed prediction and taxonomy contracts
│   ├── inference/           # stage routing and orchestration
│   └── models/              # FastText, XGBoost, DeBERTa, Llama adapters
├── configs/                 # non-secret YAML/TOML configuration
├── data/                    # .gitkeep only; never commit customer data
├── models/                  # ignored runtime artifacts and manifests
├── notebooks/               # exploratory work only, not production logic
├── scripts/                 # train, evaluate, export, and smoke-test entrypoints
├── tests/                   # unit, contract, integration, and API tests
├── Dockerfile
├── docker-compose.yml       # local service dependencies only when needed
├── .dockerignore
├── .env.example
├── pyproject.toml
└── README.md
```

Use typed Python, pinned or bounded dependencies, deterministic seeds, structured logging, and configuration from environment variables. Add a `.gitignore` that excludes data, credentials, caches, model binaries, and generated reports.

### 2. Define the taxonomy and data contract

- Represent the 17 categories in one versioned taxonomy module or config file; each label must have a stable ID, name, description, and optional examples.
- Validate labels on ingestion. Fail with a useful error for unknown, missing, or contradictory labels.
- Normalize text conservatively: preserve information useful for classification, remove only known boilerplate, and make PII handling explicit.
- Deduplicate before splitting and prevent the same case, thread, or customer-group leakage across train/validation/test.
- Use stratified splits where possible. Record split seed, dataset hash, taxonomy version, and preprocessing version in a manifest.
- Preserve the original case identifier only as metadata; do not use IDs or target-derived fields as features.

### 3. Implement the first-stage classifier

- Build a FastText adapter that supports loading a pretrained vector model and a documented fallback/training path. Make out-of-vocabulary handling deterministic.
- Convert each case into a fixed-size embedding using a clear pooling strategy and record the embedding dimension.
- Train XGBoost on the embeddings with multiclass objectives and reproducible parameters. Handle class imbalance through documented weights or sampling, not hidden behavior.
- Persist the XGBoost model, FastText model reference, preprocessing settings, taxonomy version, and feature schema together in a versioned artifact manifest.
- Expose calibrated or otherwise documented confidence scores. If probabilities are not calibrated, label them as model scores and do not call them probabilities.

### 4. Add evaluation and taxonomy optimization

Report, at minimum, Top-1, Top-2, and Top-3 accuracy, macro and weighted F1, per-class precision/recall, support counts, calibration or confidence-bin performance, and a normalized confusion matrix. Include a no-data and single-class fixture test.

Use confusion-matrix findings to propose taxonomy changes, but do not silently change labels. A taxonomy change must include:

- The proposed merge/split or renamed labels.
- Evidence: confusion counts/rates, representative anonymized examples, and business rationale.
- A migration mapping for historical labels.
- A new taxonomy version and a before/after evaluation comparison.

LLM-assisted error analysis must be an offline, reproducible analysis job. Redact sensitive text, constrain the prompt to evidence supplied, save the prompt/template version, and treat model suggestions as hypotheses requiring human review. Do not use an LLM to manufacture ground truth.

### 5. Implement the ambiguity path

Route a case to the second stage only when explicit policy conditions are met, such as low first-stage confidence, a small top-1/top-2 margin, an out-of-distribution signal, or an escalation category. Keep thresholds in configuration and include the routing reason in the response.

- Generate DeBERTa embeddings through an adapter with batching, device selection, truncation, and model version recorded.
- Define how DeBERTa features combine with the FastText/XGBoost result: a separately trained reranker/classifier, feature concatenation, or another documented strategy. Do not imply that embeddings alone improve accuracy.
- Use a Llama 3.3 70B adapter behind a provider-neutral interface. Validate structured output against a schema containing category, confidence/decision status, rationale, and model metadata.
- Constrain Llama outputs to the known taxonomy, reject invalid labels, bound token/time usage, redact prompts, and never log secrets or raw sensitive tickets.
- Make external LLM calls opt-in for production, timeout-bounded, retry-limited, and observable. Define a deterministic fallback when the provider is unavailable.
- Evaluate the low-confidence cohort separately and compare against the first stage, including coverage, latency, cost, abstention rate, and accuracy. Report the observed improvement only.

### 6. Build the service contract

Provide FastAPI endpoints with Pydantic schemas and OpenAPI documentation:

- `GET /health/live` for process liveness.
- `GET /health/ready` for loaded artifacts and dependency readiness.
- `POST /v1/predict` for one case, returning final category, ranked candidates, stage used, confidence semantics, taxonomy/model versions, routing reason, and request ID.
- `POST /v1/predict/batch` with bounded batch size and per-item validation.
- `GET /v1/model-info` for non-secret artifact and taxonomy metadata.

Keep prediction logic independent of HTTP so it can be tested and used by batch jobs. Return stable error shapes, enforce request size limits, validate content types, and add correlation IDs. Do not expose raw prompts, customer text, credentials, or internal stack traces.

### 7. Containerize and operate it

- Use a small multi-stage Docker build, a non-root runtime user, a pinned Python base image, and a predictable application entrypoint.
- Add a Docker healthcheck against the liveness or readiness endpoint and document CPU/GPU variants if relevant.
- Load model artifacts at startup or lazily with explicit readiness state; fail fast on incompatible manifests.
- Add JSON logs, latency and stage counters, error counters, escalation/provider metrics, and request IDs. Avoid logging ticket text by default.
- Document model download/licensing assumptions, environment variables, resource requirements, retraining, rollback, and artifact promotion.
- Add CI checks for formatting, linting, type checking, unit tests, API contract tests, Docker build, and a smoke test with fixture artifacts.

### 8. Validate before declaring completion

Run the narrowest relevant check after each implementation slice, then the full validation suite. Completion requires:

- A clean import and application startup with fixture artifacts.
- Unit tests for preprocessing, taxonomy validation, stage routing, confidence thresholds, and provider failure behavior.
- Evaluation tests proving Top-K ranking semantics and metric calculations.
- API tests for valid, invalid, oversized, batch, readiness, and model-info requests.
- A Docker build and container smoke test where Docker is available.
- A README with exact commands for setup, training, evaluation, local serving, prediction, and deployment.
- A report that distinguishes measured metrics, unavailable metrics, and future targets.

## Design Decisions to Preserve

- FastText plus XGBoost is the default low-latency path; expensive models are not called for every request.
- DeBERTa and Llama are adapters behind interfaces so they can be mocked, replaced, or disabled without changing API/domain code.
- Taxonomy, thresholds, model versions, and metric definitions are explicit and versioned.
- Abstention or human review is preferable to an invalid or fabricated category when confidence is inadequate.
- Customer text and model provider credentials are sensitive by default.

## Final Response Format

When using this skill, summarize:

1. What was implemented and the key files.
2. The measured evaluation results, or exactly which results could not be measured.
3. Commands run and their outcomes.
4. Configuration, model-artifact, data, and credential prerequisites.
5. Any known limitations, especially unvalidated accuracy, latency, cost, or provider behavior.
