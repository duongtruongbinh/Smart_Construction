# Smart Construction

This pack contains a minimal ZenML pipeline to run rule-based Active/Idle classification from IMU (accelerometer + gyroscope).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
zenml up
python -m scripts.run_pipeline_rule_based
```

Configure input CSV paths inside `scripts/run_pipeline_rule_based.py` or wire a config reader.

## Layout
- `src/core`: shared interfaces and utilities (time prep, norms, resample)
- `src/predictors/rule_based.py`: robust z-score + hysteresis + debounce model
- `src/zen_pipelines/rule_based/steps`: ZenML steps (ingest → normalize → denoise → resample → feature_window → rule_classify → persist)
- `src/zen_pipelines/rule_based/pipeline.py`: the orchestrated pipeline
- `src/configs/pipeline.yaml`: minimal configuration
- `tests/`: basic contract tests

# 🏗️ Smart Construction

This project contains a minimal **ZenML** pipeline for rule-based *Active / Idle* classification from IMU data (accelerometer + gyroscope).

---

## ⚡ Quickstart

```bash
# 1️⃣ Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Enable pre-commit checks for code quality
pre-commit install
pre-commit run --all-files

# 4️⃣ Start ZenML server and run pipeline
zenml up
python -m scripts.run_pipeline_rule_based
