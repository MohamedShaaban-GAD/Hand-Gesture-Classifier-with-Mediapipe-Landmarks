# Hand Gesture Classification using Mediapipe Landmarks

A structured ML pipeline for classifying hand gestures from MediaPipe landmark data, with full **MLflow** experiment tracking.

## Project Structure

```
structured_project/
├── config.py            # Configuration: paths, hyperparams, MLflow settings
├── data_loader.py       # Downloads & loads the HAGRID dataset
├── preprocessing.py     # Landmark normalization, train/test split, label encoding
├── train.py             # RandomizedSearchCV across 4 model families
├── evaluate.py          # Test metrics, confusion matrix, classification report
├── inference.py         # Real-time webcam gesture recognition
├── main.py              # Pipeline orchestrator + MLflow logging
├── requirements.txt     # Dependencies
└── artifacts/           # Saved models & evaluation outputs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (train + evaluate + log to MLflow)
python main.py

# Run with live webcam inference after training
python main.py --infer
```

## Pipeline Overview

| Step | Description |
|------|-------------|
| 1. **Load Data** | Downloads HAGRID hand-landmarks CSV (25,675 samples, 63 features, 18 gesture classes) |
| 2. **Preprocess** | Normalize x/y coordinates relative to wrist landmark, 80/20 stratified split |
| 3. **Train** | `RandomizedSearchCV` (60 iterations, 5-fold CV) across XGBoost, SVM, Random Forest, Logistic Regression |
| 4. **Evaluate** | Computes test metrics for top 5 models, generates confusion matrices and reports |
| 5. **Save** | Best model saved to `artifacts/gestures_model.pkl` |

## MLflow Experiment Tracking

All experiments are tracked with MLflow using a local **SQLite** backend.

### Viewing the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open **http://localhost:5000** in your browser.

### Run Structure

The pipeline creates a **parent run** with **5 nested child runs** — one for each of the top 5 models found during hyperparameter search.

```
experiment_overview (parent run)
├── rank_1_XGBClassifier      (child run)
├── rank_2_XGBClassifier      (child run)
├── rank_3_XGBClassifier      (child run)
├── rank_4_RandomForest...    (child run)
└── rank_5_XGBClassifier      (child run)
```

### What Gets Logged

#### Parent Run — `experiment_overview`

| Category   | Logged Items |
|------------|-------------|
| **Params** | `cv_folds`, `n_iter`, `random_state`, `scoring`, `train_samples`, `test_samples`, `num_classes` |
| **Artifacts** | The full dataset CSV |

#### Each Child Run — `rank_N_ModelName`

| Category    | Logged Items |
|-------------|-------------|
| **Params**  | `rank`, `model_type`, and all model-specific hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `C`, `kernel`, `subsample`, `colsample_bytree`) |
| **Metrics** | `cv_accuracy`, `test_accuracy`, `test_f1_macro`, `test_precision_macro`, `test_recall_macro` |
| **Artifacts** | Serialized sklearn pipeline, confusion matrix PNG, classification report TXT, dataset CSV |

### Comparing Models in MLflow

1. Open the MLflow UI
2. Expand the `experiment_overview` parent run to see all 5 child runs
3. Select the runs you want to compare
4. Click **Compare** to see side-by-side metrics, parameters, and artifacts

## Model Families Searched

| Model | Key Hyperparameters |
|-------|-------------------|
| **XGBClassifier** | `n_estimators` (100–500), `max_depth` (3–10), `learning_rate` (0.01–0.31), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0) |
| **SVC** | `C` (0.1–20.1), `kernel` (rbf, linear) |
| **RandomForestClassifier** | `n_estimators` (50–300), `max_depth` (5–30) |
| **LogisticRegression** | `C` (0.01–100.01) |

## Gesture Classes

The model classifies 18 hand gestures:

`call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`
