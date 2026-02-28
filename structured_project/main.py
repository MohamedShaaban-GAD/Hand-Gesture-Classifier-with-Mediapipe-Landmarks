"""
Hand Gesture Classification — structured pipeline.
Reproduces the notebook workflow with MLflow experiment tracking.

Usage:
    python main.py              # train + evaluate
    python main.py --infer      # run webcam inference after training
"""

import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from config import (
    CV_FOLDS,
    DATASET_DIR,
    DATASET_FILENAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_OUTPUT_PATH,
    N_ITER,
    RANDOM_STATE,
)
from data_loader import load_hagrid
from evaluate import evaluate
from inference import run_webcam
from preprocessing import split_data
from train import train


def main(run_inference: bool = False):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    df = load_hagrid()

    # ── 2. Split & encode ─────────────────────────────────────
    print("\n[2/4] Splitting and encoding data ...")
    train_features, test_features, train_labels_enc, test_labels_enc, encoder = split_data(df)
    print(f"  Train samples : {len(train_labels_enc)}")
    print(f"  Test samples  : {len(test_labels_enc)}")
    print(f"  Classes       : {list(encoder.classes_)}")

    # ── 3. Train ──────────────────────────────────────────────
    print("\n[3/4] Training ...")
    search, top_k_models = train(train_features, train_labels_enc, top_k=5)

    # ── 4. Evaluate & log top 5 to MLflow ─────────────────────
    print("\n[4/4] Evaluating top 5 models & logging to MLflow ...")

    dataset_path = str(Path(DATASET_DIR) / DATASET_FILENAME)
    best_pipeline = None

    with mlflow.start_run(run_name="experiment_overview") as parent_run:
        # Parent: log search config + dataset
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_iter", N_ITER)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("scoring", "accuracy")
        mlflow.log_param("train_samples", len(train_labels_enc))
        mlflow.log_param("test_samples", len(test_labels_enc))
        mlflow.log_param("num_classes", len(encoder.classes_))
        mlflow.log_artifact(dataset_path, artifact_path="dataset")

        for entry in top_k_models:
            rank = entry["rank"]
            name = entry["model_name"]
            params = entry["params"]
            cv_acc = entry["cv_accuracy"]
            pipeline = entry["fitted_pipeline"]

            tag = f"rank{rank}_{name}"

            # Evaluate on test set
            eval_result = evaluate(
                pipeline, test_features, test_labels_enc, encoder, tag=tag,
            )
            metrics = eval_result["metrics"]

            # ── Child run ──
            with mlflow.start_run(
                run_name=f"rank_{rank}_{name}",
                nested=True,
            ):
                # Params
                mlflow.log_param("rank", rank)
                mlflow.log_param("model_type", name)
                for key, val in params.items():
                    param_name = key.replace("model__", "")
                    if param_name == "model":
                        continue
                    mlflow.log_param(param_name, str(val))

                # Metrics
                mlflow.log_metric("cv_accuracy", cv_acc)
                mlflow.log_metrics(metrics)

                # Artifacts: model, confusion matrix, classification report
                mlflow.sklearn.log_model(pipeline, "model")
                mlflow.log_artifact(eval_result["cm_path"])
                mlflow.log_artifact(eval_result["report_path"])
                mlflow.log_artifact(dataset_path, artifact_path="dataset")

                print(
                    f"  Rank {rank} | {name:25s} | "
                    f"CV={cv_acc:.4f}  Test={metrics['test_accuracy']:.4f}"
                )

            # Keep rank-1 as the best
            if rank == 1:
                best_pipeline = pipeline

    # ── 5. Save best model ────────────────────────────────────
    output_path = Path(MODEL_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_path)
    print(f"\nBest model saved to {output_path}")

    # ── 6. Optional live inference ────────────────────────────
    if run_inference:
        print("\nLaunching webcam inference ...")
        run_webcam(model_path=str(output_path), encoder=encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Classification Pipeline")
    parser.add_argument("--infer", action="store_true", help="Run webcam inference after training")
    args = parser.parse_args()
    main(run_inference=args.infer)
