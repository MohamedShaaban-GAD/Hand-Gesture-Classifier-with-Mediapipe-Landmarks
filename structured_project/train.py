import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import (
    CV_FOLDS,
    N_ITER,
    PARAM_DISTRIBUTIONS,
    RANDOM_STATE,
)
from preprocessing import build_preprocessor


def _model_name(params: dict) -> str:
    """Extract a readable model class name from a param dict."""
    return type(params["model"]).__name__


def train(
    train_features: pd.DataFrame,
    train_labels_encoded: np.ndarray,
    top_k: int = 5,
) -> tuple[RandomizedSearchCV, list[dict]]:
    """
    Run RandomizedSearchCV, then refit the top-k models on the full
    training set so they can each be evaluated and logged individually.

    Returns
    -------
    search : the fitted RandomizedSearchCV object
    top_k_models : list of dicts, each with keys
        "rank", "model_name", "params", "cv_accuracy", "fitted_pipeline"
    """
    base_pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", RandomForestClassifier()),
    ])

    random_search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        cv=CV_FOLDS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="accuracy",
    )

    print("Starting RandomizedSearchCV ...")
    random_search.fit(train_features, train_labels_encoded)

    # --- Extract and refit top-k candidates ---
    results_df = pd.DataFrame(random_search.cv_results_)
    top_rows = results_df.sort_values("rank_test_score").head(top_k)

    top_k_models = []
    for _, row in top_rows.iterrows():
        params = row["params"]
        cv_acc = row["mean_test_score"]
        rank = int(row["rank_test_score"])
        name = _model_name(params)

        # Rebuild pipeline with these exact params and fit
        pipe = clone(base_pipeline)
        pipe.set_params(**params)
        pipe.fit(train_features, train_labels_encoded)

        top_k_models.append({
            "rank": rank,
            "model_name": name,
            "params": params,
            "cv_accuracy": cv_acc,
            "fitted_pipeline": pipe,
        })

        print(f"  Rank {rank} | {name} | CV acc = {cv_acc:.4f}")

    return random_search, top_k_models
