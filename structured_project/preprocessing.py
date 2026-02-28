import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from config import RANDOM_STATE, TEST_SIZE

# Column groups
X_COLS = [f"x{i}" for i in range(1, 22)]
Y_COLS = [f"y{i}" for i in range(1, 22)]
Z_COLS = [f"z{i}" for i in range(1, 22)]


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize x/y coordinates relative to landmark 1, scaled by landmark 13."""
    data = data.copy()
    data[X_COLS] = data[X_COLS].subtract(data["x1"], axis=0) / data["x13"].values.reshape(-1, 1)
    data[Y_COLS] = data[Y_COLS].subtract(data["y1"], axis=0) / data["y13"].values.reshape(-1, 1)
    return data


def build_preprocessor() -> Pipeline:
    """Return the preprocessing pipeline (normalization step)."""
    return Pipeline([
        ("normalized", FunctionTransformer(normalize, validate=False, feature_names_out="one-to-one")),
    ])


def split_data(df: pd.DataFrame):
    """Split into train/test features and encoded labels."""
    features = df.drop("label", axis=1)
    labels = df["label"]

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    encoder = LabelEncoder()
    train_labels_encoded = encoder.fit_transform(train_labels)
    test_labels_encoded = encoder.transform(test_labels)

    return (
        train_features, test_features,
        train_labels_encoded, test_labels_encoded,
        encoder,
    )
