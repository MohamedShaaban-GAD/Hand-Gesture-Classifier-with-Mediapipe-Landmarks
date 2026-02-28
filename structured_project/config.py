import logging
import warnings

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Silence noisy logs that PowerShell shows as red
logging.getLogger("alembic").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

# --- Paths ---
DATASET_URL = (
    "https://raw.githubusercontent.com/MohamedShaaban-GAD/"
    "Hand-Gesture-Classifier-with-Mediapipe-Landmarks/refs/heads/main/"
    "hand_landmarks_data%20-%20hand_landmarks_data.csv"
)
DATASET_DIR = "./datasets"
DATASET_FILENAME = "hand_landmarks_data.csv"
MODEL_OUTPUT_PATH = "./artifacts/gestures_model.pkl"

# --- Data split ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Search ---
CV_FOLDS = 5
N_ITER = 60

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "Hand_Gesture_Classification"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# --- Hyperparameter search distributions ---
PARAM_DISTRIBUTIONS = [
    {
        "model": [
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
        ],
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    {
        "model": [SVC(gamma="auto")],
        "model__C": uniform(0.1, 20),
        "model__kernel": ["rbf", "linear"],
    },
    {
        "model": [RandomForestClassifier(random_state=RANDOM_STATE)],
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(5, 30),
    },
    {
        "model": [LogisticRegression(solver="lbfgs", max_iter=1000)],
        "model__C": uniform(0.01, 100),
    },
]
