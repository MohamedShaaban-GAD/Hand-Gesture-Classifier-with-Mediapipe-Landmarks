import urllib.request
from pathlib import Path

import pandas as pd

from config import DATASET_DIR, DATASET_FILENAME, DATASET_URL


def load_hagrid(
    path: str | None = None,
    url: str = DATASET_URL,
) -> pd.DataFrame:
    """Download (if needed) and return the HAGRID hand-landmarks dataset."""
    if path is None:
        path = str(Path(DATASET_DIR) / DATASET_FILENAME)

    dataset_path = Path(path)
    if not dataset_path.is_file():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {dataset_path} ...")
        urllib.request.urlretrieve(url, dataset_path)

    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
