import hashlib
import pandas as pd
from src.config import TRAIN_CSV, TEST_CSV, TARGET


def _load(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode antes: python -m src.make_split")
    df = pd.read_csv(path)
    return df.drop(TARGET, axis=1), df[TARGET]


def load_train():
    return _load(TRAIN_CSV)


def load_test():  # usar SOMENTE em final_test.py
    return _load(TEST_CSV)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
