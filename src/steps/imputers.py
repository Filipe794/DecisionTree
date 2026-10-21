"""Etapa 4 (imputer)."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from src.config import SEED


def median():
    return SimpleImputer(strategy="median")


def mean():
    return SimpleImputer(strategy="mean")


def knn(n_neighbors=5):
    return KNNImputer(n_neighbors=n_neighbors)


def iterative(max_iter=10):
    return IterativeImputer(max_iter=max_iter, random_state=SEED)
