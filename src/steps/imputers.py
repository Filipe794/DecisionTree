"""Etapa 3 (imputer)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import SEED


class KNNScaledImputer(BaseEstimator, TransformerMixin):
    """KNN com colunas padronizadas para escolher os vizinhos.

    O KNNImputer puro mede distância nas unidades originais, então Insulin e Glucose
    dominam. Aqui a distância é calculada em escala padronizada, os valores voltam para
    as unidades originais e só as células ausentes são trocadas (observados ficam idênticos).
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Ignora NaN no fit; coluna de desvio zero recebe scale_ = 1 (sem divisão por zero).
        # set_output("default"): internos em numpy, independente do set_config global
        self.scaler_ = StandardScaler().set_output(transform="default").fit(X)
        self.knn_ = (KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
                     .set_output(transform="default")
                     .fit(self.scaler_.transform(X)))
        return self

    def transform(self, X):
        back = self.scaler_.inverse_transform(self.knn_.transform(self.scaler_.transform(X)))
        mask = X.isna().to_numpy()
        out = X.to_numpy(dtype=float, copy=True)
        out[mask] = back[mask]
        return pd.DataFrame(out, columns=X.columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


def median():
    return SimpleImputer(strategy="median")


def mean():
    return SimpleImputer(strategy="mean")


def knn(n_neighbors=5):
    return KNNImputer(n_neighbors=n_neighbors)


def knn_scaled(n_neighbors=5):
    return KNNScaledImputer(n_neighbors)


def iterative(max_iter=10):
    return IterativeImputer(max_iter=max_iter, random_state=SEED)
