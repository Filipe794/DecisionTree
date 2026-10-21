"""Etapa 3 (outliers). Limites calculados SÓ no fit (treino do fold)."""
import numpy as np
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, TransformerMixin


def _numeric_cols(X):
    # Não mexe nas colunas de flag criadas por missing_flags
    return [c for c in X.columns if not c.endswith("_missing")]


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.cols_ = _numeric_cols(X)
        vals = X[self.cols_].to_numpy(dtype=float)
        self.low_ = np.nanquantile(vals, self.lower, axis=0)
        self.high_ = np.nanquantile(vals, self.upper, axis=0)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_] = X[self.cols_].clip(self.low_, self.high_, axis=1)
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        self.cols_ = _numeric_cols(X)
        vals = X[self.cols_].to_numpy(dtype=float)
        q1, q3 = np.nanquantile(vals, [0.25, 0.75], axis=0)
        self.low_ = q1 - self.k * (q3 - q1)
        self.high_ = q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_] = X[self.cols_].clip(self.low_, self.high_, axis=1)
        return X


def _drop_iqr_rows(X, y, k):
    # Executado só no fit (samplers são ignorados no predict), logo usa só o treino
    cols = _numeric_cols(X)
    q1, q3 = X[cols].quantile(0.25), X[cols].quantile(0.75)
    low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    outlier = ((X[cols] < low) | (X[cols] > high)).any(axis=1)  # NaN não conta
    return X[~outlier], y[~outlier]


def winsor(lower=0.01, upper=0.99):
    return Winsorizer(lower, upper)


def iqr_clip(k=1.5):
    return IQRClipper(k)


def iqr_drop(k=1.5):
    return FunctionSampler(func=_drop_iqr_rows, kw_args={"k": k}, validate=False)
