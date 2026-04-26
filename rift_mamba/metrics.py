"""Small metric helpers for experiment smoke tests."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true, y_pred) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape[0] == 0:
        raise ValueError("cannot score empty arrays")
    return float((true == pred).mean())


def mae(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.abs(true - pred).mean())


def rmse(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.square(true - pred).mean()))
