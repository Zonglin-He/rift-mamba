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


def roc_auc_score(y_true, y_score) -> float:
    true = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    positives = score[true == 1]
    negatives = score[true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    wins = 0.0
    for value in positives:
        wins += np.sum(value > negatives)
        wins += 0.5 * np.sum(value == negatives)
    return float(wins / (len(positives) * len(negatives)))


def average_precision_score(y_true, y_score) -> float:
    true = np.asarray(y_true).astype(bool)
    score = np.asarray(y_score, dtype=np.float64)
    if true.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked_true = true[order]
    precision = np.cumsum(ranked_true) / (np.arange(len(ranked_true)) + 1)
    return float(np.sum(precision * ranked_true) / true.sum())


def recall_at_k(pred_isin, dst_count, k: int | None = None) -> float:
    hits = np.asarray(pred_isin, dtype=bool)
    if k is not None:
        hits = hits[:, :k]
    counts = np.asarray(dst_count, dtype=np.float64)
    denom = np.maximum(counts, 1.0)
    return float(np.mean(hits.sum(axis=1) / denom))


def precision_at_k(pred_isin, k: int | None = None) -> float:
    hits = np.asarray(pred_isin, dtype=bool)
    if k is not None:
        hits = hits[:, :k]
    denom = hits.shape[1] if k is None else min(k, hits.shape[1])
    return float(np.mean(hits.sum(axis=1) / max(denom, 1)))


def mrr_at_k(pred_isin, k: int | None = None) -> float:
    hits = np.asarray(pred_isin, dtype=bool)
    if k is not None:
        hits = hits[:, :k]
    reciprocal = []
    for row in hits:
        indices = np.flatnonzero(row)
        reciprocal.append(0.0 if len(indices) == 0 else 1.0 / float(indices[0] + 1))
    return float(np.mean(reciprocal))


def ndcg_at_k(pred_isin, dst_count, k: int | None = None) -> float:
    hits = np.asarray(pred_isin, dtype=bool)
    if k is not None:
        hits = hits[:, :k]
    counts = np.asarray(dst_count, dtype=np.int64)
    discounts = 1.0 / np.log2(np.arange(hits.shape[1]) + 2.0)
    dcg = (hits * discounts).sum(axis=1)
    ideal = []
    for count in counts:
        top = min(max(int(count), 0), hits.shape[1])
        ideal.append(float(discounts[:top].sum()) if top else 0.0)
    ideal_arr = np.asarray(ideal, dtype=np.float64)
    scores = np.divide(dcg, ideal_arr, out=np.zeros_like(dcg, dtype=np.float64), where=ideal_arr > 0)
    return float(np.mean(scores))
