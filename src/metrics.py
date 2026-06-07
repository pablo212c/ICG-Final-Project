from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def _safe_float(value: float) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return float("nan")
    return float(value)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Metric inputs must have the same shape: {true.shape} vs {pred.shape}")

    diff = pred - true
    metrics = {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
    }

    if len(true) < 2 or np.std(true) == 0 or np.std(pred) == 0:
        metrics.update({"spearman": float("nan"), "kendall": float("nan")})
        return metrics

    metrics["spearman"] = _safe_float(stats.spearmanr(true, pred).correlation)
    metrics["kendall"] = _safe_float(stats.kendalltau(true, pred).correlation)
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key in ("mse", "mae", "spearman", "kendall", "attr_mae"):
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key}={value:.4f}" if not math.isnan(value) else f"{key}=nan")
    return ", ".join(parts)


def pairwise_accuracy(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    correct = 0
    total = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] == true[j]:
                continue
            total += 1
            correct += int((true[i] - true[j]) * (pred[i] - pred[j]) > 0)
    return float(correct / total) if total else float("nan")


def group_ranking_metrics(
    rows: list[dict[str, object]],
    group_size: int = 5,
    group_count: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2.")
    if len(rows) < group_size:
        return {
            "group_spearman": float("nan"),
            "group_kendall": float("nan"),
            "pairwise_accuracy": float("nan"),
            "top1_accuracy": float("nan"),
        }

    rng = np.random.default_rng(seed)
    spearman_values: list[float] = []
    kendall_values: list[float] = []
    pairwise_values: list[float] = []
    top1_hits = 0
    top1_total = 0

    for _ in range(group_count):
        indices = rng.choice(len(rows), size=group_size, replace=False)
        true = np.asarray([float(rows[i]["true_score"]) for i in indices], dtype=np.float64)
        pred = np.asarray([float(rows[i]["pred_score"]) for i in indices], dtype=np.float64)

        if np.std(true) > 0 and np.std(pred) > 0:
            spearman = stats.spearmanr(true, pred).correlation
            kendall = stats.kendalltau(true, pred).correlation
            if not math.isnan(spearman):
                spearman_values.append(float(spearman))
            if not math.isnan(kendall):
                kendall_values.append(float(kendall))

        pair_acc = pairwise_accuracy(true, pred)
        if not math.isnan(pair_acc):
            pairwise_values.append(pair_acc)

        true_best = np.flatnonzero(true == true.max())
        pred_best = int(np.argmax(pred))
        top1_hits += int(pred_best in set(true_best.tolist()))
        top1_total += 1

    return {
        "group_spearman": float(np.mean(spearman_values)) if spearman_values else float("nan"),
        "group_kendall": float(np.mean(kendall_values)) if kendall_values else float("nan"),
        "pairwise_accuracy": float(np.mean(pairwise_values)) if pairwise_values else float("nan"),
        "top1_accuracy": float(top1_hits / top1_total) if top1_total else float("nan"),
    }


def format_group_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key in ("group_spearman", "group_kendall", "pairwise_accuracy", "top1_accuracy"):
        value = metrics[key]
        parts.append(f"{key}={value:.4f}" if not math.isnan(value) else f"{key}=nan")
    return ", ".join(parts)
