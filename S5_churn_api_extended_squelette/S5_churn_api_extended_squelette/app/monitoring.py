"""Monitoring & drift detection for the Churn API.

🎯 Mission: implement the 4 functions below.

Drift metric: PSI (Population Stability Index).
- PSI < 0.10 : ok
- 0.10 <= PSI < 0.25 : warning (moderate drift)
- PSI >= 0.25 : critical (significant drift)
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import (
    LOG_PATH, BASELINE_TRAIN_PATH,
    PSI_OK_THRESHOLD, PSI_WARNING_THRESHOLD,
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]


def log_prediction(features: dict, prediction: dict) -> None:
    """Append one prediction event to LOG_PATH (JSONL format).

    Each line: {"timestamp": ISO8601, "features": {...}, "prediction": {...}}
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_recent_logs(n: int = 1000) -> list[dict]:
    """Read the last n entries from the JSONL log (returns [] if file missing)."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    records = [json.loads(line) for line in lines]
    return records[-n:]


def psi_numeric(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI for numeric features.

    Formula: sum over bins of (actual_pct - expected_pct) * log(actual_pct / expected_pct)
    Use np.linspace on expected to define bin edges, then np.histogram for both.
    Replace zeros with a small epsilon (1e-6) to avoid division/log issues.
    """
    eps = 1e-6
    edges = np.linspace(expected.min(), expected.max(), n_bins + 1)
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for categorical features (sum over modalities)."""
    eps = 1e-6
    categories = sorted(set(expected.unique()) | set(actual.unique()))

    expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0)

    expected_pct = expected_pct.replace(0, eps)
    actual_pct = actual_pct.replace(0, eps)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_drift(baseline: pd.DataFrame, recent: pd.DataFrame) -> dict:
    """Compute PSI per monitored feature. Return dict {feature: psi_value}."""
    scores = {}
    for feature in NUMERIC_FEATURES:
        scores[feature] = psi_numeric(
            baseline[feature].to_numpy(), recent[feature].to_numpy()
        )
    for feature in CATEGORICAL_FEATURES:
        scores[feature] = psi_categorical(baseline[feature], recent[feature])
    return scores


def status_from_max_psi(scores: dict) -> str:
    """Translate the worst PSI into a global status: ok / warning / critical / no_data."""
    if not scores:
        return "no_data"
    max_psi = max(scores.values())
    if max_psi < PSI_OK_THRESHOLD:
        return "ok"
    elif max_psi < PSI_WARNING_THRESHOLD:
        return "warning"
    else:
        return "critical"


def get_baseline() -> pd.DataFrame:
    """Load the training-time baseline DataFrame."""
    return pd.read_csv(BASELINE_TRAIN_PATH)
