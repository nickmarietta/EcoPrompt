"""V1 eco-score: relative quality-adjusted compute efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TOKEN_WEIGHT = 1.0
LATENCY_WEIGHT = 0.5
RETRIEVAL_WEIGHT = 0.2
DEFAULT_QUALITY_SCORE = 1.0
NORMALIZATION_BASELINE = 0.001
ECO_VERSION = "v1"

MODEL_SIZE_FACTORS = {
    "small": 1.0,
    "medium": 2.0,
    "large": 4.0,
}


@dataclass
class RunMetrics:
    input_tokens: float
    output_tokens: float
    attempts: int = 1
    latency_ms: float | None = None
    retrieval_count: int = 0
    model_size: str = "small"
    quality_score: float = DEFAULT_QUALITY_SCORE

    @property
    def total_tokens(self) -> float:
        return max(0.0, self.input_tokens) + max(0.0, self.output_tokens)


def infer_model_size(model_name: str | None) -> str:
    normalized = (model_name or "").lower()
    if any(marker in normalized for marker in ("0.5b", "1b", "1.5b", "2b", "3b", "mini", "small")):
        return "small"
    if any(marker in normalized for marker in ("7b", "8b", "9b", "10b", "11b", "12b", "13b", "14b")):
        return "medium"
    if any(marker in normalized for marker in ("30b", "32b", "34b", "70b", "72b", "large")):
        return "large"
    return "medium"


def compute_compute_proxy(run: RunMetrics) -> float:
    model_factor = MODEL_SIZE_FACTORS.get(run.model_size, MODEL_SIZE_FACTORS["medium"])
    token_component = run.total_tokens * TOKEN_WEIGHT
    attempt_component = max(1, run.attempts)
    latency_component = (max(0.0, run.latency_ms or 0.0) / 1000.0) * LATENCY_WEIGHT
    retrieval_component = max(0, run.retrieval_count) * RETRIEVAL_WEIGHT
    return (token_component * model_factor * attempt_component) + latency_component + retrieval_component


def compute_eco_score_raw(run: RunMetrics) -> float:
    compute_proxy = compute_compute_proxy(run)
    if compute_proxy <= 0:
        return 0.0
    return max(0.0, run.quality_score) / compute_proxy


def normalize_score(score_raw: float, baseline: float = NORMALIZATION_BASELINE) -> float:
    if baseline <= 0:
        baseline = NORMALIZATION_BASELINE
    return round(min(100.0, (max(0.0, score_raw) / baseline) * 100.0), 2)


def build_eco_score_payload(run: RunMetrics) -> dict[str, Any]:
    compute_proxy = compute_compute_proxy(run)
    eco_raw = compute_eco_score_raw(run)
    eco_score = normalize_score(eco_raw)
    return {
        "eco_version": ECO_VERSION,
        "eco_score": eco_score,
        "eco_score_raw": round(eco_raw, 8),
        "eco_breakdown": {
            "input_tokens": round(max(0.0, run.input_tokens), 2),
            "output_tokens": round(max(0.0, run.output_tokens), 2),
            "attempts": max(1, run.attempts),
            "latency_ms": round(max(0.0, run.latency_ms or 0.0), 2),
            "retrieval_count": max(0, run.retrieval_count),
            "model_size": run.model_size,
            "quality_score": round(max(0.0, run.quality_score), 4),
            "compute_proxy": round(compute_proxy, 6),
        },
    }
