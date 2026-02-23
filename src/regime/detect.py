"""
Regime detection from activation snapshots.

Takes activation vectors captured at sequential checkpoints and classifies
each interval into STABLE, CHAOTIC, or TRANSITION regimes based on:
1. Activation similarity (cosine) between consecutive checkpoints
2. Activation variance over a sliding window
3. Loss curvature proxy (second finite difference)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    STABLE = "stable"
    CHAOTIC = "chaotic"
    TRANSITION = "transition"


@dataclass
class RegimeLabel:
    step: int
    regime: Regime
    similarity: float
    variance: float
    curvature: float


@dataclass
class RegimeResult:
    """Full result of regime detection across a training run."""
    labels: list[RegimeLabel]
    steps: np.ndarray
    similarities: np.ndarray
    variances: np.ndarray
    curvatures: np.ndarray
    losses: np.ndarray
    thresholds: dict

    @property
    def regime_labels(self) -> list[str]:
        return [l.regime.value for l in self.labels]

    @property
    def stable_fraction(self) -> float:
        if not self.labels:
            return 0.0
        return sum(1 for l in self.labels if l.regime == Regime.STABLE) / len(self.labels)

    @property
    def chaotic_fraction(self) -> float:
        if not self.labels:
            return 0.0
        return sum(1 for l in self.labels if l.regime == Regime.CHAOTIC) / len(self.labels)

    def summary(self) -> str:
        n = len(self.labels)
        if n == 0:
            return "No regime labels computed."
        stable = sum(1 for l in self.labels if l.regime == Regime.STABLE)
        chaotic = sum(1 for l in self.labels if l.regime == Regime.CHAOTIC)
        transition = sum(1 for l in self.labels if l.regime == Regime.TRANSITION)
        return (
            f"Regimes ({n} intervals):\n"
            f"  STABLE:     {stable:3d} ({100*stable/n:.1f}%)\n"
            f"  CHAOTIC:    {chaotic:3d} ({100*chaotic/n:.1f}%)\n"
            f"  TRANSITION: {transition:3d} ({100*transition/n:.1f}%)\n"
            f"  Similarity: mean={self.similarities.mean():.4f}, "
            f"std={self.similarities.std():.4f}\n"
            f"  Thresholds: high={self.thresholds['high']:.4f}, "
            f"low={self.thresholds['low']:.4f}"
        )


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between corresponding rows of a and b.

    Args:
        a, b: Arrays of shape (num_probes, hidden_dim)

    Returns:
        Array of shape (num_probes,) with per-probe cosine similarities
    """
    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    # Dot product per row
    return np.sum(a_norm * b_norm, axis=1)


def compute_similarities(snapshots: list[np.ndarray]) -> np.ndarray:
    """
    Compute mean cosine similarity between consecutive checkpoint activations.

    Args:
        snapshots: List of arrays, each shape (num_probes, hidden_dim),
                   one per checkpoint in chronological order.

    Returns:
        Array of shape (num_checkpoints - 1,) with mean similarity per interval.
    """
    similarities = []
    for i in range(len(snapshots) - 1):
        per_probe_sim = cosine_similarity_batch(snapshots[i], snapshots[i + 1])
        similarities.append(per_probe_sim.mean())
    return np.array(similarities)


def compute_variance(snapshots: list[np.ndarray], window: int = 5) -> np.ndarray:
    """
    Compute activation variance over a sliding window of checkpoints.

    For each checkpoint, look at the window of recent checkpoints and compute
    how much each probe's activation is varying.

    Args:
        snapshots: List of arrays, each (num_probes, hidden_dim)
        window: Number of checkpoints in the sliding window

    Returns:
        Array of shape (num_checkpoints - 1,) with mean variance per interval.
    """
    n = len(snapshots)
    variances = []
    for i in range(1, n):
        start = max(0, i - window + 1)
        window_acts = np.stack(snapshots[start:i + 1], axis=0)  # (w, probes, dim)
        # Variance across the window dimension, then mean over probes and dims
        var = window_acts.var(axis=0).mean()
        variances.append(var)
    return np.array(variances)


def compute_curvature(losses: np.ndarray) -> np.ndarray:
    """
    Loss curvature proxy via second finite difference.

    curvature[i] = L(i+1) - 2*L(i) + L(i-1)

    Near zero = smooth descent. Large magnitude = curvature change.

    Args:
        losses: Array of loss values at each checkpoint

    Returns:
        Array of shape (num_checkpoints - 2,) with curvature at each interior point.
        Padded with NaN at the start to align with similarity array.
    """
    if len(losses) < 3:
        return np.full(max(len(losses) - 1, 0), np.nan)

    curv = losses[2:] - 2 * losses[1:-1] + losses[:-2]

    # Pad with NaN at the start so index aligns with similarities
    # similarities[i] corresponds to interval (checkpoint_i, checkpoint_i+1)
    # curvature[i] corresponds to checkpoint_i+1 (needs i, i+1, i+2)
    padded = np.concatenate([[np.nan], curv])
    # Trim or pad to match similarities length
    target_len = len(losses) - 1
    if len(padded) < target_len:
        padded = np.concatenate([padded, np.full(target_len - len(padded), np.nan)])
    return padded[:target_len]


def detect_regimes(
    snapshots: list[np.ndarray],
    losses: np.ndarray,
    steps: np.ndarray,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    percentile_high: float = 70,
    percentile_low: float = 30,
    variance_window: int = 5,
) -> RegimeResult:
    """
    Detect training regimes from activation snapshots and loss values.

    If thresholds are not provided, they are computed from the similarity
    distribution using the specified percentiles.

    Args:
        snapshots: List of activation arrays, one per checkpoint
        losses: Loss values at each checkpoint
        steps: Training step numbers for each checkpoint
        threshold_high: Similarity above this = STABLE. Auto-computed if None.
        threshold_low: Similarity below this = CHAOTIC. Auto-computed if None.
        percentile_high: Percentile for auto threshold_high
        percentile_low: Percentile for auto threshold_low
        variance_window: Window size for variance computation

    Returns:
        RegimeResult with labels, metrics, and thresholds
    """
    similarities = compute_similarities(snapshots)
    variances = compute_variance(snapshots, window=variance_window)
    curvatures = compute_curvature(losses)

    # Auto-compute thresholds from similarity distribution
    if threshold_high is None:
        threshold_high = float(np.percentile(similarities, percentile_high))
    if threshold_low is None:
        threshold_low = float(np.percentile(similarities, percentile_low))

    # Ensure high > low
    if threshold_high <= threshold_low:
        mid = (threshold_high + threshold_low) / 2
        threshold_high = mid + 0.001
        threshold_low = mid - 0.001

    # Classify each interval
    labels = []
    for i in range(len(similarities)):
        sim = similarities[i]
        var = variances[i] if i < len(variances) else 0.0
        curv = curvatures[i] if i < len(curvatures) else 0.0

        if sim >= threshold_high:
            regime = Regime.STABLE
        elif sim <= threshold_low:
            regime = Regime.CHAOTIC
        else:
            regime = Regime.TRANSITION

        labels.append(RegimeLabel(
            step=int(steps[i + 1]),  # Label refers to the destination checkpoint
            regime=regime,
            similarity=float(sim),
            variance=float(var),
            curvature=float(curv) if not np.isnan(curv) else 0.0,
        ))

    # Steps for intervals (use the destination checkpoint step)
    interval_steps = steps[1:len(similarities) + 1]

    return RegimeResult(
        labels=labels,
        steps=interval_steps,
        similarities=similarities,
        variances=variances,
        curvatures=curvatures,
        losses=losses,
        thresholds={'high': threshold_high, 'low': threshold_low},
    )
