"""
baselines.py

The two simplest diagnostic baselines already scoped in Table 3.2 of the
thesis proposal:

1. Magnitude / residual ranking -- picks the variable with the largest
   post-fault deviation from its own pre-fault behaviour. Tests the naive
   assumption that "biggest anomaly == the source".

2. Onset-time ranking -- picks the variable whose deviation crosses a
   threshold earliest after the fault window opens. Tests whether timing
   alone is a reliable source signal.

Both operate on z-scores computed from PRE-FAULT statistics only (mean/std
estimated on t < fault_start), which is the standard, leakage-free way to
score anomalies against "normal" behaviour.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from synthetic_system import VARIABLES


def zscores(df: pd.DataFrame, fault_start: int) -> pd.DataFrame:
    """Per-variable z-score using only pre-fault mean/std as the reference."""
    z = pd.DataFrame(index=df.index)
    pre = df.loc[df.index < fault_start, VARIABLES]
    for v in VARIABLES:
        mu, sd = pre[v].mean(), pre[v].std()
        sd = sd if sd > 1e-8 else 1e-8
        z[v] = (df[v] - mu) / sd
    return z


def magnitude_ranking(df: pd.DataFrame, fault_start: int) -> str:
    """Predicted source = variable with the largest |z| during the post-fault window."""
    z = zscores(df, fault_start)
    post = z.loc[df.index >= fault_start, VARIABLES].abs()
    peak_per_var = post.max()
    return peak_per_var.idxmax()


def onset_time_ranking(df: pd.DataFrame, fault_start: int, threshold: float = 3.0,
                        sustain: int = 3) -> str:
    """Predicted source = variable whose |z| first exceeds `threshold` for
    `sustain` consecutive points after the fault window opens (a small
    sustain requirement avoids single-sample noise spikes being picked)."""
    z = zscores(df, fault_start)
    post = z.loc[df.index >= fault_start, VARIABLES].abs().reset_index(drop=True)

    onset = {}
    for v in VARIABLES:
        above = post[v] >= threshold
        crossed_at = None
        run = 0
        for t, flag in enumerate(above):
            run = run + 1 if flag else 0
            if run >= sustain:
                crossed_at = t - sustain + 1
                break
        onset[v] = crossed_at if crossed_at is not None else np.inf

    # earliest onset wins; if none crossed, fall back to magnitude ranking
    best = min(onset, key=onset.get)
    if onset[best] == np.inf:
        return magnitude_ranking(df, fault_start)
    return best
