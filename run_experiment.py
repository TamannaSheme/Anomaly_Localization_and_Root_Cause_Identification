"""
run_experiment.py

Runs the source-vs-symptom proof-of-concept:
  1. Generates N synthetic cases (known source = pump, known propagation path).
  2. Scores each case with two simple baselines (magnitude ranking, onset-time ranking).
  3. Reports how often each baseline correctly identifies the TRUE source vs.
     a downstream symptom.
  4. Saves a results table (CSV) and one illustrative figure (PNG).

Usage:
    python run_experiment.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_system import simulate_case, random_case_config, VARIABLES, TRUE_SOURCE
from baselines import magnitude_ranking, onset_time_ranking, zscores

N_CASES = 30
OUT_CSV = "results_table.csv"
OUT_FIG = "source_vs_symptom_example.png"


def run_all_cases(n_cases: int = N_CASES) -> pd.DataFrame:
    rows = []
    for seed in range(n_cases):
        cfg = random_case_config(seed)
        df = simulate_case(cfg)
        fault_start = cfg.fault_start

        mag_pred = magnitude_ranking(df, fault_start)
        onset_pred = onset_time_ranking(df, fault_start)

        rows.append({
            "case_id": seed,
            "true_source": TRUE_SOURCE,
            "delays": cfg.delays,
            "amplification": tuple(round(g, 2) for g in cfg.amplification),
            "magnitude_pred": mag_pred,
            "magnitude_correct": mag_pred == TRUE_SOURCE,
            "onset_pred": onset_pred,
            "onset_correct": onset_pred == TRUE_SOURCE,
        })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> None:
    n = len(results)
    mag_acc = results["magnitude_correct"].mean()
    onset_acc = results["onset_correct"].mean()

    print(f"\n=== Source-identification accuracy over {n} synthetic cases ===")
    print(f"True source in every case: '{TRUE_SOURCE}'\n")
    print(f"Magnitude ranking : {results['magnitude_correct'].sum():>2d}/{n}  "
          f"({mag_acc:.0%} correct)")
    print(f"Onset-time ranking: {results['onset_correct'].sum():>2d}/{n}  "
          f"({onset_acc:.0%} correct)")

    print("\nWhen magnitude ranking is WRONG, what does it pick instead?")
    wrong = results.loc[~results["magnitude_correct"], "magnitude_pred"]
    if len(wrong):
        print(wrong.value_counts().to_string())
    else:
        print("(never wrong)")

    print("\nWhen onset-time ranking is WRONG, what does it pick instead?")
    wrong2 = results.loc[~results["onset_correct"], "onset_pred"]
    if len(wrong2):
        print(wrong2.value_counts().to_string())
    else:
        print("(never wrong)")


def make_figure(example_seed: int = 0) -> None:
    """Plot one representative case: all four signals, fault window shaded,
    with the magnitude-ranking prediction annotated against the true source."""
    from synthetic_system import CaseConfig

    cfg = CaseConfig(seed=example_seed)  # fixed, readable parameters for the figure
    df = simulate_case(cfg)
    fault_start = cfg.fault_start
    z = zscores(df, fault_start)

    mag_pred = magnitude_ranking(df, fault_start)
    onset_pred = onset_time_ranking(df, fault_start)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    colors = {"pump": "#c0392b", "flow": "#27ae60", "pressure": "#e67e22", "temperature": "#2980b9"}

    ax = axes[0]
    for v in VARIABLES:
        ax.plot(df.index, df[v], label=v.capitalize(), color=colors[v], linewidth=1.4)
    ax.axvspan(fault_start, cfg.T, color="grey", alpha=0.12, label="Fault window")
    ax.axvline(fault_start, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Raw signal")
    ax.set_title(
        f"Synthetic case (seed={example_seed}): fault injected at '{TRUE_SOURCE}', "
        f"propagates to flow \u2192 pressure \u2192 temperature"
    )
    ax.legend(loc="upper left", ncol=4, fontsize=9)

    ax2 = axes[1]
    for v in VARIABLES:
        ax2.plot(df.index, z[v].abs(), label=v.capitalize(), color=colors[v], linewidth=1.4)
    ax2.axvspan(fault_start, cfg.T, color="grey", alpha=0.12)
    ax2.axvline(fault_start, color="grey", linestyle="--", linewidth=1)
    ax2.axhline(3.0, color="black", linestyle=":", linewidth=1, label="onset threshold (|z|=3)")
    ax2.set_ylabel("|z-score| vs. pre-fault normal")
    ax2.set_xlabel("Time step")
    ax2.legend(loc="upper left", ncol=3, fontsize=9)

    correct_mark = lambda pred: "correct" if pred == TRUE_SOURCE else f"WRONG (picked '{pred}')"
    fig.text(
        0.5, -0.02,
        f"True source: '{TRUE_SOURCE}'   |   Magnitude ranking: {correct_mark(mag_pred)}"
        f"   |   Onset-time ranking: {correct_mark(onset_pred)}",
        ha="center", fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {OUT_FIG}")


if __name__ == "__main__":
    results = run_all_cases(N_CASES)
    results.to_csv(OUT_CSV, index=False)
    print(f"Saved per-case results to {OUT_CSV}")
    summarise(results)
    make_figure(example_seed=0)
