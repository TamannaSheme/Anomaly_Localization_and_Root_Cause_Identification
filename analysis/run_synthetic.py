"""Run the reconstructed 30-case demonstration and prespecified 160-case sweep."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import time
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_system import CaseConfig, simulate_case, random_case_config, GENERATOR_VERSION
from baselines import magnitude_ranking, onset_time_ranking, zscores

OUT = ROOT / "analysis/results"
FIG = ROOT / "analysis/figures"
METHODS = ["Magnitude", "Onset"]


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return center-radius, center+radius


def evaluate(cfg, **metadata):
    df = simulate_case(cfg)
    post = zscores(df, cfg.fault_start).iloc[cfg.fault_start:].abs()
    above = post.ge(3).rolling(3).sum().eq(3)
    starts = [np.flatnonzero(above[v].values)[0]-2 if above[v].any() else np.inf for v in df]
    tied = sum(t == min(starts) for t in starts) if min(starts) < np.inf else 0
    row = dict(seed=cfg.seed, **metadata,
               magnitude_pred=magnitude_ranking(df, cfg.fault_start),
               onset_pred=onset_time_ranking(df, cfg.fault_start),
               earliest_onset_tie_count=tied,
               onset_fallback=not np.isfinite(min(starts)),
               **{f"onset_{v}": None if not np.isfinite(t) else int(t) for v,t in zip(df, starts)})
    row["magnitude_correct"] = row["magnitude_pred"] == "pump"
    row["onset_correct"] = row["onset_pred"] == "pump"
    return row, df.values


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    configs = [random_case_config(seed) for seed in range(30)]
    rows, arrays = zip(*(evaluate(cfg) for cfg in configs))
    demo = pd.DataFrame(rows)
    demo.to_csv(OUT / "demonstration_cases.csv", index=False)
    np.savez_compressed(OUT / "demonstration_data.npz", signals=np.stack(arrays))
    table = []
    for method in METHODS:
        key = method.lower()
        n = int(demo[f"{key}_correct"].sum())
        wrong = demo.loc[~demo[f"{key}_correct"], f"{key}_pred"]
        lo, hi = wilson(n, 30)
        table.append(dict(method=method, correct=n, total=30, acc1=n/30,
                          ci95_low=lo, ci95_high=hi,
                          most_common_wrong=wrong.mode().iloc[0] if len(wrong) else "none"))
    pd.DataFrame(table).to_csv(OUT / "table_7_1.csv", index=False)
    # First actual demonstration case, chosen by seed before scoring.
    cfg=configs[0]; df=simulate_case(cfg); z=zscores(df,cfg.fault_start)
    fig,axes=plt.subplots(2,1,figsize=(10,6.8),sharex=True,layout="constrained")
    colors=["#b43e3e","#397848","#b66d20","#286c9b"]
    for j,v in enumerate(df):
        axes[0].plot(df.index,df[v],label=v.capitalize(),color=colors[j],lw=1.2)
        axes[1].plot(df.index,z[v].abs(),label=v.capitalize(),color=colors[j],lw=1.2)
    for ax in axes:
        ax.axvspan(cfg.fault_start,cfg.T-1,color="black",alpha=.07)
        ax.axvline(cfg.fault_start,color="grey",ls="--",lw=1)
        ax.legend(ncol=4,fontsize=9,loc="upper left")
    axes[0].set_ylabel("Raw signal (arbitrary units)")
    axes[1].set_ylabel("|z-score| against pre-fault normal")
    axes[1].set_xlabel("Time step")
    axes[1].axhline(3,color="black",ls=":",lw=1)
    axes[0].set_title("Reconstructed source-versus-symptom experiment: case 0")
    axes[1].set_title(f"True source: pump | Magnitude: {rows[0]['magnitude_pred']} | Onset: {rows[0]['onset_pred']}",fontsize=11)
    fig.savefig(FIG/"figure_7_1.png",dpi=200)
    fig.savefig(FIG/"figure_7_1.pdf")
    plt.close(fig)

    sweep_rows, sweep_arrays, sweep_configs = [], [], []
    # Main sweep equalizes stationary noise. Sensitivity repeats the original quieter-downstream profile.
    for profile, sd in [("equal_noise", (1.,1.,1.,1.)), ("quieter_downstream", (1.,.8,.65,.5))]:
        for g in [.5, 1., 1.5, 2.]:
            for d in [1, 3, 5, 10]:
                for seed in range(10):
                    cfg = CaseConfig(seed=seed, amplification=(g,g,g), delays=(d,d,d), noise_std=sd)
                    row, signals = evaluate(cfg, gain=g, delay=d, noise_profile=profile)
                    sweep_rows.append(row)
                    sweep_arrays.append(signals)
                    sweep_configs.append(asdict(cfg))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "sweep_cases.csv", index=False)
    np.savez_compressed(OUT / "sweep_data.npz", signals=np.stack(sweep_arrays))
    cells = []
    for (profile, g, d), group in sweep.groupby(["noise_profile", "gain", "delay"], sort=True):
        for method in METHODS:
            n = int(group[f"{method.lower()}_correct"].sum())
            lo,hi = wilson(n,len(group))
            cells.append(dict(noise_profile=profile, gain=g, delay=d, method=method,
                              correct=n, total=len(group), acc1=n/len(group), ci95_low=lo, ci95_high=hi))
    cell_df = pd.DataFrame(cells)
    cell_df.to_csv(OUT / "sweep_cells.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, method in zip(axes, METHODS):
        subset = cell_df.query("noise_profile == 'equal_noise' and method == @method")
        values = subset.pivot(index="gain", columns="delay", values="acc1")
        im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4), [1,3,5,10]); ax.set_yticks(range(4), [.5,1.,1.5,2.])
        ax.set_xlabel("Delay per link (time steps)"); ax.set_ylabel("Gain per link (multiplier)")
        ax.set_title(f"{method} ranking")
        for i in range(4):
            for j in range(4):
                value = values.iloc[i,j]
                ax.text(j,i,f"{round(value*10)}/10",ha="center",va="center",color="white" if value>.55 else "black")
    fig.colorbar(im, ax=axes, label="Source-identification accuracy (Acc@1)", shrink=.8)
    fig.suptitle("Propagation strength and delay: reconstructed response chain", fontsize=13)
    fig.savefig(FIG / "figure_7_3.png", dpi=200)
    fig.savefig(FIG / "figure_7_3.pdf")
    plt.close(fig)
    manifest = dict(generator=GENERATOR_VERSION, original_results_reproduced=False,
                    original_report_claims={"magnitude":"0/30", "onset":"17/30"},
                    demonstration_configs=[asdict(c) for c in configs], representative_config=asdict(configs[0]),
                    sweep_configs=sweep_configs, sweep_order="matches CSV and NPZ first axis",
                    primary_cases=160, sensitivity_cases=160,
                    threshold=3.0, sustain=3, normalization="per-case pre-fault mean/sample std (ddof=1)",
                    tie_policy="original baseline: variable order pump, flow, pressure, temperature",
                    selection="seed 0 figure and all seeds selected before inspection",
                    seconds=time.perf_counter()-start)
    (OUT / "synthetic_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(cell_df.query("noise_profile == 'equal_noise'").to_string(index=False))


if __name__ == "__main__":
    main()
