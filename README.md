# Semester-final anomaly-localisation analysis

Completed on 14 September 2026. Start with [the Results section (PDF)](analysis/Preliminary_Results.pdf), [the self-contained HTML version](analysis/Preliminary_Results.html), or [editable Markdown](analysis/Preliminary_Results.md).

The original `Anomaly-Localisation.docx`, `source_vs_symptom_example.png`, `baselines.py` and `run_experiment.py` are preserved. The new Results section is separate so that the revised evidence can be reviewed before incorporating it into the full report.

## What was completed

- A **newly reconstructed**, explicitly specified generator and 30-case demonstration: magnitude **0/30**, onset **8/30**. This does **not** reproduce the historical onset claim of 17/30; the original generator and CSV were missing.
- A **160-case gain x delay sweep**, plus 160 separate noise-profile sensitivity cases. Ten seeds are paired across conditions.
- The **unmodified official AERCA linear experiment**, repeated for three seeds with 100 test sequences each, using its default architecture and training protocol on CPU. AC@1 and AC@3 were 100% for all three runs. These are supported-data reproduction results, not a comparison on our custom generator or an independent reproduction of every paper result.
- An **SMD machine-1-1 case study** covering all eight interpretation intervals: primary Hit@1 62.5%, Hit@3 100%, NDCG@3 0.750. This is affected-variable localisation with supplied intervals, not physical root-cause analysis. Numerical-floor sensitivity and an annotation mismatch are documented.
- Eight automated checks passed, covering intervention support, reproducibility, paired noise, pre-fault reference leakage, onset sustain, multilabel metrics, raw-data hashes, interval alignment, saved predictions and AERCA score alignment.

## Main report assets

- `analysis/figures/figure_7_1.png` and `.pdf`: first actual reconstructed demonstration case, seed 0.
- `analysis/results/table_7_1.csv`: reconstructed demonstration accuracy and 95% Wilson intervals.
- `analysis/figures/figure_7_2.png` and `.pdf`: the main equal-noise sweep; each cell shows correct / 10.
- `analysis/results/table_7_2.csv`: AERCA reproduction by seed, including training epochs and explicitly distinguished AC versus AC* metrics.
- `analysis/figures/figure_7_3.png` and `.pdf`: first chronological SMD interpretation interval.
- `analysis/results/table_7_3.csv`: SMD event-macro metrics and exact uniform-random expectations.

There are three main figures and three main tables. Detailed CSVs and arrays support the analysis without expanding the Results section with additional figures.

## Reproduce the experiments

Run from this folder. `analysis/requirements-observed.txt` records the package versions actually used with Python 3.9.6 on arm64 macOS. A fresh installation should use an isolated environment; the official AERCA CUDA requirements are intentionally not installed on this Mac.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r analysis/requirements-observed.txt
.venv/bin/python analysis/run_synthetic.py
.venv/bin/python analysis/run_aerca.py --seeds 0 1 2
.venv/bin/python analysis/summarize_aerca.py
.venv/bin/python analysis/run_smd.py
.venv/bin/python analysis/validate_pipeline.py
```

The current local environment can run these with `python3` directly. The new synthetic harness writes to `analysis/`; the original `run_experiment.py` is retained as supplied and is not the recommended report-reproduction entry point.

If the external source checkout is absent, fetch the exact version first:

```sh
git clone https://github.com/hanxiao0607/AERCA.git external/AERCA
git -C external/AERCA checkout c3dd49d88568d77c3e1ecb9a40472cd77a79b930
```

`run_smd.py` downloads only the four files for the selected machine if they are absent. Downloads are pinned to OmniAnomaly commit `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`; file hashes and URLs are stored in `analysis/data/smd/sources.json`. Nothing is trained on SMD. All eight intervals are evaluated and the first is plotted without performance-based selection.

Regenerate the report using its separate lightweight rendering environment:

```sh
analysis/.venv-report/bin/python analysis/build_report.py
pdftoppm -scale-to 1300 -png analysis/Preliminary_Results.pdf analysis/qa/results
```

The rendering environment contains Python 3.11.15, ReportLab 5.0.1 and Pillow 12.3.0 (plus pypdf 6.18.1 for inspection). It does not alter the experiment environment. If absent, create it with `python3.11 -m venv analysis/.venv-report` and install `reportlab==5.0.1 pillow==12.3.0 pypdf==6.18.1`. The narrative describes this dated run; revise it if changing the protocol.

## Experimental definitions

### Reconstructed generator

`synthetic_system.py` reconstructs the description, not missing original source code. Its version is `reconstructed-response-chain-v1`. Independent AR(1) backgrounds have persistence 0.65 and specified stationary standard deviations, with stationary initial samples. An additive pump response of 2.5 begins at time 100. The other responses are recursively delayed and multiplied copies of the upstream response. Observations are the sum of background and response, for 200 time steps.

The normal backgrounds are independent: **the generator does not offer a learnable normal causal graph**. It is intentionally used only for the simple diagnostic study. It is not a mechanistic industrial simulator. The source is fixed to pump, so an identity-only always-pump rule trivially achieves 100%; the experiment tests evidence heuristics rather than general source classification.

Demonstration random parameters use a separate SeedSequence stream `[seed, 20260914]`. Gains are uniform in `[1.15, 1.6)`, delays are uniform integers 1-5, and noise scales are `(1, .8, .65, .5)` multiplied independently by uniform factors in `[.85, 1.15)`. The first actual demonstration case is plotted.

The main sweep fixes noise scales to one and uses per-link `g in {0.5, 1, 1.5, 2}` and `d in {1, 3, 5, 10}`. Thus total pump-to-temperature amplification is `g^3` and delay is `3d`. Every cell uses seeds 0-9. The response leaves the background RNG unchanged, so conditions have exactly paired background realizations. The sensitivity sweep changes only stationary noise scales to `(1, .8, .65, .5)`.

The original baselines remain unchanged: sample-standardized pre-fault z-scores, peak post-fault absolute magnitude, and earliest threshold crossing sustained for three samples at `|z| >= 3`. Ties follow column order. In four of 160 main-sweep cases, earliest onsets tie; two ties favour pump. Random tie resolution would lower onset cells `(g=1,d=3)` and `(g=1,d=10)` by 0.05 each. No main-sweep case uses onset's magnitude fallback. These caveats are reported; no seed selection or tuning was used to match the historical 17/30.

Wilson intervals are pointwise descriptive intervals within each condition. Since seeds are reused between cells, one must not treat all 160 cases as independent replicates for a pooled inference about gain or delay.

### AERCA reproduction

The repository at `external/AERCA` is unchanged. `analysis/run_aerca.py` invokes its official entry point, isolates each seed's files, sets CPU execution and one PyTorch thread, and saves a run manifest. No model or objective was changed. The dependency snapshot differs from the repository's documented Ubuntu/CUDA environment and is recorded separately.

The linear defaults have 500 samples per series, 10 normal series for an 8/2 training/validation split, 100 held-out test series, four variables, a one-sample exogenous pulse of size 3 and innovation noise scale 0.4. The generator chooses 1-4 injected sources per event. Seeds 0, 1 and 2 had 24, 30 and 28 single-source test cases respectively. Early stopping at 791, 1114 and 1266 epochs selected checkpoints from epochs 771, 1094 and 1246. Each seed changes both model randomness and system coefficients.

The official ranking score is signed standardized encoder residual evidence for its positive exogenous interventions. At an annotated anomaly time:

`AC@K = number of labelled sources in top K / min(K, number of labelled sources)`.

This is averaged over anomalous times and series. It differs from Hit@K (which tests whether **any** relevant dimension is retrieved). `summarize_aerca.py` independently verifies AC@1 and AC@3 from exported scores and the two-step temporal offset, preserving original variable IDs 0-3.

The official starred `AC*@K` ranks flattened time-variable entries after POT thresholding, with thresholds estimated from **pooled test scores**. It is transductive and must not be conflated with the normal-only calibration of our custom diagnostic scores. The report preserves the star and explains the distinction. The AC@K result itself does not need the POT thresholds.

Upstream implementation observations are retained rather than silently corrected: `compute_kl_divergence` uses a positive log-determinant where the standard Gaussian KL expression has a negative log-determinant; its `_smoothness_loss` differences the lag axis (at default window size one this contributes zero). These runs establish execution reproducibility of released code, not complete mathematical validation against the paper. Any future corrective study should version its changes and compare them separately.

The first seed was launched directly before the reusable runner was written; its total wall runtime was not captured, so its manifest explicitly leaves that value null. Full epoch history and the successful command's logs remain available. The later two runs have measured wall runtimes.

### SMD validation

Dimensions are kept as the release's 1-based IDs for presentation, mapped to zero-based columns for scoring. Time ranges use zero-based, end-exclusive slices, supported by checking all starts and ends against binary labels. Raw interpretation labels remain available so this indexing choice can be audited. The first annotated interval ends 27 samples before its binary anomaly run ends; only the interpretation-covered interval is scored. All eight annotated intervals pass the binary-label overlap check.

Normalization uses only training mean and sample standard deviation, floored at `1e-8`. Eight training channels are constant: 5, 8, 17, 18, 27, 29, 37 and 38. The sensitivity file records floors `1e-8`, `0.001`, `0.01` and `0.1`; these were inspected after discovering the constant-channel issue and are not independently selected optimal settings. First-event D29 scores are large because its training variance is zero, not because its raw physical deviation is necessarily large. The plot uses symmetric-log and log scales to make the trajectories and score ranges visible.

Hit@K is binary event success, NDCG@3 uses binary relevance and logarithmic rank discounts, and Recall@3 divides retrieved labelled dimensions by the label count. All are macro-averaged across events. The random reference is calculated exactly for a uniform permutation of 38 dimensions, accounting for each event's label count. It is not another trained model. These results support pipeline validation on affected-variable labels only.

## Detailed outputs and provenance

- `analysis/results/synthetic_manifest.json`: every generator configuration, ordering and runtime.
- `analysis/results/demonstration_cases.csv`, `demonstration_data.npz`: all reconstructed demonstration cases.
- `analysis/results/sweep_cases.csv`, `sweep_cells.csv`, `sweep_data.npz`: primary and noise-sensitivity cases, scores, onsets and uncertainty intervals.
- `analysis/results/aerca_seed*/`: generated data, trained checkpoints, calibration arrays, oracle scores, run manifests and training histories.
- `analysis/results/aerca_cases.csv`: all 300 per-sequence AERCA metrics and true source sets.
- `analysis/results/smd_events.csv`, `smd_rankings.csv`, `smd_audit.json`, `smd_normalization.npz`: per-event scores and label audit.
- `analysis/results/smd_floor_sensitivity*.csv`: separately disclosed normalization sensitivity.
- `analysis/results/environment.json`, `analysis/requirements-observed.txt`: experiment environment.
- `analysis/results/validation.json`, `analysis/logs/validation.log`: automated verification.
- `analysis/results/artifact_hashes.json`: original-input and analysis-code hashes.

Sources: [AERCA paper](https://openreview.net/forum?id=k38Th3x4d9), [official AERCA code](https://github.com/hanxiao0607/AERCA), [official SMD release](https://github.com/NetManAIOps/OmniAnomaly). Preserve upstream licence notices when redistributing code or data.
