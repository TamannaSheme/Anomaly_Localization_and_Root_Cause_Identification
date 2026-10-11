"""Reconstructed demonstration generator, version 1 (2026-09-14).

The original generator was absent. This implements the report's description,
not its unknown original parameters or random-number sequence. Independent
stationary AR(1) backgrounds receive a delayed, multiplicatively propagated
step response. Normal backgrounds do NOT encode a learnable causal chain.
Use the official AERCA linear generator for causal-model reproduction.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

VARIABLES = ["pump", "flow", "pressure", "temperature"]
TRUE_SOURCE = "pump"
GENERATOR_VERSION = "reconstructed-response-chain-v1"


@dataclass(frozen=True)
class CaseConfig:
    seed: int = 0
    T: int = 200
    fault_start: int = 100
    phi: float = 0.65
    fault_size: float = 2.5
    delays: tuple = (3, 3, 3)
    amplification: tuple = (1.3, 1.3, 1.3)
    noise_std: tuple = (1.0, 0.8, 0.65, 0.5)


def simulate_components(cfg: CaseConfig):
    if not (2 <= cfg.fault_start < cfg.T and abs(cfg.phi) < 1):
        raise ValueError("Need pre-fault observations, a fault window, and stationary AR(1).")
    if len(cfg.delays) != 3 or any(d < 1 or int(d) != d for d in cfg.delays):
        raise ValueError("Three positive integer per-link delays are required.")
    if len(cfg.amplification) != 3 or any(g < 0 for g in cfg.amplification):
        raise ValueError("Three nonnegative per-link gains are required.")
    sd = np.asarray(cfg.noise_std, dtype=float)
    if sd.shape != (4,) or (sd <= 0).any():
        raise ValueError("Four positive stationary background standard deviations are required.")
    if cfg.fault_start + sum(cfg.delays) >= cfg.T:
        raise ValueError("The observation window must include the last downstream arrival.")
    rng = np.random.default_rng(cfg.seed)
    background = np.empty((cfg.T, 4))
    background[0] = rng.normal(size=4) * sd
    innovations = rng.normal(size=(cfg.T - 1, 4)) * sd * np.sqrt(1 - cfg.phi**2)
    for t in range(1, cfg.T):
        background[t] = cfg.phi * background[t - 1] + innovations[t - 1]
    response = np.zeros_like(background)
    response[cfg.fault_start:, 0] = cfg.fault_size
    for j, (delay, gain) in enumerate(zip(cfg.delays, cfg.amplification), start=1):
        response[delay:, j] = gain * response[:-delay, j - 1]
    return background, response


def simulate_case(cfg: CaseConfig) -> pd.DataFrame:
    background, response = simulate_components(cfg)
    return pd.DataFrame(background + response, columns=VARIABLES)


def random_case_config(seed: int) -> CaseConfig:
    # Independent streams for parameters and background; chosen before inspecting results.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260914]))
    return CaseConfig(seed=seed,
                      delays=tuple(int(x) for x in rng.integers(1, 6, 3)),
                      amplification=tuple(float(x) for x in rng.uniform(1.15, 1.6, 3)),
                      noise_std=tuple(np.array([1.0, .8, .65, .5]) * rng.uniform(.85, 1.15, 4)))
