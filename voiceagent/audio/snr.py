"""Blind SNR estimation with WADA-SNR (Kim & Stern, Interspeech 2008).

Clean speech amplitude is modeled as Gamma-distributed (shape 0.4), noise as Gaussian.
The statistic G = log(mean|x|) - mean(log|x|) moves monotonically with SNR, so a
table of G vs. SNR maps a measured G back to an SNR. The original uses a precomputed
table; we rebuild it once by Monte Carlo under the same model."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from voiceagent.audio.wav import read_wav

_ALPHA = 0.4
_GRID_DB = np.arange(-20.0, 100.5, 0.5)


def _g_stat(x: np.ndarray) -> float:
    a = np.maximum(np.abs(x), 1e-10)
    return float(np.log(a.mean()) - np.log(a).mean())


@lru_cache(maxsize=1)
def _table() -> np.ndarray:
    rng = np.random.default_rng(1234)
    n = 100_000
    speech = rng.gamma(_ALPHA, 1.0, n) * rng.choice([-1.0, 1.0], n)
    speech /= np.sqrt(np.mean(speech**2))
    noise = rng.standard_normal(n)
    g = np.array([_g_stat(speech + noise * 10 ** (-snr / 20)) for snr in _GRID_DB])
    return np.maximum.accumulate(g)


def estimate_snr_db(x: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean()
    if not np.any(x):
        return float(_GRID_DB[0])
    return float(np.interp(_g_stat(x), _table(), _GRID_DB))


def estimate_snr_from_wav(data: bytes) -> float:
    x, _ = read_wav(data)
    return estimate_snr_db(x)
