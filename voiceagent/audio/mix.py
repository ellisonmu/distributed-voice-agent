from __future__ import annotations

import numpy as np


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    n = noise[start : start + len(speech)].astype(np.float64)
    p_speech = np.mean(speech.astype(np.float64) ** 2)
    p_noise = np.mean(n**2)
    if p_noise == 0:
        raise ValueError("noise segment is silent")
    out = speech + n * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)
