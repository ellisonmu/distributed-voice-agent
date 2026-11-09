import numpy as np
import pytest

from voiceagent.audio.mix import mix_at_snr
from voiceagent.audio.snr import estimate_snr_db, estimate_snr_from_wav
from voiceagent.audio.wav import read_wav, resample, write_wav


def _speech_like(n, rng):
    x = rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)
    return x / np.sqrt(np.mean(x**2))


@pytest.mark.parametrize("snr", [0.0, 5.0, 10.0, 20.0])
def test_wada_snr_recovers_known_snr(snr):
    rng = np.random.default_rng(42)
    s = _speech_like(48000, rng)
    x = s + rng.standard_normal(len(s)) * 10 ** (-snr / 20)
    assert abs(estimate_snr_db(x) - snr) < 2.0


def test_wada_snr_orders_noise_levels():
    rng = np.random.default_rng(0)
    s = _speech_like(32000, rng)
    noise = rng.standard_normal(len(s))
    estimates = [estimate_snr_db(s + noise * 10 ** (-snr / 20)) for snr in (0, 10, 20, 30)]
    assert estimates == sorted(estimates)


def test_mix_hits_requested_snr():
    rng = np.random.default_rng(1)
    speech = 0.01 * _speech_like(16000, rng).astype(np.float32)  # quiet enough that no peak rescaling happens
    noise = rng.standard_normal(5000).astype(np.float32)
    mixed = mix_at_snr(speech, noise, 10.0, rng)
    added = mixed - speech
    measured = 10 * np.log10(np.mean(speech**2) / np.mean(added**2))
    assert abs(measured - 10.0) < 0.1


def test_wav_roundtrip_and_resample():
    x = (0.5 * np.sin(np.linspace(0, 200, 24000))).astype(np.float32)
    y, sr = read_wav(write_wav(x, 24000))
    assert sr == 24000 and np.max(np.abs(x - y)) < 1e-3
    assert len(resample(y, 24000, 16000)) == 16000
    assert estimate_snr_from_wav(write_wav(np.zeros(1000, dtype=np.float32), 16000)) == -20.0
