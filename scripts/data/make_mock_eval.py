"""Makes a small fake eval set for local runs against the mock servers: speech-like
(Gamma-amplitude) signals with Gaussian noise at known SNRs, so the SNR estimator and
the ASR routing are exercised without real audio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voiceagent.audio.wav import write_wav


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    conditions = [None, 20.0, 10.0, 5.0, 0.0]
    (a.out_dir / "audio").mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "eval.jsonl", "w") as f:
        for i in range(a.count):
            n = 16000 * 2
            x = rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)
            x *= 0.05 / np.sqrt(np.mean(x**2))
            snr = conditions[i % len(conditions)]
            if snr is not None:
                x = x + rng.standard_normal(n) * 0.05 * 10 ** (-snr / 20)
            name = f"audio/{i:05d}.wav"
            (a.out_dir / name).write_bytes(write_wav(x.astype(np.float32), 16000))
            f.write(json.dumps({
                "question_id": f"mock-{i}",
                "question": "What are the early symptoms of asthma in adults?",
                "answer": "Wheezing, coughing, chest tightness and shortness of breath.",
                "qtype": "symptoms",
                "audio_path": name,
                "snr_db": snr,
            }) + "\n")
    print(f"wrote {a.count} mock clips to {a.out_dir}")


if __name__ == "__main__":
    main()
