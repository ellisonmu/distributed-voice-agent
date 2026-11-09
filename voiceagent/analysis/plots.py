"""Figures for the report from `summarize --out summary.csv` (and training/train_router.py outputs).

  latency_vs_n.png        p95 time to first audio and to first answer vs. N, per policy, with the deadline
  quality_vs_latency.png  the deck's trade-off figure with real data: judged quality vs. p95 TTFA
  routing_mix.png         adaptive policy: which tier answered, by N
  stalls.png              share of turns with an audible gap mid-answer, by N
  training.png            DDP vs. FSDP: peak GPU memory and throughput (needs --train-runs)

  python -m voiceagent.analysis.plots --summary logs/run-123/summary.csv --out-dir figures \
      --train-runs $SCRATCH/router-ddp $SCRATCH/router-fsdp"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated for color-vision deficiency); every policy keeps its color
# and marker in every figure. Low-contrast hues always get a direct label as well.
POLICY_ORDER = ["always_small", "always_large", "always_turbo", "adaptive", "adaptive_no_filler"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
LABELS = {"always_small": "Always 8B", "always_large": "Always 70B", "always_turbo": "Always turbo",
          "adaptive": "Adaptive", "adaptive_no_filler": "Adaptive, no filler"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def _num(v: str):
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def load_summary(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rows


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def _by_policy(rows: list[dict]) -> dict[str, list[dict]]:
    out = {}
    for p in POLICY_ORDER:
        runs = sorted((r for r in rows if r["policy"] == p), key=lambda r: r["n"])
        if runs:
            out[p] = runs
    return out


def _line(ax, policy: str, xs, ys, end_label: bool = True) -> None:
    i = POLICY_ORDER.index(policy)
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pts:
        return
    ax.plot(*zip(*pts), color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[policy])
    if end_label:
        x, y = pts[-1]
        ax.annotate(LABELS[policy], (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)


def _n_axis(ax, ns) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.minorticks_off()


def _deadline(ax, deadline: float) -> None:
    ax.axhline(deadline, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"{deadline:g} s deadline", (0.01, deadline), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)


def latency_vs_n(rows, out: Path, deadline: float) -> None:
    groups = _by_policy(rows)
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, key, title in ((axes[0], "ttfa_p95", "p95 time to first audio"),
                           (axes[1], "ttsa_p95", "p95 time to the first answer sentence")):
        for p, runs in groups.items():
            _line(ax, p, [r["n"] for r in runs], [r[key] for r in runs])
        _deadline(ax, deadline)
        _style(ax, "concurrent callers (N)", "seconds")
        _n_axis(ax, ns)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def quality_vs_latency(rows, out: Path, deadline: float) -> bool:
    groups = {p: [r for r in runs if r.get("quality") is not None] for p, runs in _by_policy(rows).items()}
    groups = {p: runs for p, runs in groups.items() if runs}
    if not groups:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="white")
    for p, runs in groups.items():
        _line(ax, p, [r["ttfa_p95"] for r in runs], [r["quality"] for r in runs])
        for r in (runs[0], runs[-1]) if len(runs) > 1 else runs[:1]:
            ax.annotate(f"N={int(r['n'])}", (r["ttfa_p95"], r["quality"]), xytext=(0, 8),
                        textcoords="offset points", fontsize=7.5, color=MUTED, ha="center")
    ax.axvline(deadline, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"{deadline:g} s deadline", (deadline, 0.98), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED, va="top")
    _style(ax, "p95 time to first audio (s)", "judged answer quality (0–1)")
    ax.set_xlim(left=0)
    ax.set_title("Answer quality vs. latency, one point per N", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def routing_mix(rows, out: Path) -> bool:
    runs = sorted((r for r in rows if r["policy"] == "adaptive"), key=lambda r: r["n"])
    if not runs:
        return False
    parts = [("8B answered", lambda r: (r["llm_small"] or 0) - (r["hedged"] or 0)),
             ("70B", lambda r: r["llm_large"] or 0),
             ("Turbo", lambda r: r.get("llm_turbo") or 0),
             ("8B, hedged fallback", lambda r: r["hedged"] or 0)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    xs = list(range(len(runs)))
    bottom = [0.0] * len(runs)
    for i, (name, f) in enumerate(parts):
        vals = [f(r) for r in runs]
        if not any(vals):
            continue
        ax.bar(xs, vals, bottom=bottom, width=0.6, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(r["n"])) for r in runs])
    _style(ax, "concurrent callers (N)", "share of turns")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Adaptive policy: which model answered", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def stalls(rows, out: Path) -> bool:
    groups = {p: runs for p, runs in _by_policy(rows).items() if any(r.get("stalled") is not None for r in runs)}
    if not groups:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    for p, runs in groups.items():
        _line(ax, p, [r["n"] for r in runs], [r["stalled"] for r in runs])
    _style(ax, "concurrent callers (N)", "share of turns")
    _n_axis(ax, sorted({r["n"] for r in rows}))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Turns with an audible gap (> 150 ms) mid-answer", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def training(run_dirs: list[str], out: Path) -> bool:
    summaries = [json.loads((Path(d) / "summary.json").read_text()) for d in run_dirs]
    if not summaries:
        return False
    names = [f"{s['strategy'].upper()} · {s['world_size']} GPUs" for s in summaries]
    colors = [SERIES[0] if s["strategy"] == "ddp" else SERIES[1] for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
    for ax, key, title, fmt in ((axes[0], "peak_mem_gb", "Peak GPU memory per GPU", "{:.1f} GB"),
                                (axes[1], "mean_tokens_per_s", "Training throughput", "{:,.0f} tok/s")):
        _style(ax, "", "")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        if all(s.get(key) is None for s in summaries):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "not measured (CPU run)", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        vals = [s.get(key) or 0 for s in summaries]
        bars = ax.bar(names, vals, width=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", nargs="*", default=[], help="CSV(s) from voiceagent.analysis.summarize --out")
    p.add_argument("--train-runs", nargs="*", default=[], help="output dirs of training/train_router.py")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--deadline", type=float, default=1.0)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if a.summary:
        rows = load_summary(a.summary)
        latency_vs_n(rows, a.out_dir / "latency_vs_n.png", a.deadline)
        made.append("latency_vs_n.png")
        for name, fn in (("quality_vs_latency.png", lambda o: quality_vs_latency(rows, o, a.deadline)),
                         ("routing_mix.png", lambda o: routing_mix(rows, o)), ("stalls.png", lambda o: stalls(rows, o))):
            if fn(a.out_dir / name):
                made.append(name)
    if a.train_runs and training(a.train_runs, a.out_dir / "training.png"):
        made.append("training.png")
    print(f"wrote {', '.join(made) or 'nothing'} to {a.out_dir}")


if __name__ == "__main__":
    main()
