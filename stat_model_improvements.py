#!/usr/bin/env python3
"""Summarize Section 3.7 model-improvement runs.

Run from the repository root after run_model_improvements.py:
    python stat_model_improvements.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "improvement_runs"
OUT_TXT = RUN_ROOT / "summary_report.txt"
OUT_CSV = RUN_ROOT / "summary_table.csv"
RESULTS_CSV = RUN_ROOT / "results.csv"

TRAIN_RE = re.compile(
    r"step_idx:\s*(\d+).*?epoch:\s*(\d+).*?batch:\s*(\d+).*?"
    r"avg loss:\s*([0-9.eE+-]+).*?normalized loss:\s*([0-9.eE+-]+).*?"
    r"ppl:\s*([0-9.eE+-]+).*?speed:\s*([0-9.eE+-]+)"
)
BLEU_RE = re.compile(r"BLEU\s*=\s*([0-9.]+)")


def parse_train(path: Path):
    pts = []
    if not path.exists():
        return pts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            pts.append({
                "step": int(m.group(1)),
                "epoch": int(m.group(2)),
                "batch": int(m.group(3)),
                "avg_loss": float(m.group(4)),
                "norm_loss": float(m.group(5)),
                "ppl": float(m.group(6)),
                "speed": float(m.group(7)),
            })
    return pts


def parse_bleu(path: Path):
    if not path.exists():
        return None
    m = BLEU_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    return float(m.group(1)) if m else None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def main():
    if not RUN_ROOT.exists():
        raise SystemExit("improvement_runs/ not found")

    status_map = {}
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                status_map[r.get("name", "")] = r.get("train_status", "")

    rows = []
    for exp_dir in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir()):
        cfg_path = exp_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        attention_variant = cfg.get("attention_variant", "normal")
        if attention_variant not in ("normal", "drophead", "relative_bias"):
            continue

        pts = parse_train(exp_dir / "train.log")
        b17 = parse_bleu(exp_dir / "predictions" / "bleu_2017.log")
        b18 = parse_bleu(exp_dir / "predictions" / "bleu_2018.log")
        last = pts[-1] if pts else None
        last5 = pts[-5:]

        rows.append({
            "name": exp_dir.name,
            "status": status_map.get(exp_dir.name, ""),
            "attention_variant": attention_variant,
            "ffn_variant": cfg.get("ffn_variant", "relu"),
            "drophead_rate": cfg.get("drophead_rate", ""),
            "swiglu_inner_hid": cfg.get("swiglu_inner_hid", ""),
            "max_relative_position": cfg.get("max_relative_position", ""),
            "bleu_2017": b17,
            "bleu_2018": b18,
            "last_step": last["step"] if last else None,
            "last_norm": last["norm_loss"] if last else None,
            "last5_norm": mean([p["norm_loss"] for p in last5]),
            "mean_speed": mean([p["speed"] for p in pts]),
        })

    # Control first, then sort improvements by 2017 (validation) BLEU.
    control = next((r for r in rows if r["name"] == "control_lr150"), None)
    others = [r for r in rows if r["name"] != "control_lr150"]
    others.sort(key=lambda r: (r["bleu_2017"] is not None, r["bleu_2017"] or -1), reverse=True)
    ordered = ([control] if control else []) + others

    control17 = control["bleu_2017"] if control else None
    control18 = control["bleu_2018"] if control else None

    lines = []
    lines.append("=" * 92)
    lines.append("Transformer 3.7 模型结构改进实验统计报告")
    lines.append("=" * 92)
    lines.append("")
    if control:
        lines.append(
            f"控制组 control_lr150: BLEU17={fmt(control17)}, BLEU18={fmt(control18)}"
        )
    else:
        lines.append("警告：未发现 control_lr150，无法计算相对提升。")
    lines.append("")
    lines.append("%-20s %-12s %-10s %-8s %8s %8s %9s %9s %8s" % (
        "Experiment", "Status", "Attention", "FFN",
        "BLEU17", "BLEU18", "Δ17", "Δ18", "Speed"
    ))
    lines.append("-" * 106)

    for r in ordered:
        d17 = (r["bleu_2017"] - control17) if (r["bleu_2017"] is not None and control17 is not None) else None
        d18 = (r["bleu_2018"] - control18) if (r["bleu_2018"] is not None and control18 is not None) else None
        raw_status = r.get("status", "")
        if raw_status.startswith("failed") or raw_status.startswith("exception"):
            status = "FAILED"
        elif raw_status in ("ok", "skipped_existing", "control_existing"):
            status = "OK"
        elif raw_status:
            status = raw_status[:12]
        else:
            status = "-"
        lines.append("%-20s %-12s %-10s %-8s %8s %8s %9s %9s %8s" % (
            r["name"], status, r["attention_variant"], r["ffn_variant"],
            fmt(r["bleu_2017"]), fmt(r["bleu_2018"]),
            "-" if d17 is None else f"{d17:+.2f}",
            "-" if d18 is None else f"{d18:+.2f}",
            fmt(r["mean_speed"]),
        ))

    valid = [r for r in others if r["bleu_2017"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["bleu_2017"])
        lines += [
            "", "-" * 106,
            f"按 newstest2017 选择的最佳结构: {best['name']} (BLEU17={best['bleu_2017']:.2f}, BLEU18={fmt(best['bleu_2018'])})",
        ]
        if control17 is not None:
            lines.append(f"相对 control_lr150 的 BLEU17 变化: {best['bleu_2017'] - control17:+.2f}")
        if best["bleu_2018"] is not None and control18 is not None:
            lines.append(f"相对 control_lr150 的 BLEU18 变化: {best['bleu_2018'] - control18:+.2f}")

    lines += [
        "", "说明：",
        "1. 3.7 应优先用 BLEU-2017 选择结构，再用 BLEU-2018 看泛化。",
        "2. Relative Position Bias 只修改 encoder/decoder self-attention，cross-attention 保持普通 attention。",
        "3. 不同结构的训练 loss/PPL 不能单独作为翻译质量结论，最终以 BLEU 为主。",
    ]

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = [
        "name", "status", "attention_variant", "ffn_variant", "drophead_rate",
        "swiglu_inner_hid", "max_relative_position",
        "bleu_2017", "bleu_2018", "last_step",
        "last_norm", "last5_norm", "mean_speed",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ordered:
            writer.writerow({k: r.get(k, "") for k in fields})

    print(OUT_TXT.read_text(encoding="utf-8"))
    print(f"written: {OUT_TXT}")
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()
