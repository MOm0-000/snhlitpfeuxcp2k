#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动统计 Transformer 超参实验结果。

默认目录结构：
NLPlab2/
├── stat.py
└── hparam_runs/
    ├── results.csv
    ├── dropout_020/
    │   ├── config.json
    │   ├── train.log
    │   └── predictions/
    │       ├── bleu_2017.log
    │       └── predict_2017.log
    └── ...

运行：
    python stat.py

输出：
    hparam_runs/summary_report.txt
    hparam_runs/summary_table.csv
    hparam_runs/training_points.csv

把 summary_report.txt 的内容发给 ChatGPT 即可做进一步实验分析。
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "hparam_runs"

SUMMARY_TXT = RUNS / "summary_report.txt"
SUMMARY_CSV = RUNS / "summary_table.csv"
POINTS_CSV = RUNS / "training_points.csv"

# 兼容你当前 train.py 的日志格式
TRAIN_RE = re.compile(
    r"step_idx:\s*(?P<step>\d+),\s*"
    r"epoch:\s*(?P<epoch>\d+),\s*"
    r"batch:\s*(?P<batch>\d+),\s*"
    r"avg loss:\s*(?P<avg>[0-9eE+.\-]+),\s*"
    r"normalized loss:\s*(?P<norm>[0-9eE+.\-]+),\s*"
    r"ppl:\s*(?P<ppl>[0-9eE+.\-]+),\s*"
    r"speed:\s*(?P<speed>[0-9eE+.\-]+)\s*step/s"
)

BLEU_RE = re.compile(
    r"BLEU\s*=\s*(?P<bleu>[0-9.]+),\s*"
    r"(?P<precisions>[0-9./]+)\s*"
    r"\(BP=(?P<bp>[0-9.]+),\s*"
    r"ratio=(?P<ratio>[0-9.]+),\s*"
    r"hyp_len=(?P<hyp>\d+),\s*"
    r"ref_len=(?P<ref>\d+)\)"
)

# 这些 warning 会使预测结果无效，尤其是 FFN/d_model/n_layer 等结构参数改变后
BAD_PRED_PATTERNS = [
    "Skip loading for",
    "receives a shape",
    "expected shape is",
]

ERROR_PATTERNS = [
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "No space left on device",
    "nan",
    "inf",
]


def safe_float(x: Any):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def fmt(x, digits=4):
    if x is None or x == "":
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def parse_training(log_path: Path):
    points = []
    text = read_text(log_path)

    for line in text.splitlines():
        m = TRAIN_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        points.append({
            "step": int(d["step"]),
            "epoch": int(d["epoch"]),
            "batch": int(d["batch"]),
            "avg_loss": float(d["avg"]),
            "norm_loss": float(d["norm"]),
            "ppl": float(d["ppl"]),
            "speed": float(d["speed"]),
        })

    return points, text


def parse_bleu_file(path: Path):
    text = read_text(path)
    m = BLEU_RE.search(text)
    if not m:
        return None
    d = m.groupdict()
    return {
        "bleu": float(d["bleu"]),
        "precisions": d["precisions"],
        "bp": float(d["bp"]),
        "ratio": float(d["ratio"]),
        "hyp_len": int(d["hyp"]),
        "ref_len": int(d["ref"]),
    }


def mean_last(points, key, n=5):
    vals = [p[key] for p in points[-n:] if key in p]
    return statistics.mean(vals) if vals else None


def mean_all(points, key):
    vals = [p[key] for p in points if key in p]
    return statistics.mean(vals) if vals else None


def find_best_min(points, key):
    if not points:
        return None, None
    p = min(points, key=lambda x: x[key])
    return p[key], p["step"]


def find_bleu(exp_dir: Path, year: str):
    p = exp_dir / "predictions" / f"bleu_{year}.log"
    return parse_bleu_file(p)


def prediction_validity(exp_dir: Path):
    pred_dir = exp_dir / "predictions"
    if not pred_dir.exists():
        return "no_prediction_log", []

    warnings = []
    for p in pred_dir.glob("predict_*.log"):
        text = read_text(p)
        for pattern in BAD_PRED_PATTERNS:
            if pattern in text:
                warnings.append(f"{p.name}: {pattern}")

    if warnings:
        return "INVALID_SHAPE_WARNING", warnings
    return "ok", []


def training_errors(text: str):
    found = []
    low = text.lower()

    # nan / inf 单独处理，避免普通英文单词误匹配
    for line in text.splitlines():
        ll = line.lower()
        if "traceback (most recent call last)" in ll:
            found.append("Traceback")
        if "cuda out of memory" in ll or "outofmemoryerror" in ll:
            found.append("OOM")
        if "no space left on device" in ll:
            found.append("disk_full")
        if re.search(r"\b(?:avg loss|normalized loss|ppl):\s*(?:nan|inf)\b", ll):
            found.append("NaN/Inf")

    return sorted(set(found))


def flatten_override(config: dict):
    # 只抽取报告中最关心的超参数
    keys = [
        "learning_rate",
        "warmup_steps",
        "label_smooth_eps",
        "prepostprocess_dropout",
        "attention_dropout",
        "relu_dropout",
        "d_model",
        "d_inner_hid",
        "d_key",
        "d_value",
        "n_head",
        "n_layer",
        "batch_size",
        "epoch",
        "beam_size",
    ]
    return {k: config.get(k) for k in keys if k in config}


def collect_results():
    if not RUNS.exists():
        raise SystemExit(
            f"找不到目录：{RUNS}\n"
            "请把 stat.py 放在 NLPlab2 根目录，与 hparam_runs 同级。"
        )

    rows = []
    all_points = []

    for exp_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        name = exp_dir.name
        config = load_json(exp_dir / "config.json")
        points, train_text = parse_training(exp_dir / "train.log")

        # beam-only 实验可能没有 config.json
        if not config and name.startswith("beam_"):
            try:
                config = {"beam_size": int(name.split("_", 1)[1])}
            except Exception:
                config = {}

        b17 = find_bleu(exp_dir, "2017")
        b18 = find_bleu(exp_dir, "2018")

        pred_status, pred_warnings = prediction_validity(exp_dir)
        errors = training_errors(train_text)

        last = points[-1] if points else {}

        min_norm, min_norm_step = find_best_min(points, "norm_loss")
        min_avg, min_avg_step = find_best_min(points, "avg_loss")

        row = {
            "name": name,
            "num_log_points": len(points),
            "last_step": last.get("step"),
            "last_epoch": last.get("epoch"),
            "last_avg_loss": last.get("avg_loss"),
            "last_norm_loss": last.get("norm_loss"),
            "last_ppl": last.get("ppl"),
            "last_speed": last.get("speed"),
            "mean_speed": mean_all(points, "speed"),
            "mean_last5_avg_loss": mean_last(points, "avg_loss", 5),
            "mean_last5_norm_loss": mean_last(points, "norm_loss", 5),
            "mean_last5_ppl": mean_last(points, "ppl", 5),
            "min_avg_loss": min_avg,
            "min_avg_loss_step": min_avg_step,
            "min_norm_loss": min_norm,
            "min_norm_loss_step": min_norm_step,
            "bleu_2017": b17["bleu"] if b17 else None,
            "bleu_2018": b18["bleu"] if b18 else None,
            "bp_2017": b17["bp"] if b17 else None,
            "ratio_2017": b17["ratio"] if b17 else None,
            "hyp_len_2017": b17["hyp_len"] if b17 else None,
            "ref_len_2017": b17["ref_len"] if b17 else None,
            "prediction_status": pred_status,
            "prediction_warnings": "; ".join(pred_warnings),
            "training_errors": "; ".join(errors),
            "config": config,
            "important_config": flatten_override(config),
        }

        rows.append(row)

        for p in points:
            all_points.append({
                "experiment": name,
                **p
            })

    return rows, all_points


def write_summary_csv(rows):
    fields = [
        "name",
        "bleu_2017",
        "bleu_2018",
        "last_step",
        "last_epoch",
        "last_avg_loss",
        "last_norm_loss",
        "last_ppl",
        "mean_last5_avg_loss",
        "mean_last5_norm_loss",
        "mean_last5_ppl",
        "min_norm_loss",
        "min_norm_loss_step",
        "mean_speed",
        "prediction_status",
        "training_errors",
        "important_config",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = {k: r.get(k, "") for k in fields}
            d["important_config"] = json.dumps(
                r["important_config"], ensure_ascii=False
            )
            w.writerow(d)


def write_points_csv(points):
    fields = [
        "experiment", "step", "epoch", "batch",
        "avg_loss", "norm_loss", "ppl", "speed"
    ]
    with POINTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(points)


def make_report(rows):
    lines = []
    lines.append("=" * 88)
    lines.append("Transformer 超参数实验自动统计报告")
    lines.append("=" * 88)
    lines.append("")

    valid17 = [
        r for r in rows
        if r["bleu_2017"] is not None
        and r["prediction_status"] == "ok"
    ]

    invalid = [
        r for r in rows
        if r["prediction_status"] == "INVALID_SHAPE_WARNING"
    ]

    lines.append(f"发现实验目录数: {len(rows)}")
    lines.append(f"具有有效 BLEU-2017 的实验数: {len(valid17)}")
    lines.append(f"检测到结构加载异常的预测数: {len(invalid)}")
    lines.append("")

    if invalid:
        lines.append("[重要] 以下预测检测到 shape mismatch / Skip loading，BLEU 不应使用：")
        for r in invalid:
            lines.append(f"  - {r['name']}: {r['prediction_warnings']}")
        lines.append("")

    # 按 BLEU 排名
    if valid17:
        ranked = sorted(valid17, key=lambda r: r["bleu_2017"], reverse=True)
        lines.append("-" * 88)
        lines.append("BLEU-2017 排名（仅统计预测日志无 shape mismatch 的实验）")
        lines.append("-" * 88)
        lines.append(
            f"{'Rank':<5} {'Experiment':<24} {'BLEU17':>8} "
            f"{'LastNorm':>10} {'LastPPL':>10} {'Speed':>9}"
        )
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"{i:<5} {r['name']:<24} "
                f"{fmt(r['bleu_2017'],2):>8} "
                f"{fmt(r['last_norm_loss'],4):>10} "
                f"{fmt(r['last_ppl'],2):>10} "
                f"{fmt(r['mean_speed'],2):>9}"
            )
        lines.append("")

        best = ranked[0]
        lines.append(
            f"当前 BLEU-2017 最佳实验: {best['name']} "
            f"(BLEU={best['bleu_2017']:.2f})"
        )
        lines.append(
            "对应关键配置: "
            + json.dumps(best["important_config"], ensure_ascii=False)
        )
        lines.append("")

    # 每组详细统计
    lines.append("-" * 88)
    lines.append("逐实验详细统计")
    lines.append("-" * 88)

    for r in rows:
        lines.append("")
        lines.append(f"[{r['name']}]")
        lines.append(
            "config = "
            + json.dumps(r["important_config"], ensure_ascii=False)
        )

        if r["num_log_points"]:
            lines.append(
                f"train: points={r['num_log_points']}, "
                f"last_step={r['last_step']}, last_epoch={r['last_epoch']}"
            )
            lines.append(
                f"last: avg_loss={fmt(r['last_avg_loss'])}, "
                f"norm_loss={fmt(r['last_norm_loss'])}, "
                f"ppl={fmt(r['last_ppl'],2)}"
            )
            lines.append(
                f"last5 mean: avg_loss={fmt(r['mean_last5_avg_loss'])}, "
                f"norm_loss={fmt(r['mean_last5_norm_loss'])}, "
                f"ppl={fmt(r['mean_last5_ppl'],2)}"
            )
            lines.append(
                f"minimum logged norm_loss={fmt(r['min_norm_loss'])} "
                f"@ step {fmt(r['min_norm_loss_step'],0)}"
            )
            lines.append(
                f"mean speed={fmt(r['mean_speed'],2)} step/s"
            )
        else:
            lines.append("train: 无 train.log 训练点（可能是 beam-only 实验）")

        lines.append(
            f"BLEU: 2017={fmt(r['bleu_2017'],2)}, "
            f"2018={fmt(r['bleu_2018'],2)}"
        )
        lines.append(f"prediction_status: {r['prediction_status']}")

        if r["training_errors"]:
            lines.append(f"training_errors: {r['training_errors']}")
        if r["prediction_warnings"]:
            lines.append(f"prediction_warnings: {r['prediction_warnings']}")

    # 参数分组，便于后续分析单变量规律
    lines.append("")
    lines.append("-" * 88)
    lines.append("按实验名称粗分组的 BLEU-2017")
    lines.append("-" * 88)

    groups = {
        "dropout": [],
        "label_smooth": [],
        "learning_rate": [],
        "warmup": [],
        "ffn": [],
        "beam": [],
        "other": [],
    }

    for r in valid17:
        n = r["name"].lower()
        if n.startswith("dropout"):
            groups["dropout"].append(r)
        elif n.startswith("label"):
            groups["label_smooth"].append(r)
        elif n.startswith("lr"):
            groups["learning_rate"].append(r)
        elif n.startswith("warmup"):
            groups["warmup"].append(r)
        elif n.startswith("ffn"):
            groups["ffn"].append(r)
        elif n.startswith("beam"):
            groups["beam"].append(r)
        else:
            groups["other"].append(r)

    for g, items in groups.items():
        if not items:
            continue
        lines.append(f"{g}:")
        for r in sorted(items, key=lambda x: x["name"]):
            lines.append(
                f"  {r['name']}: BLEU17={fmt(r['bleu_2017'],2)}, "
                f"last5_norm={fmt(r['mean_last5_norm_loss'])}, "
                f"mean_speed={fmt(r['mean_speed'],2)}"
            )

    lines.append("")
    lines.append("=" * 88)
    lines.append("建议发给 ChatGPT 的内容")
    lines.append("=" * 88)
    lines.append(
        "直接上传或复制本 summary_report.txt。若还需要画 loss 曲线，"
        "再附上 training_points.csv。"
    )
    lines.append(
        "注意：若 prediction_status=INVALID_SHAPE_WARNING，"
        "对应 BLEU 必须先重新预测后再分析。"
    )

    return "\n".join(lines) + "\n"


def main():
    rows, points = collect_results()

    write_summary_csv(rows)
    write_points_csv(points)

    report = make_report(rows)
    SUMMARY_TXT.write_text(report, encoding="utf-8")

    print(report)
    print("")
    print("已生成：")
    print(f"  {SUMMARY_TXT}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {POINTS_CSV}")


if __name__ == "__main__":
    main()
