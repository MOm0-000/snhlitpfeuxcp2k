#!/usr/bin/env python3
"""Section 3.7 model-structure experiments for NLPlab2.

Run from the repository root after replacing transformer.py and adding the
new scalar keys to transformer.yaml:

    python run_model_improvements.py

The script keeps the tuned training setup fixed (learning_rate=1.5) and changes
ONLY one model-structure switch at a time:

    drophead_010       : existing Scheduled DropHead, p_max=0.10
    drophead_005       : existing Scheduled DropHead, p_max=0.05
    swiglu_1360        : existing parameter-matched SwiGLU
    swiglu_1536        : existing wider SwiGLU
    relative_bias_16   : learned per-head relative-position bias, max distance 16

The already trained lr_150 checkpoint is reused as the control. All BLEU scores
are recomputed against the current references. Before BLEU evaluation, the
script verifies that hypothesis and reference line counts are identical.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MAIN = ROOT / "main.py"
DATA = ROOT / "wmt18_zhen_data_bpe3"
MOSES = ROOT / "mosesdecoder" / "scripts" / "generic" / "multi-bleu.perl"
EXP_ROOT = ROOT / "improvement_runs"
RESULT_CSV = EXP_ROOT / "results.csv"

# 3.6 best single-factor model, used as the control for Section 3.7.
CONTROL_NAME = "control_lr150"
CONTROL_MODEL_DIR = ROOT / "hparam_runs" / "lr_150" / "models" / "step_final"
CONTROL_TRAIN_LOG = ROOT / "hparam_runs" / "lr_150" / "train.log"
CONTROL_PRED_DIR = ROOT / "hparam_runs" / "lr_150" / "predictions"

EVAL_YEARS = ("2017", "2018")

# Keep ALL ordinary training hyperparameters fixed at the 3.6 lr_150 setup.
BASE = {
    "epoch": 10,
    "batch_size": 4096,
    "save_step": 1000000,
    "print_step": 100,
    "learning_rate": 1.5,
    "warmup_steps": 8000,
    "label_smooth_eps": 0.1,
    "prepostprocess_dropout": 0.1,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "d_model": 512,
    "d_inner_hid": 2048,
    "d_key": 64,
    "d_value": 64,
    "n_head": 8,
    "n_layer": 6,
    "max_length": 256,
    "beam_size": 5,
    "max_out_len": 255,  # keep identical to the 3.6 automated evaluation

    # Section 3.7 switches. 'normal/relu' reproduces the old Transformer path.
    "attention_variant": "normal",
    "ffn_variant": "relu",
    "drophead_rate": 0.1,
    "drophead_schedule": "v",
    "drophead_warmup_steps": 8000,
    "drophead_total_steps": 18400,
    "swiglu_inner_hid": 1360,
    "max_relative_position": 16,
}

# Existing completed experiments remain in the list so reruns reuse their
# checkpoints/predictions. relative_bias_16 is the new Section-3.7 experiment.
EXPERIMENTS = [
    {
        "name": "drophead_010",
        "description": "Scheduled DropHead p_max=0.10 (existing run)",
        "enabled": True,
        "override": {
            "attention_variant": "drophead",
            "ffn_variant": "relu",
            "drophead_rate": 0.10,
            "drophead_schedule": "v",
            "drophead_warmup_steps": 8000,
            "drophead_total_steps": 18400,
        },
    },
    {
        "name": "drophead_005",
        "description": "Scheduled DropHead p_max=0.05 (existing run)",
        "enabled": True,
        "override": {
            "attention_variant": "drophead",
            "ffn_variant": "relu",
            "drophead_rate": 0.05,
            "drophead_schedule": "v",
            "drophead_warmup_steps": 8000,
            "drophead_total_steps": 18400,
        },
    },
    {
        "name": "swiglu_1360",
        "description": "Parameter-matched SwiGLU FFN (existing run)",
        "enabled": True,
        "override": {
            "attention_variant": "normal",
            "ffn_variant": "swiglu",
            "swiglu_inner_hid": 1360,
        },
    },
    {
        "name": "swiglu_1536",
        "description": "Wider SwiGLU FFN, inner_hid=1536 (existing run)",
        "enabled": True,
        "override": {
            "attention_variant": "normal",
            "ffn_variant": "swiglu",
            "swiglu_inner_hid": 1536,
        },
    },
    {
        "name": "relative_bias_16",
        "description": "Learned per-head relative-position bias, max distance=16",
        "enabled": True,
        "override": {
            "attention_variant": "relative_bias",
            "ffn_variant": "relu",
            "max_relative_position": 16,
        },
    },
]

TRAIN_LOG_RE = re.compile(
    r"step_idx:\s*(\d+).*?avg loss:\s*([0-9.eE+-]+).*?"
    r"normalized loss:\s*([0-9.eE+-]+).*?ppl:\s*([0-9.eE+-]+)"
)
BLEU_RE = re.compile(r"BLEU\s*=\s*([0-9.]+)")

# Every argument here is consumed by TransformerModel.from_args() and therefore
# must be identical at training and prediction time.
MODEL_BUILD_KEYS = (
    "d_model", "d_inner_hid", "d_key", "d_value", "n_head", "n_layer",
    "max_length", "prepostprocess_dropout", "attention_dropout", "relu_dropout",
    "label_smooth_eps",
    "attention_variant", "ffn_variant",
    "drophead_rate", "drophead_schedule", "drophead_warmup_steps",
    "drophead_total_steps", "swiglu_inner_hid",
    "max_relative_position",
)


def ensure_layout() -> None:
    required = [MAIN, DATA, MOSES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    # The 2018 file should now be the corrected/aligned reference. We don't
    # hard-code 3979 here; line-count equality is checked per prediction below.
    for year in EVAL_YEARS:
        ref = DATA / f"newstest{year}.tok.zh"
        if not ref.exists():
            raise FileNotFoundError(f"Missing reference: {ref}")

    EXP_ROOT.mkdir(parents=True, exist_ok=True)


def arg_pairs(params: dict) -> list[str]:
    out: list[str] = []
    for key, value in params.items():
        out.extend([f"--{key}", str(value)])
    return out


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fout.write(line)
        return proc.wait()


def parse_last_train_metrics(log_path: Path) -> dict:
    if not log_path.exists():
        return {
            "last_step": "", "last_avg_loss": "", "last_norm_loss": "",
            "last_ppl": "",
        }
    last = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_LOG_RE.search(line)
            if m:
                last = m
    if last is None:
        return {
            "last_step": "", "last_avg_loss": "", "last_norm_loss": "",
            "last_ppl": "",
        }
    return {
        "last_step": int(last.group(1)),
        "last_avg_loss": float(last.group(2)),
        "last_norm_loss": float(last.group(3)),
        "last_ppl": float(last.group(4)),
    }


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def debpe(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"(@@ )|(@@ ?$)")
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
         dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(pattern.sub("", line.rstrip("\n")) + "\n")


def evaluate_bleu(hyp_tok: Path, ref: Path, bleu_log: Path) -> float:
    hyp_n = count_lines(hyp_tok)
    ref_n = count_lines(ref)
    if hyp_n != ref_n:
        raise RuntimeError(
            f"Ref/hyp line mismatch: hyp={hyp_n}, ref={ref_n}. "
            f"hyp={hyp_tok}, ref={ref}. Refuse to compute invalid BLEU."
        )

    print(f"[BLEU] line alignment OK: {hyp_n} hypothesis == {ref_n} reference")
    with hyp_tok.open("rb") as fin:
        proc = subprocess.run(
            ["perl", str(MOSES), str(ref)],
            cwd=ROOT,
            stdin=fin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = proc.stdout.decode("utf-8", errors="replace")
    bleu_log.parent.mkdir(parents=True, exist_ok=True)
    bleu_log.write_text(text, encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"BLEU evaluation failed, return code={proc.returncode}")
    m = BLEU_RE.search(text)
    if not m:
        raise RuntimeError("Could not parse BLEU from multi-bleu.perl output")
    return float(m.group(1))


def train_experiment(name: str, override: dict) -> tuple[Path, dict, dict]:
    exp_dir = EXP_ROOT / name
    model_root = exp_dir / "models"
    final_model = model_root / "step_final"
    log_path = exp_dir / "train.log"
    exp_dir.mkdir(parents=True, exist_ok=True)

    params = BASE.copy()
    params.update(override)
    params["save_model_path"] = str(model_root)

    with (exp_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    pdparams = final_model / "transformer.pdparams"
    pdopt = final_model / "transformer.pdopt"

    if pdparams.exists():
        print(f"\n[{name}] final checkpoint already exists; skip training.")
        if pdopt.exists():
            pdopt.unlink()
            print(f"[{name}] removed optimizer state to save disk: {pdopt}")
        status = "skipped_existing"
        elapsed = 0.0
    else:
        train_params = {
            k: v for k, v in params.items()
            if k not in ("beam_size", "max_out_len")
        }
        cmd = [
            PYTHON, "-u", str(MAIN),
            "--do_train", "True",
            "--src_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
            "--trg_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
            "--special_token", "<s>", "<e>", "<unk>",
            "--training_file", str(DATA / "train.tok.clean.bpe.32000.en-zh"),
        ] + arg_pairs(train_params)

        t0 = time.time()
        rc = run_and_tee(cmd, log_path)
        elapsed = time.time() - t0
        if rc != 0:
            metrics = parse_last_train_metrics(log_path)
            metrics.update({
                "train_status": f"failed_rc_{rc}",
                "train_seconds": round(elapsed, 1),
            })
            return final_model, metrics, params
        status = "ok"

        if pdopt.exists():
            pdopt.unlink()
            print(f"[{name}] removed optimizer state to save disk: {pdopt}")

    metrics = parse_last_train_metrics(log_path)
    metrics.update({"train_status": status, "train_seconds": round(elapsed, 1)})
    return final_model, metrics, params


def predict_and_eval(name: str, model_dir: Path, model_params: dict,
                     beam_size: int = 5) -> dict:
    checkpoint = model_dir / "transformer.pdparams"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    exp_dir = EXP_ROOT / name
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    build_params = {k: model_params[k] for k in MODEL_BUILD_KEYS if k in model_params}
    scores: dict[str, float] = {}

    for year in EVAL_YEARS:
        raw_pred = pred_dir / f"predict_{year}.txt"
        tok_pred = pred_dir / f"predict_{year}.tok.txt"
        pred_log = pred_dir / f"predict_{year}.log"
        bleu_log = pred_dir / f"bleu_{year}.log"
        meta_path = pred_dir / f"predict_{year}.meta.json"

        wanted_meta = {
            "checkpoint": str(checkpoint.resolve()),
            "beam_size": beam_size,
            "max_out_len": model_params.get("max_out_len", BASE["max_out_len"]),
            "model_build_params": build_params,
        }

        reuse = False
        if raw_pred.exists() and meta_path.exists():
            try:
                reuse = json.loads(meta_path.read_text(encoding="utf-8")) == wanted_meta
            except Exception:
                reuse = False

        if reuse:
            print(f"[{name}] {raw_pred.name} matches checkpoint/config; skip prediction.")
        else:
            if raw_pred.exists():
                print(f"[{name}] stale/mismatched {raw_pred.name}; regenerate.")
            cmd = [
                PYTHON, "-u", str(MAIN),
                "--do_predict", "True",
                "--src_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
                "--trg_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
                "--special_token", "<s>", "<e>", "<unk>",
                "--predict_file", str(DATA / f"newstest{year}.tok.bpe.32000.en-zh"),
                "--batch_size", "32",
                "--init_from_params", str(model_dir),
                "--beam_size", str(beam_size),
                "--max_out_len", str(model_params.get("max_out_len", BASE["max_out_len"])),
                "--output_file", str(raw_pred),
            ] + arg_pairs(build_params)

            rc = run_and_tee(cmd, pred_log)
            if rc != 0:
                raise RuntimeError(
                    f"Prediction failed for {name}/{year}, return code={rc}"
                )
            meta_path.write_text(
                json.dumps(wanted_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        debpe(raw_pred, tok_pred)
        ref = DATA / f"newstest{year}.tok.zh"
        scores[f"bleu_{year}"] = evaluate_bleu(tok_pred, ref, bleu_log)

    return scores


def evaluate_control() -> dict:
    """Recompute the lr_150 control BLEU against the CURRENT references."""
    exp_dir = EXP_ROOT / CONTROL_NAME
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(
        json.dumps(BASE, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    scores = {}
    can_reuse_hparam_preds = all(
        (CONTROL_PRED_DIR / f"predict_{year}.txt").exists()
        for year in EVAL_YEARS
    )

    if can_reuse_hparam_preds:
        print("\n[control_lr150] reuse existing 3.6 predictions; recompute BLEU only.")
        for year in EVAL_YEARS:
            src_raw = CONTROL_PRED_DIR / f"predict_{year}.txt"
            raw_pred = pred_dir / f"predict_{year}.txt"
            tok_pred = pred_dir / f"predict_{year}.tok.txt"
            # Do not duplicate hundreds of MB: create/update a small relative symlink.
            if raw_pred.exists() or raw_pred.is_symlink():
                raw_pred.unlink()
            rel_target = os.path.relpath(src_raw, pred_dir)
            raw_pred.symlink_to(rel_target)
            debpe(src_raw, tok_pred)
            ref = DATA / f"newstest{year}.tok.zh"
            scores[f"bleu_{year}"] = evaluate_bleu(
                tok_pred, ref, pred_dir / f"bleu_{year}.log"
            )
    else:
        print("\n[control_lr150] old predictions missing; run prediction from checkpoint.")
        scores = predict_and_eval(CONTROL_NAME, CONTROL_MODEL_DIR, BASE, BASE["beam_size"])

    metrics = parse_last_train_metrics(CONTROL_TRAIN_LOG)
    metrics.update({"train_status": "control_existing", "train_seconds": 0})
    return {**metrics, **scores}


def upsert_result(row: dict) -> None:
    fields = [
        "name", "kind", "description", "override", "model_dir", "beam_size",
        "train_status", "train_seconds", "last_step", "last_avg_loss",
        "last_norm_loss", "last_ppl", "bleu_2017", "bleu_2018",
    ]
    normalized = {k: row.get(k, "") for k in fields}
    rows = []

    if RESULT_CSV.exists():
        with RESULT_CSV.open("r", newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                if old.get("name") == str(normalized["name"]):
                    continue
                rows.append({k: old.get(k, "") for k in fields})
    rows.append(normalized)

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ensure_layout()

    print("=" * 80)
    print("Section 3.7 Transformer model-improvement experiments")
    print(f"Python: {PYTHON}")
    print(f"Repo:   {ROOT}")
    print(f"Eval:   {', '.join(EVAL_YEARS)}")
    print(f"Root:   {EXP_ROOT}")
    print("Control: learning_rate=1.5, attention=normal, FFN=relu")
    print("=" * 80)

    run_status: dict[str, str] = {}

    # Control first: no training, only recompute BLEU under the current references.
    if not (CONTROL_MODEL_DIR / "transformer.pdparams").exists():
        msg = f"control checkpoint missing: {CONTROL_MODEL_DIR}"
        print(f"[ERROR] {msg}", file=sys.stderr)
        run_status[CONTROL_NAME] = "FAILED: missing checkpoint"
    else:
        try:
            control = evaluate_control()
            upsert_result({
                "name": CONTROL_NAME,
                "kind": "control",
                "description": "3.6 best single-factor control (lr=1.5)",
                "override": json.dumps({"learning_rate": 1.5}, ensure_ascii=False),
                "model_dir": str(CONTROL_MODEL_DIR),
                "beam_size": BASE["beam_size"],
                **control,
            })
            run_status[CONTROL_NAME] = "OK"
        except Exception as e:
            print(f"[ERROR] {CONTROL_NAME}: {e}", file=sys.stderr)
            run_status[CONTROL_NAME] = f"FAILED: {e}"

    for exp in EXPERIMENTS:
        name = exp["name"]
        override = exp["override"]

        if not exp.get("enabled", True):
            print(f"\n[{name}] disabled; skip.")
            run_status[name] = "SKIPPED (disabled)"
            continue

        print(f"\n\n######## MODEL EXPERIMENT: {name} ########")
        print(exp["description"])
        print("override =", override)

        try:
            model_dir, train_metrics, params = train_experiment(name, override)

            if train_metrics["train_status"].startswith("failed"):
                upsert_result({
                    "name": name,
                    "kind": "model_improvement",
                    "description": exp["description"],
                    "override": json.dumps(override, ensure_ascii=False),
                    "model_dir": str(model_dir),
                    "beam_size": BASE["beam_size"],
                    **train_metrics,
                })
                run_status[name] = f"FAILED: {train_metrics['train_status']}"
                continue

            bleu = predict_and_eval(name, model_dir, params, BASE["beam_size"])
            upsert_result({
                "name": name,
                "kind": "model_improvement",
                "description": exp["description"],
                "override": json.dumps(override, ensure_ascii=False),
                "model_dir": str(model_dir),
                "beam_size": BASE["beam_size"],
                **train_metrics,
                **bleu,
            })
            run_status[name] = "OK"

        except Exception as e:
            print(f"[ERROR] {name}: {e}", file=sys.stderr)
            upsert_result({
                "name": name,
                "kind": "model_improvement",
                "description": exp["description"],
                "override": json.dumps(override, ensure_ascii=False),
                "model_dir": "",
                "beam_size": BASE["beam_size"],
                "train_status": f"exception: {e}",
            })
            run_status[name] = f"FAILED: {e}"

    print("\n" + "=" * 80)
    print("Section 3.7 run summary")
    print("=" * 80)
    for name, status in run_status.items():
        print(f"{name:<20} {status}")

    failed = [name for name, status in run_status.items()
              if status.startswith("FAILED")]

    print(f"\nResults: {RESULT_CSV}")
    print("Next: python stat_model_improvements.py")

    if failed:
        print(
            f"\nCompleted with {len(failed)} failed experiment(s): "
            + ", ".join(failed)
        )
        return 1

    print("\nAll enabled Section-3.7 experiments completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
