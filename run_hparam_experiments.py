#!/usr/bin/env python3
"""Sequential one-factor-at-a-time hyperparameter experiments for NLPlab2.

Run this script from the repository root:
    python run_hparam_experiments.py

It does NOT modify train.py / transformer.py / reader.py. It only calls main.py
with different command-line hyperparameters, then predicts on newstest2017,
computes BLEU with multi-bleu.perl, and appends results to CSV.
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
EXP_ROOT = ROOT / "hparam_runs"
RESULT_CSV = EXP_ROOT / "results.csv"

# The already-trained baseline model from your 3.3 experiment.
BASELINE_MODEL_DIR = ROOT / "saved_models" / "step_final"

# During hyperparameter tuning, use 2017 as validation data.
# After selecting the best configuration, change to ("2017", "2018")
# and evaluate the final best model once on 2018.
EVAL_YEARS = ("2017", "2018")

# Keep the baseline setup fixed unless the experiment below overrides one item.
BASE = {
    "epoch": 10,
    "batch_size": 4096,
    "save_step": 1000000,  # only save step_final in tuning runs to save disk
    "print_step": 100,
    "learning_rate": 2.0,
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
    "max_out_len": 255,
}

# Strict one-factor-at-a-time training experiments.
# Baseline value is kept for every field not listed in "override".
TRAIN_EXPERIMENTS = [
    {"name": "dropout_020", "override": {"prepostprocess_dropout": 0.20}},
    {"name": "dropout_030", "override": {"prepostprocess_dropout": 0.30}},
    {"name": "label_smooth_005", "override": {"label_smooth_eps": 0.05}},
    {"name": "label_smooth_020", "override": {"label_smooth_eps": 0.20}},
    {"name": "lr_150", "override": {"learning_rate": 1.5}},
    {"name": "warmup_12000", "override": {"warmup_steps": 12000}},
    {"name": "ffn_1024", "override": {"d_inner_hid": 1024}},
    {"name": "ffn_1536", "override": {"d_inner_hid": 1536}},
]

# Inference-only experiments: reuse the SAME baseline checkpoint.
# beam=5 is already the baseline, so only test alternatives here.
BEAM_EXPERIMENTS = [1, 3, 8]

TRAIN_LOG_RE = re.compile(
    r"step_idx:\s*(\d+).*?avg loss:\s*([0-9.eE+-]+).*?"
    r"normalized loss:\s*([0-9.eE+-]+).*?ppl:\s*([0-9.eE+-]+)"
)
BLEU_RE = re.compile(r"BLEU\s*=\s*([0-9.]+)")

# Parameters that determine the model instantiated by TransformerModel.from_args().
# These MUST match training when loading a checkpoint for prediction.
MODEL_BUILD_KEYS = (
    "d_model",
    "d_inner_hid",
    "d_key",
    "d_value",
    "n_head",
    "n_layer",
    "max_length",
    "prepostprocess_dropout",
    "attention_dropout",
    "relu_dropout",
    "label_smooth_eps",
)


def ensure_layout() -> None:
    required = [MAIN, DATA, MOSES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))
    EXP_ROOT.mkdir(parents=True, exist_ok=True)


def arg_pairs(params: dict) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        args.extend([f"--{key}", str(value)])
    return args


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    """Run a command, show output live, and save the same output to a log file."""
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
    last = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_LOG_RE.search(line)
            if m:
                last = m
    if last is None:
        return {
            "last_step": "",
            "last_avg_loss": "",
            "last_norm_loss": "",
            "last_ppl": "",
        }
    return {
        "last_step": int(last.group(1)),
        "last_avg_loss": float(last.group(2)),
        "last_norm_loss": float(last.group(3)),
        "last_ppl": float(last.group(4)),
    }


def train_experiment(name: str, override: dict) -> tuple[Path, dict]:
    exp_dir = EXP_ROOT / name
    model_root = exp_dir / "models"
    final_model = model_root / "step_final"
    log_path = exp_dir / "train.log"
    exp_dir.mkdir(parents=True, exist_ok=True)

    params = BASE.copy()
    params.update(override)
    params["save_model_path"] = str(model_root)

    # Save an exact machine-readable record of the configuration.
    with (exp_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # Resume-friendly behavior: if the final checkpoint exists, do not train again.
    pdparams = final_model / "transformer.pdparams"
    pdopt = final_model / "transformer.pdopt"
    if pdparams.exists():
        print(f"\n[{name}] final checkpoint already exists; skip training.")
        # A previous run may have filled the disk while writing .pdopt.
        # Prediction only needs .pdparams, so remove any optimizer state here.
        if pdopt.exists():
            pdopt.unlink()
            print(f"[{name}] removed optimizer state to save disk: {pdopt}")
        status = "skipped_existing"
        elapsed = 0.0
    else:
        cmd = [
            PYTHON, "-u", str(MAIN),
            "--do_train", "True",
            "--src_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
            "--trg_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
            "--special_token", "<s>", "<e>", "<unk>",
            "--training_file", str(DATA / "train.tok.clean.bpe.32000.en-zh"),
        ] + arg_pairs({k: v for k, v in params.items() if k != "beam_size" and k != "max_out_len"})

        t0 = time.time()
        rc = run_and_tee(cmd, log_path)
        elapsed = time.time() - t0
        if rc != 0:
            return final_model, {
                "train_status": f"failed_rc_{rc}",
                "train_seconds": round(elapsed, 1),
                **parse_last_train_metrics(log_path),
            }
        status = "ok"

        # For hyperparameter evaluation we only need model weights.
        # Adam state is useful for resuming training, but consumes a lot of disk.
        if pdopt.exists():
            pdopt.unlink()
            print(f"[{name}] removed optimizer state to save disk: {pdopt}")

    metrics = parse_last_train_metrics(log_path) if log_path.exists() else {
        "last_step": "", "last_avg_loss": "", "last_norm_loss": "", "last_ppl": ""
    }
    metrics.update({"train_status": status, "train_seconds": round(elapsed, 1)})
    return final_model, metrics


def debpe(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"(@@ )|(@@ ?$)")
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
         dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            # Apply the same transformation as:
            # sed -r 's/(@@ )|(@@ ?$)//g'
            fout.write(pattern.sub("", line.rstrip("\n")) + "\n")


def evaluate_bleu(hyp_tok: Path, ref: Path, bleu_log: Path) -> float:
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
    bleu_log.write_text(text, encoding="utf-8")
    print(text, end="" if text.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"BLEU evaluation failed with return code {proc.returncode}")
    m = BLEU_RE.search(text)
    if not m:
        raise RuntimeError("Could not parse BLEU score from multi-bleu.perl output")
    return float(m.group(1))


def predict_and_eval(
    name: str,
    model_dir: Path,
    beam_size: int,
    model_params: dict,
) -> dict:
    """Predict with exactly the same model-shape parameters used for training.

    This matters for structural experiments such as d_inner_hid/n_layer/n_head/
    d_model. Loading a checkpoint into a baseline-shaped model can otherwise
    silently skip mismatched tensors in Paddle and produce an invalid BLEU.
    """
    if not (model_dir / "transformer.pdparams").exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_dir / 'transformer.pdparams'}"
        )

    exp_dir = EXP_ROOT / name
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}

    # Exact architecture/configuration used to instantiate the model for inference.
    build_params = {k: model_params[k] for k in MODEL_BUILD_KEYS if k in model_params}

    for year in EVAL_YEARS:
        raw_pred = pred_dir / f"predict_{year}.txt"
        tok_pred = pred_dir / f"predict_{year}.tok.txt"
        pred_log = pred_dir / f"predict_{year}.log"
        bleu_log = pred_dir / f"bleu_{year}.log"
        meta_path = pred_dir / f"predict_{year}.meta.json"

        wanted_meta = {
            "model_dir": str(model_dir.resolve()),
            "beam_size": beam_size,
            "max_out_len": model_params.get("max_out_len", BASE["max_out_len"]),
            "model_build_params": build_params,
        }

        # Only reuse a prediction when we can prove it was produced with the
        # same checkpoint/config/beam. Old predictions made by the buggy script
        # have no matching metadata and are therefore regenerated automatically.
        reuse_prediction = False
        if raw_pred.exists() and meta_path.exists():
            try:
                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                reuse_prediction = old_meta == wanted_meta
            except Exception:
                reuse_prediction = False

        if reuse_prediction:
            print(f"[{name}] {raw_pred.name} matches config; skip prediction.")
        else:
            if raw_pred.exists():
                print(
                    f"[{name}] existing {raw_pred.name} has missing/mismatched "
                    "metadata; regenerate it."
                )

            cmd = [
                PYTHON, "-u", str(MAIN),
                "--do_predict", "True",
                "--src_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
                "--trg_vocab_fpath", str(DATA / "vocab_all.bpe.32000"),
                "--special_token", "<s>", "<e>", "<unk>",
                "--predict_file", str(
                    DATA / f"newstest{year}.tok.bpe.32000.en-zh"
                ),
                "--batch_size", "32",
                "--init_from_params", str(model_dir),
                "--beam_size", str(beam_size),
                "--max_out_len", str(
                    model_params.get("max_out_len", BASE["max_out_len"])
                ),
                "--output_file", str(raw_pred),
            ] + arg_pairs(build_params)

            rc = run_and_tee(cmd, pred_log)
            if rc != 0:
                raise RuntimeError(
                    f"Prediction failed for {name}, {year}, return code={rc}"
                )

            # Write metadata only after prediction succeeds.
            meta_path.write_text(
                json.dumps(wanted_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        debpe(raw_pred, tok_pred)
        ref = DATA / f"newstest{year}.tok.zh"
        scores[f"bleu_{year}"] = evaluate_bleu(tok_pred, ref, bleu_log)

    return scores


def append_result(row: dict) -> None:
    """Insert or replace one experiment row in results.csv.

    The old script appended blindly. This version replaces an existing row
    with the same (name, kind), so rerunning after a prediction/config bug
    automatically removes the stale BLEU instead of creating duplicates.
    """
    fields = [
        "name", "kind", "override", "model_dir", "beam_size",
        "train_status", "train_seconds",
        "last_step", "last_avg_loss", "last_norm_loss", "last_ppl",
        "bleu_2017", "bleu_2018",
    ]
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

    normalized = {k: row.get(k, "") for k in fields}
    old_rows = []

    if RESULT_CSV.exists():
        with RESULT_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for old in reader:
                # Replace the same logical experiment.
                if (
                    old.get("name") == str(normalized["name"])
                    and old.get("kind") == str(normalized["kind"])
                ):
                    continue
                old_rows.append({k: old.get(k, "") for k in fields})

    old_rows.append(normalized)

    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old_rows)


def main() -> None:
    ensure_layout()

    print("=" * 72)
    print("Hyperparameter experiments (one factor at a time)")
    print(f"Python: {PYTHON}")
    print(f"Repo:   {ROOT}")
    print(f"Eval:   {', '.join(EVAL_YEARS)}")
    print(f"CSV:    {RESULT_CSV}")
    print("=" * 72)

    # 1) Training-time hyperparameters.
    for exp in TRAIN_EXPERIMENTS:
        name = exp["name"]
        override = exp["override"]
        print(f"\n\n######## TRAIN EXPERIMENT: {name}  override={override} ########")
        try:
            model_dir, train_metrics = train_experiment(name, override)
            if train_metrics["train_status"].startswith("failed"):
                append_result({
                    "name": name,
                    "kind": "train",
                    "override": json.dumps(override, ensure_ascii=False),
                    "model_dir": str(model_dir),
                    "beam_size": BASE["beam_size"],
                    **train_metrics,
                })
                continue

            model_params = BASE.copy()
            model_params.update(override)
            bleu = predict_and_eval(
                name, model_dir, BASE["beam_size"], model_params
            )
            append_result({
                "name": name,
                "kind": "train",
                "override": json.dumps(override, ensure_ascii=False),
                "model_dir": str(model_dir),
                "beam_size": BASE["beam_size"],
                **train_metrics,
                **bleu,
            })
        except Exception as e:
            print(f"[ERROR] {name}: {e}", file=sys.stderr)
            append_result({
                "name": name,
                "kind": "train",
                "override": json.dumps(override, ensure_ascii=False),
                "model_dir": "",
                "beam_size": BASE["beam_size"],
                "train_status": f"exception: {e}",
            })

    # 2) Inference-only beam-size sweep using the baseline checkpoint.
    if not (BASELINE_MODEL_DIR / "transformer.pdparams").exists():
        print(
            f"\n[WARN] Baseline checkpoint not found at {BASELINE_MODEL_DIR}; "
            "skip beam-size experiments."
        )
    else:
        for beam in BEAM_EXPERIMENTS:
            name = f"beam_{beam}"
            print(f"\n\n######## BEAM EXPERIMENT: beam={beam} ########")
            try:
                bleu = predict_and_eval(
                    name, BASELINE_MODEL_DIR, beam, BASE
                )
                append_result({
                    "name": name,
                    "kind": "beam",
                    "override": json.dumps({"beam_size": beam}),
                    "model_dir": str(BASELINE_MODEL_DIR),
                    "beam_size": beam,
                    "train_status": "not_applicable",
                    "train_seconds": 0,
                    **bleu,
                })
            except Exception as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)
                append_result({
                    "name": name,
                    "kind": "beam",
                    "override": json.dumps({"beam_size": beam}),
                    "model_dir": str(BASELINE_MODEL_DIR),
                    "beam_size": beam,
                    "train_status": f"exception: {e}",
                })

    print("\nAll scheduled experiments finished.")
    print(f"Results: {RESULT_CSV}")


if __name__ == "__main__":
    main()
