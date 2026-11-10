"""Checkpoint -> continuations for the dev set (scored with the official evaluator) or the test set (zip).

  python -m phase1.eval.predict --run runs/run_A.py --ckpt <runs>/A_decoder_v1/ckpt/step_N.pt \
      --data <p1data> --decode decode_A.json --split dev  --out preds/A_v1_dev
  python -m phase1.eval.predict ... --split test --out preds/A_v1_test     # -> submission.zip

--decode is a JSON file with DecodeConfig values (phase1/eval/generate.py), plus
"max_prompt_tokens" and "batch_size", e.g. {"temperature": ..., "top_p": ..., "max_prompt_tokens": ...}.
Missing keys keep DecodeConfig's defaults (plain sampling, nothing suppressed).

The model is driven through `decoder_api(model)` from the run file if it defines one (fast
KV-cache / recurrent-state prefill+step); otherwise through the slow no-cache adapters.

dev : writes prediction.json and predict_record.json with the official scores (BERTScore F1,
      ROUGE-L F1, BLEU-4 from course_files/eval/Evaluation Script/evaluate_phase1.py, which needs
      `pip install bert-score`). An empty continuation is scored as "." (the evaluator rejects
      empty text) and counted in empty_outputs.
test: writes prediction.json + submission.zip after phase1.eval.submission.validate passes.
"""
import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time

import torch

from ..data.a1_bpe import load_tokenizer
from ..train.launch import load_run
from .generate import CausalNoCache, DecodeConfig, EncDecNoCache, generate
from .submission import validate, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVALUATOR = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase1.py")


def load_decode(path):
    with open(path, encoding="utf-8") as f:
        d = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    extra = {k: d.pop(k) for k in ("max_prompt_tokens", "batch_size") if k in d}
    if "max_prompt_tokens" not in extra:
        raise SystemExit(f"{path}: set max_prompt_tokens (the model's usable prompt length)")
    fields = {f.name for f in dataclasses.fields(DecodeConfig)}
    unknown = set(d) - fields
    if unknown:
        raise SystemExit(f"{path}: unknown decode keys {sorted(unknown)}; valid: {sorted(fields)}")
    return DecodeConfig(**d), extra


def official_scores(truth_path, pred_path):
    if os.environ.get("PHASE1_FAKE_SCORER") == "1":
        from .generate import _unigram_f1
        with open(truth_path, encoding="utf-8") as f:
            truth = json.load(f)
        with open(pred_path, encoding="utf-8") as f:
            pred = json.load(f)
        f1 = sum(_unigram_f1(pred[k]["text"].split(), truth[k]["text"].split()) for k in truth) / len(truth)
        return {"bertscore_f1": f1, "rougeL_f1": None, "bleu4": None, "fake": True}
    r = subprocess.run([sys.executable, EVALUATOR, "--truth", truth_path, "--prediction", pred_path],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit("official evaluator failed")
    out = {}
    for key, name in [("BERTScore F1", "bertscore_f1"), ("ROUGE-L F1", "rougeL_f1"), ("BLEU-4", "bleu4")]:
        m = re.search(re.escape(key) + r":\s+([0-9.]+)", r.stdout)
        out[name] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="prep output dir (tokenizer/, dev/, test.json)")
    ap.add_argument("--decode", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (quick checks)")
    args = ap.parse_args()

    run = load_run(args.run)
    with open(os.path.join(args.data, "prep_manifest.json"), encoding="utf-8") as f:
        tok = load_tokenizer(os.path.join(args.data, json.load(f)["tokenizer"]))
    cfg, extra = load_decode(args.decode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = run.build_model(tok.get_vocab_size())
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"loaded {args.ckpt} (step {state.get('step')}, {state.get('tokens', 0):,} tokens)")
    if hasattr(run, "decoder_api"):
        api = run.decoder_api(model)
    else:
        api = CausalNoCache(model) if run.KIND == "causal" else EncDecNoCache(model)
        print("using the no-cache adapter (slow); define decoder_api(model) in the run file for cached decoding")

    prompts_path = os.path.join(args.data, "dev", "prompts.json") if args.split == "dev" \
        else os.path.join(args.data, "test.json")
    with open(prompts_path, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = dict(list(prompts.items())[:args.limit])
    t = time.time()
    preds, stats = generate(prompts, tok, api, cfg, extra["max_prompt_tokens"],
                            batch_size=extra.get("batch_size", 16), device=device)
    os.makedirs(args.out, exist_ok=True)
    empty = sorted(k for k, v in preds.items() if not v["text"].strip())
    record = {"run": run.NAME, "ckpt": os.path.abspath(args.ckpt), "step": state.get("step"),
              "tokens": state.get("tokens"), "decode": dataclasses.asdict(cfg), **extra, "split": args.split,
              "n": len(preds), "gen_stats": stats, "empty_outputs": len(empty), "seconds": round(time.time() - t)}

    if args.split == "test":
        if empty:
            raise SystemExit(f"{len(empty)} empty continuations (e.g. {empty[:5]}): Codabench rejects empty text. "
                             "Change the decoding (e.g. suppress_eos) and rerun.")
        with open(os.path.join(args.data, "test.json"), encoding="utf-8") as f:
            test = json.load(f)
        if args.limit:
            test = {k: test[k] for k in prompts}
        for w in validate(preds, test):
            print("WARNING:", w)
        pred_path, zip_path = write_submission(preds, args.out)
        print(f"wrote {zip_path}")
    else:
        pred_path = os.path.join(args.out, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump({k: v if v["text"].strip() else {"text": "."} for k, v in preds.items()}, f,
                      ensure_ascii=False, indent=1)
        truth_path = os.path.join(args.data, "dev", "truth.json")
        if args.limit:
            with open(truth_path, encoding="utf-8") as f:
                truth = {k: v for k, v in json.load(f).items() if k in prompts}
            truth_path = os.path.join(args.out, "truth_subset.json")
            with open(truth_path, "w", encoding="utf-8") as f:
                json.dump(truth, f, ensure_ascii=False)
        record["scores"] = official_scores(truth_path, pred_path)
    with open(os.path.join(args.out, "predict_record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print("RECORD " + json.dumps({k: record[k] for k in ("run", "step", "split", "empty_outputs", "seconds")}
                                 | ({"scores": record["scores"]} if "scores" in record else {})))


if __name__ == "__main__":
    main()
