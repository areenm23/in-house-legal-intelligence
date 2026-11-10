"""Build, validate and zip a Phase-1 Codabench submission.

In code:
    from phase1.eval.submission import to_prediction_text, validate, write_submission
    pred = {id_: {"text": to_prediction_text(generated)} for id_, generated in outputs.items()}
    validate(pred, test)            # raises on anything Codabench would reject
    write_submission(pred, "runs/decoder_v1")   # -> prediction.json + submission.zip

From the shell (validate an existing file or zip against the official test prompts):
    python -m phase1.eval.submission --check runs/x/submission.zip --test data/raw/Phase_1_Pretraining/Test/test.json
"""
import argparse
import json
import os
import zipfile

TARGET_WORDS = 128


def to_prediction_text(generated, n_words=TARGET_WORDS):
    """Keep the first n whitespace-separated words of the generated continuation (prompt excluded)."""
    return " ".join(generated.split()[:n_words])


def validate(pred, test, n_words=TARGET_WORDS):
    """Mirror the checks in evaluate_phase1.py and the rules in the course document; return warnings."""
    if not isinstance(pred, dict) or not pred:
        raise ValueError("prediction must be a non-empty JSON object")
    missing, extra = set(test) - set(pred), set(pred) - set(test)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={sorted(missing)[:5]} ({len(missing)}), "
                         f"extra={sorted(extra)[:5]} ({len(extra)})")
    warnings, short = [], 0
    for id_, rec in pred.items():
        if not isinstance(rec, dict) or not isinstance(rec.get("text"), str) or not rec["text"].strip():
            raise ValueError(f"ID {id_!r}: needs a non-empty string field 'text'")
        n = len(rec["text"].split())
        if n > n_words:
            raise ValueError(f"ID {id_!r}: {n} words > {n_words}")
        short += n < n_words
        head = rec["text"].split()[:20]
        if head == test[id_]["text"].split()[:len(head)]:
            warnings.append(f"ID {id_!r}: text looks like it starts with the prompt")
    if short:
        warnings.append(f"{short} predictions have fewer than {n_words} words")
    return warnings


def write_submission(pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=1)
    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    return pred_path, zip_path


def load_prediction(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if names != ["prediction.json"]:
                raise ValueError(f"zip must contain exactly prediction.json at its root, found {names}")
            return json.loads(z.read("prediction.json"))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", required=True, help="prediction.json or submission.zip")
    ap.add_argument("--test", required=True, help="official test prompts JSON")
    args = ap.parse_args()
    with open(args.test, encoding="utf-8") as f:
        test = json.load(f)
    pred = load_prediction(args.check)
    for w in validate(pred, test):
        print("WARNING:", w)
    print(f"OK: {len(pred)} predictions match the {len(test)} test IDs")


if __name__ == "__main__":
    main()
