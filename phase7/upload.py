"""Numbered Codabench upload folder for Phase 7 (hybrid zips, ranked by dev micro-F1@7 of the hybrid ranking).

    python -m phase7.upload --results results/p7 --out kaggle/p7_upload [--max 40]
"""
from phase6.upload import run_cli

from .score import K

SIX = [("LoRMA", "A", "lorma", False), ("full", "B", "full", False), ("IA3", "C", "ia3", False),
       ("partial", "D", "partial", False), ("QLoRA", "G", "qlora", False), ("KD", "As", None, True)]


def metrics(r, _run_dir):
    h, d = r.get("dev_hybrid") or {}, r.get("dev_dense") or {}
    if h.get("f1") is None:
        return None
    return {"f1_hybrid": h["f1"], "f1_dense": d.get("f1"), "p@7": h.get("precision"), "r@7": h.get("recall"), "alpha": r.get("alpha")}


def check(pred):
    if not isinstance(pred, dict) or not pred:
        raise ValueError("prediction must map query ids to lists")
    for q, v in pred.items():
        if not isinstance(v, list) or len(v) != K or len(set(v)) != K or not all(isinstance(c, str) for c in v):
            raise ValueError(f"{q}: need exactly {K} unique candidate ids")
        if q in v:
            raise ValueError(f"{q}: a query is ranked as its own prior case")
    return frozenset(pred)


def main(argv=None):
    return run_cli(argv, "results/p7", "kaggle/p7_upload", {"metrics": metrics, "score": "f1_hybrid"}, SIX, check)


if __name__ == "__main__":
    main()
