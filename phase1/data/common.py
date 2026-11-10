"""Shared helpers for the Phase-1 data pipeline.

The Phase-1 training files are multi-GB JSON arrays of records like {"text": ...}.
`json.load` on them is what crashes Kaggle (and would need ~3-5x the file size in
RAM), so everything here streams one record at a time.
"""
import hashlib
import json
import os
import re

SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
HF_TRAIN_DIR = "Phase_1_Pretraining/Train"


def source_file(raw_dir, source):
    return os.path.join(raw_dir, HF_TRAIN_DIR, f"train_{source}.json")


_SEP = re.compile(r"[\s,]*")


def iter_json_records(path, chunk_chars=1 << 22):
    """Yield the elements of a top-level JSON array (or of a JSON-lines file) one by one.

    Memory stays at roughly one chunk plus the largest single record.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk_chars)
        pos = _SEP.match(buf, 0).end()
        while pos == len(buf):
            more = f.read(chunk_chars)
            if not more:
                return
            buf, pos = more, _SEP.match(more, 0).end()
        in_array = buf[pos] == "["
        if in_array:
            pos += 1
        while True:
            pos = _SEP.match(buf, pos).end()
            if in_array and pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk_chars)
                if not more:
                    if not in_array and pos == len(buf):
                        return
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end


def record_text(rec, field="text"):
    if not isinstance(rec, dict) or not isinstance(rec.get(field), str):
        raise ValueError(f"record has no string field {field!r}: keys={list(rec)[:10] if isinstance(rec, dict) else type(rec)}")
    return rec[field]


def _unit_hash(*parts):
    """Deterministic float in [0, 1) from the given parts (same on every machine)."""
    h = hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def is_val_doc(source, index, val_frac):
    """Document-level split: a whole document is either train or val, never both."""
    return _unit_hash("split", source, index) < val_frac


def keep_train_doc(source, index, keep_frac):
    """Deterministic subsampling of training documents (independent of the val split)."""
    return _unit_hash("keep", source, index) < keep_frac
