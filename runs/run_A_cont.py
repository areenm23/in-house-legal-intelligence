"""Continued training of run_A.py (run_A_cont.py): starts from the newest A_decoder checkpoint in the attached
inputs (weights only), fresh optimizer, re-warmed cosine schedule at half the original peak LR,
new data seed (different token windows)."""
import importlib.util
import os

from phase1.models.optim import make_adamw, warmup_cosine

_spec = importlib.util.spec_from_file_location("_base_A", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        "run_A.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

NAME = _base.NAME + "_cont"
INIT_FROM = _base.NAME
KIND = _base.KIND
CFG = dict(_base.CFG, seed=4321, token_milestones=[])
build_model = _base.build_model
decoder_api = _base.decoder_api


def make_optimizer(model):
    return make_adamw(model, lr=3e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(100, max(1, total_steps // 20)), final_frac=0.1)
