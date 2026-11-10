"""Build self-contained Kaggle notebooks (code embedded, no code dataset needed) and optionally push them.

  python kaggle/build.py prep --section smoke                     # CPU: data prep check on acts only
  python kaggle/build.py prep --section full                      # CPU: full data prep
  python kaggle/build.py train --slug cs781-a-pilot --gpu0 runs/run_A.py --pilot 200
  python kaggle/build.py train --slug cs781-ab-s1 --gpu0 runs/run_A.py --gpu1 runs/run_B.py
  python kaggle/build.py train --slug cs781-ab-s2 --gpu0 ... --resume-from cs781-ab-s1   # next 12 h session
  python kaggle/build.py generate --slug cs781-a-gen1 --run runs/run_A.py --decode decode/A1.json       --from cs781-ab-s1 [--ckpt step_12000.pt] [--splits dev test]          # dev scores + submission.zip
  python kaggle/build.py audit                                    # CPU: corpus audit (stats, dedup, fertility, dev/test overlap)
  add --push to upload and start it with the Kaggle CLI (account = KAGGLE_CONFIG_DIR / default config)

Each build writes kaggle/out/<slug>/{notebook.ipynb, kernel-metadata.json}. The notebook can
also be imported by hand (Kaggle: File -> Import Notebook). Settings it needs:
  prep : CPU, Internet on, secret HF_TOKEN attached (Add-ons -> Secrets; once per notebook)
  generate: GPU, Internet on (pip install bert-score), inputs = prep output + the training output
  train: GPU T4 x2, Internet on, inputs = the prep notebook's output (+ the previous session's
         output when resuming); the metadata file lists them for a CLI push.
Run with "Save Version -> Save & Run All (Commit)" so /kaggle/working is kept as the output.
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "out")
PREP_SLUG = {"smoke": "cs781-p1-prep-smoke", "full": "cs781-p1-prep"}
BUNDLE_DIRS = [("phase1", True), (os.path.join("study", "Areen_implementations"), True), ("runs", False),
               ("decode", False), ("ft", True), ("phase2", True),
               ("phase3", True), ("phase4", True), ("phase5", True), ("phase6", True), ("phase7", True)]
EXTRA_FILES = [os.path.join("course_files", "eval", "Evaluation Script", f"evaluate_phase{k}.py") for k in (1, 2, 3, 4, 5, 6, 7)]
P2_BACKBONES = ["A", "Amuon", "As", "B", "C", "D", "G"]


def bundle(extra=()):
    """Zip of the code the notebooks need (no tests, caches or notebooks) -> (base64 text, sha1).
    extra: (source path, path inside the zip) pairs added as well."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d, recursive in BUNDLE_DIRS:
            base = os.path.join(ROOT, d)
            for cur, dirs, files in os.walk(base):
                dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", "tests", ".ipynb_checkpoints"))
                if not recursive:
                    dirs[:] = []
                for f in sorted(files):
                    if f.endswith((".py", ".json")):
                        p = os.path.join(cur, f)
                        z.write(p, os.path.relpath(p, ROOT))
        for f in EXTRA_FILES:
            z.write(os.path.join(ROOT, f), f)
        for src, arc in extra:
            z.write(src, arc)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha1(raw).hexdigest()[:12]


SETUP = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, time, zipfile
CODE = "/tmp/cs781/code"   # outside /kaggle/working: the output keeps only results
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(CODE)
print("code bundle", BUNDLE_SHA, "->", CODE)
os.chdir(CODE)
'''

PREP = r'''
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as e:
    raise SystemExit(f"attach the HF_TOKEN secret to this notebook (Add-ons -> Secrets): {type(e).__name__}: {e}")
subprocess.run("df -h /tmp /kaggle/working; nproc; free -g", shell=True)
subprocess.run([sys.executable, "-u", "-m", "phase1.data.prep", "--section", SECTION,
                "--raw-dir", "/tmp/cs781/raw", "--out", "/kaggle/working/p1data"], check=True)
with open("/kaggle/working/p1data/BUNDLE_SHA", "w") as f:
    f.write(BUNDLE_SHA)
subprocess.run("du -sh /kaggle/working/p1data/*", shell=True)
'''

TRAIN = r'''
metas = sorted(glob.glob("/kaggle/input/**/p1data/shards/meta.json", recursive=True))
assert metas, "attach the data-prep notebook's output (it contains p1data/) as an input"
DATA = os.path.dirname(os.path.dirname(metas[0]))
print("data:", DATA, json.load(open(os.path.join(DATA, "prep_manifest.json")))["code_hash"])
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working", shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (T4 x2)"
if len(JOBS) > n_gpu:
    raise SystemExit(f"{len(JOBS)} jobs but only {n_gpu} GPU(s)")
OUTD = "/kaggle/working/runs"
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for gpu, run_file in enumerate(JOBS):
    tag = os.path.splitext(os.path.basename(run_file))[0]
    logf = open(f"/kaggle/working/logs/{tag}.log", "a")
    cmd = [sys.executable, "-u", "-m", "phase1.train.launch", "--run", run_file, "--data", DATA,
           "--out", OUTD, "--inputs", "/kaggle/input", "--max-hours", str(MAX_HOURS)]
    if PILOT:
        cmd += ["--pilot", str(PILOT)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {run_file}")

def tail(tag, n=4):
    try:
        return "".join(open(f"/kaggle/working/logs/{tag}.log").readlines()[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(60 if PILOT else 600)
    for tag, _ in procs:
        print(f"--- {tag} ---\n{tail(tag)}", flush=True)
for tag, p in procs:
    print(f"==== {tag}: exit {p.returncode}\n{tail(tag, 40 if p.returncode else 8)}")
for f in sorted(glob.glob(f"{OUTD}/*/pilot.json")):
    print(open(f).read())
subprocess.run("du -sh /kaggle/working/runs/* 2>/dev/null", shell=True)
'''


GENERATE = r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bert-score", "rouge-score"], check=False)
metas = sorted(glob.glob("/kaggle/input/**/p1data/shards/meta.json", recursive=True))
assert metas, "attach the data-prep notebook's output (it contains p1data/) as an input"
DATA = os.path.dirname(os.path.dirname(metas[0]))
name = __import__("phase1.train.launch", fromlist=["load_run"]).load_run(RUN).NAME
cands = glob.glob(f"/kaggle/input/**/runs/{name}/ckpt/{CKPT}", recursive=True)
assert cands, f"no checkpoint runs/{name}/ckpt/{CKPT} in the attached inputs"
step = lambda p: int(__import__("re").search(r"_(\d+)\.pt$", p).group(1))
ckpt = max(cands, key=step)
print("checkpoint:", ckpt)
with open("/tmp/cs781/decode.json", "w") as f:
    json.dump(DECODE, f)
for split in SPLITS:
    out = f"/kaggle/working/{TAG}_{split}"
    subprocess.run([sys.executable, "-u", "-m", "phase1.eval.predict", "--run", RUN, "--ckpt", ckpt, "--data", DATA,
                    "--decode", "/tmp/cs781/decode.json", "--split", split, "--out", out], check=True)
print(open(f"/kaggle/working/{TAG}_dev/predict_record.json").read() if "dev" in SPLITS else "")
'''


SESSION = r'''
# ---------------------------------------------------------------- 1. token, packages, machine
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
pip = subprocess.Popen([sys.executable, "-m", "pip", "install", "-q", "bert-score", "rouge-score"])
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h /tmp /kaggle/working",
               shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (Settings -> Accelerator -> GPU T4 x2)"
if len(JOBS) > n_gpu:
    print(f"WARNING: {len(JOBS)} jobs but {n_gpu} GPU(s); running only {JOBS[:n_gpu]} (choose GPU T4 x2)")
    JOBS = JOBS[:n_gpu]

# ---------------------------------------------------------------- 2. data (reuse an attached prep output if any)
metas = sorted(glob.glob("/kaggle/input/**/p1data/shards/meta.json", recursive=True))
if metas:
    DATA = os.path.dirname(os.path.dirname(metas[0]))
    print("using attached data:", DATA)
else:
    assert os.environ["HF_TOKEN"], "set HF_TOKEN at the top of the notebook (or attach a secret named HF_TOKEN)"
    DATA = "/tmp/cs781/p1data"
    # reuse the round-1 tokenizer from an attached session output (p1data_small/tokenizer) if there is one
    toks = sorted(glob.glob("/kaggle/input/**/p1data_small/tokenizer/a1bpe_*.json", recursive=True))
    toks = [t for t in toks if not t.endswith(".report.json")]
    tok_args = ["--tokenizer-from", os.path.dirname(toks[0])] if toks else []
    print("tokenizer:", toks[0] if toks else "fit in this session")
    if not os.path.exists(os.path.join(DATA, "prep_manifest.json")):
        subprocess.run([sys.executable, "-u", "-m", "phase1.data.prep", "--section", "full",
                        "--raw-dir", "/tmp/cs781/raw", "--out", DATA, *tok_args], check=True)
    shutil.rmtree("/tmp/cs781/raw", ignore_errors=True)
keep = "/kaggle/working/p1data_small"  # tokenizer, dev set, manifests: small, reused in Phases 2-7
for sub in ("tokenizer", "dev"):
    if os.path.isdir(os.path.join(DATA, sub)):
        shutil.copytree(os.path.join(DATA, sub), os.path.join(keep, sub), dirs_exist_ok=True)
for f in ("prep_manifest.json", "stats.json", "shards/meta.json"):
    if os.path.exists(os.path.join(DATA, f)):
        os.makedirs(os.path.dirname(os.path.join(keep, f)), exist_ok=True)
        shutil.copy(os.path.join(DATA, f), os.path.join(keep, f))
tok_file = [p for p in glob.glob(os.path.join(DATA, "tokenizer", "a1bpe_*.json")) if not p.endswith(".report.json")]
TOK_SHA = __import__("hashlib").sha1(open(tok_file[0], "rb").read()).hexdigest()[:12]
print("tokenizer sha1:", TOK_SHA)
if EXPECTED_TOKENIZER_SHA and TOK_SHA != EXPECTED_TOKENIZER_SHA:
    raise SystemExit(f"tokenizer {TOK_SHA} != expected {EXPECTED_TOKENIZER_SHA}: models would not share a "
                     "vocabulary. Attach a round-1 session output (it holds p1data_small/tokenizer).")
print(f"prep finished at {(time.time() - T0) / 3600:.2f} h")
pip.wait()

# ---------------------------------------------------------------- 3. one job per GPU: train -> test zip -> dev scores
train_until = T0 + (SESSION_HOURS - GEN_RESERVE_HOURS) * 3600
hard_stop = T0 + SESSION_HOURS * 3600
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for gpu, run_file in enumerate(JOBS):
    tag = os.path.splitext(os.path.basename(run_file))[0]
    logf = open(f"/kaggle/working/logs/{tag}.log", "a")
    cmd = [sys.executable, "-u", "-m", "phase1.job", "--run", run_file, "--data", DATA,
           "--out", "/kaggle/working/runs", "--train-until", str(train_until), "--hard-stop", str(hard_stop),
           "--decodes", *DECODES, "--preds", "/kaggle/working/preds", "--subs", "/kaggle/working/submissions"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {run_file}")
print(f"training until {(train_until - T0) / 3600:.2f} h, everything stops by {SESSION_HOURS} h")

def tail(tag, n=4):
    try:
        return "".join(open(f"/kaggle/working/logs/{tag}.log", errors="replace").readlines()[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(600)
    print(f"===== {(time.time() - T0) / 3600:.2f} h", flush=True)
    for tag, _ in procs:
        print(f"--- {tag} ---\n{tail(tag)}", flush=True)
for tag, p in procs:
    print(f"==== {tag}: exit {p.returncode}\n{tail(tag, 40 if p.returncode else 12)}")
subprocess.run("ls -la /kaggle/working/submissions; du -sh /kaggle/working/*", shell=True)
for f in sorted(glob.glob("/kaggle/working/submissions/*__summary.json")):
    print(open(f).read())
'''


SWEEP = r'''
# ---------------------------------------------------------------- 1. token, packages
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bert-score", "rouge-score"], check=False)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (GPU T4 x2)"

# ---------------------------------------------------------------- 2. minimal data dir: tokenizer + dev set (attached) + test prompts (HF)
smalls = sorted(glob.glob("/kaggle/input/**/p1data_small/prep_manifest.json", recursive=True))
assert smalls, "attach a training session's output (it holds p1data_small/ with the tokenizer and dev set)"
DATA = "/tmp/cs781/p1data"
shutil.copytree(os.path.dirname(smalls[0]), DATA, dirs_exist_ok=True)
from huggingface_hub import hf_hub_download
from phase1.data.common import HF_DATASET_REPO
src = hf_hub_download(HF_DATASET_REPO, "Phase_1_Pretraining/Test/test.json", repo_type="dataset",
                      local_dir="/tmp/cs781/raw")
shutil.copy(src, os.path.join(DATA, "test.json"))
tok = [p for p in glob.glob(os.path.join(DATA, "tokenizer", "a1bpe_*.json")) if not p.endswith(".report.json")][0]
print("tokenizer sha1:", __import__("hashlib").sha1(open(tok, "rb").read()).hexdigest()[:12])

# ---------------------------------------------------------------- 3. one sweep per GPU
hard_stop = T0 + SESSION_HOURS * 3600
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for gpu, job in enumerate(JOBS[:n_gpu]):
    cks = glob.glob(f"/kaggle/input/**/runs/{job['model']}/ckpt/step_*.pt", recursive=True)
    if not cks:
        print(f"no checkpoint runs/{job['model']}/ckpt/step_*.pt in the inputs; skipping {job}")
        continue
    ck = max(cks, key=lambda p: int(__import__("re").search(r"step_(\d+)\.pt$", p).group(1)))
    tag = job["model"]
    cmd = [sys.executable, "-u", "-m", "phase1.sweep", "--run", job["run"], "--ckpt", ck, "--data", DATA,
           "--decodes", *DECODES, "--top-k", str(TOP_K), "--hard-stop", str(hard_stop),
           "--preds", "/kaggle/working/preds", "--subs", "/kaggle/working/submissions"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=open(f"/kaggle/working/logs/sweep_{tag}.log", "a"),
                                        stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {job['run']} with {ck}")

def tail(tag, n=4):
    try:
        return "".join(open(f"/kaggle/working/logs/sweep_{tag}.log", errors="replace").readlines()[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(300)
    print(f"===== {(time.time() - T0) / 3600:.2f} h", flush=True)
    for tag, _ in procs:
        print(f"--- {tag} ---\n{tail(tag)}", flush=True)
for tag, p in procs:
    print(f"==== {tag}: exit {p.returncode}\n{tail(tag, 30 if p.returncode else 6)}")
for f in sorted(glob.glob("/kaggle/working/submissions/*__sweep.json")):
    print(open(f).read())
'''


AUDIT = r'''
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as e:
    raise SystemExit(f"attach the HF_TOKEN secret to this notebook (Add-ons -> Secrets): {type(e).__name__}: {e}")
subprocess.run("nproc; free -g; df -h /tmp /kaggle/working", shell=True)
OUT = "/kaggle/working/data_audit"
cmd = [sys.executable, "-u", "-m", "phase1.data.audit", "--raw-dir", "/tmp/cs781/raw", "--ref", "round1",
       "--out", OUT, "--download", "--delete-raw", "--workers", str(os.cpu_count())]
if MAX_DOCS:
    cmd += ["--max-docs", str(MAX_DOCS)]
subprocess.run(cmd, check=True)
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{OUT}/audit.md", encoding="utf-8").read().split("## Figures")[0]))
for f in sorted(glob.glob(f"{OUT}/figures/*.png")):
    display(Image(f))
'''


P2 = r"""
sys.path.insert(0, CODE)  # in-process imports of the bundle (Jupyter has the cwd on sys.path; be explicit)

# ---------------------------------------------------------------- 1. token, machine
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h /kaggle/working", shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (Settings -> Accelerator -> GPU T4 x2)"

# ---------------------------------------------------------------- 2. Phase-2 data (HF) + backbones (attached dataset / outputs)
from phase2.data import download
assert os.environ["HF_TOKEN"], "set HF_TOKEN at the top (or attach a Kaggle secret named HF_TOKEN)"
DATA = download("/tmp/cs781/p2raw", token=os.environ["HF_TOKEN"])
print("data:", DATA, os.listdir(os.path.join(DATA, "Train")))
ROOTS = ["/kaggle/input"]
from ft.backbones import REGISTRY, find_checkpoint
from phase2.train import find_tokenizer, sha12
tok = find_tokenizer(ROOTS)
print("tokenizer:", tok, sha12(tok))
for k, (_r, name) in REGISTRY.items():
    print(f"  backbone {k:10s} {name:22s} {find_checkpoint(name, ROOTS) or '-'}")
OUTD = "/kaggle/working/runs_p2"
os.makedirs(OUTD, exist_ok=True)
subprocess.run([sys.executable, "-u", "-m", "phase2.report", "--data", DATA, "--roots", *ROOTS,
                "--out", f"{OUTD}/data_report.json"], check=True)

# ---------------------------------------------------------------- 3. one experiment queue per GPU
queues = [list(q) for q in QUEUES]
while len(queues) > n_gpu:  # fewer GPUs than queues: interleave the extra queue into the others
    extra = queues.pop()
    for i, item in enumerate(extra):
        queues[i % n_gpu].append(item)
deadline = T0 + SESSION_HOURS * 3600
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for gpu, q in enumerate(queues):
    cmd = [sys.executable, "-u", "-m", "phase2.session", "--queue", *q, "--data", DATA, "--roots", *ROOTS,
           "--out", OUTD, "--deadline", str(deadline), "--tag", f"gpu{gpu}"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=open(f"/kaggle/working/logs/gpu{gpu}.log", "a"),
                                        stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {q}")

def tail(gpu, n=3):
    try:
        lines = open(f"/kaggle/working/logs/gpu{gpu}.log", errors="replace").readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(300)
    print(f"===== {(time.time() - T0) / 3600:.2f} h", flush=True)
    for gpu, _ in procs:
        print(f"--- gpu{gpu} ---\n{tail(gpu)}", flush=True)
for gpu, p in procs:
    print(f"==== gpu{gpu}: exit {p.returncode}")
    subprocess.run(f"grep -h SESSION /kaggle/working/logs/gpu{gpu}.log", shell=True)

# ---------------------------------------------------------------- 4. summary + submissions in one folder
subprocess.run([sys.executable, "-m", "phase2.session", "--summary", "--out", OUTD])
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for z in sorted(glob.glob(f"{OUTD}/*/submission.zip")):
    shutil.copy(z, f"/kaggle/working/submissions/{os.path.basename(os.path.dirname(z))}.zip")
shutil.rmtree(f"{OUTD}/_cache", ignore_errors=True)
subprocess.run("ls -la /kaggle/working/submissions; du -sh /kaggle/working/*", shell=True)
"""


P2ENS = r"""
sys.path.insert(0, CODE)
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
from phase2.data import download
DATA = download("/tmp/cs781/p2raw", token=os.environ["HF_TOKEN"])
OUTD = "/kaggle/working/ens"
os.makedirs(OUTD, exist_ok=True)
need = sorted({m for e in ENSEMBLES for m in e["runs"]})
for m in need:  # gather member folders (config + saved scores) from the attached grid outputs
    hits = [h for h in glob.glob(f"/kaggle/input/**/runs_p2/{m}/config.json", recursive=True)]
    if not hits:
        print("MISSING member", m)
        continue
    src = os.path.dirname(hits[0])
    dst = os.path.join(OUTD, m)
    os.makedirs(dst, exist_ok=True)
    for f in ("config.json", "scores_dev.npz", "scores_test.npz"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print("member", m, "<-", src)
for e in ENSEMBLES:
    cmd = [sys.executable, "-u", "-m", "phase2.ensemble", "--runs", *e["runs"], "--data", DATA,
           "--roots", "/kaggle/input", "--out", OUTD, "--name", e["name"]]
    if e.get("weights"):
        cmd += ["--weights", *map(str, e["weights"])]
    subprocess.run(cmd)
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for e in ENSEMBLES:
    z = os.path.join(OUTD, e["name"], "submission.zip")
    if os.path.exists(z):
        shutil.copy(z, f"/kaggle/working/submissions/{e['name']}.zip")
    j = os.path.join(OUTD, e["name"], "ensemble.json")
    if os.path.exists(j):
        r = json.load(open(j))
        f1 = r.get("dev", {}).get("macro_f1")
        print(f"{e['name']:28s} dev strict macro-F1 {f1 if f1 is None else round(f1, 4)}  test entities {r.get('test_entities')}")
for m in need:
    shutil.rmtree(os.path.join(OUTD, m), ignore_errors=True)
subprocess.run("ls -la /kaggle/working/submissions", shell=True)
"""


P3 = r"""
sys.path.insert(0, CODE)  # in-process imports of the bundle

# ---------------------------------------------------------------- 1. token, machine
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h /kaggle/working", shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (Settings -> Accelerator -> GPU T4 x2)"

# ---------------------------------------------------------------- 2. Phase-3 data (HF) + backbones (attached dataset)
from phase3.data import download, describe
assert os.environ["HF_TOKEN"], "set HF_TOKEN at the top (or attach a Kaggle secret named HF_TOKEN)"
DATA = download("/tmp/cs781/p3raw", token=os.environ["HF_TOKEN"])
print("data:", DATA)
print(describe(DATA))
ROOTS = ["/kaggle/input"]
from ft.backbones import REGISTRY, find_checkpoint
from phase3.train import find_tokenizer, sha12
tok = find_tokenizer(ROOTS)
print("tokenizer:", tok, sha12(tok))
for k, (_r, name) in REGISTRY.items():
    print(f"  backbone {k:10s} {name:22s} {find_checkpoint(name, ROOTS) or '-'}")
OUTD = "/kaggle/working/runs_p3"
os.makedirs(OUTD, exist_ok=True)
subprocess.run([sys.executable, "-u", "-m", "phase3.report", "--data", DATA, "--roots", *ROOTS,
                "--out", f"{OUTD}/data_report.json"], check=True)

# ---------------------------------------------------------------- 3. one experiment queue per GPU
queues = [list(q) for q in QUEUES]
while len(queues) > n_gpu:  # fewer GPUs than queues: interleave the extra queue into the others
    extra = queues.pop()
    for i, item in enumerate(extra):
        queues[i % n_gpu].append(item)
deadline = T0 + SESSION_HOURS * 3600
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for gpu, q in enumerate(queues):
    cmd = [sys.executable, "-u", "-m", "phase3.session", "--queue", *q, "--data", DATA, "--roots", *ROOTS,
           "--out", OUTD, "--deadline", str(deadline), "--tag", f"gpu{gpu}"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=open(f"/kaggle/working/logs/gpu{gpu}.log", "a"),
                                        stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {q}")

def tail(gpu, n=3):
    try:
        return "".join(open(f"/kaggle/working/logs/gpu{gpu}.log", errors="replace").readlines()[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(120)
    print(f"===== {(time.time() - T0) / 3600:.2f} h", flush=True)
    for gpu, _ in procs:
        print(f"--- gpu{gpu} ---\n{tail(gpu)}", flush=True)
for gpu, p in procs:
    print(f"==== gpu{gpu}: exit {p.returncode}")
    subprocess.run(f"grep -h SESSION /kaggle/working/logs/gpu{gpu}.log", shell=True)

# ---------------------------------------------------------------- 4. ensembles (CPU, seconds), summary, submissions in one folder
for e in ENSEMBLES:
    if all(os.path.exists(f"{OUTD}/{m}/scores_test.npz") for m in e["runs"]):
        cmd = [sys.executable, "-m", "phase3.ensemble", "--runs", *e["runs"], "--data", DATA, "--roots", *ROOTS,
               "--out", OUTD, "--name", e["name"], "--bias"]
        subprocess.run(cmd + (["--weights", *map(str, e["weights"])] if e.get("weights") else []))
    else:
        print("ensemble", e["name"], "skipped: a member has no scores")
subprocess.run([sys.executable, "-m", "phase3.session", "--summary", "--out", OUTD])
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for z in sorted(glob.glob(f"{OUTD}/*/submission*.zip")):
    run_name = os.path.basename(os.path.dirname(z))
    suffix = "__bias" if os.path.basename(z) == "submission_bias.zip" else ""
    shutil.copy(z, f"/kaggle/working/submissions/{run_name}{suffix}.zip")
shutil.rmtree(f"{OUTD}/_cache", ignore_errors=True)
subprocess.run("ls -la /kaggle/working/submissions; du -sh /kaggle/working/*", shell=True)
"""


P4 = r"""
sys.path.insert(0, CODE)  # in-process imports of the bundle

# ---------------------------------------------------------------- 1. token, machine
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h /kaggle/working", shell=True)
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "turn on the GPU accelerator (Settings -> Accelerator -> GPU T4 x2)"

# ---------------------------------------------------------------- 2. Phase-4 data (HF) + backbones (attached dataset)
from phase4.data import download
assert os.environ["HF_TOKEN"], "set HF_TOKEN at the top (or attach a Kaggle secret named HF_TOKEN)"
DATA = download("/tmp/cs781/p4raw", token=os.environ["HF_TOKEN"])
print("data:", DATA)
subprocess.run(f"ls -la {DATA}/Train {DATA}/Test", shell=True)
ROOTS = ["/kaggle/input"]
from ft.backbones import REGISTRY, find_checkpoint
from phase3.train import find_tokenizer, sha12
tok = find_tokenizer(ROOTS)
print("tokenizer:", tok, sha12(tok))
for k, (_r, name) in REGISTRY.items():
    print(f"  backbone {k:10s} {name:22s} {find_checkpoint(name, ROOTS) or '-'}")
OUTD = "/kaggle/working/runs_p4"
os.makedirs(OUTD, exist_ok=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
# data report (format, counts, lengths, duplicates, baselines, evaluator parity); it also builds the token cache every run uses
subprocess.run([sys.executable, "-u", "-m", "phase4.report", "--data", DATA, "--roots", *ROOTS,
                "--out", f"{OUTD}/data_report.json"], check=True)
bl = None
if BASELINES:  # TF-IDF + LR per document view, on the CPU next to the GPU queues
    bl = subprocess.Popen(["nice", "-n", "15", sys.executable, "-u", "-m", "phase4.report", "--data", DATA, "--roots",
                           *ROOTS, "--out", f"{OUTD}/tfidf_baselines.json", "--baselines-only"],
                          stdout=open("/kaggle/working/logs/baselines.log", "a"), stderr=subprocess.STDOUT)

# ---------------------------------------------------------------- 3. one experiment queue per GPU
queues = [list(q) for q in QUEUES]
while len(queues) > n_gpu:  # fewer GPUs than queues: interleave the extra queue into the others
    extra = queues.pop()
    for i, item in enumerate(extra):
        queues[i % n_gpu].append(item)
deadline = T0 + SESSION_HOURS * 3600
procs = []
for gpu, q in enumerate(queues):
    cmd = [sys.executable, "-u", "-m", "phase4.session", "--queue", *q, "--data", DATA, "--roots", *ROOTS,
           "--out", OUTD, "--deadline", str(deadline), "--tag", f"gpu{gpu}"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=open(f"/kaggle/working/logs/gpu{gpu}.log", "a"),
                                        stderr=subprocess.STDOUT)))
    print(f"gpu{gpu}: {q}")

def tail(gpu, n=3):
    try:
        return "".join(open(f"/kaggle/working/logs/gpu{gpu}.log", errors="replace").readlines()[-n:])
    except OSError:
        return ""
while any(p.poll() is None for _, p in procs):
    time.sleep(120)
    print(f"===== {(time.time() - T0) / 3600:.2f} h", flush=True)
    for gpu, _ in procs:
        print(f"--- gpu{gpu} ---\n{tail(gpu)}", flush=True)
for gpu, p in procs:
    print(f"==== gpu{gpu}: exit {p.returncode}")
    subprocess.run(f"grep -h SESSION /kaggle/working/logs/gpu{gpu}.log", shell=True)
if bl is not None:
    try:
        bl.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        bl.kill()
    subprocess.run("tail -n 8 /kaggle/working/logs/baselines.log", shell=True)

# ---------------------------------------------------------------- 4. ensembles (CPU, seconds), summary, submissions in one folder
for e in ENSEMBLES:
    if all(os.path.exists(f"{OUTD}/{m}/scores_test.npz") for m in e["runs"]):
        cmd = [sys.executable, "-m", "phase4.ensemble", "--runs", *e["runs"], "--out", OUTD, "--name", e["name"]]
        subprocess.run(cmd + (["--weights", *map(str, e["weights"])] if e.get("weights") else []))
    else:
        print("ensemble", e["name"], "skipped: a member has no scores")
subprocess.run([sys.executable, "-m", "phase4.session", "--summary", "--out", OUTD])
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for z in sorted(glob.glob(f"{OUTD}/*/submission*.zip")):
    run_name = os.path.basename(os.path.dirname(z))
    tag = os.path.basename(z)[len("submission"):-len(".zip")].lstrip("_")  # "", "thr", "pl", "g1"
    shutil.copy(z, f"/kaggle/working/submissions/{run_name}{'__' + tag if tag else ''}.zip")
shutil.rmtree(f"{OUTD}/_cache", ignore_errors=True)
if not KEEP_CHUNKS:
    for f in glob.glob(f"{OUTD}/*/chunks_*.npz"):
        os.remove(f)
subprocess.run("ls -la /kaggle/working/submissions; du -sh /kaggle/working/runs_p4/* | sort -h | tail -25", shell=True)
"""


P5 = (P4.replace("phase4.", "phase5.").replace("from phase4.data", "from phase5.data").replace("runs_p4", "runs_p5")
      .replace("/tmp/cs781/p4raw", "/tmp/cs781/p5raw").replace("Phase-4 data", "Phase-5 data"))


REPORT_FATAL = '''"--out", f"{OUTD}/data_report.json"], check=True)'''
REPORT_SOFT = '''"--out", f"{OUTD}/data_report.json"], check=False)'''

P6 = (P4.replace("phase4.", "phase6.").replace("from phase4.data", "from phase6.data").replace("runs_p4", "runs_p6")
      .replace("/tmp/cs781/p4raw", "/tmp/cs781/p6raw").replace("Phase-4 data", "Phase-6 data")
      .replace("sys.path.insert(0, CODE)  # in-process imports of the bundle",
               "sys.path.insert(0, CODE)  # in-process imports of the bundle\n"
               "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rouge-score', 'bert-score'], check=False)")
      .replace(REPORT_FATAL, REPORT_SOFT))


P7 = (P4.replace("phase4.", "phase7.").replace("from phase4.data", "from phase7.data").replace("runs_p4", "runs_p7")
      .replace("/tmp/cs781/p4raw", "/tmp/cs781/p7raw").replace("Phase-4 data", "Phase-7 data")
      .replace('''"--out", OUTD, "--name", e["name"]]''', '''"--out", OUTD, "--name", e["name"], "--data", DATA, "--roots", *ROOTS]''')
      .replace(REPORT_FATAL, REPORT_SOFT))


P3ENS = r"""
sys.path.insert(0, CODE)
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__})")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
from phase3.data import download
DATA = download("/tmp/cs781/p3raw", token=os.environ["HF_TOKEN"])
OUTD = "/kaggle/working/ens"
os.makedirs(OUTD, exist_ok=True)
need = sorted({m for e in ENSEMBLES for m in e["runs"]})
for m in need:  # member folders (config + saved scores) from the attached session outputs
    hits = glob.glob(f"/kaggle/input/**/runs_p3/{m}/config.json", recursive=True)
    if not hits:
        print("MISSING member", m)
        continue
    src = os.path.dirname(hits[0])
    dst = os.path.join(OUTD, m)
    os.makedirs(dst, exist_ok=True)
    for f in ("config.json", "scores_dev.npz", "scores_test.npz"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print("member", m, "<-", src)
for e in ENSEMBLES:
    cmd = [sys.executable, "-u", "-m", "phase3.ensemble", "--runs", *e["runs"], "--data", DATA,
           "--roots", "/kaggle/input", "--out", OUTD, "--name", e["name"], "--bias"]
    subprocess.run(cmd + (["--weights", *map(str, e["weights"])] if e.get("weights") else []))
os.makedirs("/kaggle/working/submissions", exist_ok=True)
for e in ENSEMBLES:
    for f, suf in (("submission.zip", ""), ("submission_bias.zip", "__bias")):
        z = os.path.join(OUTD, e["name"], f)
        if os.path.exists(z):
            shutil.copy(z, f"/kaggle/working/submissions/{e['name']}{suf}.zip")
for m in need:
    shutil.rmtree(os.path.join(OUTD, m), ignore_errors=True)
shutil.rmtree(os.path.join(OUTD, "_cache"), ignore_errors=True)
subprocess.run("ls -la /kaggle/working/submissions", shell=True)
"""


def _cell(code):
    lines = code.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}


def notebook(title, header, body, b64, sha, separate_header=False):
    """separate_header: the editable settings get their own first cell, apart from the code bundle."""
    if separate_header:
        cells = [_cell(header), _cell(f'BUNDLE_SHA = "{sha}"\nBUNDLE = "{b64}"\n' + SETUP + body)]
    else:
        cells = [_cell(f'BUNDLE_SHA = "{sha}"\n{header}\nBUNDLE = "{b64}"\n' + SETUP + body)]
    return {"nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]}] + cells}


def write(slug, nb, meta):
    d = os.path.join(OUT, slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "notebook.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {os.path.relpath(d, ROOT)} ({os.path.getsize(os.path.join(d, 'notebook.ipynb')) / 1e3:.0f} kB)")
    return d


def metadata(user, slug, title, gpu, kernel_sources):
    m = {"id": f"{user}/{slug}", "title": title, "code_file": "notebook.ipynb", "language": "python",
         "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
         "enable_internet": True, "dataset_sources": [], "competition_sources": [],
         "kernel_sources": kernel_sources, "model_sources": []}
    if gpu:
        m["machine_shape"] = "NvidiaTeslaT4"
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["prep", "train", "generate", "session", "sweep", "audit", "p2", "backbones",
                                     "p2ens", "p3", "p3ens", "p4", "p5", "p6", "p7"])
    ap.add_argument("--ensembles", help="p2ens: JSON list of {name, runs, weights?}")
    ap.add_argument("--queue0", nargs="*", default=[], help="p2: experiment queue for GPU 0 (EXP[:MODE][@k=v,...])")
    ap.add_argument("--queue1", nargs="*", default=[], help="p2: experiment queue for GPU 1")
    ap.add_argument("--title", default="")
    ap.add_argument("--baselines", action="store_true", help="p4: also run the TF-IDF baselines (CPU) during the queues")
    ap.add_argument("--drop-chunks", action="store_true", help="p4: delete chunk vectors at the end (smaller output)")
    ap.add_argument("--ref-dir", default=os.path.join("results", "results", "p1data_small"),
                    help="audit: round-1 p1data_small (tokenizer/, dev/, shards/meta.json) embedded in the notebook")
    ap.add_argument("--max-docs", type=int, default=0, help="audit: first N documents per source only (smoke test)")
    ap.add_argument("--sweep-jobs", nargs="+", help="sweep: run_file=model_folder, one per GPU")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--jobs", nargs="+", help="session: run files, one per GPU")
    ap.add_argument("--decodes", nargs="*", default=["decode/greedy_nr4.json", "decode/nucleus_t07.json"])
    ap.add_argument("--session-hours", type=float, default=11.6)
    ap.add_argument("--gen-reserve-hours", type=float, default=0.85)
    ap.add_argument("--tokenizer-sha", default="", help="session: expected tokenizer sha1[:12] from round 1")
    ap.add_argument("--run", help="generate: run file of the model")
    ap.add_argument("--decode", help="generate: decode JSON (DecodeConfig values + max_prompt_tokens)")
    ap.add_argument("--from", dest="from_slugs", nargs="*", default=[], help="generate: training notebook slug(s)")
    ap.add_argument("--ckpt", default="step_*.pt", help="generate: checkpoint file name/glob (newest step wins)")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--user", default="swadha1509", help="Kaggle username that owns the notebooks")
    ap.add_argument("--section", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--slug")
    ap.add_argument("--gpu0")
    ap.add_argument("--gpu1")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--prep-slug", default=PREP_SLUG["full"], help="data-prep notebook whose output holds p1data/")
    ap.add_argument("--resume-from", nargs="*", default=[], help="previous training notebook slug(s)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if args.kind == "backbones":
        sys.path.insert(0, ROOT)
        from ft.backbones import export_backbones
        slug = args.slug or "cs781-p2-backbones"
        d = os.path.join(OUT, slug)
        roots = [os.path.join(ROOT, "results")]
        done = export_backbones(P2_BACKBONES, roots, os.path.join(d, "backbones"))
        tok = [t for t in glob.glob(os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_*.json"))
               if not t.endswith(".report.json")]
        os.makedirs(os.path.join(d, "tokenizer"), exist_ok=True)
        for t in tok:
            with open(t, "rb") as fi, open(os.path.join(d, "tokenizer", os.path.basename(t)), "wb") as fo:
                fo.write(fi.read())
        with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
            json.dump({"title": slug, "id": f"{args.user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, f, indent=1)
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(done, f, indent=1)
        total = sum(v["bytes"] for v in done.values())
        print(f"wrote {os.path.relpath(d, ROOT)}: {len(done)} backbones, {total / 1e9:.2f} GB")
        return
    extra = []
    if args.kind == "audit":
        ref = os.path.join(ROOT, args.ref_dir)
        files = [p for p in glob.glob(os.path.join(ref, "tokenizer", "a1bpe_*.json")) if not p.endswith(".report.json")]
        files += [os.path.join(ref, "dev", f"{n}.json") for n in ("prompts", "truth", "sources")]
        files += [os.path.join(ref, "shards", "meta.json")]
        for f in files:
            if not os.path.exists(f):
                ap.error(f"{f} not found")
        extra = [(f, os.path.join("round1", os.path.relpath(f, ref))) for f in files]
    b64, sha = bundle(extra)
    if args.kind == "p3ens":
        ens = json.loads(open(args.ensembles).read()) if os.path.exists(args.ensembles) else json.loads(args.ensembles)
        header = "\n".join([
            "import time; T0 = time.time()",
            'HF_TOKEN = ""  # or attach the HF_TOKEN secret',
            f"ENSEMBLES = {json.dumps(ens, indent=1)}  # members = run folders in the attached session outputs"])
        slug = args.slug or "cs781-p3-ensemble"
        nb = notebook("CS781 Phase 3 ensembles (CPU)", header, P3ENS, b64, sha, separate_header=True)
        d = write(slug, nb, metadata(args.user, slug, slug, False, []))
    elif args.kind in ("p4", "p5", "p6", "p7"):
        if not (args.slug and args.queue0):
            ap.error(f"{args.kind} needs --slug and --queue0 (and usually --queue1)")
        sys.path.insert(0, ROOT)
        if args.kind == "p4":
            from phase4.experiments import names as p4_names
            from phase4.session import parse_item as p4_parse
        elif args.kind == "p5":
            from phase5.experiments import names as p4_names
            from phase5.session import parse_item as p4_parse
        elif args.kind == "p6":
            from phase6.experiments import names as p4_names
            from phase6.session import parse_item as p4_parse
        else:
            from phase7.experiments import names as p4_names
            from phase7.session import parse_item as p4_parse
        for item in args.queue0 + args.queue1:
            exp, mode, _ = p4_parse(item)
            if exp not in p4_names() or mode not in ("train", "hier", "decode"):
                ap.error(f"unknown experiment or mode in {item!r}")
        ens = (json.loads(open(args.ensembles).read()) if args.ensembles and os.path.exists(args.ensembles)
               else json.loads(args.ensembles or "[]"))
        queues = [q for q in (args.queue0, args.queue1) if q]
        header = "\n".join([
            "import time; T0 = time.time()  # session clock starts here",
            'HF_TOKEN = ""  # Hugging Face READ token, or leave "" and attach a Kaggle secret named HF_TOKEN',
            f"QUEUES = {json.dumps(queues, indent=1)}  # one queue per GPU: EXP[:train|hier][@key=value,...]",
            f"ENSEMBLES = {json.dumps(ens)}  # built at the end from this session's runs (if all members finished)",
            f"SESSION_HOURS = {args.session_hours}  # queues stop starting experiments near this limit",
            f"BASELINES = {args.baselines}  # TF-IDF + LR baselines per document view (CPU, in the background)",
            f"KEEP_CHUNKS = {not args.drop_chunks}  # keep chunk vectors (phase4.hier inputs, ~250 MB per extracting run)"])
        titles = {"p4": "Phase 4 CJPE", "p5": "Phase 5 LSI", "p6": "Phase 6 summarisation", "p7": "Phase 7 PCR"}
        nb = notebook(args.title or f"CS781 {titles[args.kind]}: {args.slug}", header,
                      {"p4": P4, "p5": P5, "p6": P6, "p7": P7}[args.kind], b64, sha, separate_header=True)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, []))
    elif args.kind == "p3":
        if not (args.slug and args.queue0):
            ap.error("p3 needs --slug and --queue0 (and usually --queue1)")
        sys.path.insert(0, ROOT)
        from phase3.experiments import names as p3_names
        from phase3.session import parse_item as p3_parse
        for item in args.queue0 + args.queue1:
            if p3_parse(item)[0] not in p3_names():
                ap.error(f"unknown experiment in {item!r}")
        ens = (json.loads(open(args.ensembles).read()) if args.ensembles and os.path.exists(args.ensembles)
               else json.loads(args.ensembles or "[]"))
        queues = [q for q in (args.queue0, args.queue1) if q]
        header = "\n".join([
            "import time; T0 = time.time()  # session clock starts here",
            'HF_TOKEN = ""  # Hugging Face READ token, or leave "" and attach a Kaggle secret named HF_TOKEN',
            f"QUEUES = {json.dumps(queues, indent=1)}  # one queue per GPU: EXP[:MODE][@key=value,...]",
            f"ENSEMBLES = {json.dumps(ens)}  # built at the end from this session's runs (if all members finished)",
            f"SESSION_HOURS = {args.session_hours}  # queues stop starting experiments near this limit"])
        nb = notebook(args.title or f"CS781 Phase 3 RR: {args.slug}", header, P3, b64, sha, separate_header=True)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, []))
    elif args.kind == "p2ens":
        ens = json.loads(open(args.ensembles).read()) if os.path.exists(args.ensembles) else json.loads(args.ensembles)
        header = "\n".join([
            "import time; T0 = time.time()",
            'HF_TOKEN = ""  # or attach the HF_TOKEN secret',
            f"ENSEMBLES = {json.dumps(ens, indent=1)}  # members = run folders in the attached grid outputs"])
        slug = args.slug or "cs781-p2-ensemble"
        nb = notebook("CS781 Phase 2 ensembles (CPU)", header, P2ENS, b64, sha, separate_header=True)
        d = write(slug, nb, metadata(args.user, slug, slug, False, []))
    elif args.kind == "p2":
        if not (args.slug and args.queue0):
            ap.error("p2 needs --slug and --queue0 (and usually --queue1)")
        sys.path.insert(0, ROOT)
        from phase2.experiments import names
        from phase2.session import parse_item
        for item in args.queue0 + args.queue1:
            if parse_item(item)[0] not in names():
                ap.error(f"unknown experiment in {item!r}")
        queues = [q for q in (args.queue0, args.queue1) if q]
        header = "\n".join([
            "import time; T0 = time.time()  # session clock starts here",
            'HF_TOKEN = ""  # Hugging Face READ token, or leave "" and attach a Kaggle secret named HF_TOKEN',
            f"QUEUES = {json.dumps(queues, indent=1)}  # one queue per GPU: EXP[:MODE][@key=value,...]",
            f"SESSION_HOURS = {args.session_hours}  # queues stop starting experiments near this limit"])
        nb = notebook(args.title or f"CS781 Phase 2 L-NER: {args.slug}", header, P2, b64, sha, separate_header=True)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, []))
    elif args.kind == "audit":
        slug = args.slug or "cs781-p1-data-audit"
        nb = notebook("CS781 Phase 1 corpus audit", f"MAX_DOCS = {args.max_docs}  # >0: quick test on the first N documents per source",
                      AUDIT, b64, sha, separate_header=True)
        d = write(slug, nb, metadata(args.user, slug, slug, False, []))
    elif args.kind == "sweep":
        if not (args.slug and args.sweep_jobs):
            ap.error("sweep needs --slug and --sweep-jobs run_file=model_folder ...")
        jobs = []
        for j in args.sweep_jobs:
            run_file, model = j.split("=")
            if not os.path.exists(os.path.join(ROOT, run_file)):
                ap.error(f"{run_file} not found")
            jobs.append({"run": run_file.replace(os.sep, "/"), "model": model})
        for d in args.decodes:
            if not os.path.exists(os.path.join(ROOT, d)):
                ap.error(f"{d} not found")
        header = "\n".join([
            "import time; T0 = time.time()",
            'HF_TOKEN = ""  # Hugging Face READ token (only the test prompts are downloaded), or a secret named HF_TOKEN',
            f"JOBS = {json.dumps(jobs)}  # one per GPU: run file + model folder (runs/<model>/ckpt) in the inputs",
            f"DECODES = {json.dumps([d.replace(os.sep, '/') for d in args.decodes])}  # dev order: cheap first",
            f"TOP_K = {args.top_k}  # test submissions for the best TOP_K decodes by dev BERTScore",
            f"SESSION_HOURS = {args.session_hours}  # everything stops by then"])
        nb = notebook(f"CS781 Phase 1 decoding sweep: {', '.join(j['model'] for j in jobs)}", header, SWEEP, b64, sha,
                      separate_header=True)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, []))
    elif args.kind == "session":
        if not (args.slug and args.jobs):
            ap.error("session needs --slug and --jobs")
        jobs = [j.replace(os.sep, "/") for j in args.jobs]
        for j in jobs + args.decodes:
            if not os.path.exists(os.path.join(ROOT, j)):
                ap.error(f"{j} not found (paths are relative to the project root)")
        header = "\n".join([
            "import time; T0 = time.time()  # session clock starts here",
            'HF_TOKEN = ""  # paste a Hugging Face READ token here, or leave "" and attach a Kaggle secret named HF_TOKEN',
            f"JOBS = {json.dumps(jobs)}  # one run file per GPU (all runs/*.py are in the bundle)",
            f"DECODES = {json.dumps([d.replace(os.sep, '/') for d in args.decodes])}",
            f"SESSION_HOURS = {args.session_hours}  # everything ends by then (Kaggle kills the session at 12 h)",
            f"GEN_RESERVE_HOURS = {args.gen_reserve_hours}  # kept after training for test zips + dev scores",
            f"EXPECTED_TOKENIZER_SHA = {json.dumps(args.tokenizer_sha)}  # round-1 tokenizer; stop if this session's differs"])
        nb = notebook(f"CS781 Phase 1 session: {', '.join(jobs)}", header, SESSION, b64, sha, separate_header=True)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, []))
    elif args.kind == "prep":
        slug = args.slug or PREP_SLUG[args.section]
        title = slug
        nb = notebook(f"CS781 Phase 1 data prep ({args.section})", f'SECTION = "{args.section}"', PREP, b64, sha)
        d = write(slug, nb, metadata(args.user, slug, title, False, []))
    elif args.kind == "generate":
        if not (args.slug and args.run and args.decode and args.from_slugs):
            ap.error("generate needs --slug, --run, --decode and --from")
        with open(args.decode, encoding="utf-8") as f:
            decode = json.load(f)
        tag = os.path.splitext(os.path.basename(args.decode))[0]
        header = "\n".join([f"RUN = {json.dumps(args.run.replace(os.sep, '/'))}", f"CKPT = {json.dumps(args.ckpt)}",
                            f"DECODE = {decode!r}", f"SPLITS = {json.dumps(args.splits)}",
                            f"TAG = {json.dumps(tag)}"])
        sources = [f"{args.user}/{args.prep_slug}"] + [s if "/" in s else f"{args.user}/{s}" for s in args.from_slugs]
        nb = notebook(f"CS781 Phase 1 generation: {args.run} with {args.decode}", header, GENERATE, b64, sha)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, sources))
    else:
        if not args.slug or not args.gpu0:
            ap.error("train needs --slug and --gpu0")
        jobs = [j.replace(os.sep, "/") for j in (args.gpu0, args.gpu1) if j]
        for j in jobs:
            if not os.path.exists(os.path.join(ROOT, j)):
                ap.error(f"{j} not found (paths are relative to the project root)")
        header = f"JOBS = {json.dumps(jobs)}\nPILOT = {args.pilot}\nMAX_HOURS = {args.max_hours}"
        sources = [f"{args.user}/{args.prep_slug}"] + [s if "/" in s else f"{args.user}/{s}" for s in args.resume_from]
        nb = notebook(f"CS781 Phase 1 training: {', '.join(jobs)}" + (f" (pilot {args.pilot})" if args.pilot else ""),
                      header, TRAIN, b64, sha)
        d = write(args.slug, nb, metadata(args.user, args.slug, args.slug, True, sources))
    print(f"bundle {sha}")
    if args.push:
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", d], check=True)


if __name__ == "__main__":
    main()
