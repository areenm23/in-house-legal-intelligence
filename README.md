# CS781 Capstone — In-House Legal Intelligence System

Areen Mahich (230188) · Codabench user `a_230188` · CS781, IIT Kanpur, 2026

Seven-phase Indian-legal NLP system built on our own language models. The models were trained **from scratch** on the
course corpus (Phase 1) and then adapted to six downstream tasks (Phases 2–7). Everything here uses only the official
competition data and our own checkpoints. No pretrained weights and no external APIs are used for training, inference
or evaluation.

| Phase | Task | Metric | Package | Best test (rank) |
|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt → 128-word continuation) | BERTScore F1 | `phase1/` | 0.8355 (31/150) |
| 2 | Legal NER (character spans, 12 labels) | strict macro-F1 | `phase2/` | 0.5449 (50) |
| 3 | Rhetorical roles (13 labels per sentence, CL + IT) | macro-F1 | `phase3/` | 0.5606 (22) |
| 4 | Court judgment prediction (binary) | macro-F1 | `phase4/` | 0.7578 (41) |
| 5 | Statute identification (multi-label, 100 statutes) | macro-F1 | `phase5/` | 0.3768 (63) |
| 6 | Abstractive summarisation | BERTScore F1 | `phase6/` | 0.8428 (54) |
| 7 | Prior case retrieval (top 7) | micro-F1@7 | `phase7/` | 0.4542 (30) |

Ranks are from the Codabench leaderboard on 9 November 2026; 89 valid submissions in total.

Model weights, checkpoints, data and Kaggle outputs are **not** included (see `.gitignore`).

---

## Repository layout

```
phase1/            data pipeline, tokenizer, models A–G, training runner, generation, evaluation
  data/            streaming JSON reader, A-1 ByteBPE tokenizer (a1_bpe.py), shards, dev-set builder, corpus audit
  models/          decoder.py (A), encdec.py (B), retnet.py (C), mamba2.py (D), moe.py (E), hybrid.py (F), mlm.py (G),
                   layers.py (RoPE, RMSNorm, SwiGLU, GQA attention), optim.py (AdamW, Muon, warmup-cosine)
  train/           runner.py (fp16, grad accumulation, atomic checkpoints, auto-resume), launch.py (time-boxed plans), data.py
  eval/            generate.py (KV / recurrent-state cached decoding, nucleus, n-gram blocking, MBR), predict.py, submission.py
  checks/          parameter counts, causal-leak tests, parallel == recurrent equivalence, throughput
  configs/data.json  tokenizer size and per-source token budgets
runs/              one run file per Phase-1 model (architecture size, optimizer, LR schedule, batch)
decode/            decoding configs (JSON)
ft/                shared fine-tuning library for Phases 2–7
  backbones.py     Phase-1 checkpoint → (B, T, d) feature extractor (bidirectional / echo modes, prompt slots)
  peft.py          full, partial, LoRA, rsLoRA, AdaLoRA, QLoRA (own NF4), LoRMA / LoRMA-π, IA3, BitFit,
                   Houlsby / Pfeiffer adapters, prompt, prefix, gradual unfreezing, layer-wise LR decay
  heads.py         linear / MLP / BiLSTM heads, CRF, BIO-constrained Viterbi
phase2/ … phase7/  one package per task: data, model/train, experiments (named configs), session (GPU queues),
                   ensemble, report (data report + official-evaluator parity), score (official metric), synth, tests
kaggle/build.py    builds self-contained Kaggle notebooks (code embedded as a zip, two GPU queues, session deadline)
kaggle/*_ensembles*.json  ensemble definitions used in the notebooks
```

Every task package follows the same pattern. One experiment is one run folder containing `config.json`, `log.jsonl`,
`result.json`, `scores_{dev,test}.npz` and `submission.zip`. `session.py` runs a queue per GPU and writes `summary.md`.
`report.py` writes the data report and checks that our metric matches the official evaluator.

## Setup and running

Python 3.10 (Kaggle) or 3.13 (local CPU tests). BERTScore downloads `roberta-large` on first use.

```bash
pip install -r requirements.txt
python -m pytest phase1/tests phase2/tests phase3/tests phase4/tests phase5/tests phase6/tests phase7/tests -q
```

The data is the gated HF dataset `Exploration-Lab/CS-781-Capstone`; the loaders download it with an `HF_TOKEN`.
The official course evaluation scripts are not redistributed here. Put them at
`course_files/eval/Evaluation Script/evaluate_phase{1..7}.py` (and the Phase-1 starter `submission.zip` at
`course_files/starter/Starter Code/`). `kaggle/build.py` embeds them in every notebook and fails without them, and the
evaluator-parity tests are skipped when they are missing.
All training ran on **Kaggle T4 x2** notebooks built with `kaggle/build.py`. Example commands:

```bash
# Phase 1: data prep (download → tokenizer → shards → dev set), then one model
python -m phase1.data.prep --config phase1/configs/data.json --section full
python -m phase1.job --run runs/run_A.py --data <p1data> --out <out> --decodes decode/greedy_nr4.json decode/nucleus_t07.json
# Phases 2–7: a queue of named experiments (see phaseK/experiments.py), e.g.
python -m phase4.session --queue A_lorma A_lorma:hier As_kd_A --data <phase4 data> --roots <backbone dir> --out runs_p4
python -m phase4.session --summary --out runs_p4
# local end-to-end check on synthetic data in the official layout
python -m phase2.synth --out data/p2synth
python -m phase2.session --queue "smoke_A@tiny=true" --data data/p2synth --roots <backbone dir> --out data/p2synth_runs
# Phase 7 CPU score fusion over finished runs
python -m phase7.fusion --data <phase7 data> --runs <run outputs> --roots <backbone dir> --out runs_p7/fusion
# Notebooks
python kaggle/build.py p4 --slug cs781-p4-gridA --session-hours 6.5 --queue0 ... --queue1 ...
```

Queue items take the form `EXP[:MODE][@key=value,...]`. For example, `A_lorma:hier` trains the two-stage hierarchical
aggregator on a finished run, and `A_lorma:decode@len_scale=0.7` re-generates summaries with new length settings.

---

## Phase 1 — Legal language model pretraining

**Data.** The Phase-1 corpus has four sources: Acts, High Courts, Supreme Court and Tribunals (19.2 GB raw).
It is split into train and validation at document level (0.5% held out). Token budgets per source are acts = all
(134M), High Courts 450M, Supreme Court 250M and tribunals 350M, about 1.17B training tokens in total. The dev set
holds 300 examples (75 per source), built from held-out documents as a 512-word prompt plus a 128-word continuation.

**Tokenizer.** Our A-1 byte-level BPE (Assignment 1), refit on the legal training split. It has 32,755 base ids plus
13 specials (`<mask>`, `<extra_0..11>`), 32,768 in total. It round-trips losslessly and has no `<unk>`. Fertility
ranges from 1.11 tokens/word (SC) to 1.50 (tribunals). All models in every phase share this one tokenizer.

**Architectures** (all use pre-norm RMSNorm, SwiGLU FFNs, tied embeddings and 1024-token windows):

| Key | Model | Params | Details (`runs/run_*.py`) |
|---|---|---|---|
| A | Decoder-only (Llama-style) | 100.7M | 12 × 768, GQA 12q/4kv, RoPE, QK-norm, SwiGLU 2048 |
| A_small | Same design, smaller (KD student) | 64.0M | 10 × 640, GQA 10q/2kv, SwiGLU 1728 |
| A_muon | A trained with Muon + AdamW | 100.7M | Muon lr 0.02 on hidden matrices |
| B | Encoder–decoder, continuation objective | 138.6M | 8 + 8 × 768; encoder reads a 768-token prefix, decoder writes 256 tokens; biases on |
| C | RetNet | 117.2M | 12 × 768, 6 retention heads, fixed decays γ = 1 − 2^(−5−h); parallel training, recurrent decoding |
| D | Mamba-2 (SSD) | 113.2M | 24 × 768, d_state 64, expand 2, chunked scan in pure PyTorch |
| E | Sparse MoE decoder | 271M total / ~101M active | 8 SwiGLU experts, top-2, load-balance + router z-loss |
| F | Hybrid Mamba-2 + attention | ~85M | 6 × [1 Mamba-2 + 1 attention] |
| G | Bidirectional MLM encoder | 110.1M | 30% BERT masking + masked-suffix rows; generates by left-to-right mask filling |

**Training.** AdamW (lr 6e-4; A_small 8e-4), weight decay 0.1, warmup then cosine decay to 10%, grad clip 1.0,
131k tokens per step, fp16. Each model trained for one ~10.5 h Kaggle session on one T4, with checkpoints and
auto-resume. Continued-training runs (`*_cont`) restart from the round-1 weights at half the peak LR with a new data
seed. **Decoding:** greedy with no-repeat-4-gram, or nucleus (T 0.7, top-p 0.9, no-repeat-4-gram). EOS is suppressed
and the output is trimmed to 128 words. MBR decoding is also implemented (`decode/mbr*.json`).

**Submissions** (dev BERTScore, 300 examples; Codabench test score):

| Submission | Dev | Test |
|---|---|---|
| A_decoder_cont nucleus | 0.8460 | 0.8355 |
| A_decoder_cont greedy | 0.8466 | 0.8343 |
| A_decoder nucleus / greedy | 0.8448 / 0.8440 | 0.8345 / 0.8325 |
| E_moe nucleus / greedy | — | 0.8345 / 0.8331 |
| F_hybrid nucleus / greedy | — | 0.8333 / 0.8319 |
| D_mamba2 nucleus / greedy | 0.8330 / 0.8318 | 0.8286 / 0.8262 |
| C_retnet greedy | 0.8324 | 0.8261 |
| B_encdec nucleus / greedy | 0.8264 / 0.8223 | 0.8246 / 0.8212 |
| G_mlm_encoder mask-fill | 0.7662 | 0.7653 |
| A_decoder_muon_cont nucleus | — | 0.8352 |
| C_retnet_cont nucleus | — | 0.8313 |

## Shared transfer protocol (Phases 2–7)

Every task model starts from a Phase-1 checkpoint loaded through `ft/backbones.py`. The fp16 exports of A
(= A_decoder_cont), A_small, B, C, D and G live in one Kaggle dataset. A task head sits on top of the backbone, and one
PEFT method from `ft/peft.py` decides which parameters train. LoRMA is implemented as LoRMA+,
`W0·x + (α/r)·B(A(W0·x))` with B initialised to 0. QLoRA stores the frozen base in our own NF4 4-bit format.
Every PEFT method except QLoRA is the identity at initialisation (tested). For knowledge distillation, a teacher run
saves its soft predictions, and the 64M student A_small trains on `α·CE + (1−α)·T²·KL` (the form depends on the task).
Every phase was run with all of A/B/C/D and a KD student. The lists below give the runs that were actually submitted
to Codabench (dev score / test score). Runs evaluated on dev only are marked as such. Phases 3 and 4 have no KD zip
among their submissions: the KD students there were evaluated on dev only.

## Phase 2 — Legal NER

- **Alignment.** `phase2/data.py` `OffsetTokenizer` gives exact character offsets for every BPE piece. With
  `split="edge"`, leading and trailing punctuation becomes separate pieces. This raises the gold round-trip
  (oracle) F1 from 0.627 (Phase-1 tokenisation) to 0.987. Learned per-label boundary-trim rules fix remaining edges.
- **Model.** Backbone over 512-token sliding windows (50% overlap, logits averaged). Causal A drops its mask
  (bidirectional); C/D can use an "echo" (window read twice). Head: linear / BiLSTM (+ optional CRF), then
  BIO-constrained Viterbi. Dev split: about 10 of the 70 training documents.
- **Submissions** (dev strict macro-F1 / test): A_frozen_bilstm (frozen backbone + BiLSTM) 0.467 / 0.4566 ·
  B_lora (LoRA) 0.314 / 0.3536 · C_lorma_bilstm__full (**LoRMA** + BiLSTM) 0.467 / 0.5182 · D_full_echo__full (full FT + echo)
  0.437 / 0.4550 · G_full (full FT) 0.529 / 0.5449 · G_lora__full (LoRA) 0.513 / 0.5438 · As_kd_G__full (**KD**, teacher G_lora)
  0.507 / 0.5377. `__full` variants are retrained on all 70 documents, so their dev score is the one of the run trained on
  the dev split. Dev only: D_lora_bilstm 0.491, A_houlsby 0.483, plus A_bitfit, G_lorma and As_kd.

## Phase 3 — Rhetorical roles

- **Model.** Whole sentences are packed into 512-token windows. Each sentence vector is the mean of its token
  features, plus relative-position and domain embeddings. A BiLSTM (optional CRF) runs over the document's sentence
  sequence. CL and IT are trained jointly with a domain tag. The loss is √-inverse-frequency weighted CE. KD uses
  T²·KL on sentence logits.
- **Submissions** (dev macro-F1 on 10 documents / test): A_qlora (QLoRA) 0.473 / 0.5606 · D_lorma (**LoRMA**) 0.489 / 0.5490 ·
  C_lora (LoRA) 0.474 / 0.5233 · D_lorma__bias (D_lorma + per-class logit bias fitted on dev) — / 0.5085 · A_adalora (AdaLoRA) 0.476 / 0.5010 ·
  A_prefix (prefix) 0.444 / 0.4958 · B_rslora (rsLoRA) 0.428 / 0.4643 · D_prefix (prefix) 0.436 / 0.4619 · G_adalora (AdaLoRA)
  0.460 / 0.4605 · B_lorma (LoRMA) 0.431 / 0.4213 · A_scratch (no pretraining, control) 0.339 / 0.3413.
  Dev only: As_kd (**KD**, teacher G_lora) 0.496, G_lora_focal 0.499, G_lora 0.493, ens_G3 0.500. The G_lora and
  C_lorma_bilstm uploads failed on Codabench, so they have no test score.

## Phase 4 — Court judgment prediction

- **Views.** first / last / middle / headtail windows, or hierarchical chunks of 512 tokens. Chunk pooling feeds a
  document aggregator (mean / attention / BiGRU), followed by a linear head with balanced class weights.
- **Two-stage hierarchical (`phase4/hier.py`).** A finished run stores chunk vectors for up to 32 chunks per document.
  Small aggregators then train on those vectors. Section probes use chunks from each quarter of the document.
- **Data note.** The course `dev.json` contains all test documents with labels, plus duplicated rows. We report and
  select on the 994 clean dev documents and never use dev labels for test.
- **Submissions** (clean-dev macro-F1 / test): A_lorma__h_mean (**LoRMA** + two-stage mean) 0.766 / 0.7578 ·
  G_gradual (gradual unfreezing) 0.709 / 0.7013 · C_houlsby (adapters) 0.686 / 0.6762 · B_full (full) 0.656 / 0.6638 ·
  C_lorma (LoRMA) 0.651 / 0.6601 · D_partial (partial FT, top 6/24 layers) 0.612 / 0.6055 · G_ab_first (LoRMA, first window
  only, ablation) — / 0.5770 · D_adalora (AdaLoRA) — / 0.5724. Dev only: As_kd_A (**KD**, teacher A_lorma) 0.736,
  ens_hmean 0.788, ens_A5 0.768, A_lorma__h_gru 0.751, A_lorma 0.727, G_lorma two-stage 0.744. The `__thr` zips in
  `kaggle/p4_upload` have thresholds tuned on the leaked dev rows and were not submitted.

## Phase 5 — Statute identification

- **Model.** The Phase-4 document model with 100 sigmoid outputs (prior-initialised bias). Loss is BCE with a capped
  √(neg/pos) positive weight; focal / ASL and a label-interaction layer are ablations.
- **Thresholds.** Options are fixed 0.5, one global threshold, per-label thresholds, and "at least one label". They are
  fitted on clean dev (97 dev rows that duplicate test texts are removed), and honest estimates come from 2-fold
  cross-fitting. A guard never predicts an id above the largest gold id seen in dev.
- **Submissions** (per-label thresholds; cross-fitted clean-dev macro-F1 / test): A_lorma__h_gru (**LoRMA**) 0.359 / 0.3593 ·
  B_full (full) 0.320 / 0.3210 · C_houlsby (adapters) 0.312 / 0.3086 · D_partial (partial) 0.235 / 0.2317 · A_qlora (QLoRA)
  0.350 / 0.3492 · As_kd (**KD**) 0.365 / 0.3634 · ens_top4 (A_full + As_full + As_kd + A_lorma two-stage) 0.380 / 0.3768 ·
  A_full 0.373 / 0.3685. Also submitted (test): ens_top7 0.3720, ens_hier4 0.3646, As_kd_feat 0.3618, As_full 0.3570,
  G_full 0.3424, A_adalora 0.3180, A_prompt 0.3009.

## Phase 6 — Abstractive summarisation

- **Model (`phase6/genlm.py`).** The full Phase-1 LMs act as generators. Decoder-only models read
  `[doc] <extra_0> [summary]`; B encodes the document and decodes the summary. The loss covers summary tokens only.
  Generation uses the cached prefill/step decoding of each family.
- **Long documents.** Input selection is `central` (lead 25% + TF-IDF-central sentences; default), `head`,
  `headtail`, `oracle` or `dancer`. Input budgets: A 1,536, B 2,048, C/D 1,024 tokens. The length target is
  0.22 × document tokens (clipped 128–1024) × `len_scale`, with no-repeat-4-gram blocking.
- **KD.** Word-level T²·KL to a saved PEFT teacher's next-token distribution.
- **Submissions** (`kaggle/p6_upload`, dev BERTScore on 100 held-out training documents / test): A_lorma__dec_s07 (**LoRMA**) 0.8442 / 0.8412 ·
  B_lora (LoRA) 0.8219 / 0.8224 · C_qlora (QLoRA) 0.8235 / 0.8228 · D_adalora (AdaLoRA) 0.8203 / 0.8200 · B_houlsby (Houlsby)
  0.8229 / 0.8221 · As_kd (**KD**) 0.8434 / 0.8424; extras include A_full 0.8453 / 0.8418, As_full 0.8438 / 0.8428, A_qlora,
  A_lora, A_houlsby, A_pfeiffer, As_scratch and long-document ablations (31 zips built, 19 submitted).

## Phase 7 — Prior case retrieval

- **Bi-encoder (`phase7/train.py`).** The Phase-4 chunk encoder (spread 512-token chunks, pooled and aggregated) plus a
  projection, L2-normalised, with a learned temperature. It trains with InfoNCE over in-batch negatives plus BM25 hard
  negatives (other relevant candidates are masked). KD is a KL divergence to the teacher's in-batch similarity
  distribution. Ranking uses `α·z(dense) + (1−α)·z(BM25)`, with α chosen on dev, and returns 7 unique ids per query
  (the query itself is excluded).
- **Score fusion (`phase7/fusion.py`, CPU).** The dense scores of the best runs are fused with BM25 over hashed token
  1–5-grams, a citation-count prior and a "candidate is newer than the query" date check. Weights are tuned on dev
  micro-F1@7 by coordinate ascent; the reported estimate comes from a 2-fold cross-fit.
- **Submissions** (dev micro-F1@7 on 118 queries / test): A_lorma (**LoRMA**) 0.311 / 0.3018 · B_full (full) 0.289 / 0.2581 ·
  C_ia3 (IA3) 0.266 / 0.2516 · D_partial (partial) 0.239 / 0.2268 · G_qlora (QLoRA) 0.297 / 0.2817 · As_kd_A (**KD**, teacher
  A_lorma) 0.316 / 0.2882 · A_qlora (QLoRA) 0.332 / 0.2995 · ens_top4 — / 0.2965. Fusion zips: fusion2_fused (dense +
  3/4/5-gram BM25 + priors) 0.4594 cross-fit / 0.4542 · fusion2_lexical_3gram (BM25 3-gram alone) 0.428 / 0.4335 ·
  fusion_fused (pass 1) 0.4197 cross-fit / 0.4306 · fusion_lexical — / 0.4141 · fusion_dense — / 0.2953.

---

## Tests

Each package has a `tests/` folder (pytest, CPU). The tests cover:
- metric parity with the course evaluators;
- tokenizer round-trips and character-offset alignment;
- causal-leak and parallel-vs-recurrent equivalence checks for every architecture;
- PEFT identity at initialisation, trainable-parameter sets and merge equivalence;
- device placement on a meta device;
- KD and ensemble paths;
- submission validators, run end to end on synthetic data in the official layouts.

Tests that need the Phase-1 tokenizer (`results/results/p1data_small/tokenizer/a1bpe_32755.json`, written by
`phase1.data.prep`) or the course evaluators are skipped when those files are absent.

## References

See the report bibliography. Main sources: Transformer / Llama, RetNet, Mamba-2, Mixture-of-Experts (Switch, Mixtral),
ModernBERT, LoRA, rsLoRA, AdaLoRA, QLoRA, LoRMA, IA3, prefix / prompt tuning, Houlsby / Pfeiffer adapters, BitFit,
ULMFiT, knowledge distillation, BM25, IL-TUR benchmark.
