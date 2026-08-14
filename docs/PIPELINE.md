# Data pipeline contract

This document is the team agreement on what each stage consumes and produces.
If you change a format, a path convention, or a default here, you are changing
an interface other people's jobs depend on - update this file in the same PR.

```
raw archives ──► Stage 1 ──► Parquet corpus ──► Stage 2 ──► packed blocks ──► Stage 3 ──► model ──► Stage 4 ──► results CSVs
              extract_corpus            pack_tokens                train_dapt              eval harnesses
```

Filesystem note: **code paths are relative to the repo root; data paths are
machine-specific and flow in via CLI args** (locally) **or `hpc/paths.env`**
(on Delta). Never hardcode an absolute path in a script - thread it through an
argument or a `CB2_*` variable.

## Stage 0 - raw corpus (input, not in git)

The EN-Politics conflict corpus: `*.json.tar.gz` archives of JSONL articles,
organized as `<corpus-root>/1945-2021 json_files_with_metadata/<source>/...`
with five source folders: `1.News`, `2.Organization`, `3.Gigaword`,
`4.UTDstory`, `5.wikipedia`. Ask the team for the share/Globus location; do not
re-download sources independently.

## Stage 1 - Parquet corpus (`src/data/extract_corpus.py`)

```bash
python src/data/extract_corpus.py --corpus-root <raw-root> --out <corpus-out> [--workers N]
```

**Output contract** - one zstd Parquet shard per input archive, laid out as
`<corpus-out>/parquet/<Source>/<stem>.parquet` where `<Source>` in
`{News, Organization, Gigaword, UTDstory, Wikipedia}`. Downstream stages infer
the source from this **parent directory name** - do not flatten the layout.

Row schema (all stages depend on these columns):

| column | meaning |
|---|---|
| `id` | stable article id |
| `source` | source name, same as the folder |
| `date`, `year` | publication date |
| `n_chars`, `n_words` | length metadata |
| `text` | ftfy-cleaned article text |
| `text_hash` | exact-dedup key |
| `split` | `train` \| `eval_random` (0.3% random) \| `eval_temporal` (year 2021) |

The script is resumable (skips shards whose output exists). Validate with
`python src/data/corpus_stats.py --corpus .../parquet --out analysis/data`.

## Stage 2 - packed token blocks (`src/data/pack_tokens.py`)

```bash
python src/data/pack_tokens.py --corpus .../parquet --split train \
  --seqlen 1024 --source-weights "News=1.0,Organization=1.0,UTDstory=1.0,Gigaword=0.7,Wikipedia=0.25" \
  --max-tokens <budget> --dedup --seed 7 --out <packed>/train_1024_<tag>
```

**Output contract** - a directory containing exactly two files:

- `tokens.u16` - uint16 memmap, shape `(n_blocks, seqlen)`, docs concatenated
  with `[SEP]`, no padding, no special tokens beyond the separator;
- `meta.json` - `seqlen`, `n_blocks`, `n_tokens`, `n_docs`, `tokenizer`,
  `source_weights`, `max_tokens`, `dedup` - the pack's provenance. Never edit
  it by hand; never mix packs made with different tokenizers.

**Naming convention:** `train_<seqlen>_<tag>` / `eval_random_<seqlen>_<tag>`.
The `<tag>` says what is inside (`native`, `aug`, `5b`, `hpc`...). Packs must
live on fast local storage (Lustre `/scratch` on Delta; ext4 in the WSL pilot -
never a network/9p mount, memmap random reads will crawl).

**Standard source weights** (used by every headline run): `News=1.0,
Organization=1.0, UTDstory=1.0, Gigaword=0.7, Wikipedia=0.25`. Changing them is
an experiment, not a default - tag the pack accordingly.

## Stage 3 - DAPT (`src/pretrain/train_dapt.py`)

Single GPU and `torchrun` DDP both work; see `hpc/pretrain.sbatch` for the
canonical Delta invocation and `docs/HPC_DELTA.md` for scaling rules.

**Reference recipes** (pilot-validated; global batch 256 blocks ≈ 262k tokens/step):

| recipe | base | scheduler | peak LR | notes |
|---|---|---|---|---|
| native (A2) | `answerdotai/ModernBERT-base` | cosine | 5e-5 | simplest baseline |
| **wsd (A3, headline)** | pre-decay stable ckpt via `convert_stable_ckpt.py` | wsd | 2e-4 | warmup 0.03, decay tail 0.20 |
| aug-wsd (A6) | stable ckpt + `augment_model.py --rescale-norms` | wsd | 2e-4 | + `--new-embed-warmup-steps 400` |

The WSD recipes need the ModernBERT **pre-decay stable checkpoint** (a Composer
`.pt`, not in git - team share) converted once with
`src/pretrain/convert_stable_ckpt.py`.

**Output contract** - the run dir contains an HF model (`config.json`,
`model.safetensors`, tokenizer files), rolling `checkpoint-*` dirs (deleted on
finalize), plus two files you must preserve with every run:

- `metrics.csv` - step, tokens_seen, losses, lr, tokens/sec (feeds the R figures);
- `run_summary.json` - full args + tokens seen + GPU hours (the run's provenance).

Finalized models are named `conflibert-v2-<recipe>[-<budget>]` and keep only the
model + tokenizer + the two provenance files.

## Stage 4 - evaluation

**Two intrinsic rules that are easy to get wrong:**

1. Pseudo-PPL (`src/eval/pseudo_ppl.py`) is only comparable **between models
   sharing a tokenizer**; pin it with `--tokenizer`.
2. NER fine-tunes on ByteLevel-BPE models require `--prefix-space`, and the LR
   grid must extend to 2.4e-4. Without both, ModernBERT-family scores are
   silently depressed (this bug cost us a wrong conclusion once - see
   `docs/STATUS_AND_NEXT.md`, 2026-08-06).

**Benchmark protocol** (`scripts/corrected_bench.py`, wraps
`src/eval/finetune_benchmark.py`): per (task, model), sweep the LR grid with
seed 123 and dev selection, then run seeds 124/125 at the best-dev LR. Nine
tasks total (7 classification + 2 NER) drawn from `external/ConfliBERT/data`.

**Results contract** - every eval harness **appends** rows to a CSV under
`analysis/data/`, keyed by (model name, task, seed, lr). These CSVs are
committed to git: they are the experimental record, and every figure and table
regenerates from them. Consequences:

- pick a **globally unique model name** before running (`ConfliBERT-v2-<recipe>`;
  suffix `-psfix` for prefix-space NER rows) - collisions corrupt the record;
- reruns append: if a job died mid-config, **dedupe by key before analysis**
  (keep-last), or delete the partial rows first;
- never edit a committed results CSV by hand; if a run was wrong, delete its
  rows in a commit whose message says why.

Paper experiment harnesses (`ipe_crossfit`, `tsv_crossfit`, `win_crossfit`,
`cw_finetune`, `llm_*_baseline`, `throughput_bench`) follow the same
append-to-`analysis/data` contract. LLM baselines read the API key from
`$OPENROUTER_API_KEY` (preferred) or a local `.openrouter_key` file - which is
gitignored and must stay that way.

## External data (not in git - fetch on setup)

| path | what | how to get it |
|---|---|---|
| `external/ConfliBERT/` | benchmark tasks + configs | `git clone https://github.com/eventdata/ConfliBERT external/ConfliBERT` |
| `external/IndiaPoliceEvents/` | IPE corpus (E1/E2/E4/E5) | `git clone https://github.com/slanglab/IndiaPoliceEvents external/IndiaPoliceEvents` |
| `external/crisiswatch/` | CrisisWatch PDFs + derived task | team share, or rebuild: `crisiswatch_download.py` → `crisiswatch_parse.py` → `crisiswatch_task.py` (downloader needs internet + `curl`; run it on a login node, be polite: it rate-limits itself) |
| ModernBERT stable ckpt (`.pt`) | WSD recipes' starting point | team share; convert with `convert_stable_ckpt.py` |
| raw corpus archives | Stage 0 | team share / Globus |

## Known warts (fix welcome, coordinate first)

- `src/data/crisiswatch_task.py` has no CLI args; run it from the repo root.
- `src/eval/throughput_bench.py` hardcodes its model list in `CONFIGS`; edit
  before running on new models, and re-run on the target hardware (timings are
  hardware-specific).
- `analysis/*.R` figure scripts still `setwd()` to the pilot Windows path;
  results CSVs are portable, the R scripts are not yet.
- `scripts/run_queue*.sh` chain jobs by polling log files for sentinel strings,
  pilot-rig style. On Delta use `sbatch --dependency=afterok:<jobid>` instead.
