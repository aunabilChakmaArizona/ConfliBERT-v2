# Contributing

This is a small research team repo. The rules below exist so that months from
now we can still tell which number came from which run.

## Ground rules

1. **Never commit**: model weights, packed token files, corpora, logs, or
   anything under `outputs/`, `external/`, `backups/` - the `.gitignore`
   enforces this; don't fight it. **Never** commit API keys (`.openrouter_key`
   is gitignored; use `$OPENROUTER_API_KEY` on shared machines).
2. **Always commit** with a finished run: its `metrics.csv` +
   `run_summary.json` (renamed into `analysis/data/`), and any rows it appended
   to the results CSVs. A run that isn't in `analysis/data/` didn't happen.
3. **Interfaces live in `docs/PIPELINE.md`.** Changing a file format, a naming
   convention, a default hyperparameter, or the standard source weights means
   updating that doc in the same PR.
4. **Model names are permanent.** Results CSVs key on the model-name string;
   pick `ConfliBERT-v2-<recipe>` style names, never reuse one for a different
   set of weights.

## Workflow

- Branch from `main`, name it `<yourname>/<topic>`. PR when it runs end-to-end.
- One experiment = one PR where feasible: the script/config change, the
  appended results rows, and a paragraph in the PR description saying what
  moved and what you concluded.
- New evaluation harnesses go in `src/eval/`, follow the existing pattern
  (argparse; `bf16`; append-to-CSV under `analysis/data/`; `--tmp` honoring
  `$CB2_TMP`; import shared plumbing from `finetune_benchmark.py`).
- New Slurm jobs go in `hpc/`, source `hpc/paths.env`, and take tunables as
  `CB2_*` environment variables with sane defaults.
- Python: match the existing style - argparse CLIs, no framework sprawl,
  paths in through arguments, prints flushed. There is no linter gate; keep
  diffs reviewable instead.

## Before you burn GPU hours

- Pack on the CPU partition, not the GPU one (`hpc/pack_corpus.sbatch`).
- Sanity-check the pack (`meta.json` token count) before submitting training.
- Reproduce a known cell first: one seed of one benchmark task on a published
  model should land within noise of its row in `analysis/data/corrected_bench.csv`.
- Read the pitfalls sections of `docs/HPC_DELTA.md` and `docs/PIPELINE.md`;
  they are all lessons already paid for.
