# ConfliBERT-v2: research design

## One-line thesis
ConfliBERT (2021) modernized *the domain* for encoders; it is time to modernize *the encoder*
for the domain. ConfliBERT-v2 = domain-adaptive pretraining (DAPT) of ModernBERT on the
EN-Politics conflict corpus, delivering long-context, faster, tokenizer-efficient conflict
understanding, with an ablation ladder reproducible on a single consumer GPU.

## Why v2 is necessary (the unique, practitioner-facing contribution)
Original ConfliBERT is a BERT-base from 2021 and inherits three limits that *bite event-data
practitioners specifically*:
1. **512-token ceiling -> the "truncation gap."** Conflict news articles routinely exceed 512
   tokens. ConfliBERT silently truncates, discarding the tail of the article (often where
   casualties, perpetrators, and outcomes are stated). We *quantify* how much of the corpus is
   truncated and how much downstream signal is lost, then show v2 (8192 ctx) recovers it. This
   is the paper's signature result.
2. **Generic-era tokenizer fertility.** Conflict jargon fragments into many subwords, wasting
   context and blurring entity boundaries. We measure fertility on conflict terms and show the
   modern tokenizer + DAPT helps.
3. **Throughput.** Event coders run over millions of documents. ModernBERT's efficiency (packed
   seqs, flash-attn) is a deployment argument. We benchmark tokens/sec.

The framing that ties it together and is itself novel: **a reproducible recipe for domain
encoders on ONE consumer GPU.** Most political scientists / conflict researchers do not have a
cluster. Our whole ablation ladder is designed to fit a 4090.

## Data (already on disk; no re-download)
Root: `F:\Confli_2\Corpus`
- `1945-2021 json_files_with_metadata/`  -> `.json.tar.gz`, fields `{title,date,text}`, 5 sources:
  News(1447), Gigaword(879), Wikipedia(200), UTDstory(100), Organization(119). Use for
  metadata-aware selection/curriculum. (~13G compressed)
- `1945-2021 plain_text_one_sentence_per_line/` -> 2745 `.txt`, pre-segmented. (~34G) Fast path
  for MLM if we do not need metadata.
- `subset_2018-2021 (story with date in the end)/` -> 343 `.txt`, temporal holdout (~3.6G).
- `Conflicting Corpus .xlsx` -> manifest.
Splits: hold out a *temporal* eval slice (e.g. 2020-2021) for pseudo-perplexity + concept-drift
tests, in addition to a random held-out slice. Never train on the eval slices.

## Models compared
- ModernBERT-base (untuned)            -- modern arch, no domain.
- ConfliBERT-scr-uncased (untuned)     -- domain, old arch. The incumbent to beat.
- **ConfliBERT-v2 (ModernBERT + DAPT)** -- our headline model.
- (stretch) ModernBERT-large + DAPT.

## Experiment ladder (ordered; stop-anywhere, each earns its compute)
E0. **Baselines, zero training.** All untuned models on: (a) MLM pseudo-perplexity on held-out
    conflict text; (b) downstream tasks below; (c) tokenizer fertility; (d) truncation-gap
    measurement. Establishes the gap v2 must close.
E1. **Vanilla DAPT (headline).** ModernBERT-base, continued MLM on selected corpus, bf16, packed
    seqs, short ctx. Produces ConfliBERT-v2. Re-run E0 evals.
E2. **Data selection ablation.** Selected vs random, *matched token budget*. Efficiency result.
E3. **Masking ablation.** Standard 30% vs salient-span (conflict-entity) masking, matched budget.
E4. **PEFT-DAPT.** LoRA-DAPT vs full DAPT: quality-vs-cost frontier.
E5. **Merging sweep.** Interpolate v2 with base (alpha 0..1); Pareto of domain vs general.
E6. **Forgetting probe.** All models on a general benchmark (GLUE subset) + general PPL.
E7. (stretch) vocab augmentation; long-context DAPT phase; ModernBERT-large.

## Downstream evaluation tasks
- **GTD attack-type multi-label** (already have it via Confli-mBERT; multi-label F1). Primary.
- **Temporal split** of GTD (train <=2016, test >=2017) to test concept drift + long-context.
- (if feasible) one or two tasks from the original ConfliBERT benchmark suite (binary conflict
  classification, NER) for comparability with the incumbent.
Report: macro/micro F1, per-class F1 (class imbalance matters), and a long-vs-short document
breakdown to expose the truncation gap.

## Metrics & rigor
- Intrinsic: pseudo-perplexity / MLM loss on held-out conflict + general text (forgetting).
- Extrinsic: downstream F1, with multiple seeds and mean +/- sd; significance where feasible.
- Efficiency: tokens/sec throughput, GPU-hours per run, peak VRAM. Report tokens-seen for every
  training run so ablations are budget-comparable.
- Everything logged so R can regenerate every figure from CSVs.

## Compute budget philosophy (one 4090, do not run it dry)
- Prefer *short, matched-budget* controlled runs for ablations over train-to-convergence.
- One or two "heavy" runs total (the headline DAPT, maybe large). Everything else cheap/medium.
- Checkpoint + resume everything. Log tokens-seen. Kill early if curves flatten.

## Deliverables
- Models: ConfliBERT-v2 (+ ablation checkpoints) pushed to HF under `shreyasmeher/`.
- Paper: LaTeX, **no em-dashes**, figures generated in R from logged CSVs (clean, consistent).
- Repo: this `conflibert-v2/` tree, reproducible end-to-end from `scripts/`.

## Open decisions -> see chat (env, model size, contribution emphasis, HF push).
