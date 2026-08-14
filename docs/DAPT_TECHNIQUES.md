# Advanced Domain-Adaptive Pretraining (DAPT): technique survey

Scope: what the field has developed for adapting an already-pretrained encoder to a
specialized domain, filtered for what is *feasible and worth ablating on a single RTX 4090*
and *novel in the conflict / political-violence setting*. Each entry notes the idea, the key
reference, the compute cost on our hardware, and whether we should run it.

Legend for cost (one ModernBERT-base DAPT pass, bf16, packed seqs, ~1-3B tokens):
  cheap  = < 4 GPU-hours or no training (eval / merge only)
  medium = ~ 8-24 GPU-hours
  heavy  = > 24 GPU-hours (budget carefully; at most one or two of these)

--------------------------------------------------------------------------------
## 1. The core: continued MLM (vanilla DAPT)
- Idea: keep the pretrained weights, continue masked-language-modeling on in-domain text.
- Ref: Gururangan et al., "Don't Stop Pretraining", ACL 2020. Original ConfliBERT `-cont`.
- Cost: medium-heavy (this is the backbone run).
- Decision: YES. This is the headline v2 model. Everything else is an ablation around it.

## 2. Tokenizer / vocabulary adaptation  ***(a strong novelty axis)***
- Idea: a generic tokenizer over-fragments domain terms (actor names, weapon types, place
  names, org acronyms). Measure *fertility* (subwords per word) on conflict text; optionally
  add domain tokens and continue training their embeddings.
- Refs: exBERT (Tai et al. 2020), AVocaDo (Hong et al. 2021), "FVT" fast vocab transfer
  (Gee et al. 2022), Portes et al. on tokenizer fertility.
- Why it matters here: ModernBERT ships a modern BPE (OLMo-style, ~50k). Conflict jargon
  ("IED", "JeM", "cantonment", "cadre", "exfil") likely fragments badly. Fertility is a
  clean, cheap, *practitioner-legible* argument for why v2 helps and where.
- Cost: fertility analysis = cheap (no training). Full vocab-augmented DAPT = medium.
- Decision: YES for the fertility *analysis* (goes in the paper as a motivating figure).
  Vocab-augmentation *training* = optional stretch experiment.

## 3. Masking strategy ablations
- 3a. Masking rate. BERT used 15%; ModernBERT uses 30%. "Should You Mask 15%?" (Wettig
  et al. 2022) argues higher is often better for larger models. Cheap-ish to ablate.
- 3b. Whole-word masking (WWM). Mask all subwords of a word together. Standard, cheap.
- 3c. Span masking (SpanBERT, Joshi et al. 2020). Mask contiguous spans.
- 3d. Salient-span / entity masking  ***(novelty axis)***. Preferentially mask
  conflict-relevant entities (actors, weapons, locations, event nouns) using an NER pass or
  a PMI/domain-term list. Refs: Guu et al. REALM (2020), Ye et al. "salient span masking".
  In our domain this is very natural: force the model to predict *who did what to whom*.
- Cost: each = medium (needs its own run to be comparable). Do these as *short controlled*
  runs on a fixed token budget rather than to convergence.
- Decision: run standard-30% vs salient-span as the headline masking ablation. WWM/rate as
  optional.

## 4. Data selection & curriculum  ***(the compute-optimal story; essential on 1 GPU)***
- Idea: 77G is far more than we can train on well with one GPU. Which tokens are worth the
  gradient steps? Select the most in-domain / task-relevant subset.
- Refs: DSIR importance resampling (Xie et al. 2023), Moore-Lewis / cross-entropy difference
  selection (Moore & Lewis 2010), DoReMi domain reweighting (Xie et al. 2023),
  data curriculum (Bengio et al. 2009; source-difficulty ordering).
- Concrete here: our corpus is *labeled by source* (News / Gigaword / Wikipedia / UTDstory /
  Organization) and *by date*. That gives free axes for reweighting and curriculum without
  any classifier: e.g. upweight News+Organization (most conflict-dense), schedule Wikipedia
  early (easy, general) then conflict-dense sources late.
- Cost: DSIR scoring = cheap (n-gram features). A selected-vs-random matched-budget run pair
  = medium each.
- Decision: YES. "Same token budget, selected beats random" is a headline efficiency result
  and directly serves the single-GPU practitioner narrative.

## 5. Parameter-efficient DAPT (PEFT)  ***(efficiency novelty)***
- Idea: adapt with LoRA/DoRA or adapters instead of full fine-tuning of all params. Far less
  memory and often competitive; lets a practitioner adapt on a laptop-class GPU.
- Refs: LoRA (Hu et al. 2021), DoRA (Liu et al. 2024), adapter-DAPT (Pfeiffer et al.),
  "AdaLoRA". Note: PEFT for *MLM continued-pretraining* (not just downstream FT) is
  under-explored, especially for encoders - genuinely publishable angle.
- Cost: cheap-medium (much lower memory; larger batch possible).
- Decision: YES as an ablation - "how close does LoRA-DAPT get to full DAPT, at what cost?"

## 6. Catastrophic forgetting: measure & mitigate
- Idea: DAPT can erode general-language ability. Measure it (eval v2 on a general benchmark,
  e.g. a GLUE subset, and on general-domain pseudo-perplexity) and mitigate.
- Mitigations: (a) replay / mix in general-domain data during DAPT; (b) EWC (Kirkpatrick
  2017); (c) conservative LR; (d) model merging (see 7).
- Cost: measurement = cheap. Replay = folds into a DAPT run (no extra run).
- Decision: YES for measurement (it is a rigor point reviewers expect). Replay as the
  cheapest mitigation to include.

## 7. Model merging / souping  ***(cheap, trendy, effective)***
- Idea: interpolate v2-DAPT weights with the base ModernBERT (spherical/linear) to trade off
  domain gain vs general retention, with NO extra training.
- Refs: Model Soups (Wortsman et al. 2022), WiSE-FT (Wortsman et al. 2022), TIES/DARE merging
  (Yadav 2023; Yu 2024), task arithmetic (Ilharco 2023).
- Cost: cheap (weight arithmetic + eval only).
- Decision: YES. A merge sweep (alpha 0..1) is nearly free and often gives a Pareto win.

## 8. Long-context adaptation  ***(the flagship practitioner argument)***
- Idea: original BERT/ConfliBERT cap at 512 tokens; conflict news articles routinely exceed
  that, so ConfliBERT *truncates* and loses signal. ModernBERT natively handles 8192.
  Quantify the "truncation gap" on our corpus and show v2 reads whole documents.
- Refs: ModernBERT (Warner et al. 2024) context extension; Longformer/BigBird lineage.
- Cost: measurement = cheap. Optional long-context DAPT phase = medium.
- Decision: YES for the truncation-gap analysis (headline figure). Long-context DAPT phase
  optional (ModernBERT already handles length; may not need extra training).

## 9. Objective / architecture extras (mostly SKIP for v1 of this project)
- Replaced-token-detection (ELECTRA), TEAMS, contrastive objectives (SimCSE-style), NER-aware
  pretraining. Powerful but each is a large detour. Note as future work.

--------------------------------------------------------------------------------
## Efficiency toolkit (applies to every training run on the 4090)
- bf16 mixed precision; ModernBERT is bf16-native.
- Sequence packing / unpadded batches (ModernBERT supports this) -> big throughput win.
- Flash-Attention 2 if installable on Windows; otherwise SDPA fallback (works, slower).
- Gradient checkpointing + gradient accumulation to fit effective batch in 24GB.
- Short context (512-1024) for the bulk of DAPT; long context only for a final phase/eval.
- Warmup-Stable-Decay or cosine LR; low peak LR (continued pretraining, not from scratch).
- Checkpoint + resumable; log tokens-seen (not just steps) so runs are budget-comparable.

## What makes THIS project's experiment set unique (not just "DAPT again")
1. Source- and date-labeled corpus enables *classifier-free* data selection & curriculum.
2. Salient-span masking of conflict entities - domain-motivated objective.
3. Tokenizer-fertility motivation specific to conflict jargon.
4. Truncation-gap quantification - a concrete, measurable practitioner harm that v2 fixes.
5. Full ablation ladder reproducible on ONE consumer GPU - democratizing domain encoders
   for social scientists, who rarely have clusters. This framing is itself a contribution.
