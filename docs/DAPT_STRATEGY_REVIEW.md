# ConfliBERT-v2: what went wrong, what the field does, and the fix

Consolidated research brief, 2026-07-31. Context: the first headline ConfliBERT-v2 (DAPT of an
augmented-tokenizer ModernBERT on 2.5B conflict tokens) improved intrinsic pseudo-perplexity a lot
(5.84 to 3.97) but underperformed downstream, landing below the untuned ModernBERT-base on 5 of 7
benchmark tasks. A cheap diagnostic (fine-tuning the pre-DAPT init model) isolated the cause: the
deficit is present before any DAPT and is neutral-to-positive after it, so the augmented tokenizer,
not the continued pretraining, is the drag. This brief checks that finding against the recent
literature and lays out a concrete, single-GPU fix.

## TL;DR recommendation

1. Drop the augmented tokenizer from the headline model. Keep ModernBERT's native tokenizer. The
   augmented tokenizer stays only as a controlled ablation, reported with the honest efficiency vs
   accuracy trade-off below.
2. Re-run DAPT the ModernBERT-correct way: resume from an Answer.AI pre-decay (stable-phase)
   checkpoint at the stable learning rate, train on conflict data, then apply a short decay phase.
   This is the Warmup-Stable-Decay (WSD) continued-pretraining recipe the ModernBERT authors
   designed for exactly this, and it is what every successful ModernBERT domain variant used.
3. Spend the token budget on the data, not the vocabulary. A tokenizer change needs about 50B
   tokens of continued pretraining to break even, which is out of reach on one 4090. In-domain MLM
   with the stock tokenizer pays off at far smaller budgets.
4. Treat the intrinsic (pseudo-PPL) win and the downstream result as a real finding, not a bug:
   pretraining loss and downstream accuracy are known to diverge, so we report both.

## What we observed (recap)

- Intrinsic: pseudo-PPL 5.84 to 3.97, so DAPT genuinely learned conflict language.
- Downstream: v2 below ModernBERT-base on 5 of 7 tasks; the gap is largest on the conflict tasks
  (insightCrime, IndiaPoliceEvents_docs) and absent or reversed on satp_relevant and cameo_class.
- Decomposition: the pre-DAPT init model (augmented tokenizer, zero DAPT) already sits about 1.3 to
  1.9 F1 below ModernBERT-base on the conflict tasks, and DAPT does not move it further. So the
  augmented tokenizer is the cause; DAPT is neutral-to-helpful.

## What the literature says

### 1. The tokenizer result is a known, quantified effect (this is the key one)

Dagan et al., "Getting the most out of your tokenizer for pre-training and domain adaptation"
(arXiv:2402.01035), measure the crossover directly: after changing a tokenizer, the adapted model
loses to the original tokenizer until roughly 50B tokens of continued pretraining, at which point
"this difference almost disappears and even inverts." At 5B tokens the gap is still substantial.
They also find that vocabulary extension (adding domain tokens to the existing vocab) gives only
small gains over a full replacement, and that freezing to train only the new embeddings gave no
improvement over full fine-tuning in their setup. Fast Vocabulary Transfer (FVT) init helps versus
random, which we already used, but init quality is second-order to the token budget.

Implication for us: at 2.5B tokens, and with no realistic path to 50B on a single 4090 (2.5B took
about 14h, so 50B is roughly 12 days), an extended tokenizer cannot pay off. Our result is exactly
what this paper predicts. Multilingual vocabulary-expansion work (arXiv:2402.14714) reaches break-
even faster (about 2B tokens) but only with parameter freezing plus subword init and for a much
larger vocabulary shift; the direction of the effect is the same.

### 2. The ModernBERT-correct CPT recipe (what we should have done)

ModernBERT (Warner et al., arXiv:2412.13663) trains with a Warmup-Stable-Decay schedule and, like
Pythia, releases every intermediate pre-decay (stable-phase) checkpoint specifically so others can
restart continued pretraining from the stable phase and then anneal on domain data. The intended
recipe: resume from a stable-phase checkpoint, reuse the stable-phase learning rate (no new warmup,
no cold restart), train on domain data, then run a decay phase for final specialization.

Every strong ModernBERT domain variant follows this and keeps the native tokenizer:
- BioClinical ModernBERT (arXiv:2506.10896): 53.5B tokens, resumed from ModernBERT's pre-decay
  checkpoint at stable LR (base 3e-4), masking 30% then 15%, native tokenizer, gains of +0.4 to
  +9.7 F1 over base ModernBERT.
- Clinical ModernBERT (arXiv:2504.03964), moBERTo for Portuguese (arXiv:2606.22722), RexBERT for
  e-commerce (arXiv:2602.04605), and ModernBERT-bio (almanach) all use stable-checkpoint CPT.

What we did differently, and should change: we continued from the fully decayed final ModernBERT
release with a fresh cosine schedule, which re-warms an already-annealed model. Resuming from a
pre-decay checkpoint at stable LR is the supported path and avoids that.

### 3. Why pseudo-PPL improved but downstream did not

This disconnect is well documented and is not evidence of a bug. Liu et al., "Same Pre-training
Loss, Better Downstream: Implicit Bias Matters" (arXiv:2210.14199), show models with identical
pretraining loss can differ downstream, and that flatness of the solution correlates with downstream
quality where loss does not. Continuing to pretrain after convergence is one of their explicit ways
to move loss without moving (or while hurting) downstream. Related work links weight decay and
sharpness-aware training to better plasticity and less forgetting. Practical levers for us: modest
weight decay, not over-training, and the WSD decay phase (which flattens the solution).

### 4. Masking rate

15% is not sacred. "Should You Mask 15% in MLM?" (Wettig et al., arXiv:2202.08005) shows larger
models tolerate much higher masking, and domain CPT commonly uses 20 to 30%. BioClinical used 30%
then 15% in a two-phase schedule. Our 30% is defensible; a 30% then 15% two-phase is a low-risk
refinement, with the lower rate late to sharpen downstream-relevant representations.

### 5. Forgetting and replay

For heavier domain shift, mixing in general-domain text (a 50/50 replay split is a common robust
default) reduces catastrophic forgetting; self-supervised objectives forget less than supervised
ones. Our forgetting was mild (DAPT was near-neutral downstream), so replay is a secondary lever,
worth a small fraction of general tokens if we see general-task regressions.

### 6. Alternatives to full DAPT (considered, mostly deprioritized for our case)

- Parameter-efficient CPT (LoRA-DAPT): cheaper and forgets less, but LoRA underperforms full
  fine-tuning for knowledge storage unless the rank is large (about 256), because it touches
  attention but not the feed-forward layers where knowledge lives. Useful as a cost ablation, not
  as the headline.
- Contrastive or denoising objectives (SimCSE, TSDAE, Condenser): strong for sentence-embedding and
  retrieval tasks, less aligned with our token- and document-classification benchmark. Out of scope
  for the headline, possible future work.
- Causal-LM detour for encoder CPT (arXiv:2605.12438) and selective layer expansion (ADEPT,
  arXiv:2510.10071): promising but adds complexity and risk; not for the first fix.

## Recommended plan for one RTX 4090 (ranked, compute-aware)

Reality check: we cannot reach the roughly 50B tokens a tokenizer change needs. So the plan spends
compute on in-domain MLM with the native tokenizer, done the ModernBERT-correct way.

- R1 (headline, do first). Locate an Answer.AI ModernBERT pre-decay stable-phase checkpoint. Resume
  CPT from it at the stable LR on the conflict corpus with the native tokenizer, masking 30%, packed
  sequences, then a short decay phase. Budget: as many tokens as we can afford (2.5B as a first cut,
  matching the current run for comparability; extend toward 5B if the curve is still moving). Expect
  modest but real gains over ModernBERT-base on conflict tasks, in the BioClinical range. If the
  stable checkpoint is unavailable or awkward to load, fall back to CPT from the final release with a
  short-warmup, stable, then decay schedule rather than a plain cosine.
- R2 (tokenizer ablation, reframed). Keep the augmented-tokenizer DAPT we already have as the
  controlled ablation. Report it honestly: the augmented tokenizer buys about 19% fewer tokens on
  conflict text (efficiency and a smaller truncation gap) but costs roughly 1 to 2 downstream F1 at
  a 2.5B-token budget, consistent with the approximately 50B-token break-even in the literature.
  This is a genuine, citable contribution, not a failure.
- R3 (cheap refinements, optional, matched budget). Two-phase masking (30% then 15%); a small
  general-text replay fraction if we see general-task regressions; modest weight decay. Each is a
  one-factor ablation.
- R4 (efficiency ablation). LoRA-DAPT at rank 256 versus full, for the quality-versus-cost frontier.

## Decisions to confirm

1. Headline model: adopt R1 (native tokenizer, stable-checkpoint WSD CPT). Yes or no.
2. Token budget for R1: 2.5B (about 14h, comparable to the current run) or push toward 5B (about
   28h) for a stronger result.
3. Whether to fetch and resume from the pre-decay checkpoint (preferred) or fall back to CPT from
   the final release with a WSD-like schedule.

## References

- Gururangan et al., 2020. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. ACL.
- Warner et al., 2024. ModernBERT. arXiv:2412.13663. Blog: huggingface.co/blog/modernbert.
- Dagan et al., 2024. Getting the most out of your tokenizer for pre-training and domain adaptation.
  arXiv:2402.01035.
- Sounack et al., 2025. BioClinical ModernBERT. arXiv:2506.10896.
- Clinical ModernBERT. arXiv:2504.03964. moBERTo (Portuguese). arXiv:2606.22722. RexBERT
  (e-commerce). arXiv:2602.04605. A Causal LM Detour Improves Encoder CPT. arXiv:2605.12438.
- Gee et al., 2022. Fast Vocabulary Transfer / Fine-Tuning Transformers: Vocabulary Transfer.
  arXiv:2112.14569. FOCUS. arXiv:2305.14481. Efficient Vocabulary Expansion. arXiv:2402.14714.
  TokAlign (two-stage vocab adaptation). arXiv:2506.03523.
- Wettig et al., 2022. Should You Mask 15% in Masked Language Modeling? arXiv:2202.08005.
- Liu et al., 2022. Same Pre-training Loss, Better Downstream: Implicit Bias Matters.
  arXiv:2210.14199.
- ADEPT. arXiv:2510.10071. Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting.
  arXiv:2605.02105.
