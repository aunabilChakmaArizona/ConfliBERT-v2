# Marginal detection curves: probability an instrument detects a gold event as a
# smooth function of where the evidence begins in the article. Spline logit per
# instrument with 95% confidence bands; pooled over the five labels.
# Rebuild: Rscript analysis/fig_ipe_dose.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr)); suppressMessages(library(tidyr))
suppressMessages(library(splines))

LAB <- c("KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION")

ev <- read.csv("analysis/data/ipe_evidence_position.csv") |>
  select(doc_id, label, pos = first_evidence_tok)

cf <- read.csv("analysis/data/ipe_crossfit_preds.csv") |>
  filter((model == "ConfliBERT-2021" & seq_len == "512") |
         (model == "ConfliBERT-2021-win") |
         (model == "ConfliBERT-v2" & seq_len == "2048")) |>
  mutate(instrument = ifelse(model == "ConfliBERT-2021-win",
                             "ConfliBERT '21 sliced", SHORT_NAMES[model]))

llm <- read.csv("analysis/data/ipe_llm_preds.csv") |>
  filter(parse_error == 0) |>
  mutate(instrument = "GPT-5.6", seed = 0)
for (L in LAB) llm[[paste0("prob_", L)]] <- llm[[paste0("pred_", L)]]

long_hits <- function(df) {
  bind_rows(lapply(LAB, function(L) {
    df |>
      filter(get(paste0("gold_", L)) == 1) |>
      transmute(instrument, seed, doc_id = as.character(doc_id), label = L,
                hit = as.integer(get(paste0("prob_", L)) > 0.5))
  }))
}

d <- bind_rows(long_hits(cf), long_hits(llm)) |>
  inner_join(ev |> mutate(doc_id = as.character(doc_id)),
             by = c("doc_id", "label")) |>
  filter(pos <= 1500)

grid <- data.frame(pos = seq(1, 1500, by = 10))
curves <- d |>
  group_by(instrument) |>
  group_modify(function(g, key) {
    fit <- glm(hit ~ ns(pos, 3), family = binomial, data = g)
    pr <- predict(fit, newdata = grid, type = "link", se.fit = TRUE)
    data.frame(pos = grid$pos,
               p = plogis(pr$fit),
               lo = plogis(pr$fit - 1.96 * pr$se.fit),
               hi = plogis(pr$fit + 1.96 * pr$se.fit))
  }) |> ungroup()

COLORS <- c("ConfliBERT '21" = "#0072B2", "ConfliBERT '21 sliced" = "#0072B2",
            "ConfliBERT-Modern" = "#D55E00", "GPT-5.6" = "#CC79A7")
LTYPES <- c("ConfliBERT '21" = "solid", "ConfliBERT '21 sliced" = "42",
            "ConfliBERT-Modern" = "solid", "GPT-5.6" = "solid")

p <- ggplot(curves, aes(x = pos, y = p, color = instrument,
                        linetype = instrument, fill = instrument)) +
  geom_vline(xintercept = 512, color = INK_FAINT, linetype = "13",
             linewidth = 0.5) +
  annotate("text", x = 545, y = 0.06, label = "512-token window ends",
           hjust = 0, size = 2.7, family = PAPER_FAMILY, color = INK_FAINT) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.13, color = NA) +
  geom_line(linewidth = 0.75) +
  geom_rug(data = d |> distinct(doc_id, label, pos),
           aes(x = pos), inherit.aes = FALSE, sides = "b",
           alpha = 0.25, length = unit(4, "pt"), color = INK_MUTED) +
  scale_color_manual(values = COLORS) +
  scale_fill_manual(values = COLORS) +
  scale_linetype_manual(values = LTYPES) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = c("0", ".25", ".50", ".75", "1")) +
  labs(x = "Token position where the evidence begins",
       y = "Probability event is detected") +
  theme_paper(grid = "y") +
  theme(legend.key.width = unit(20, "pt"))

save_paper_fig(p, "fig_ipe_dose", w = 6.0, h = 3.4)
