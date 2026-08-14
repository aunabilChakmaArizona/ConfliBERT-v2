# Corpus-level recall by event type and instrument (IndiaPoliceEvents).
# PRELIMINARY while E1's 2048-token configs run: encoders shown at 512 tokens.
# Rebuild: Rscript analysis/fig_ipe_recall.R   (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr)); suppressMessages(library(tidyr))

LAB <- c("KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION")
DISP <- c(KILL = "Police killed someone", ARREST = "Police made arrests",
          FAIL = "Police failed to act", FORCE = "Police used other force",
          ANY_ACTION = "Any police action")

cf <- read.csv("analysis/data/ipe_crossfit_preds.csv") |>
  filter((model == "ConfliBERT-2021" & seq_len == 512) |
         (model == "ConfliBERT-v2" & seq_len == 2048) |
         (model == "ModernBERT-base" & seq_len == 2048))
enc <- lapply(LAB, function(L) {
  cf |>
    group_by(model, seed) |>
    summarise(recall = sum(get(paste0("gold_", L)) == 1 &
                             get(paste0("prob_", L)) > 0.5) /
                       sum(get(paste0("gold_", L)) == 1), .groups = "drop") |>
    mutate(label = L)
}) |> bind_rows() |>
  group_by(model, label) |>
  summarise(mid = mean(recall), lo = min(recall), hi = max(recall),
            .groups = "drop")

llm <- read.csv("analysis/data/ipe_llm_preds.csv") |>
  filter(parse_error == 0)
llm <- lapply(LAB, function(L) {
  data.frame(model = "GPT-5.6", label = L,
             mid = sum(llm[[paste0("gold_", L)]] == 1 &
                         llm[[paste0("pred_", L)]] == 1) /
                   sum(llm[[paste0("gold_", L)]] == 1),
             lo = NA, hi = NA)
}) |> bind_rows()

d <- bind_rows(enc, llm) |>
  mutate(model = factor(model, levels = INSTRUMENT_ORDER),
         panel = DISP[label])
row_order <- d |> group_by(panel) |> summarise(m = mean(mid)) |> arrange(m)
d$panel <- factor(d$panel, levels = row_order$panel)

# identification via the legend only; no internal titles or name labels
p <- ggplot(d, aes(x = mid, y = panel, color = model)) +
  geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.6, alpha = 0.55,
                 na.rm = TRUE) +
  geom_point(size = 2.4) +
  scale_color_instrument(labels = SHORT_NAMES[INSTRUMENT_ORDER]) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = c("0", ".25", ".50", ".75", "1"),
                     expand = expansion(mult = c(0.01, 0.03))) +
  labs(x = "Corpus-level recall", y = NULL) +
  theme_paper(grid = "x") +
  theme(axis.text.y = element_text(color = INK, size = 9))

save_paper_fig(p, "fig_ipe_recall", w = 6.0, h = 3.3)
