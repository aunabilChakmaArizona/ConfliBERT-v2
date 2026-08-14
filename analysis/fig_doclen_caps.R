# Share of documents longer than each context window, per benchmark corpus.
# Rebuild: Rscript analysis/fig_doclen_caps.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr))

d <- read.csv("analysis/data/doclen_caps.csv") |>
  mutate(cap = factor(cap, levels = c(512, 1024, 2048)),
         corpus = factor(corpus, levels = c("IndiaPoliceEvents", "InsightCrime")))

CORPUS_COLORS <- c("IndiaPoliceEvents" = "#56B4E9", "InsightCrime" = "#E69F00")

p <- ggplot(d, aes(x = cap, y = pct_beyond, fill = corpus)) +
  geom_col(position = position_dodge(width = 0.62), width = 0.55) +
  geom_text(aes(label = sprintf("%.0f%%", pct_beyond)),
            position = position_dodge(width = 0.62), vjust = -0.45,
            size = 2.9, family = PAPER_FAMILY, color = INK) +
  scale_fill_manual(values = CORPUS_COLORS) +
  scale_y_continuous(limits = c(0, 68), expand = expansion(mult = c(0, 0.02)),
                     breaks = NULL) +
  labs(x = "Reading window (tokens)", y = NULL) +
  theme_paper(grid = "none")

save_paper_fig(p, "fig_doclen_caps", w = 5.6, h = 3.2)
