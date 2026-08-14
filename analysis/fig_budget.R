# Token budgets on one axis: this pilot, the published BioClinical ModernBERT
# adaptation, and the planned HPC run. Dashed line: the ~50B-token break-even
# for changing the tokenizer (Dagan et al. 2024).
suppressMessages({library(ggplot2); library(dplyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

d <- tibble::tibble(
  name = c("This pilot (per run)", "BioClinical ModernBERT", "HPC run (planned)"),
  lo   = c(2.5, 53.5, 60),
  hi   = c(2.5, 53.5, 150),
  col  = c(unname(OI["blue"]), unname(OI["green"]), unname(OI["blue"])),
  labx = c(2.5, 53.5, 150),
  lab  = c("2.5B", "53.5B", "60-150B"))
d$ypos <- 3:1

p <- ggplot(d, aes(y = ypos)) +
  geom_vline(xintercept = 50, linetype = "22", color = "grey50", linewidth = 0.5) +
  annotate("text", x = 45, y = 3.42, hjust = 1, size = 3.1, color = "grey35",
           label = "tokenizer break-even: ~50B") +
  geom_segment(aes(x = lo, xend = hi, yend = ypos, color = col),
               linewidth = 3.2, alpha = 0.85, lineend = "round") +
  geom_point(aes(x = lo, color = col), size = 3.2) +
  geom_point(aes(x = hi, color = col), size = 3.2) +
  geom_text(aes(x = labx, label = lab), hjust = -0.35, size = 3.4, color = "grey20") +
  scale_color_identity() +
  scale_x_log10(breaks = c(2.5, 10, 25, 50, 100, 150),
                labels = c("2.5", "10", "25", "50", "100", "150"),
                limits = c(2, 500)) +
  scale_y_continuous(breaks = d$ypos, labels = as.character(d$name)) +
  coord_cartesian(ylim = c(0.7, 3.55)) +
  labs(x = "Training tokens (billions, log scale)", y = NULL) +
  theme_cb(base_size = 13) +
  theme(panel.grid.major.y = element_blank())

save_fig(p, "fig_budget", w = 6.8, h = 2.4)
