# What task-adaptive pretraining (A5) changes, task by task. Arrows run from
# A3 (open circle) to A3+TAPT (arrowhead); labels give the shift in F1.
suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

lab <- c(re3d = "re3d (NER)", IndiaPoliceEvents_docs = "IPE documents",
         insightCrime = "insightCrime", satp_relevant = "SATP relevance",
         cameo_class = "CAMEO class", cameo_ner = "CAMEO NER",
         IndiaPoliceEvents_sents = "IPE sentences", `20news` = "20 Newsgroups",
         BBC_News = "BBC News")

d <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(model %in% c("ConfliBERT-v2-wsd", "ConfliBERT-v2-wsd-TAPT"), n_seeds >= 3) |>
  mutate(model = ifelse(model == "ConfliBERT-v2-wsd", "a3", "tapt"),
         f1 = 100 * mean_primary, task_lab = lab[task]) |>
  select(task_lab, model, f1) |>
  pivot_wider(names_from = model, values_from = f1) |>
  mutate(delta = tapt - a3, gain = delta > 0)
d$task_lab <- reorder(d$task_lab, d$delta)

p <- ggplot(d, aes(y = task_lab, color = gain)) +
  geom_segment(aes(x = a3, xend = tapt, yend = task_lab),
               arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
               linewidth = 0.9) +
  geom_point(aes(x = a3), shape = 21, fill = "white", color = "grey45",
             size = 2.1, stroke = 0.9) +
  geom_text(aes(x = tapt, label = sprintf("%+.1f", delta),
                hjust = ifelse(gain, -0.35, 1.35)),
            size = 3.3, color = "grey20", show.legend = FALSE) +
  scale_color_manual(values = c(`TRUE` = unname(OI["blue"]), `FALSE` = unname(OI["orange"])),
                     breaks = c(TRUE, FALSE),
                     labels = c("TAPT gains", "TAPT loses")) +
  scale_x_continuous(limits = c(58, 99), breaks = seq(60, 95, 10)) +
  labs(x = "Test F1 (three seeds at the best-dev learning rate)", y = NULL) +
  theme_cb(base_size = 13) +
  theme(legend.position = "top", legend.justification = "left",
        panel.grid.major.y = element_blank())

save_fig(p, "fig_tapt", w = 6.6, h = 3.8)
