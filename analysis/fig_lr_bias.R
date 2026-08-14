# Why a single shared fine-tuning learning rate biases the comparison.
# Test F1 across the LR grid for ConfliBERT-2021 and ConfliBERT-v2 (A2),
# all nine tasks. The filled dot is the LR the corrected protocol selects
# on dev; the two model families peak at different rates.
suppressMessages({library(ggplot2); library(dplyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

lab <- c(re3d = "re3d (NER)", cameo_ner = "CAMEO NER",
         IndiaPoliceEvents_docs = "IPE documents",
         IndiaPoliceEvents_sents = "IPE sentences", insightCrime = "insightCrime",
         satp_relevant = "SATP relevance", cameo_class = "CAMEO class",
         `20news` = "20 Newsgroups", BBC_News = "BBC News")

d <- read.csv("analysis/data/lr_curves.csv") |>
  mutate(lr5 = lr * 1e5, f1 = 100 * test_primary,
         task_lab = factor(lab[task], levels = unname(lab)))
labmap <- c("ConfliBERT-2021" = "ConfliBERT-2021",
            "ConfliBERT-v2-native" = "ConfliBERT-v2 (A2)")
d$model_lab <- factor(labmap[d$model], levels = labmap)
best <- d |> group_by(task, model) |>
  slice_max(dev_primary, n = 1, with_ties = FALSE) |> ungroup()

p <- ggplot(d, aes(lr5, f1, color = model_lab)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 1.7, stroke = 0.9, shape = 21, fill = "white") +
  geom_point(data = best, size = 2.7) +
  scale_x_log10(breaks = c(2, 3, 5, 8)) +
  scale_color_manual(values = c("ConfliBERT-2021" = unname(OI["vermillion"]),
                                "ConfliBERT-v2 (A2)" = unname(OI["blue"]))) +
  facet_wrap(~task_lab, scales = "free_y", nrow = 3) +
  labs(x = "Fine-tuning learning rate (x 1e-5, log scale)",
       y = "Test F1  (filled dot = LR selected on dev)") +
  theme_cb(base_size = 11.5) +
  theme(legend.position = "top", legend.justification = "left",
        strip.text = element_text(face = "bold", color = "grey25", size = 10),
        panel.spacing.x = unit(0.7, "lines"))

save_fig(p, "fig_lr_bias", w = 6.3, h = 4.3)
