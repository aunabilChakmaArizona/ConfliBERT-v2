# The corrected protocol's selected learning rate for every task and model.
# The selected rate spans the full grid and differs across tasks and models; a
# single shared rate cannot be fair to all of them. On the two NER tasks the
# grid extends to 16 and 24 (x1e-5): the ModernBERT-family models were still
# improving at the old 8e-5 edge under the fixed tokenization protocol.
suppressMessages({library(ggplot2); library(dplyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

tlab <- c(re3d = "re3d (NER)", cameo_ner = "CAMEO NER",
          IndiaPoliceEvents_docs = "IPE documents",
          IndiaPoliceEvents_sents = "IPE sentences", insightCrime = "insightCrime",
          satp_relevant = "SATP relevance", cameo_class = "CAMEO class",
          `20news` = "20 Newsgroups", BBC_News = "BBC News")
morder <- c("ConfliBERT-2021", "ModernBERT-base", "ConfliBERT-v2-native",
            "ConfliBERT-v2-wsd", "ConfliBERT-v2-wsd-core", "ConfliBERT-v2-wsd-TAPT",
            "ConfliBERT-v2-wsd-5b")
mlab <- c("CB-2021", "MB-base", "v2 A2", "v2 A3", "v2 A4", "v2+TAPT", "v2 A7")

d <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(n_seeds >= 3, model %in% morder) |>
  mutate(lr5 = factor(round(best_lr * 1e5), levels = c(2, 3, 5, 8, 16, 24)),
         task_lab = factor(tlab[task], levels = rev(unname(tlab))),
         model_lab = factor(mlab[match(model, morder)], levels = mlab))

FILLS <- c(`2` = "#D6E6F4", `3` = "#9CC3E4", `5` = "#4E93C6", `8` = "#0B5FA5",
           `16` = "#084A80", `24` = "#06355C")
d$txtcol <- ifelse(d$lr5 %in% c("5", "8", "16", "24"), "white", "grey25")

p <- ggplot(d, aes(model_lab, task_lab, fill = lr5)) +
  geom_tile(color = "white", linewidth = 1.4) +
  geom_text(aes(label = as.character(lr5), color = txtcol), size = 3.6) +
  scale_fill_manual(values = FILLS, guide = "none") +
  scale_color_identity() +
  scale_x_discrete(position = "top") +
  labs(x = NULL, y = NULL) +
  theme_cb(base_size = 12.5) +
  theme(panel.grid.major = element_blank(),
        axis.text.x = element_text(face = "bold", color = "grey25"),
        plot.margin = margin(6, 10, 6, 6)) +
  coord_fixed(ratio = 0.55)

save_fig(p, "fig_lr_matrix", w = 6.2, h = 3.9)
