# Where every run landed: nine-task mean F1 under the corrected protocol.
# One dot per model; dashed line marks ModernBERT-base, the bar to clear.
suppressMessages({library(ggplot2); library(dplyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

d <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(n_seeds >= 3) |>
  group_by(model) |>
  summarise(mean_f1 = 100 * mean(mean_primary), .groups = "drop")
stopifnot(all(table(read.csv("analysis/data/corrected_summary.csv")$model) == 9))

mlab <- c("ConfliBERT-2021"        = "ConfliBERT-2021",
          "ModernBERT-base"        = "ModernBERT-base",
          "ConfliBERT-v2-native"   = "A2: cosine restart",
          "ConfliBERT-v2-wsd"      = "A3: WSD schedule",
          "ConfliBERT-v2-wsd-core" = "A4: conflict-core decay",
          "ConfliBERT-v2-wsd-TAPT" = "A5: A3 + TAPT",
          "ConfliBERT-v2-aug-wsd"  = "A6: bigger vocabulary, fixed",
          "ConfliBERT-v2-wsd-5b"   = "A7: twice the data",
          "ConfliBERT-v2-wsd-5b-TAPT" = "A8: A7 + TAPT")
grp <- c("ConfliBERT-2021" = "cb", "ModernBERT-base" = "base",
         "ConfliBERT-v2-native" = "v2", "ConfliBERT-v2-wsd" = "v2",
         "ConfliBERT-v2-wsd-core" = "v2", "ConfliBERT-v2-wsd-TAPT" = "v2best",
         "ConfliBERT-v2-aug-wsd" = "v2", "ConfliBERT-v2-wsd-5b" = "v2",
         "ConfliBERT-v2-wsd-5b-TAPT" = "v2")
d <- d |> filter(model %in% names(mlab))
d <- d |> mutate(lab = mlab[model], g = grp[model])
d$lab <- reorder(d$lab, d$mean_f1)
base_mean <- d$mean_f1[d$model == "ModernBERT-base"]

p <- ggplot(d, aes(mean_f1, lab, color = g)) +
  geom_vline(xintercept = base_mean, linetype = "22", color = "grey60", linewidth = 0.5) +
  geom_segment(aes(x = 73, xend = mean_f1, yend = lab), color = "grey88", linewidth = 0.7) +
  geom_point(aes(shape = g, fill = g), size = 3.4, stroke = 1.2) +
  geom_text(aes(label = sprintf("%.1f", mean_f1)), hjust = -0.7,
            size = 3.7, color = "grey20") +
  scale_color_manual(values = c(cb = unname(OI["vermillion"]), base = "grey55",
                                v2 = unname(OI["blue"]), v2best = unname(OI["blue"])),
                     guide = "none") +
  scale_fill_manual(values = c(cb = unname(OI["vermillion"]), base = "grey55",
                               v2 = "white", v2best = unname(OI["blue"])), guide = "none") +
  scale_shape_manual(values = c(cb = 21, base = 21, v2 = 21, v2best = 21), guide = "none") +
  scale_x_continuous(limits = c(73, 79.6), breaks = 73:79) +
  labs(x = "Nine-task mean test F1 (corrected protocol)", y = NULL) +
  theme_cb(base_size = 13) +
  theme(panel.grid.major.y = element_blank())

save_fig(p, "fig_ladder", w = 6.8, h = 2.9)
