# Main results under the corrected protocol. Left: test F1 per task (three
# seeds at the best-dev LR, one-SD whiskers) for the three headline models.
# Right: the per-task gap, v2+TAPT minus ConfliBERT-2021, aligned row by row.
# The nine-task mean sits in the shaded top row.
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

lab <- c(re3d = "re3d (NER)", IndiaPoliceEvents_docs = "IPE documents",
         insightCrime = "insightCrime", satp_relevant = "SATP relevance",
         cameo_class = "CAMEO class", cameo_ner = "CAMEO NER",
         IndiaPoliceEvents_sents = "IPE sentences", `20news` = "20 Newsgroups",
         BBC_News = "BBC News")

d <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(model %in% c("ConfliBERT-2021", "ModernBERT-base", "ConfliBERT-v2-wsd-TAPT"),
         n_seeds >= 3) |>
  mutate(task_lab = lab[task])

means <- d |> group_by(model) |>
  summarise(mean_primary = mean(mean_primary), sd_primary = NA, .groups = "drop") |>
  mutate(task_lab = "All tasks (mean)")
d <- bind_rows(d |> select(task_lab, model, mean_primary, sd_primary), means)

gap <- d |> pivot_wider(id_cols = task_lab, names_from = model,
                        values_from = mean_primary) |>
  mutate(delta = 100 * (`ConfliBERT-v2-wsd-TAPT` - `ConfliBERT-2021`))

ord <- gap |> filter(task_lab != "All tasks (mean)") |> arrange(delta)
lev <- c(ord$task_lab, "All tasks (mean)")      # bottom to top; mean on top
d$task_lab   <- factor(d$task_lab, levels = lev)
gap$task_lab <- factor(gap$task_lab, levels = lev)

labmap <- c("ConfliBERT-2021" = "ConfliBERT-2021",
            "ModernBERT-base" = "ModernBERT-base",
            "ConfliBERT-v2-wsd-TAPT" = "ConfliBERT-v2 + TAPT")
d$model_lab <- factor(labmap[d$model], levels = labmap)
n <- length(lev)
band <- annotate("rect", xmin = -Inf, xmax = Inf, ymin = n - 0.5, ymax = n + 0.5,
                 fill = "grey93")

p_dots <- ggplot(d, aes(100 * mean_primary, task_lab, color = model_lab)) +
  band +
  geom_line(aes(group = task_lab), color = "grey80", linewidth = 0.8) +
  geom_errorbar(aes(xmin = 100 * (mean_primary - sd_primary),
                    xmax = 100 * (mean_primary + sd_primary)),
                width = 0, alpha = 0.4, linewidth = 1.4, na.rm = TRUE) +
  geom_point(size = 2.7, stroke = 1.1, shape = 21, fill = "white") +
  geom_point(data = filter(d, task_lab == "All tasks (mean)"), size = 3.2) +
  scale_color_manual(values = c("ConfliBERT-2021" = unname(OI["vermillion"]),
                                "ModernBERT-base" = "grey55",
                                "ConfliBERT-v2 + TAPT" = unname(OI["blue"]))) +
  labs(x = "Test F1", y = NULL) +
  theme_cb(base_size = 12.5) +
  theme(panel.grid.major.y = element_blank())

p_gap <- ggplot(gap, aes(delta, task_lab, fill = delta > 0)) +
  band +
  geom_col(width = 0.55) +
  geom_vline(xintercept = 0, color = "grey45", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+.1f", delta),
                hjust = ifelse(delta > 0, -0.18, 1.15)),
            size = 3.1, color = "grey20") +
  scale_fill_manual(values = c(`TRUE` = unname(OI["blue"]),
                               `FALSE` = unname(OI["vermillion"])), guide = "none") +
  scale_x_continuous(limits = c(-11.5, 5.5), breaks = c(-8, -4, 0, 4)) +
  labs(x = "v2+TAPT minus CB-2021", y = NULL) +
  theme_cb(base_size = 12.5) +
  theme(axis.text.y = element_blank(), panel.grid.major.y = element_blank())

p <- p_dots + p_gap +
  plot_layout(widths = c(2.05, 1), guides = "collect") &
  theme(legend.position = "top", legend.justification = "left")
save_fig(p, "fig_results", w = 7.4, h = 4.4)
