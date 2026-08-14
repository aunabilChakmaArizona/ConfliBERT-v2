# Does more context help? Document tasks fine-tuned at 512/1024/2048 for
# ModernBERT-base and ConfliBERT-v2 (A3); the shaded band is the gap between
# them. ConfliBERT-2021 is capped at 512 (diamond). End labels give F1 at 2,048.
suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

ctx <- read.csv("analysis/data/context_corrected.csv") |>
  group_by(task, model, seq_len) |>
  summarise(f1 = 100 * mean(primary), .groups = "drop")
anchors <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(task %in% c("IndiaPoliceEvents_docs", "insightCrime"),
         model %in% c("ConfliBERT-2021", "ModernBERT-base", "ConfliBERT-v2-wsd")) |>
  transmute(task, model, seq_len = 512, f1 = 100 * mean_primary)
d <- bind_rows(ctx, anchors)

lab <- c(IndiaPoliceEvents_docs = "IPE documents", insightCrime = "insightCrime")
d$task_lab <- factor(lab[d$task], levels = c("IPE documents", "insightCrime"))
labmap <- c("ConfliBERT-2021" = "ConfliBERT-2021 (512 cap)",
            "ModernBERT-base" = "ModernBERT-base",
            "ConfliBERT-v2-wsd" = "ConfliBERT-v2 (A3)")
d$model_lab <- factor(labmap[d$model], levels = labmap)

lines  <- filter(d, model != "ConfliBERT-2021")
anchor <- filter(d, model == "ConfliBERT-2021")
ends   <- filter(lines, seq_len == 2048)
band <- lines |>
  pivot_wider(id_cols = c(task_lab, seq_len), names_from = model, values_from = f1) |>
  mutate(lo = pmin(`ModernBERT-base`, `ConfliBERT-v2-wsd`),
         hi = pmax(`ModernBERT-base`, `ConfliBERT-v2-wsd`))

p <- ggplot(lines, aes(seq_len, f1, color = model_lab)) +
  geom_ribbon(data = band, aes(x = seq_len, ymin = lo, ymax = hi),
              inherit.aes = FALSE, fill = "grey70", alpha = 0.18) +
  geom_line(linewidth = 0.95) +
  geom_point(size = 2.7, stroke = 1.1, shape = 21, fill = "white") +
  geom_point(data = anchor, size = 3.6, shape = 18) +
  geom_text(data = ends, aes(label = sprintf("%.1f", f1)), hjust = -0.35,
            size = 3.3, show.legend = FALSE) +
  scale_x_continuous(transform = "log2", breaks = c(512, 1024, 2048),
                     expand = expansion(mult = c(0.07, 0.22))) +
  scale_color_manual(values = c("ConfliBERT-2021 (512 cap)" = unname(OI["vermillion"]),
                                "ModernBERT-base" = "grey55",
                                "ConfliBERT-v2 (A3)" = unname(OI["blue"]))) +
  facet_wrap(~task_lab, scales = "free_y") +
  labs(x = "Fine-tuning context length (tokens)", y = "Test F1") +
  theme_cb(base_size = 13) +
  theme(legend.position = "top", legend.justification = "left",
        strip.text = element_text(face = "bold", color = "grey25"))

save_fig(p, "fig_context", w = 7.0, h = 3.8)
