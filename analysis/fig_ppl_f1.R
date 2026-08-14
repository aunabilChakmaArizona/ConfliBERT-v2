# Intrinsic fit vs downstream mean for the native-tokenizer models. Better
# pseudo-perplexity (right) does not order the downstream means by itself.
suppressMessages({library(ggplot2); library(dplyr); library(ggrepel)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

f1 <- read.csv("analysis/data/corrected_summary.csv") |>
  filter(n_seeds >= 3) |>
  group_by(model) |> summarise(mean_f1 = 100 * mean(mean_primary), .groups = "drop")
g <- function(m) f1$mean_f1[f1$model == m]

d <- tibble::tibble(
  lab = c("ModernBERT-base", "A2: cosine", "A4: core decay", "A3: WSD",
          "A7: twice the data"),
  ppl = c(4.2971, 3.6240, 3.5304, 3.4477, 3.3496),
  f1  = c(g("ModernBERT-base"), g("ConfliBERT-v2-native"),
          g("ConfliBERT-v2-wsd-core"), g("ConfliBERT-v2-wsd"),
          g("ConfliBERT-v2-wsd-5b")),
  col = c("grey55", unname(OI["skyblue"]), unname(OI["skyblue"]), unname(OI["blue"]),
          unname(OI["skyblue"])))

p <- ggplot(d, aes(ppl, f1, color = col)) +
  geom_point(size = 3.4) +
  geom_text_repel(aes(label = lab), size = 3.5, seed = 7, box.padding = 0.5,
                  show.legend = FALSE) +
  scale_x_reverse(limits = c(4.45, 3.3)) +
  scale_color_identity() +
  labs(x = "Pseudo-perplexity on held-out conflict text (lower = better fit)",
       y = "Nine-task mean test F1") +
  theme_cb(base_size = 12.5)

save_fig(p, "fig_ppl_f1", w = 5.4, h = 3.4)
