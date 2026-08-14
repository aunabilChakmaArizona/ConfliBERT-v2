# Training dynamics, run A2 (cosine from the final release) vs run A3
# (Warmup-Stable-Decay from the pre-decay checkpoint). Same data, tokenizer
# and packing. Top: smoothed training loss with final values labeled.
# Bottom: learning-rate schedule; A3's decay window is shaded.
suppressMessages({library(ggplot2); library(dplyr); library(patchwork)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

roll <- function(v, k = 9) as.numeric(stats::filter(v, rep(1/k, k), sides = 1))
prep <- function(path, run) {
  read.csv(path) |>
    filter(train_loss != "" & !is.na(train_loss)) |>
    mutate(tokens_b = tokens_seen / 1e9, loss = roll(as.numeric(train_loss)),
           lr = as.numeric(lr), run = run) |>
    filter(!is.na(loss))
}
a2 <- prep("analysis/data/dapt_metrics_native.csv", "A2: cosine, final release")
a3 <- prep("analysis/data/dapt_metrics_wsd.csv",    "A3: WSD, stable checkpoint")
d  <- bind_rows(a2, a3)
COLS <- c("A2: cosine, final release"  = "grey60",
          "A3: WSD, stable checkpoint" = unname(OI["blue"]))

lrmax <- max(a3$lr, na.rm = TRUE)
decay_from <- max(a3$tokens_b[a3$lr >= 0.99 * lrmax], na.rm = TRUE)
ends <- d |> group_by(run) |> slice_tail(n = 1) |> ungroup()

p_loss <- ggplot(d, aes(tokens_b, loss, color = run)) +
  annotate("rect", xmin = decay_from, xmax = 2.5, ymin = -Inf, ymax = Inf,
           fill = unname(OI["blue"]), alpha = 0.06) +
  geom_line(linewidth = 0.9) +
  geom_text(data = ends, aes(label = sprintf("%.2f", loss)), hjust = -0.25,
            size = 3.3, show.legend = FALSE) +
  scale_color_manual(values = COLS) +
  coord_cartesian(xlim = c(0, 2.5), clip = "off") +
  labs(x = NULL, y = "Training loss") +
  theme_cb(base_size = 13) +
  theme(legend.position = "top", legend.justification = "left",
        axis.text.x = element_blank(), plot.margin = margin(6, 34, 2, 10))

p_lr <- ggplot(d, aes(tokens_b, lr * 1e4, color = run)) +
  annotate("rect", xmin = decay_from, xmax = 2.5, ymin = -Inf, ymax = Inf,
           fill = unname(OI["blue"]), alpha = 0.06) +
  annotate("text", x = 1.05, y = 2.28, label = "hold 2e-4", size = 3.0, color = "grey40") +
  annotate("text", x = decay_from + 0.125, y = 2.28, label = "decay", size = 3.0, color = "grey40") +
  geom_line(linewidth = 0.9, show.legend = FALSE) +
  scale_color_manual(values = COLS) +
  scale_y_continuous(breaks = c(0, 1, 2), limits = c(0, 2.5)) +
  coord_cartesian(xlim = c(0, 2.5), clip = "off") +
  labs(x = "Tokens seen (billions)", y = "LR (x 1e-4)") +
  theme_cb(base_size = 13) +
  theme(plot.margin = margin(2, 34, 6, 10))

p <- p_loss / p_lr + plot_layout(heights = c(2.1, 1))
save_fig(p, "fig_dynamics", w = 6.8, h = 4.3)
