# What a fixed context window costs. From a 200k-article sample (ModernBERT
# tokenizer): share of articles longer than the window, and share of all corpus
# tokens that fall beyond it. Model caps marked at 512 and 8,192.
suppressMessages({library(ggplot2); library(dplyr); library(scales)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

t <- read.csv("analysis/data/token_lengths.csv")$modernbert_tokens
wins <- c(128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
d <- bind_rows(lapply(wins, function(w) {
  tibble::tibble(
    win = w,
    `Articles longer than the window` = 100 * mean(t > w),
    `Tokens beyond the window`        = 100 * sum(pmax(t - w, 0)) / sum(t))
})) |>
  tidyr::pivot_longer(-win, names_to = "series", values_to = "pct")

COLS <- c("Articles longer than the window" = unname(OI["blue"]),
          "Tokens beyond the window"        = unname(OI["green"]))
at512 <- filter(d, win == 512)
cat(sprintf("at 512: %s\n", paste(sprintf("%s=%.1f", at512$series, at512$pct), collapse=", ")))

p <- ggplot(d, aes(win, pct, color = series)) +
  geom_vline(xintercept = 512,  linetype = "22", color = unname(OI["vermillion"]), linewidth = 0.55) +
  geom_vline(xintercept = 8192, linetype = "22", color = "grey55", linewidth = 0.55) +
  geom_line(linewidth = 0.95) +
  geom_point(size = 2.2, stroke = 1.0, shape = 21, fill = "white") +
  geom_point(data = at512, size = 2.6) +
  annotate("text", x = 1300, y = 47, hjust = 0, size = 3.4, color = "grey20",
           lineheight = 1.05, fontface = "bold",
           label = "at 512:\n44.6% of articles cut\n45.2% of tokens lost") +
  annotate("text", x = 512 * 1.09, y = 97, hjust = 0, size = 3.2,
           color = unname(OI["vermillion"]), label = "ConfliBERT-2021 cap: 512") +
  annotate("text", x = 8192 / 1.09, y = 97, hjust = 1, size = 3.2,
           color = "grey35", label = "ModernBERT cap: 8,192") +
  scale_x_continuous(transform = "log2", breaks = c(128, 256, 512, 1024, 2048, 4096, 8192),
                     labels = comma) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_color_manual(values = COLS) +
  labs(x = "Context window (tokens, log scale)", y = "Share of corpus (%)") +
  theme_cb(base_size = 13) +
  theme(legend.position = "top", legend.justification = "left")

save_fig(p, "fig_truncation", w = 6.9, h = 3.9)
