# Pretraining corpus in one view. Left: words per source (train split).
# Right: dated articles per year for the three dated sources, same colors.
suppressMessages({library(ggplot2); library(dplyr); library(patchwork); library(scales)})
setwd("F:/Confli_2/Corpus/conflibert-v2")
source("analysis/theme_conflibert.R")

st <- read.csv("analysis/data/corpus_stats.csv") |>
  filter(split == "train") |>
  mutate(words_b = n_words / 1e9)
yr <- read.csv("analysis/data/corpus_by_year.csv") |>
  filter(year >= 1990, year <= 2025, source %in% c("News", "Gigaword", "Organization")) |>
  mutate(source = factor(source, levels = c("News", "Gigaword", "Organization")))

p_src <- ggplot(st, aes(words_b, reorder(source, words_b), fill = source)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.2fB", words_b)), hjust = -0.15,
            size = 3.4, color = "grey20") +
  scale_fill_manual(values = SOURCE_COLORS, guide = "none") +
  scale_x_continuous(limits = c(0, 3.9), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Words in the train split (billions)", y = NULL) +
  theme_cb(base_size = 12.5) +
  theme(panel.grid.major.y = element_blank())

p_yr <- ggplot(yr, aes(year, n_articles / 1e3, fill = source)) +
  geom_area(alpha = 0.92, color = "white", linewidth = 0.25) +
  scale_fill_manual(values = SOURCE_COLORS, guide = "none") +
  scale_x_continuous(breaks = seq(1990, 2025, 10)) +
  labs(x = "Publication year (dated sources)", y = "Articles per year (thousands)") +
  theme_cb(base_size = 12.5)

p <- p_src + p_yr + plot_layout(widths = c(1, 1.35))
save_fig(p, "fig_corpus", w = 7.2, h = 3.1)
