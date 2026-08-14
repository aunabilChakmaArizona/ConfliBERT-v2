# Calibration (reliability) curves: do the instruments' probabilities mean what
# they say? Predicted probability (binned) against observed event frequency,
# pooled over the five labels and all documents; Wilson 95% intervals per bin.
# The zero-shot LLM emits only hard 0/1 codes and so cannot appear here, which is
# itself part of the argument for probability-releasing instruments.
# Rebuild: Rscript analysis/fig_ipe_calibration.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr)); suppressMessages(library(tidyr))

LAB <- c("KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION")

cf <- read.csv("analysis/data/ipe_crossfit_preds.csv") |>
  filter((model == "ConfliBERT-2021" & seq_len == "512") |
         (model == "ConfliBERT-2021-win") |
         (model == "ConfliBERT-v2" & seq_len == "2048")) |>
  mutate(instrument = ifelse(model == "ConfliBERT-2021-win",
                             "ConfliBERT '21 sliced", SHORT_NAMES[model]))

d <- bind_rows(lapply(LAB, function(L)
  cf |> transmute(instrument,
                  p = get(paste0("prob_", L)),
                  y = as.integer(get(paste0("gold_", L)) == 1))))

wilson <- function(k, n, z = 1.96) {
  ph <- k / n
  den <- 1 + z^2 / n
  ctr <- (ph + z^2 / (2 * n)) / den
  hw <- z * sqrt(ph * (1 - ph) / n + z^2 / (4 * n^2)) / den
  cbind(lo = ctr - hw, hi = ctr + hw)
}

cal <- d |>
  mutate(bin = pmin(floor(p * 10), 9)) |>
  group_by(instrument, bin) |>
  summarise(p_mean = mean(p), n = n(), k = sum(y), .groups = "drop") |>
  filter(n >= 30) |>
  mutate(obs = k / n, lo = wilson(k, n)[, "lo"], hi = wilson(k, n)[, "hi"])

COLORS <- c("ConfliBERT '21" = "#0072B2", "ConfliBERT '21 sliced" = "#56B4E9",
            "ConfliBERT-Modern" = "#D55E00")

p <- ggplot(cal, aes(x = p_mean, y = obs, color = instrument)) +
  geom_abline(slope = 1, intercept = 0, color = INK_FAINT,
              linetype = "13", linewidth = 0.5) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.55, alpha = 0.7) +
  geom_line(linewidth = 0.55, alpha = 0.7) +
  geom_point(size = 1.9) +
  scale_color_manual(values = COLORS) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = c("0", ".25", ".50", ".75", "1")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = c("0", ".25", ".50", ".75", "1")) +
  coord_equal() +
  labs(x = "Predicted probability (bin mean)",
       y = "Observed frequency") +
  theme_paper(grid = "both")

save_paper_fig(p, "fig_ipe_calibration", w = 4.6, h = 4.2)
