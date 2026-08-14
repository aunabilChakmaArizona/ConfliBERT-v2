# Expected cumulative event-document counts over March 2002 with simulation-based
# 95% intervals (King-Tomz-Wittenberg-style expected values): for each instrument,
# documents are drawn as Bernoulli trials from the model's per-document probability,
# pooling draws across fine-tuning seeds; the ribbon spans the 2.5th-97.5th
# percentile of 300 simulated histories per seed.
# Rebuild: Rscript analysis/fig_ipe_timeseries.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr)); suppressMessages(library(tidyr))

set.seed(42)
N_SIM <- 300
LABS <- c(KILL = "Police killed someone", FAIL = "Police failed to act")

cf <- read.csv("analysis/data/ipe_crossfit_preds.csv") |>
  filter((model == "ConfliBERT-2021-win") |
         (model == "ConfliBERT-v2" & seq_len == "2048")) |>
  mutate(instrument = ifelse(model == "ConfliBERT-2021-win",
                             "ConfliBERT '21 sliced", SHORT_NAMES[model]),
         day = as.integer(substr(date, 9, 10)))

llm <- read.csv("analysis/data/ipe_llm_preds.csv") |>
  filter(parse_error == 0) |>
  mutate(instrument = "GPT-5.6", seed = 0,
         day = as.integer(substr(date, 9, 10)),
         prob_KILL = pred_KILL, prob_FAIL = pred_FAIL)

sim_band <- function(df, L) {
  df |>
    group_by(instrument, seed) |>
    group_modify(function(g, key) {
      probs <- g[[paste0("prob_", L)]]
      days <- g$day
      sims <- replicate(N_SIM, {
        hits <- rbinom(length(probs), 1, probs)
        cumsum(sapply(1:31, function(d) sum(hits[days == d])))
      })
      data.frame(day = 1:31, t(apply(sims, 1, quantile, c(0.025, 0.5, 0.975)))) |>
        setNames(c("day", "lo", "mid", "hi"))
    }) |> ungroup() |>
    group_by(instrument, day) |>
    summarise(lo = min(lo), mid = mean(mid), hi = max(hi), .groups = "drop") |>
    mutate(panel = LABS[L])
}

pred <- bind_rows(lapply(names(LABS), function(L)
  bind_rows(sim_band(cf, L), sim_band(llm, L))))

gold <- cf |> filter(seed == min(cf$seed), instrument == "ConfliBERT '21 sliced")
gold_d <- bind_rows(lapply(names(LABS), function(L) {
  gold |>
    group_by(day) |>
    summarise(n = sum(get(paste0("gold_", L)) == 1), .groups = "drop") |>
    complete(day = 1:31, fill = list(n = 0)) |>
    arrange(day) |> mutate(cum = cumsum(n), panel = LABS[L])
}))

COLORS <- c("ConfliBERT '21 sliced" = "#0072B2", "ConfliBERT-Modern" = "#D55E00",
            "GPT-5.6" = "#CC79A7")
LTYPES <- c("ConfliBERT '21 sliced" = "42", "ConfliBERT-Modern" = "solid",
            "GPT-5.6" = "solid")

p <- ggplot(pred, aes(x = day, color = instrument, fill = instrument)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.16, color = NA) +
  geom_line(aes(y = mid, linetype = instrument), linewidth = 0.7) +
  geom_line(data = gold_d, aes(x = day, y = cum), inherit.aes = FALSE,
            color = INK, linewidth = 1.0) +
  facet_wrap(~panel, scales = "free_y") +
  scale_color_manual(values = COLORS) +
  scale_fill_manual(values = COLORS) +
  scale_linetype_manual(values = LTYPES) +
  scale_x_continuous(breaks = c(1, 10, 20, 31)) +
  labs(x = "Day of March 2002", y = "Cumulative documents") +
  theme_paper(grid = "y") +
  theme(legend.key.width = unit(20, "pt"))

save_paper_fig(p, "fig_ipe_timeseries", w = 6.0, h = 3.1)
