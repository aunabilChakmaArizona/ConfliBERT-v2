# E2: recall by evidence position - the structural cost of a 512-token window.
# Slope chart: each instrument's pooled recall on documents whose gold evidence
# starts within the first 512 tokens vs entirely beyond them.
# Rebuild: Rscript analysis/fig_ipe_strata.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr))

d <- read.csv("analysis/data/ipe_strata.csv") |>
  mutate(
    instrument = case_when(
      config == "ConfliBERT-2021@512" ~ "ConfliBERT-2021",
      config == "ConfliBERT-2021-win@win512" ~ "ConfliBERT-2021",
      config == "ConfliBERT-v2@2048" ~ "ConfliBERT-v2",
      config == "ModernBERT-base@2048" ~ "ModernBERT-base",
      config == "GPT-5.6" ~ "GPT-5.6",
      TRUE ~ NA_character_),
    mode = ifelse(grepl("win", config), "sliced", "single pass"),
    stratum = factor(ifelse(grepl("within", stratum),
                            "Evidence within\nfirst 512 tokens",
                            "All evidence beyond\ntoken 512"),
                     levels = c("Evidence within\nfirst 512 tokens",
                                "All evidence beyond\ntoken 512"))) |>
  filter(!is.na(instrument)) |>
  mutate(instrument = factor(instrument, levels = INSTRUMENT_ORDER))

dl <- d |> filter(stratum == levels(d$stratum)[2]) |>
  mutate(short = paste0(SHORT_NAMES[as.character(instrument)],
                        ifelse(mode == "sliced", " sliced", "")),
         label_y = recall_mean +
           case_when(mode == "sliced" ~ 0.05,
                     instrument == "ConfliBERT-v2" ~ 0.045,
                     instrument == "ModernBERT-base" ~ -0.05,
                     TRUE ~ 0))

p <- ggplot(d, aes(x = stratum, y = recall_mean, color = instrument,
                   linetype = mode, group = config)) +
  geom_line(linewidth = 0.7, alpha = 0.85) +
  scale_linetype_manual(values = c("single pass" = "solid", "sliced" = "42"),
                        guide = "none") +
  geom_linerange(aes(ymin = recall_mean - recall_sd,
                     ymax = recall_mean + recall_sd),
                 linewidth = 0.6, alpha = 0.5) +
  geom_point(size = 2.6) +
  geom_text(data = dl, aes(y = label_y,
                           label = sprintf("%s  %.2f", short, recall_mean)),
            color = INK, hjust = 0, nudge_x = 0.06, size = 2.8,
            family = PAPER_FAMILY) +
  scale_color_instrument(guide = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = c("0", ".25", ".50", ".75", "1")) +
  scale_x_discrete(expand = expansion(mult = c(0.08, 0.55))) +
  labs(x = NULL, y = "Recall") +
  theme_paper(grid = "y")

save_paper_fig(p, "fig_ipe_strata", w = 5.2, h = 3.3)
