# CrisisWatch corpus overview: parsed entries per year by trend verdict.
# Rebuild: Rscript analysis/fig_crisiswatch_overview.R (from repo root)
source("analysis/theme_paper.R")
suppressMessages(library(dplyr))

d <- read.csv("analysis/data/crisiswatch_overview.csv") |>
  mutate(status = factor(status,
                         levels = c("unchanged", "deteriorated", "improved")))

STATUS_COLORS <- c(unchanged = "#b8b8b4", deteriorated = "#D55E00",
                   improved = "#009E73")

p <- ggplot(d, aes(x = factor(year), y = n, fill = status)) +
  geom_col(width = 0.68) +
  scale_fill_manual(values = STATUS_COLORS) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.03))) +
  labs(x = NULL, y = "Entries") +
  theme_paper(grid = "y")

save_paper_fig(p, "fig_crisiswatch_overview", w = 5.6, h = 3.3)
