# Shared plotting theme + palette for ConfliBERT-v2 figures.
# Okabe-Ito colorblind-safe palette (canonical for publication figures).
suppressMessages({library(ggplot2)})

# Okabe-Ito
OI <- c(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
        yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")

# Two-model comparison: ModernBERT vs ConfliBERT (well-separated, CVD-safe)
MODEL_COLORS <- c("ModernBERT" = unname(OI["blue"]),
                  "ConfliBERT" = unname(OI["vermillion"]),
                  "ConfliBERT-v2" = unname(OI["green"]))

# Source palette (5 sources) from Okabe-Ito (skip yellow: low contrast on white)
SOURCE_COLORS <- c("News" = unname(OI["blue"]), "Gigaword" = unname(OI["vermillion"]),
                   "Organization" = unname(OI["green"]), "UTDstory" = unname(OI["orange"]),
                   "Wikipedia" = unname(OI["purple"]))

theme_cb <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      axis.title = element_text(color = "grey20"),
      axis.text = element_text(color = "grey30"),
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(color = "grey35", size = base_size),
      plot.caption = element_text(color = "grey50", size = base_size - 2),
      legend.position = "top",
      legend.title = element_blank(),
      plot.margin = margin(10, 14, 10, 10)
    )
}

save_fig <- function(p, name, w = 7, h = 4.4) {
  dir <- file.path(dirname(sys.frame(1)$ofile %||% "."), "figures")
  # fallback: fixed path
  outdir <- "F:/Confli_2/Corpus/conflibert-v2/analysis/figures"
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(outdir, paste0(name, ".pdf")), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(outdir, paste0(name, ".png")), p, width = w, height = h, dpi = 200)
  cat("wrote", name, ".pdf/.png\n")
}
`%||%` <- function(a, b) if (is.null(a)) b else a
