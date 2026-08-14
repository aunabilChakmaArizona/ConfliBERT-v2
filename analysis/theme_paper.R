# Publication theme + fixed instrument palette for the Political Analysis paper.
# Source this from every figure script: source("analysis/theme_paper.R")
#
# Design contract (validated 2026-08-13 with the palette validator):
#  - Categorical hues are Okabe-Ito, assigned to INSTRUMENTS in a FIXED order and
#    never cycled. Canonical legend/factor order: ConfliBERT-2021 (blue),
#    ModernBERT-base (green), ConfliBERT-v2 (vermillion), GPT-5.6 (purple).
#    This order passes all CVD adjacent-pair checks (worst deutan dE 11.0).
#  - The purple slot is 2.98:1 against white, just under the 3:1 relief line:
#    every figure that uses it MUST carry direct labels (we direct-label all
#    <=4-series figures anyway).
#  - The human gold standard is NOT a series color: it is drawn in near-black ink
#    (GOLD_INK) as the reference the instruments are judged against.
#  - Context window (512 vs 2048) is a secondary encoding on the same hue:
#    shape 16/21 for points, linetype solid/42 for lines. Color follows the
#    instrument, never the configuration.
#  - Text (labels, legends, values) wears ink colors, never series colors.
#  - Typeface matches the paper body (Palatino / newpx): "Palatino Linotype" via
#    cairo devices; save with save_paper_fig() so fonts embed in the PDF.

suppressMessages(library(ggplot2))

# ---- inks and surfaces -------------------------------------------------------
INK       <- "#1a1a1a"   # primary text / gold-standard reference
INK_MUTED <- "#5a5a5a"   # secondary text
INK_FAINT <- "#8a8a8a"   # captions, source notes
GRID_COL  <- "#e4e4e0"   # recessive hairline grid
GOLD_INK  <- INK

# ---- fixed instrument palette (Okabe-Ito; order = canonical legend order) ----
INSTRUMENT_COLORS <- c(
  "ConfliBERT-2021" = "#0072B2",  # blue       - original BERT-based domain model
  "ModernBERT-base" = "#009E73",  # green      - generic modern architecture
  "ConfliBERT-v2"   = "#D55E00",  # vermillion - modern domain model (displays as ConfliBERT-Modern)
  "GPT-5.6"         = "#CC79A7"   # purple     - zero-shot LLM baseline
)
INSTRUMENT_ORDER <- names(INSTRUMENT_COLORS)

# Secondary encoding for context window on the same hue
CTX_SHAPES    <- c("512" = 16, "2048" = 21)       # filled vs open circle
CTX_LINETYPES <- c("512" = "solid", "2048" = "42")

scale_color_instrument <- function(...)
  scale_color_manual(values = INSTRUMENT_COLORS, breaks = INSTRUMENT_ORDER, ...)
scale_fill_instrument <- function(...)
  scale_fill_manual(values = INSTRUMENT_COLORS, breaks = INSTRUMENT_ORDER, ...)

# Short display names for in-plot annotation (endpoint tags on slope charts).
# Figure files carry NO titles/subtitles/captions: that prose belongs in the
# LaTeX \caption or the slide text, so plots stay uncluttered.
# NOTE: "ConfliBERT-v2" is the internal run ID in the logged CSVs; the paper's
# display name is "ConfliBERT-Modern" (no "v2" branding in the PA paper).
SHORT_NAMES <- c("ConfliBERT-2021" = "ConfliBERT '21",
                 "ModernBERT-base" = "ModernBERT",
                 "ConfliBERT-v2"   = "ConfliBERT-Modern",
                 "GPT-5.6"         = "GPT-5.6")

# ---- theme -------------------------------------------------------------------
PAPER_FAMILY <- if (.Platform$OS.type == "windows") "Palatino Linotype" else "Palatino"

theme_paper <- function(base_size = 9.5, grid = c("y", "x", "both", "none")) {
  grid <- match.arg(grid)
  gl <- element_line(color = GRID_COL, linewidth = 0.3)
  th <- theme_minimal(base_size = base_size, base_family = PAPER_FAMILY) +
    theme(
      text = element_text(color = INK),
      panel.grid.minor = element_blank(),
      panel.grid.major = gl,
      axis.title = element_text(color = INK_MUTED, size = base_size),
      axis.title.x = element_text(margin = margin(t = 6)),
      axis.title.y = element_text(margin = margin(r = 6)),
      axis.text = element_text(color = INK_MUTED, size = base_size - 0.5),
      axis.ticks = element_blank(),
      plot.title = element_text(face = "bold", size = base_size + 1.5,
                                margin = margin(b = 2)),
      plot.subtitle = element_text(color = INK_MUTED, size = base_size,
                                   margin = margin(b = 8)),
      plot.caption = element_text(color = INK_FAINT, size = base_size - 1.5,
                                  hjust = 0, margin = margin(t = 8)),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      legend.text = element_text(color = INK, size = base_size - 0.5),
      legend.key.height = unit(base_size, "pt"),
      legend.margin = margin(0, 0, 2, 0),
      strip.text = element_text(face = "bold", color = INK, size = base_size,
                                hjust = 0, margin = margin(b = 4)),
      panel.spacing = unit(14, "pt"),
      plot.margin = margin(8, 12, 8, 8)
    )
  if (grid == "y")    th <- th + theme(panel.grid.major.x = element_blank())
  if (grid == "x")    th <- th + theme(panel.grid.major.y = element_blank())
  if (grid == "none") th <- th + theme(panel.grid.major = element_blank())
  th
}

# Direct-label helper: last resort positions are figure-specific; this only
# standardizes the look (ink text, never series-colored).
label_ink <- function(size = 3.0)
  list(color = INK, family = PAPER_FAMILY, size = size)

# ---- output ------------------------------------------------------------------
# Text width at 11pt/1.25in margins is 6.0in; default figures fill it.
save_paper_fig <- function(p, name, w = 6.0, h = 3.6) {
  outdir <- "F:/Confli_2/Corpus/conflibert-v2/analysis/figures"
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(outdir, paste0(name, ".pdf")), p, width = w, height = h,
         device = cairo_pdf)
  ggsave(file.path(outdir, paste0(name, ".png")), p, width = w, height = h,
         dpi = 300, type = "cairo-png")
  cat("wrote", file.path(outdir, name), ".pdf/.png\n")
}
