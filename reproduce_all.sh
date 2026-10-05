#!/usr/bin/env bash
# Reproduce every figure and table of the paper into outputs/.
set -e
cd "$(dirname "$0")"
export FIG_EXPORT_DIR="$(pwd)/outputs/figures/export"   # vector PDF + 600-dpi PNG copies
python figures/make_m1_diagram.py                                          # Fig. 1
for f in D3 D0; do python figures/analysis_figures.py $f --en; done        # Figs. 2-3
python scripts/mc_catchment.py                                             # Table 4 (single trips)
python scripts/mc_states_extended.py                                       # Table 4 (pooled routes)
python scripts/verification_grid.py                                        # Table 4 (effects), data for Fig. 4
python figures/uniqueness_figure.py                                        # Fig. 4
for f in D1 D2 D4 E3 E4 E4B E6; do python figures/analysis_figures.py $f --en; done   # Figs. 5-11
python scripts/sensitivity_rstar.py                                        # Table 5
for f in F7 F8 F6; do python figures/sensitivity_figures.py $f --en; done  # Figs. 12-13, Table 6
