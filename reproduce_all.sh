#!/usr/bin/env bash
# Reproduce every figure and table of the paper into outputs/.
set -e
cd "$(dirname "$0")"
export FIG_EXPORT_DIR="$(pwd)/outputs/figures/export"   # vector PDF + 600-dpi PNG copies
python figures/make_m1_diagram.py                                     # Fig. 1
for f in D3 D0 D1 D2 D4 E3 E4 E4B E6; do python figures/analysis_figures.py $f --en; done   # Figs. 2-10
for f in F7 F8 F6; do python figures/sensitivity_figures.py $f --en; done                 # Figs. 11-12, Table 6
python scripts/mc_catchment.py                                        # Table 4
python scripts/uniqueness_check.py                                    # Section 3.4
python scripts/sensitivity_rstar.py                                   # Table 5
