#!/usr/bin/env bash
# Runs every measured part of HW2.5 on one card, in order, then rebuilds the figures and
# METRICS.md. Run this on the reserved workstation, once per GPU.
#
#   ./run_all.sh              # GPU 0, full 20 minute thermal run
#   GPU=1 ./run_all.sh        # second card in the same box
#   SMOKE=1 ./run_all.sh      # about 1 min end to end chain check, not submittable
set -euo pipefail

GPU="${GPU:-0}"
MINUTES="${MINUTES:-20}"
SMOKE="${SMOKE:-0}"
PY="${PY:-python}"
cd "$(dirname "$0")"

# SMOKE shrinks every part, not just Part E, so the pre flight really does take about a
# minute. MINUTES alone only shortens Part E and still runs the full B and D sweeps.
if [ "$SMOKE" = "1" ]; then
  B_ARGS="--sizes 1024 2048 --warmup 2 --iters 5"
  C_ARGS="--elements 16777216 --matmul-n 2048 --warmup 2 --iters 5"
  D_ARGS="--seq-lens 512 1024 2048 --warmup 1 --iters 3 --no-refine"
  MINUTES="${MINUTES_SMOKE:-1}"
  echo "SMOKE MODE: short sweeps and a ${MINUTES} minute thermal run."
  echo "These outputs are a chain check only. Delete them before the real run."
else
  B_ARGS=""; C_ARGS=""; D_ARGS=""
fi

echo "Part A: provenance"
$PY scripts/part_a_provenance.py --index "$GPU"

echo "Part B: precision and achieved throughput"
$PY scripts/part_b_precision.py --index "$GPU" $B_ARGS

echo "Part C: bandwidth bound against compute bound"
$PY scripts/part_c_roofline.py --index "$GPU" $C_ARGS

echo "Part D: the cost of attention"
$PY scripts/part_d_attention.py --index "$GPU" $D_ARGS

echo "Part E: ${MINUTES} minutes of sustained load"
$PY scripts/part_e_thermal.py --index "$GPU" --minutes "$MINUTES"

echo "Figures and METRICS.md"
$PY scripts/make_figures.py
$PY scripts/make_metrics.py

if [ "$SMOKE" = "1" ]; then
  echo
  echo "Smoke run complete. If data/, logs/ and figures/ filled in, the chain works."
  echo "Now clear it before the real run:"
  echo "  rm -f data/*.csv data/*.json logs/*.csv figures/*.png RUN_LOG.txt"
else
  echo "Done. Review RUN_LOG.txt, METRICS.md and figures/ before committing."
fi
