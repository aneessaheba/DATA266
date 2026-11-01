#!/usr/bin/env bash
# Runs HW4.a and HW4.5 on one card, then rebuilds METRICS.md and figures.
#   CARD=rtx5090 bash run_all.sh               everything on the lab 5090
#   CARD=colab_t4 bash run_all.sh              everything on Colab
#   CARD=kaggle_t4x2 STAGES="prov c" bash run_all.sh   optional Part C on a two GPU box
#   SMOKE=1 CARD=mac bash run_all.sh            chain check, delete results afterwards
set -euo pipefail
cd "$(dirname "$0")"
export CARD="${CARD:?set CARD, e.g. rtx5090 or colab_t4}"
STAGES="${STAGES:-prov hw4a serve a b d}"   # Part C is optional, add c to run it
mkdir -p provenance results figures
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has prov && [ "$NGPU" -gt 0 ]; then
  nvidia-smi -L > "provenance/nvidia-smi-L_${CARD}.txt"
  nvidia-smi -q > "provenance/nvidia-smi-q_${CARD}.txt"
  nvidia-smi topo -m > "provenance/nvidia-smi-topo_${CARD}.txt" || true
fi
has hw4a && python scripts/hw4a.py
if has serve; then
  python scripts/serve.py hf
  [ "$NGPU" -gt 0 ] && python scripts/serve.py vllm hw4
fi
has a && python scripts/part_a.py
has b && python scripts/part_b.py
if has c; then
  torchrun --master_addr=127.0.0.1 --master_port=29511 --nproc_per_node=1 scripts/part_c_ddp.py
  [ "$NGPU" -ge 2 ] && torchrun --master_addr=127.0.0.1 --master_port=29511 --nproc_per_node=2 scripts/part_c_ddp.py
fi
if has d && [ "$NGPU" -gt 0 ]; then
  python scripts/serve.py vllm partd
  python scripts/part_d_kv.py
fi
python scripts/report.py
echo "Done for $CARD. Check RUN_LOG.txt, METRICS.md, figures/."
