#!/usr/bin/env bash
set -euo pipefail

# Debug info
echo "Running under shell: $(ps -p $$ -o comm=) (PID $$)"
echo "BASH_VERSION=${BASH_VERSION:-<none>}"

# Ensure running under bash 
if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: please run this script with bash (bash run.sh or ./run.sh)."
  exit 1
fi

# --- Temporarily disable -u when sourcing/activating conda to avoid 'unbound variable' in conda's hook scripts ---
set +u
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda init script not found at ~/anaconda3/etc/profile.d/conda.sh"
  exit 1
fi

# activate environment (still in +u mode)
conda activate marl

# Re-enable -u for the rest of the script
set -u


# Configuration (edit if you need)

echo "Starting training experiments..."
# Train IGNARL
echo "Training IGNARL..."
python /home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/train_IGNARL.py

# Train IPPO
echo "Training IPPO..."
python /home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/train_IPPO.py

# Train MAPPO
echo "Training MAPPO..."
python /home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/train_mappo.py

# Train IQL
echo "Training IQL..."
python /home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/train_IQL.py

# Train VDN
echo "Training VDN..."
python /home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/train_VDN.py

echo "All experiments finished."
