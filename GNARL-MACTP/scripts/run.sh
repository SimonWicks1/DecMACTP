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
conda activate MAGNARL_env

# Re-enable -u for the rest of the script
set -u


# Configuration (edit if you need)
NUM_RUNS=5
COOLDOWN=5
GROUP1_NAME="MACTP_V3_2A2G_5_seeds"
GROUP2_NAME="MACTP_V3_3A3G_5_seeds"

SCRIPT="/home/pemb7543/MAGNARL/GNARL-MACTP/scripts/train_mactp.py"
CONFIG1="/home/pemb7543/MAGNARL/GNARL-MACTP/configs/mactp.yaml"
CONFIG2="/home/pemb7543/MAGNARL/GNARL-MACTP/configs/mactp2.yaml"

# Helper: generate a 32-bit unsigned seed from /dev/urandom
generate_seed() {
  # od prints an unsigned 32-bit integer; fallback to RANDOM if od not available
  if command -v od >/dev/null 2>&1; then
    # strip whitespace
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
  else
    echo $((1 + RANDOM % 100000))
  fi
}

run_group () {
  local script="$1"; shift
  local config="$1"; shift
  local group="$1"; shift

  echo "======================================================"
  echo " Experiment group: ${group}"
  echo " Script: ${script}"
  echo " Config: ${config}"
  echo " Runs: ${NUM_RUNS}"
  echo "======================================================"

  for i in $(seq 1 "$NUM_RUNS"); do
    SEED=$(generate_seed)
    echo "------------------------------------------------------"
    echo " RUN ${i}/${NUM_RUNS}  |  SEED=${SEED}"
    echo "------------------------------------------------------"

    python3 "$script" --config "$config" --seed "$SEED" --group "$group" --wandb

    echo "Finished run ${i}. Sleeping ${COOLDOWN}s..."
    sleep "${COOLDOWN}"
  done
}

# Run both groups
run_group "$SCRIPT" "$CONFIG1" "$GROUP1_NAME"
run_group "$SCRIPT" "$CONFIG2" "$GROUP2_NAME"

echo "All experiments finished."
