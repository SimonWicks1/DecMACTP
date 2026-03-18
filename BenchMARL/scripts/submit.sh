#! /bin/bash
#SBATCH --job-name=marl_sweep
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-59          # 6 algorithms * 10 seeds = 50 concurrent tasks
#SBATCH --output=/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs/slurm_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pemb7543@ox.ac.uk

# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
module load Anaconda3
module load Boost
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/engs-goals/pemb7543/DecMACTP
PY="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

SCRIPTS_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/BenchMARL/scripts"
LOG_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------
# Seed and Algorithm Definition
# ---------------------------------------------------------
# A hardcoded array guarantees that train_IPPO.py and train_mappo.py 
# are evaluated on the exact same graph initializations.
SEEDS=(14024 24772 18898 24030 2522 15230 23626 2866 25479 14661)
NUM_SEEDS=${#SEEDS[@]}

ALGOS=(
  "train_IGNARL.py"
  "train_IPPO.py"
  "train_mappo.py"
  "train_IQL.py"
  "train_VDN.py"
  "train_MAGNARL.py"
)
NUM_ALGOS=${#ALGOS[@]}

# ---------------------------------------------------------
# Safe Logging & Bounds Checking
# ---------------------------------------------------------
# Isolate the log writing to Task 0 to prevent 50 concurrent file overwrites
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Using deterministic SEEDS array: [ ${SEEDS[@]} ]"
    echo "${SEEDS[@]}" > "${LOG_DIR}/seeds_used_in_this_run.txt"
fi

MAX_TASKS=$(( NUM_SEEDS * NUM_ALGOS - 1 ))
if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$MAX_TASKS" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    exit 1
fi

# ---------------------------------------------------------
# Task Mapping
# ---------------------------------------------------------
# Map task ID -> (seed, algo)
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_ALGOS ))
ALGO_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_ALGOS ))

SEED="${SEEDS[$SEED_IDX]}"
SCRIPT="${ALGOS[$ALGO_IDX]}"
ALGO_NAME=$(basename "$SCRIPT" .py)
LOG_FILE="${LOG_DIR}/${ALGO_NAME}_seed_${SEED}.log"

echo "Job ID:       $SLURM_ARRAY_JOB_ID"
echo "Array TaskID: $SLURM_ARRAY_TASK_ID"
echo "Algorithm:    $ALGO_NAME"
echo "Seed:         $SEED"
echo "Log:          $LOG_FILE"

# Print the Python executable path
echo "Python Path:  $(which python)"

# Execute the training script
python "${SCRIPTS_DIR}/${SCRIPT}" --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"

echo "Done: ${ALGO_NAME} seed=${SEED}"
