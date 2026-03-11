#! /bin/bash
#SBATCH --job-name=marl_sweep
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs/slurm_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pemb7543@ox.ac.uk


# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
module load Anaconda3
module load Boost
source activate /data/engs-goals/pemb7543/DecMACTP

SCRIPTS_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/BenchMARL/scripts"
LOG_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs"
mkdir -p "$LOG_DIR"


NUM_SEEDS=10
SEEDS=()

echo "Generating ${NUM_SEEDS} random seeds for the experiments..."
for ((i=1; i<=NUM_SEEDS; i++)); do
    # $RANDOM 会生成一个 0 到 32767 之间的伪随机整数
    # 为了避免极小概率的重复，可以在前面加一个基础值或者组合
    current_seed=$RANDOM
    SEEDS+=($current_seed)
done
echo "Generated SEEDS array: [ ${SEEDS[@]} ]"
# 将生成的种子列表保存到文件中，方便后续对齐不同算法的性能
echo "${SEEDS[@]}" > "${LOG_DIR}/seeds_used_in_this_run.txt"

ALGOS=(
  "train_IGNARL.py"
  "train_IPPO.py"
  "train_mappo.py"
  "train_IQL.py"
  "train_VDN.py"
)

NUM_ALGOS=${#ALGOS[@]}

# Map task ID → (seed, algo)
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_ALGOS ))
ALGO_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_ALGOS ))

SEED="${SEEDS[$SEED_IDX]}"
SCRIPT="${ALGOS[$ALGO_IDX]}"
ALGO_NAME=$(basename "$SCRIPT" .py)
LOG_FILE="${LOG_DIR}/${ALGO_NAME}_seed_${SEED}.log"

echo "Job ID:       $SLURM_JOB_ID"
echo "Array TaskID: $SLURM_ARRAY_TASK_ID"
echo "Algorithm:    $ALGO_NAME"
echo "Seed:         $SEED"
echo "Log:          $LOG_FILE"

python "${SCRIPTS_DIR}/${SCRIPT}" --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"

echo "Done: ${ALGO_NAME} seed=${SEED}"
