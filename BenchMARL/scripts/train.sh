#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------
echo "Running under shell: $(ps -p $$ -o comm=) (PID $$)"
echo "BASH_VERSION=${BASH_VERSION:-<none>}"

if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: please run this script with bash (bash run.sh or ./run.sh)."
  exit 1
fi

set +u
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda init script not found at ~/anaconda3/etc/profile.d/conda.sh"
  exit 1
fi

conda activate marl
set -u


SCRIPTS_DIR="/home/pemb7543/DeC_MACTP/BenchMARL/scripts"
ACTIVE_RUN_PATH="$(pwd)"
LOG_DIR="${ACTIVE_RUN_PATH}/logs"

# 创建日志目录
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


# 定义需要评估的算法列表
ALGOS=(
  "train_IGNARL.py"
  "train_IPPO.py"
  "train_mappo.py"
  "train_IQL.py"
  "train_VDN.py"
)

echo "Starting rigorous empirical evaluation across ${#SEEDS[@]} seeds..."

# ---------------------------------------------------------
# 主循环
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  echo "=========================================================="
  echo " STARTING EXPERIMENT SUITE FOR SEED: ${seed}"
  echo "=========================================================="
  
  for script in "${ALGOS[@]}"; do
    # 提取无后缀的算法名用于日志命名 (例如: train_IPPO)
    algo_name=$(basename "$script" .py)
    log_file="${LOG_DIR}/${algo_name}_seed_${seed}.log"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running ${algo_name} with seed ${seed}..."
    echo "Logging stdout/stderr to: ${log_file}"
    
    # 核心：通过 --seed 传递参数，并用 tee 进行屏幕和文件双重输出
    python "${SCRIPTS_DIR}/${script}" --seed "${seed}" 2>&1 | tee "${log_file}"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finished ${algo_name} for seed ${seed}."
    echo "----------------------------------------------------------"
  done
done

echo "All 50 experiments finished successfully."