#!/usr/bin/env bash
#SBATCH --job-name=marl_eval
#SBATCH --partition=short
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pemb7543@ox.ac.uk
#SBATCH --output=/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs/eval_%A_%a.out
#SBATCH --array=0-60

set -euo pipefail   # ✅ 新增：任何错误立即终止，防止静默失败

# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
module load Anaconda3
module load Boost
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/engs-goals/pemb7543/DecMACTP
PY="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ---------------------------------------------------------
# 路径配置
# ---------------------------------------------------------
SCRIPT_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/BenchMARL/scripts"
GRAPH_DATA_ROOT="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/Test_eval"
LOG_DIR="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs"
CONFIG_DIR="${GRAPH_DATA_ROOT}/configs/test_yamls"

POLICY_SCRIPT="${SCRIPT_DIR}/test_policy.py"
RANDOM_SCRIPT="${SCRIPT_DIR}/test_random.py"

# 注意：RANDOM_CONFIG 依赖 magnarl 第一个 seed 的 config.pkl
# 仅用于初始化测试环境，不加载策略权重
RANDOM_CONFIG="/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__5a706552_26_03_18-06_58_59/config.pkl"

EPISODES=100
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

# ✅ 新增：验证关键路径存在
if [ ! -f "$RANDOM_CONFIG" ]; then
    echo "Error: RANDOM_CONFIG 不存在: $RANDOM_CONFIG"
    exit 1
fi
if [ ! -f "$POLICY_SCRIPT" ]; then
    echo "Error: POLICY_SCRIPT 不存在: $POLICY_SCRIPT"
    exit 1
fi
if [ ! -f "$RANDOM_SCRIPT" ]; then
    echo "Error: RANDOM_SCRIPT 不存在: $RANDOM_SCRIPT"
    exit 1
fi

# ---------------------------------------------------------
# 图配置矩阵（每个 array 会遍历全部 15 个配置）
# 格式: "节点数:num_starts:num_goals"
# ---------------------------------------------------------
GRAPH_CONFIGS=(
    "64:6:6"
    "64:6:9"
    "64:6:12"
    "128:6:6"
    "128:6:9"
    "128:6:12"
    "256:6:6"
    "256:6:9"
    "256:6:12"
    "512:6:6"
    "512:6:9"
    "512:6:12"
    # "1024:6:6"
    # "1024:6:9"
    # "1024:6:12"
)
NUM_GRAPHS=${#GRAPH_CONFIGS[@]}

# ---------------------------------------------------------
# 算法列表
# ---------------------------------------------------------
ALGOS=(
    "magnarl"
    "ignarl"
    "ippo"
    "mappo"
    "iql"
    "vdn"
)
NUM_ALGOS=${#ALGOS[@]}
NUM_SEEDS=10

# ---------------------------------------------------------
# 与 submit.sh 严格对齐的 seed 列表
# ---------------------------------------------------------
SEEDS=(14024 24772 18898 24030 2522 15230 23626 2866 25479 14661)

# ---------------------------------------------------------
# Checkpoint 路径（每个算法 10 个 seed，与 SEEDS 数组索引严格对齐）
# ---------------------------------------------------------
CHECKPOINTS_MAGNARL=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__5a706552_26_03_18-06_58_59/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__6ee48be6_26_03_19-12_59_36/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__6f91eae8_26_03_19-11_56_15/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__30a6bbd4_26_03_19-07_27_34/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__508a7bb2_26_03_18-12_04_42/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__1014eb9c_26_03_19-13_11_12/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__13401175_26_03_19-04_02_16/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__bc4addd6_26_03_18-21_52_54/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__c2d3f354_26_03_19-10_53_59/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/magnarl_train_magnarlactorgnn__fcce5c95_26_03_19-01_14_27/checkpoints/checkpoint_500000.pt"
)

CHECKPOINTS_IGNARL=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__1ffb4588_26_03_18-21_59_36/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__4d81af83_26_03_19-09_15_58/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__16d73fcb_26_03_19-12_59_37/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__456adfe8_26_03_19-04_29_22/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__67530bdc_26_03_18-08_47_33/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__907352f1_26_03_19-11_01_27/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__4958920c_26_03_18-12_12_15/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__93793409_26_03_19-02_35_46/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__a481956a_26_03_18-00_36_28/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ignarl_train_ignarlactorgnn__d5cc64f3_26_03_19-11_59_46/checkpoints/checkpoint_500000.pt"
)

CHECKPOINTS_IPPO=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__0db5e17e_26_03_18-10_26_03/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__2ed48486_26_03_19-09_22_47/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__9cf2d2dd_26_03_19-02_35_46/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__9181a0c0_26_03_19-11_59_46/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__393958d1_26_03_18-14_02_46/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__33120768_26_03_18-02_19_28/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__a8bef8f2_26_03_19-12_59_37/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__b00024ee_26_03_19-11_21_43/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__bb19123c_26_03_19-04_29_22/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/ippo_train_graphactorgnn__fcdb0ab2_26_03_18-22_07_13/checkpoints/checkpoint_500000.pt"
)

CHECKPOINTS_MAPPO=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__2b005dc7_26_03_19-05_52_43/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__04f8dce1_26_03_18-04_04_28/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__5b9ae424_26_03_19-11_42_52/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__203946e4_26_03_19-13_05_56/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__af3c0a72_26_03_19-09_50_17/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__b1b99a46_26_03_18-10_27_17/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__b7ce115b_26_03_18-20_21_16/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__cf0a1f60_26_03_18-23_42_45/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__d3120091_26_03_19-03_21_55/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/mappo_train_graphactorgnn__069a5bce_26_03_19-12_03_26/checkpoints/checkpoint_500000.pt"
)

CHECKPOINTS_IQL=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__4fb47c9a_26_03_19-07_03_26/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__8a578e46_26_03_19-13_05_56/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__19e1d795_26_03_18-10_37_08/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__93e959d6_26_03_19-12_14_36/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__07180f0e_26_03_18-05_44_17/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__15275e9c_26_03_18-23_50_57/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__715604ef_26_03_19-11_42_52/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__a04f13db_26_03_19-03_27_03/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__a3500787_26_03_19-10_36_03/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/iql_train_graphqnet__ea07c1f2_26_03_18-21_23_38/checkpoints/checkpoint_500000.pt"
)

CHECKPOINTS_VDN=(
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__3abf2a5a_26_03_18-11_33_58/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__4d9c8d2a_26_03_19-00_45_53/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__7f742737_26_03_18-21_32_37/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__8e773529_26_03_19-13_12_06/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__43bd4c92_26_03_19-07_06_01/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__2328f091_26_03_19-03_47_12/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__483832e9_26_03_19-12_19_00/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__b1003531_26_03_19-11_56_15/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__cc7ee3b1_26_03_19-10_38_58/checkpoints/checkpoint_500000.pt"
    "/data/engs-goals/pemb7543/DecMACTP/DecMACTP/logs_v2/Train_Server/vdn_train_graphqnet__f2e7ce32_26_03_18-06_58_58/checkpoints/checkpoint_500000.pt"
)

# ---------------------------------------------------------
# YAML 生成函数
# 所有日志 → stderr；文件路径 → stdout
# ---------------------------------------------------------
generate_yaml() {
    local nodes="$1" starts="$2" goals="$3"
    local graph_dir="graph_data_${nodes}_${starts}_${goals}"
    local yaml_file="${CONFIG_DIR}/test_${nodes}_${starts}_${goals}.yaml"

    cat > "$yaml_file" <<EOF
# 自动生成 by test_submit.sh — 请勿手动编辑
max_nodes: null
num_agents: null
seed: 0

graph_generator:
  class_path: "gnarl.envs.generate.graph_generator.RandomSetGraphGenerator"
  data:
    algorithm: "mactp"
    data_root: "${GRAPH_DATA_ROOT}"
    graph_dir: "${graph_dir}"
    split: "test"
    graph_generator: "er"
    graph_generator_kwargs:
      p_range: [1.0, 2.0]
    node_samples:
      ${nodes}: 10000
    num_starts: ${starts}
    num_goals: ${goals}
    seed: 18
  kwargs: {}
EOF
    echo "$yaml_file"
}

# ---------------------------------------------------------
# 边界检查
# 总 array 数 = 1(random) + NUM_ALGOS*NUM_SEEDS = 1 + 60 = 61
# --array=0-60
# ---------------------------------------------------------
MAX_TASK_ID=$(( 1 + NUM_ALGOS * NUM_SEEDS - 1 ))

echo "════════════════════════════════════════════════════════════"
echo "  Task ID        : ${SLURM_ARRAY_TASK_ID}"
echo "  最大 Task ID   : ${MAX_TASK_ID}"
echo "  图配置数量     : ${NUM_GRAPHS}（每个 array 内部循环全部）"
echo "  GRAPH_DATA_ROOT: ${GRAPH_DATA_ROOT}"
echo "════════════════════════════════════════════════════════════"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ] || \
   [ "$SLURM_ARRAY_TASK_ID" -gt "$MAX_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID out of bounds [0, $MAX_TASK_ID]"
    exit 1
fi

# ---------------------------------------------------------
# Phase 1: 随机 Baseline（仅 task_id = 0）
# ---------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Phase: 随机 Baseline（循环所有 ${NUM_GRAPHS} 个图配置）"

    for cfg in "${GRAPH_CONFIGS[@]}"; do
        IFS=':' read -r NODES STARTS GOALS <<< "$cfg"
        GRAPH_DIR="graph_data_${NODES}_${STARTS}_${GOALS}"
        YAML_FILE=$(generate_yaml "$NODES" "$STARTS" "$GOALS")

        echo "────────────────────────────────────"
        echo "  图配置: $GRAPH_DIR"
        echo "  YAML:   $YAML_FILE"

        python "$RANDOM_SCRIPT" \
            --config    "$RANDOM_CONFIG" \
            --graph     "$GRAPH_DIR" \
            --yaml-path "$YAML_FILE" \
            --episodes  "$EPISODES" \
            --no-gif

        echo "  ✅ Done: Random | $GRAPH_DIR"
    done

    echo "随机 Baseline 全部完成"
    exit 0
fi

# ---------------------------------------------------------
# Phase 2: 策略评估（task_id 1 ~ 60）
# 映射:
#   policy_task_id = task_id - 1
#   algo_idx = policy_task_id / NUM_SEEDS
#   seed_idx = policy_task_id % NUM_SEEDS
# ---------------------------------------------------------
POLICY_TASK_ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
ALGO_IDX=$(( POLICY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( POLICY_TASK_ID % NUM_SEEDS ))

ALGO="${ALGOS[$ALGO_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

case "$ALGO" in
    "magnarl") CKPT="${CHECKPOINTS_MAGNARL[$SEED_IDX]}" ;;
    "ignarl")  CKPT="${CHECKPOINTS_IGNARL[$SEED_IDX]}"  ;;
    "ippo")    CKPT="${CHECKPOINTS_IPPO[$SEED_IDX]}"    ;;
    "mappo")   CKPT="${CHECKPOINTS_MAPPO[$SEED_IDX]}"   ;;
    "iql")     CKPT="${CHECKPOINTS_IQL[$SEED_IDX]}"     ;;
    "vdn")     CKPT="${CHECKPOINTS_VDN[$SEED_IDX]}"     ;;
    *)
        echo "Error: 未知算法 $ALGO"
        exit 1
        ;;
esac

echo "Phase:      策略评估（循环所有 ${NUM_GRAPHS} 个图配置）"
echo "算法:       ${ALGO} (algo_idx=${ALGO_IDX})"
echo "Seed:       ${SEED} (seed_idx=${SEED_IDX})"
echo "Checkpoint: ${CKPT}"

if [ ! -f "$CKPT" ]; then
    echo "Error: Checkpoint 不存在: $CKPT"
    exit 1
fi

for cfg in "${GRAPH_CONFIGS[@]}"; do
    IFS=':' read -r NODES STARTS GOALS <<< "$cfg"
    GRAPH_DIR="graph_data_${NODES}_${STARTS}_${GOALS}"
    YAML_FILE=$(generate_yaml "$NODES" "$STARTS" "$GOALS")

    echo "────────────────────────────────────"
    echo "  图配置: $GRAPH_DIR"
    echo "  YAML:   $YAML_FILE"

    python "$POLICY_SCRIPT" \
        --checkpoints "$CKPT" \
        --graph       "$GRAPH_DIR" \
        --yaml-path   "$YAML_FILE" \
        --episodes    "$EPISODES" \
        --no-gif

    echo "  ✅ Done: ${ALGO} | seed=${SEED} | $GRAPH_DIR"
done

echo "策略评估全部完成: ${ALGO} | seed=${SEED}"