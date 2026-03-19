#!/usr/bin/env bash
# =============================================================
# eval_all.sh
#
# 用法（在 location1 运行）:
#   bash /path/to/location2/eval_all.sh [options]
#
# 必须在 location1（含 graph_data）目录下运行，
# 或通过 --data-root 显式指定图数据根目录。
#
# 选项:
#   --data-root <path>   图数据根目录（默认: 当前目录）
#   --script-dir <path>  Python 脚本目录（默认: 脚本自身所在目录）
#   --algo <name>        只评估某个算法
#   --episodes <n>       评估轮数（默认: 100）
#   --policy-only        只跑策略评估
#   --random-only        只跑随机 baseline
#   --no-gif             禁用 GIF 保存
#   --exact-oracle       使用精确 Held-Karp oracle
#   --dry-run            只打印命令，不执行
# =============================================================

set -euo pipefail

# ─── 自动推断路径 ───────────────────────────────────────────────
SCRIPT_DIR="/home/pemb7543/DeC_MACTP/BenchMARL/scripts"
GRAPH_DATA_ROOT="$(pwd)"

# ─── 默认参数 ───────────────────────────────────────────────────
POLICY_SCRIPT="${SCRIPT_DIR}/test_policy.py"
RANDOM_SCRIPT="${SCRIPT_DIR}/test_random.py"
CONFIG_DIR="${GRAPH_DATA_ROOT}/configs/test_yamls"
LOG_DIR="${GRAPH_DATA_ROOT}/eval_logs"

RANDOM_CONFIG="/home/pemb7543/DeC_MACTP/Train/magnarl_train_magnarlactorgnn__26_03_17-21_46_45_b48119e9/config.pkl"

EPISODES=100
FILTER_ALGO=""
RUN_POLICY=true
RUN_RANDOM=true
DRY_RUN=false
NO_GIF=""
EXACT_ORACLE=""

# ─── Checkpoint 映射（location3）───────────────────────────────
declare -A CHECKPOINTS
# CHECKPOINTS["magnarl"]="/home/pemb7543/DeC_MACTP/Train/magnarl_train_magnarlactorgnn__26_03_17-21_46_45_b48119e9/checkpoints/checkpoint_500000.pt"
# CHECKPOINTS["ignarl"]="/home/pemb7543/DeC_MACTP/Train/ignarl_train_ignarlactorgnn__26_03_07-15_48_05_b2070763/checkpoints/checkpoint_500000.pt"
CHECKPOINTS["ippo"]="/home/pemb7543/DeC_MACTP/Train/ippo_train_graphactorgnn__26_03_07-16_13_57_400b8d5d/checkpoints/checkpoint_500000.pt"
# CHECKPOINTS["mappo"]="/home/pemb7543/DeC_MACTP/Train/mappo_train_graphactorgnn__26_03_06-20_22_00_3920cbbe/checkpoints/checkpoint_500000.pt"
# CHECKPOINTS["iql"]="/home/pemb7543/DeC_MACTP/Train/iql_train_graphqnet__26_03_06-20_46_33_6e77aa5f/checkpoints/checkpoint_500000.pt"
# CHECKPOINTS["vdn"]="/home/pemb7543/DeC_MACTP/Train/vdn_train_graphqnet__26_03_06-22_20_37_172c509d/checkpoints/checkpoint_500000.pt"

# ─── 图配置矩阵 ─────────────────────────────────────────────────
# 格式: "节点数:num_starts:num_goals"
GRAPH_CONFIGS=(
    # "128:4:8"
    # "64:4:8"
    # "256:4:8"
    # "128:2:4"
    # "128:8:16"
    "10:2:4"
    "12:3:4"
)

# ─── 解析命令行参数 ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)    GRAPH_DATA_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
        --script-dir)   SCRIPT_DIR="$(cd "$2" && pwd)";      shift 2 ;;
        --algo)         FILTER_ALGO="$2";                     shift 2 ;;
        --episodes)     EPISODES="$2";                        shift 2 ;;
        --policy-only)  RUN_RANDOM=false;                     shift   ;;
        --random-only)  RUN_POLICY=false;                     shift   ;;
        --dry-run)      DRY_RUN=true;                         shift   ;;
        --no-gif)       NO_GIF="--no-gif";                    shift   ;;
        --exact-oracle) EXACT_ORACLE="--exact-oracle";        shift   ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 参数解析后重新赋值依赖 GRAPH_DATA_ROOT 的路径
CONFIG_DIR="${GRAPH_DATA_ROOT}/configs/test_yamls"
LOG_DIR="${GRAPH_DATA_ROOT}/eval_logs"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

# ─── 路径确认 ───────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  图数据根目录 (data_root): ${GRAPH_DATA_ROOT}"
echo "  Python 脚本目录:          ${SCRIPT_DIR}"
echo "  YAML 配置输出目录:        ${CONFIG_DIR}"
echo "  日志目录:                 ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

# ─── 工具函数 ───────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────────"; }

run_cmd() {
    local log_file="$1"; shift
    if $DRY_RUN; then
        log "[DRY-RUN] $*"
        return 0
    fi
    local t0=$(date +%s)
    if "$@" 2>&1 | tee "$log_file"; then
        log "✅ 完成 (耗时 $(( $(date +%s) - t0 ))s)"
        return 0
    else
        log "❌ 失败 — 详见 $log_file"
        return 1
    fi
}

# ─── YAML 生成 ──────────────────────────────────────────────────
# 关键原则：函数内所有日志必须输出到 stderr (>&2)
# 只有最终的文件路径通过 stdout 返回，供 $() 捕获
generate_yaml() {
    local nodes="$1"
    local starts="$2"
    local goals="$3"
    local graph_dir="graph_data_${nodes}_${starts}_${goals}"
    local yaml_file="${CONFIG_DIR}/test_${nodes}_${starts}_${goals}.yaml"

    # ✅ 所有日志重定向到 stderr，不污染 stdout
    log "生成 YAML: ${yaml_file}" >&2
    log "  graph_dir : ${graph_dir}" >&2
    log "  data_root : ${GRAPH_DATA_ROOT}" >&2
    log "  节点数=${nodes}, starts=${starts}, goals=${goals}" >&2

    cat > "$yaml_file" <<EOF
# 自动生成 by eval_all.sh — 请勿手动编辑
# 配置: nodes=${nodes}, num_starts=${starts}, num_goals=${goals}

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

    # ✅ 唯一的 stdout 输出：yaml 文件路径
    echo "$yaml_file"
}

# ─── 主循环 ─────────────────────────────────────────────────────
total=0; success=0; failed=0
t_start=$(date +%s)

for cfg in "${GRAPH_CONFIGS[@]}"; do
    IFS=':' read -r NODES STARTS GOALS <<< "$cfg"
    GRAPH_DIR="graph_data_${NODES}_${STARTS}_${GOALS}"

    # ✅ generate_yaml 的 stdout 现在只含纯路径字符串
    YAML_FILE=$(generate_yaml "$NODES" "$STARTS" "$GOALS")

    hr
    log "图配置:    $GRAPH_DIR"
    log "YAML:      $YAML_FILE"
    log "data_root: ${GRAPH_DATA_ROOT}"

    # ── 随机 Baseline ─────────────────────────────────────────
    if $RUN_RANDOM; then
        total=$((total + 1))
        LOG_FILE="${LOG_DIR}/random_${GRAPH_DIR}_$(date '+%Y%m%d_%H%M%S').log"
        log "▶ 随机 Baseline | $GRAPH_DIR"

        if run_cmd "$LOG_FILE" \
            python "$RANDOM_SCRIPT" \
                --config    "$RANDOM_CONFIG" \
                --graph     "$GRAPH_DIR" \
                --yaml-path "$YAML_FILE" \
                --episodes  "$EPISODES" \
                $NO_GIF $EXACT_ORACLE
        then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    fi

    # ── 策略评估 ──────────────────────────────────────────────
    if $RUN_POLICY; then
        for algo in "${!CHECKPOINTS[@]}"; do
            [[ -n "$FILTER_ALGO" && "$algo" != "$FILTER_ALGO" ]] && continue

            total=$((total + 1))
            LOG_FILE="${LOG_DIR}/${algo}_${GRAPH_DIR}_$(date '+%Y%m%d_%H%M%S').log"
            log "▶ 策略评估 | ${algo^^} | $GRAPH_DIR"

            if run_cmd "$LOG_FILE" \
                python "$POLICY_SCRIPT" \
                    --checkpoints "$algo" \
                    --graph       "$GRAPH_DIR" \
                    --yaml-path   "$YAML_FILE" \
                    --episodes    "$EPISODES" \
                    $NO_GIF $EXACT_ORACLE
            then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done
    fi
done

# ─── 汇总 ───────────────────────────────────────────────────────
hr
log "全部完成 | 总: $total | ✅ $success | ❌ $failed | 耗时: $(( $(date +%s) - t_start ))s"
hr