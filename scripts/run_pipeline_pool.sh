#!/usr/bin/env bash
# 批量对 pool100.csv 中所有股票依次运行 Stage 2/3/4/5
# 用法: bash scripts/run_pipeline_pool.sh [--end-date 2026-03-21] [--start-stage 2] [--end-stage 5]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_CSV="${SCRIPT_DIR}/data/pool100.csv"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 优先使用项目 venv 的 python
if [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="python"
fi

# 默认参数
END_DATE="2026-03-21"
START_DATE="2023-01-01"
START_STAGE=2
END_STAGE=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --end-date)    END_DATE="$2";    shift 2 ;;
    --start-date)  START_DATE="$2";  shift 2 ;;
    --start-stage) START_STAGE="$2"; shift 2 ;;
    --end-stage)   END_STAGE="$2";   shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 读取股票列表（跳过 header 行 "code"，兼容 bash 3.x）
STOCKS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == "code" ]] && continue
  STOCKS+=("$line")
done < "$POOL_CSV"

TOTAL=${#STOCKS[@]}
echo "=========================================="
echo "Pool: $POOL_CSV  共 $TOTAL 只股票"
echo "Python: $PYTHON"
echo "Stage 范围: $START_STAGE ~ $END_STAGE"
echo "日期范围: $START_DATE ~ $END_DATE"
echo "=========================================="

PASS=0
FAIL=0
FAIL_LIST=()

# macOS 兼容：优先用 gtimeout，否则 fallback 到直接执行（无超时保护）
if command -v gtimeout &>/dev/null; then
  _timeout() { gtimeout "$@"; }
elif command -v timeout &>/dev/null; then
  _timeout() { timeout "$@"; }
else
  _timeout() { shift; "$@"; }  # 无超时命令，直接执行
fi

for i in "${!STOCKS[@]}"; do
  STOCK="${STOCKS[$i]}"
  IDX=$((i + 1))
  echo ""
  echo "---------- [$IDX/$TOTAL] 股票: $STOCK ----------"

  # 断点续跑：若 stage5 summary 已存在则跳过整只股票
  STAGE5_SUMMARY="${SCRIPT_DIR}/../data/pipeline_runs/stage5_single_stock_prediction/${STOCK}_${END_DATE}/summary.md"
  if [[ $END_STAGE -ge 5 && -f "$STAGE5_SUMMARY" ]]; then
    echo "[跳过] $STOCK 已有 Stage-5 结果，续跑跳过"
    PASS=$((PASS + 1))
    continue
  fi

  # Stage 2: 历史因子研究
  if [[ $START_STAGE -le 2 && $END_STAGE -ge 2 ]]; then
    echo "[Stage-2] $STOCK 历史因子研究..."
    if _timeout 300 "$PYTHON" "${SCRIPT_DIR}/run_single_stock_factor_research.py" \
        --stock "$STOCK" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"; then
      echo "[Stage-2] $STOCK 完成"
    else
      echo "[Stage-2] $STOCK 失败，跳过后续 Stage"
      FAIL=$((FAIL + 1))
      FAIL_LIST+=("$STOCK(stage2)")
      continue
    fi
  fi

  # Stage 3: 近期因子挖掘
  if [[ $START_STAGE -le 3 && $END_STAGE -ge 3 ]]; then
    echo "[Stage-3] $STOCK 近期因子挖掘..."
    if _timeout 300 "$PYTHON" "${SCRIPT_DIR}/run_single_stock_recent_factor_mining.py" \
        --stock "$STOCK" \
        --end-date "$END_DATE"; then
      echo "[Stage-3] $STOCK 完成"
    else
      echo "[Stage-3] $STOCK 失败，跳过后续 Stage"
      FAIL=$((FAIL + 1))
      FAIL_LIST+=("$STOCK(stage3)")
      continue
    fi
  fi

  # Stage 4: 策略搜索
  if [[ $START_STAGE -le 4 && $END_STAGE -ge 4 ]]; then
    echo "[Stage-4] $STOCK 策略搜索..."
    if _timeout 600 "$PYTHON" "${SCRIPT_DIR}/run_single_stock_strategy_search.py" \
        --stock "$STOCK" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"; then
      echo "[Stage-4] $STOCK 完成"
    else
      echo "[Stage-4] $STOCK 失败，跳过后续 Stage"
      FAIL=$((FAIL + 1))
      FAIL_LIST+=("$STOCK(stage4)")
      continue
    fi
  fi

  # Stage 5: 当日预测
  if [[ $START_STAGE -le 5 && $END_STAGE -ge 5 ]]; then
    echo "[Stage-5] $STOCK 当日预测..."
    if _timeout 300 "$PYTHON" "${SCRIPT_DIR}/run_single_stock_prediction.py" \
        --stock "$STOCK" \
        --trade-date "$END_DATE"; then
      echo "[Stage-5] $STOCK 完成"
    else
      echo "[Stage-5] $STOCK 失败"
      FAIL=$((FAIL + 1))
      FAIL_LIST+=("$STOCK(stage5)")
      continue
    fi
  fi

  PASS=$((PASS + 1))
done

echo ""
echo "=========================================="
echo "全部完成: 成功 $PASS / 失败 $FAIL / 共 $TOTAL"
if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
  echo "失败列表:"
  for item in "${FAIL_LIST[@]}"; do
    echo "  - $item"
  done
fi
echo "=========================================="
