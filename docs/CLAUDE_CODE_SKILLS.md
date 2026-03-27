# FactorHub Claude Code Skills 指南

本文档定义了在Claude Code中使用FactorHub的7个核心Skills（命令）。

**API基础URL**: `http://localhost:8000/api`
**默认起始日期**: 2023-01-01
**默认结束日期**: 当前日期

---

## 🎯 7个核心Skills概览

| # | Skill | 功能 | 使用频率 | 延迟 |
|---|-------|------|--------|------|
| 1 | `factor_analyze` | 因子分析 (IC/IR/稳定性) | 高 | 10-60s |
| 2 | `backtest_factor` | 因子回测 | 高 | 5-30s |
| 3 | `compare_strategies` | 策略对比 | 中 | 20-90s |
| 4 | `mine_factors` | 因子自动挖掘 | 低 | 2-10min |
| 5 | `validate_formula` | 公式验证 | 中 | <1s |
| 6 | `get_stock_data` | 获取股票数据 | 高 | <1s |
| 7 | `portfolio_optimize` | 组合权重优化 | 低 | 30-120s |

---

## Skill 1: `factor_analyze`

**功能**: 分析单个因子的IC/IR、稳定性、衰减等特性

### 命令语法
```bash
claude> /factor_analyze STOCK_CODE EXPRESSION [--start START_DATE] [--end END_DATE] [--horizons H1,H2,H3]
```

### 参数说明
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `STOCK_CODE` | string | ✅ | - | 股票代码，如 `000001` |
| `EXPRESSION` | string | ✅ | - | 因子表达式，如 `RSI(close, 14)` |
| `--start` | date | - | 2023-01-01 | 开始日期 YYYY-MM-DD |
| `--end` | date | - | 今天 | 结束日期 YYYY-MM-DD |
| `--horizons` | int[] | - | [1,5,10] | 预测周期列表 |

### 使用示例

#### 示例1: 快速分析RSI因子
```bash
claude> /factor_analyze 000001 "RSI(close, 14)"

输出:
因子分析报告: 000001 - RSI(close, 14)
├─ 数据覆盖: 2023-01-01 至 2025-03-27 (631个交易日)
├─ 因子统计:
│  ├─ 平均值: 52.3
│  ├─ 标准差: 18.5
│  ├─ 范围: [15.2, 89.4]
├─ IC/IR分析:
│  ├─ IC均值: 0.032 ✅ (>0.03)
│  ├─ IC标准差: 0.12
│  ├─ IR比率: 0.47 ⚠️ (<0.5)
│  ├─ IC正比例: 62%
├─ 稳定性评级: B (中等)
├─ 衰减分析:
│  ├─ 1日IC: 0.045
│  ├─ 5日IC: 0.038 ⬇️
│  ├─ 10日IC: 0.028 ⬇️
│  └─ 20日IC: 0.015 ⬇️
└─ 综合评估: 因子有效但衰减较快，建议短期操作
```

#### 示例2: 自定义日期和周期
```bash
claude> /factor_analyze 600519 "close/SMA(close, 20)" --start 2024-01-01 --end 2024-12-31 --horizons 1,3,5,10

(返回2024年全年的分析结果和4个衰减周期)
```

### 后台API调用

```bash
# 1. 获取数据
curl -X GET "http://localhost:8000/api/data/stock/000001?start_date=2023-01-01&end_date=2025-03-27"

# 2. 验证公式
curl -X POST "http://localhost:8000/api/factors/validate" \
  -H "Content-Type: application/json" \
  -d '{"code": "RSI(close, 14)", "formula_type": "expression"}'

# 3. 计算因子
curl -X POST "http://localhost:8000/api/analysis/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_name": "RSI(close, 14)",
    "stock_codes": ["000001"],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27"
  }'

# 4. IC/IR分析
curl -X POST "http://localhost:8000/api/analysis/ic" \
  -H "Content-Type: application/json" \
  -d '{
    "factors": ["RSI(close, 14)"],
    "stock_codes": ["000001"],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27"
  }'

# 5. 稳定性测试
curl -X POST "http://localhost:8000/api/analysis/stability" \
  -H "Content-Type: application/json" \
  -d '{ ... (同上) ... }'

# 6. 衰减分析
curl -X POST "http://localhost:8000/api/analysis/decay" \
  -H "Content-Type: application/json" \
  -d '{ ... (同上) ... }'
```

---

## Skill 2: `backtest_factor`

**功能**: 对单个或多个因子进行回测，返回详细的性能指标和权益曲线

### 命令语法
```bash
claude> /backtest_factor STOCK_CODE EXPRESSION [EXPRESSION2] [--start START_DATE] [--end END_DATE] [--percentile P] [--direction long|short] [--capital AMOUNT]
```

### 参数说明
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `STOCK_CODE` | string | ✅ | - | 股票代码 |
| `EXPRESSION` | string | ✅ | - | 因子表达式，可多个 |
| `--start` | date | - | 2023-01-01 | 回测开始日期 |
| `--end` | date | - | 今天 | 回测结束日期 |
| `--percentile` | int | - | 50 | 因子值阈值 (1-99) |
| `--direction` | string | - | long | 交易方向 (long/short) |
| `--capital` | number | - | 1000000 | 初始资金 |
| `--commission` | number | - | 0.0003 | 佣金率 |

### 使用示例

#### 示例1: 单因子回测
```bash
claude> /backtest_factor 000001 "close/SMA(close, 20)"

输出:
回测结果: 000001 - close/SMA(close, 20)
├─ 回测期间: 2023-01-01 至 2025-03-27 (631交易日)
├─ 初始资金: ¥1,000,000
├─ 性能指标:
│  ├─ 总收益: 25.6% 📈
│  ├─ 年化收益: 12.8%
│  ├─ 年化波动率: 15.0%
│  ├─ 夏普比率: 0.85 ✅ (>0.7为良好)
│  ├─ 最大回撤: -12.0%
│  ├─ 卡玛比率: 1.07
│  ├─ 胜率: 62%
│  ├─ 日均收益: +0.061%
│  └─ 月均收益: +1.03%
├─ 交易统计:
│  ├─ 总交易数: 42次
│  ├─ 平均持仓: 8.3天
│  ├─ 最长连胜: 8次
│  ├─ 平均单笔: +0.61%
│  └─ 手续费成本: ¥23,450
├─ 风险指标:
│  ├─ VaR (95%): -2.5%
│  ├─ CVaR (95%): -3.5%
│  └─ 风险等级: 中等
├─ 最终资产: ¥1,256,000
└─ 综合评估: ⭐⭐⭐⭐ 优秀 (可考虑实际应用)
```

#### 示例2: 多因子回测比较
```bash
claude> /backtest_factor 000001 "RSI(close,14)" "close/SMA(close,20)" "momentum_20" --start 2024-01-01

(返回三个因子的并行回测结果，自动排序显示最优因子)
```

#### 示例3: 空头策略
```bash
claude> /backtest_factor 000001 "RSI(close, 14)" --direction short

(返回看空操作的反向收益)
```

### 后台API调用

```bash
curl -X POST "http://localhost:8000/api/backtest/single" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_codes": ["000001"],
    "factors": ["close/SMA(close, 20)"],
    "strategy_type": "single_factor",
    "start_date": "2023-01-01",
    "end_date": "2025-03-27",
    "initial_capital": 1000000,
    "percentile": 50,
    "direction": "long",
    "commission_rate": 0.0003
  }'
```

---

## Skill 3: `compare_strategies`

**功能**: 对比多个策略的性能，快速找到最优因子

### 命令语法
```bash
claude> /compare_strategies STOCK_CODE EXPRESSION1 EXPRESSION2 [EXPRESSION3 ...] [--start START_DATE] [--end END_DATE]
```

### 使用示例

#### 示例1: 三个因子对比
```bash
claude> /compare_strategies 000001 "RSI(close,14)" "close/SMA(close,20)" "momentum_20"

输出:
策略性能对比: 000001
┌──────────────────────┬──────────┬──────────┬──────────┬─────────────┬──────────┐
│ 因子                 │ 总收益   │ 年化收益 │ 夏普比   │ 最大回撤    │ 胜率     │
├──────────────────────┼──────────┼──────────┼──────────┼─────────────┼──────────┤
│ close/SMA(c,20) ⭐   │ 25.6%    │ 12.8%    │ 0.85 ✅  │ -12.0%      │ 62%      │
│ momentum_20          │ 22.1%    │ 11.0%    │ 0.78     │ -13.5%      │ 58%      │
│ RSI(close,14)        │ 18.2%    │ 9.1%     │ 0.62     │ -15.0%      │ 55%      │
└──────────────────────┴──────────┴──────────┴──────────┴─────────────┴──────────┘

🏆 最优策略: close/SMA(close,20)
   ├─ 年化超额收益: +1.8% vs 第2名
   ├─ 风险调整收益: 0.07 更高
   └─ 推荐指数: 9/10 (适合实际应用)

📊 详细对比:
   ├─ 最稳定: close/SMA(close,20) (回撤最小)
   ├─ 最激进: momentum_20 (波动最大)
   └─ 最均衡: RSI(close,14) (风险收益比中等)
```

#### 示例2: 自定义时间段
```bash
claude> /compare_strategies 600519 "RSI(close,14)" "MACD" --start 2024-01-01 --end 2024-12-31

(仅对比2024年的表现)
```

### 后台API调用

```bash
curl -X POST "http://localhost:8000/api/backtest/comparison" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_codes": ["000001"],
    "strategies": [
      {"name": "RSI14", "factors": ["RSI(close,14)"], "weights": [1.0]},
      {"name": "MA20", "factors": ["close/SMA(close,20)"], "weights": [1.0]},
      {"name": "Momentum", "factors": ["momentum_20"], "weights": [1.0]}
    ],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27"
  }'
```

---

## Skill 4: `mine_factors`

**功能**: 使用遗传算法自动发现最优因子组合

### 命令语法
```bash
claude> /mine_factors STOCK_CODE --base FACTOR1,FACTOR2 [--generations N] [--pop-size N] [--start START] [--end END]
```

### 参数说明
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `STOCK_CODE` | string | ✅ | - | 挖掘目标股票 |
| `--base` | string[] | ✅ | - | 基础因子，逗号分隔 |
| `--generations` | int | - | 10 | 进化代数 (5-100) |
| `--pop-size` | int | - | 50 | 种群规模 (10-500) |
| `--start` | date | - | 2023-01-01 | 开始日期 |
| `--end` | date | - | 今天 | 结束日期 |

### 使用示例

#### 示例1: 基于两个因子挖掘
```bash
claude> /mine_factors 000001 --base "RSI(close,14),close/SMA(close,20)" --generations 15

输出 (实时进度):
🔄 因子挖掘进行中: 000001
├─ 基础因子池: RSI(close,14), close/SMA(close,20)
├─ 种群规模: 50 个体
├─ 目标代数: 15

[进度条 ====-------- 40%] Generation 6/15

最佳个体追踪:
  Gen 1: IC=0.032 (baseline)
  Gen 2: IC=0.035 ⬆️ (+9%)
  Gen 3: IC=0.038 ⬆️ (+18%)
  Gen 4: IC=0.041 ⬆️ (+28%)
  Gen 5: IC=0.044 ⬆️ (+37%)
  Gen 6: IC=0.047 ⬆️ (+47%) 🎯

(任务完成后)
✅ 因子挖掘完成!

🏆 发现最优因子 (Top 3):
1️⃣  (RSI(close,14) * 1.5) + 0.2
   ├─ IC: 0.047 ⬆️ (提升+47%)
   ├─ IR: 0.54
   ├─ 稳定性: B
   └─ 建议: 可直接使用或进一步优化

2️⃣  (close/SMA(close,20)) ^ 0.8
   ├─ IC: 0.045
   ├─ IR: 0.52
   └─ 建议: 次优选择

3️⃣  (RSI(close,14) + close/SMA(close,20)) / 2
   ├─ IC: 0.043
   ├─ IR: 0.51
   └─ 建议: 平衡组合

📊 进化统计:
  ├─ 最优个体适应度轨迹: [0.032→0.047]
  ├─ 平均适应度轨迹: [0.020→0.035]
  ├─ 探索效率: 94%
  └─ 运行时间: 3分12秒
```

#### 示例2: 大规模优化
```bash
claude> /mine_factors 000001 --base "RSI(close,14),MACD,momentum_20" --pop-size 100 --generations 20

(更大规模的搜索，可能需要5-10分钟)
```

### 后台API调用

```bash
# 启动异步挖掘任务
curl -X POST "http://localhost:8000/api/mining/genetic" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "000001",
    "base_factors": ["RSI(close,14)", "close/SMA(close,20)"],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27",
    "population_size": 50,
    "n_generations": 10
  }'

# 返回: {"task_id": "a1b2c3d4...", "status": "pending"}

# 查询进度
curl "http://localhost:8000/api/mining/status/a1b2c3d4..."

# 获取结果
curl "http://localhost:8000/api/mining/results/a1b2c3d4..."
```

---

## Skill 5: `validate_formula`

**功能**: 快速验证因子公式的有效性和统计特性

### 命令语法
```bash
claude> /validate_formula "FORMULA_EXPRESSION" [STOCK_CODE] [--start START] [--end END]
```

### 使用示例

#### 示例1: 验证表达式语法
```bash
claude> /validate_formula "(close - SMA(close, 20)) / SMA(close, 20)"

输出:
公式验证: (close - SMA(close, 20)) / SMA(close, 20)
├─ ✅ 语法检查: 通过
├─ ✅ 函数库检查:
│  ├─ SMA(): 可用 (TALib)
│  ├─ close: 可用 (OHLC)
│  └─ 运算符: 全部有效
├─ 公式结构: 复合表达式 (正规)
└─ 评估: ✅ 公式格式正确，可用于分析
```

#### 示例2: 完整验证 (包含执行检查)
```bash
claude> /validate_formula "RSI(close, 14) > 30" 000001

输出:
公式验证: RSI(close, 14) > 30
├─ ✅ 语法检查: 通过
├─ ✅ 函数验证: 通过
├─ ✅ 执行测试 (000001, 2023-01-01 至 2025-03-27):
│  ├─ 样本执行: 成功
│  ├─ 值范围: [0, 1] (布尔值)
│  ├─ 非空值比例: 100.0% ✅
│  ├─ 计算时间: 12ms
│  └─ 内存占用: 2.3MB
├─ 统计特征分析:
│  ├─ True出现率: 45.1%
│  ├─ False出现率: 54.9%
│  ├─ 信号稳定性: 良好 (无突变)
│  └─ 自相关性: 0.28 (存在短期持续性)
└─ 综合评估: ✅ 有效公式，适合用于因子构建
```

### 后台API调用

```bash
curl -X POST "http://localhost:8000/api/factors/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "(close - SMA(close, 20)) / SMA(close, 20)",
    "formula_type": "expression"
  }'
```

---

## Skill 6: `get_stock_data`

**功能**: 快速获取股票的OHLCV数据

### 命令语法
```bash
claude> /get_stock_data STOCK_CODE [--start START_DATE] [--end END_DATE] [--format json|csv]
```

### 使用示例

#### 示例1: 获取最近一年数据
```bash
claude> /get_stock_data 000001 --start 2024-01-01 --end 2024-12-31

输出:
股票数据: 000001 (2024-01-01 至 2024-12-31)

┌──────────────┬────────┬────────┬────────┬────────┬──────────────┐
│ 日期         │ 开盘   │ 最高   │ 最低   │ 收盘   │ 成交量(万)   │
├──────────────┼────────┼────────┼────────┼────────┼──────────────┤
│ 2024-12-31   │ 15.82  │ 15.95  │ 15.62  │ 15.88  │ 24,532       │
│ 2024-12-30   │ 15.65  │ 15.90  │ 15.61  │ 15.85  │ 23,145       │
│ ...          │ ...    │ ...    │ ...    │ ...    │ ...          │
│ 2024-01-02   │ 10.25  │ 10.45  │ 10.20  │ 10.42  │ 18,234       │
└──────────────┴────────┴────────┴────────┴────────┴──────────────┘

📊 统计信息:
├─ 交易日数: 252
├─ 涨跌家数: 134涨 / 118跌
├─ 最高价: 16.28 (2024-11-15)
├─ 最低价: 10.15 (2024-02-28)
├─ 年初收盘: 10.38
├─ 年末收盘: 15.88
├─ 年度涨幅: +52.9% 📈
├─ 平均成交量: 21.3万
├─ 成交额: ¥685.4亿
└─ 振幅: 60.4% (高波动性)
```

#### 示例2: CSV格式导出
```bash
claude> /get_stock_data 600519 --start 2025-01-01 --end 2025-03-27 --format csv

(输出可直接导入Excel或其他分析工具的CSV文件)
```

### 后台API调用

```bash
curl -X GET "http://localhost:8000/api/data/stock/000001?start_date=2024-01-01&end_date=2024-12-31"
```

---

## Skill 7: `portfolio_optimize`

**功能**: 对多个因子进行权重优化，找到最优组合

### 命令语法
```bash
claude> /portfolio_optimize STOCK_CODE FACTOR1 FACTOR2 [FACTOR3...] [--method METHOD] [--start START] [--end END]
```

### 参数说明
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `STOCK_CODE` | string | ✅ | - | 优化目标股票 |
| `FACTOR` | string | ✅ | - | 因子列表，至少2个 |
| `--method` | string | - | max_sharpe | 优化方法 |
| `--start` | date | - | 2023-01-01 | 优化开始日期 |
| `--end` | date | - | 今天 | 优化结束日期 |

### 方法说明
```
1. equal_weight       - 等权重 (1/n)
2. ic_weight         - IC加权 (按IC值)
3. ir_weight         - IR加权 (按信息比率)
4. max_sharpe        - 最大化夏普比 (推荐)
5. max_return        - 最大化收益
6. min_variance      - 最小化波动率
```

### 使用示例

#### 示例1: 三因子最优权重
```bash
claude> /portfolio_optimize 000001 "RSI(close,14)" "close/SMA(close,20)" "momentum_20"

输出:
组合优化结果: 000001
├─ 优化方法: 最大化夏普比 (推荐)
├─ 回测期间: 2023-01-01 至 2025-03-27
├─ 优化因子数: 3

🎯 最优权重配置:
┌──────────────────────┬────────┬─────────┐
│ 因子                 │ 权重   │ 权重%   │
├──────────────────────┼────────┼─────────┤
│ close/SMA(close,20)  │ 0.450  │ 45.0% ★ |
│ momentum_20          │ 0.300  │ 30.0%   │
│ RSI(close,14)        │ 0.250  │ 25.0%   │
└──────────────────────┴────────┴─────────┘

📊 预期性能指标:
├─ 年化收益: 13.5% (vs单因子最好12.8%)
├─ 夏普比: 0.92 (vs单因子最好0.85)
├─ 最大回撤: -11.2%
├─ 信息比率: 0.68
├─ 波动率: 14.6%
└─ 收益/风险比: 0.93

💡 权重建议:
├─ 主导因子: close/SMA(close,20) (趋势跟踪)
├─ 辅助因子: momentum_20 (动量确认)
├─ 对冲因子: RSI(close,14) (超买超卖)
└─ 整体风格: 稳健增长 (适合保守投资者)

📈 对比其他方法:
┌─────────────────┬──────────┬──────────┬─────────┐
│ 优化方法        │ 年化收益 │ 夏普比   │ 推荐度  │
├─────────────────┼──────────┼──────────┼─────────┤
│ max_sharpe      │ 13.5%    │ 0.92 ⭐  │ ⭐⭐⭐⭐⭐│
│ equal_weight    │ 12.8%    │ 0.81     │ ⭐⭐⭐   │
│ ic_weight       │ 13.2%    │ 0.88     │ ⭐⭐⭐⭐ │
│ max_return      │ 14.1%    │ 0.78     │ ⭐⭐⭐   │
│ min_variance    │ 10.2%    │ 0.85     │ ⭐⭐    │
└─────────────────┴──────────┴──────────┴─────────┘

✅ 综合评估: 推荐采用此权重配置 (夏普比最优)
```

#### 示例2: 对比所有优化方法
```bash
claude> /portfolio_optimize 000001 "RSI" "momentum" --method all --start 2024-01-01

(自动对比所有6种方法，显示优劣对比)
```

### 后台API调用

```bash
# 权重优化
curl -X POST "http://localhost:8000/api/portfolio/optimize-weights" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_codes": ["000001"],
    "factors": ["RSI(close,14)", "close/SMA(close,20)", "momentum_20"],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27",
    "method": "max_sharpe"
  }'

# 方法对比
curl -X POST "http://localhost:8000/api/portfolio/compare-methods" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_codes": ["000001"],
    "factors": ["RSI(close,14)", "close/SMA(close,20)", "momentum_20"],
    "start_date": "2023-01-01",
    "end_date": "2025-03-27",
    "methods": ["equal_weight", "ic_weight", "ir_weight", "max_sharpe", "max_return", "min_variance"]
  }'
```

---

## 💡 使用工作流示例

### 工作流1: 快速因子筛选 (5分钟)
```
1. /validate_formula "RSI(close, 14)"
   ↓ 验证公式有效性

2. /factor_analyze 000001 "RSI(close, 14)"
   ↓ 计算IC/IR，检查因子质量

3. /backtest_factor 000001 "RSI(close, 14)"
   ↓ 回测获得实际收益数据

✅ 完成 - 得到因子的全面评估
```

### 工作流2: 多因子优化 (10分钟)
```
1. /compare_strategies 000001 "RSI(14)" "MA(20)" "Momentum(20)"
   ↓ 找到表现最好的3个因子

2. /portfolio_optimize 000001 "RSI(14)" "MA(20)" "Momentum(20)"
   ↓ 优化权重组合

3. /backtest_factor 000001 "RSI(14)" "MA(20)" "Momentum(20)"
   ↓ 验证优化效果

✅ 完成 - 获得最优多因子策略
```

### 工作流3: 自动发现因子 (30分钟)
```
1. /mine_factors 000001 --base "RSI(14),MA(20)" --generations 15
   ↓ 自动挖掘最优因子

2. /backtest_factor 000001 "[挖掘出的最优因子]"
   ↓ 验证发现的因子

3. /factor_analyze 000001 "[最优因子]"
   ↓ 深度分析因子特性

✅ 完成 - 发现高质量的新因子
```

---

## 📋 快速参考表

### 常用因子表达式
| 类别 | 因子 | 表达式 |
|------|------|--------|
| 动量 | RSI14 | `RSI(close, 14)` |
| 趋势 | MA20比值 | `close / SMA(close, 20)` |
| 动量 | 12日动量 | `close / REF(close, 12) - 1` |
| 反转 | MACD | `MACD(close)` |
| 价格位置 | 布林带位置 | `(close - BB_LOWER) / (BB_UPPER - BB_LOWER)` |

### 常用参数组合
| 场景 | --percentile | --direction | --capital |
|------|------|---------|-----------|
| 看多 | 70 | long | 1000000 |
| 看空 | 30 | short | 1000000 |
| 激进 | 90 | long | 500000 |
| 保守 | 50 | long | 5000000 |

---

**最后更新**: 2025-03-27
**版本**: 1.0
