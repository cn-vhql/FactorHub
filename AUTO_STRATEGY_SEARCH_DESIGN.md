# 自动化策略搜索与 Champion Strategy 技术方案

## 1. 目标

本方案的目标是把现有 FactorHub 从“固定 8 因子打分器”升级成“自动化策略搜索与应用平台”，并且满足下面 4 个业务目标：

- `temporal_pool` 可以读取一份配置清单，按配置对单只股票或股票池打分
- `champion_strategy` 可以独立运行，自动为“某一只股票”或“某一组股票”选出当前最优策略
- 搜索结果可以直接应用在现有的 `100 -> 10` 股票筛选场景
- 同一套配置和搜索框架可以兼容现有的截面因子过滤逻辑

本方案明确约束：

- 不引入新的第三方库
- 优先复用现有服务、回测模块、任务模式、评分模块
- 第一版不做复杂黑箱模型，不做外部训练框架
- 第一版的搜索对象是“因子组合 + 参数配置”，不是任意数学公式


## 2. 业务定义

## 2.1 temporal_pool

`temporal_pool` 的含义是：

- 输入一组股票代码
- 对每只股票独立使用同一套时序策略配置计算 `score`
- 再把所有股票按 `score` 排序
- 取前 `TopN` 或按阈值筛选

这里的关键点是：

- `100 -> 10` 本质上不是新的横截面模型
- 而是“同一套单股时序策略，对 100 支股票逐只打分，再汇总排序”

也就是说：

- 单股打分逻辑不变
- 股票池筛选只是循环调用单股打分，再统一排序

## 2.2 champion_strategy

`champion_strategy` 的含义是：

- 给定一个搜索空间
- 在历史数据上自动回测多套候选策略
- 用统一稳定性评分挑选出一套最优策略
- 输出可复用的配置文件和评估报告

`champion_strategy` 支持两种作用域：

- `single_stock`
  - 为单只股票选最适合它的时序策略
- `stock_group`
  - 为一组股票选一套通用的池内筛选策略

## 2.3 cross_sectional_filter

本方案保留现有截面过滤能力，并将其纳入统一框架：

- `temporal_pool`
  - 每只股票独立时序打分，再做池内排序
- `cross_sectional_filter`
  - 同一交易日对股票池做截面因子打分和排序

两者共用：

- 因子模板
- 搜索空间定义
- 策略评价体系
- champion 选择机制


## 3. 现状与改造原则

当前可复用模块：

- 因子定义与计算：
  [factor_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/factor_service.py)
- 时序打分：
  [temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py)
- 时序筛选验证：
  [temporal_filter_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_filter_validation_service.py)
- 截面回测：
  [vectorbt_backtest_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/vectorbt_backtest_service.py)
- 因子质量验证：
  [factor_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/factor_validation_service.py)
- 遗传搜索任务模式：
  [genetic_factor_mining_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/genetic_factor_mining_service.py)
- API 注册：
  [main.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/main.py)
- 当前时序打分 API：
  [scoring.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/routers/scoring.py)

改造原则：

- 不推翻现有 `temporal_scoring_service`
- 把“固定 8 因子”改为“按 profile 配置驱动”
- 搜索服务独立，不和打分服务强耦合
- 搜索结果落盘为 JSON / CSV / Markdown，不强依赖数据库
- 第一版 champion 使用文件注册即可，后续再决定是否入库


## 4. 交付范围

本次技术方案要求一次覆盖完整链路，交付范围包括：

- `temporal_pool` 读取配置清单
- `champion_strategy` 独立运行
- 单股 champion 搜索
- 股票组 champion 搜索
- `100 -> 10` 池内筛选
- 接入现有截面过滤模式
- API、任务、结果落盘、配置格式、评估体系、执行拆分

不在第一版交付范围内：

- 新数据库表
- 新三方库
- 强化学习 / 深度学习 / AutoML
- 任意公式级别无限制搜索


## 5. 总体架构

整体架构拆成 6 个模块：

1. `Factor Profile Registry`
- 管理内置策略模板和配置清单

2. `Temporal Pool Executor`
- 按 profile 对单股打分
- 汇总成股票池排名

3. `Strategy Search Engine`
- 枚举或搜索候选策略

4. `Evaluation Engine`
- 统一做回测、稳定性评价、显著性检验

5. `Champion Strategy Runner`
- 独立执行搜索任务
- 输出 champion 结果

6. `Champion Registry`
- 保存当前选中的 champion 配置
- 供打分和过滤接口直接调用

推荐流程：

```text
配置清单 -> 候选策略生成 -> 历史回测/验证 -> 稳定性打分 -> 选出 champion -> 应用到 temporal_pool / cross_sectional_filter
```


## 6. 配置清单设计

第一版采用文件配置，不引入数据库表。

新增配置目录：

- `config/temporal_pool_profiles.json`
- `config/cross_sectional_profiles.json`
- `config/strategy_search_space.json`
- `config/champion_runtime.json`

## 6.1 temporal_pool_profiles.json

用于定义可直接运行的时序策略模板。

示例：

```json
{
  "profiles": [
    {
      "id": "momentum_v1",
      "mode": "temporal_pool",
      "description": "趋势动量模板",
      "factors": [
        {"name": "price_vs_sma20", "direction": 1, "weight": 0.22, "role": "signal"},
        {"name": "momentum_20", "direction": 1, "weight": 0.18, "role": "signal"},
        {"name": "price_vwma_ratio", "direction": 1, "weight": 0.16, "role": "signal"},
        {"name": "force_index_ma", "direction": 1, "weight": 0.14, "role": "signal"},
        {"name": "bollinger_position", "direction": 1, "weight": 0.14, "role": "signal"},
        {"name": "stochastic_d", "direction": 1, "weight": 0.16, "role": "signal"},
        {"name": "atr_norm", "direction": -1, "weight": 0.0, "role": "risk_filter", "threshold": 0.80},
        {"name": "volume_ma_ratio", "direction": 1, "weight": 0.0, "role": "risk_filter", "threshold": 0.30}
      ],
      "params": {
        "rebalance": "W-FRI",
        "hold_days": 5,
        "top_n": 10,
        "score_threshold": 70,
        "percentile_window": 252
      }
    }
  ]
}
```

## 6.2 cross_sectional_profiles.json

用于定义截面过滤模板。

示例：

```json
{
  "profiles": [
    {
      "id": "cross_sectional_momentum_v1",
      "mode": "cross_sectional_filter",
      "description": "截面动量模板",
      "factors": [
        {"name": "momentum_20", "direction": 1, "weight": 0.35},
        {"name": "price_vs_sma20", "direction": 1, "weight": 0.25},
        {"name": "atr_norm", "direction": -1, "weight": 0.20},
        {"name": "volume_ma_ratio", "direction": 1, "weight": 0.20}
      ],
      "params": {
        "rebalance": "W-FRI",
        "top_percentile": 0.10,
        "direction": "long"
      }
    }
  ]
}
```

## 6.3 strategy_search_space.json

用于定义自动搜索范围。

示例：

```json
{
  "search_spaces": {
    "temporal_default": {
      "candidate_factors": [
        "price_vs_sma20",
        "momentum_20",
        "price_vwma_ratio",
        "force_index_ma",
        "bollinger_position",
        "stochastic_d",
        "atr_norm",
        "volume_ma_ratio",
        "rsi_14",
        "roc_10",
        "deviation_from_ma20",
        "cci_20"
      ],
      "factor_groups": {
        "trend": ["price_vs_sma20", "momentum_20", "price_vwma_ratio", "roc_10"],
        "reversal": ["deviation_from_ma20", "bollinger_position", "stochastic_d", "rsi_14", "cci_20"],
        "flow": ["force_index_ma", "volume_ma_ratio"],
        "risk": ["atr_norm"]
      },
      "min_factors": 4,
      "max_factors": 8,
      "max_per_group": 2,
      "rebalance_choices": ["W-FRI", "2W-FRI"],
      "hold_days_choices": [5, 10, 20],
      "top_n_choices": [5, 10, 15],
      "score_threshold_choices": [60, 65, 70, 75]
    }
  }
}
```


## 7. 数据模型

建议统一内部数据结构。

## 7.1 StrategyProfile

```text
id
mode
description
factors[]
params{}
version
source
```

## 7.2 StrategyCandidate

```text
candidate_id
profile_id
scope_type
scope_key
config
train_metrics
valid_metrics
test_metrics
stability_score
selected
```

## 7.3 ChampionStrategy

```text
champion_id
scope_type
scope_key
mode
profile_id
config
metrics
selected_at
effective_from
report_path
```

说明：

- `scope_type` 取值：
  - `single_stock`
  - `stock_group`
  - `cross_sectional_universe`
- `scope_key`：
  - 单股时为股票代码
  - 股票组时为股票列表 hash
  - 全市场截面时为 universe 名称


## 8. 核心模块设计

## 8.1 FactorProfileRegistryService

新文件：

- `backend/services/factor_profile_registry_service.py`

职责：

- 加载 YAML 配置
- 校验 profile 合法性
- 返回 profile 列表、单个 profile、search space

核心接口：

```python
load_profiles(mode: str) -> list[dict]
get_profile(profile_id: str) -> dict
load_search_space(space_id: str) -> dict
validate_profile(profile: dict) -> None
```

## 8.2 TemporalPoolService

新文件：

- `backend/services/temporal_pool_service.py`

职责：

- 接收 `codes + profile_id + trade_date`
- 对每只股票循环调用时序打分
- 汇总成池内排序结果
- 输出 `TopN`

注意：

- `100 -> 10` 的实现不需要新模型
- 只需要循环调用单股评分，再排序

核心接口：

```python
score_pool(codes: list[str], profile_id: str, trade_date: str) -> pd.DataFrame
select_top_n(pool_scores: pd.DataFrame, top_n: int, score_threshold: float | None) -> pd.DataFrame
score_pool_history(codes: list[str], profile_id: str, start_date: str, end_date: str, max_workers: int) -> pd.DataFrame
```

## 8.3 StrategySearchService

新文件：

- `backend/services/strategy_search_service.py`

职责：

- 读取 search space
- 生成候选策略
- 调用验证引擎
- 输出候选排行榜

搜索方法不引入新库，只使用现有和标准库：

- `template_enum`
- `constrained_random`
- `forward_selection`
- `backward_pruning`

第一版不建议直接上任意公式遗传搜索。

核心接口：

```python
generate_candidates(mode: str, base_profiles: list[dict], search_space: dict, n_candidates: int) -> list[dict]
search_for_stock(code: str, search_space_id: str, start_date: str, end_date: str) -> list[dict]
search_for_pool(codes: list[str], search_space_id: str, start_date: str, end_date: str) -> list[dict]
search_for_cross_sectional(universe: str, search_space_id: str, start_date: str, end_date: str) -> list[dict]
```

## 8.4 StrategyEvaluationService

新文件：

- `backend/services/strategy_evaluation_service.py`

职责：

- 统一时序和截面两类策略的评价逻辑
- 负责 walk-forward、显著性、稳定性总分

拆成两个执行器：

- `evaluate_temporal_pool_candidate`
- `evaluate_cross_sectional_candidate`

可复用：

- [temporal_filter_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_filter_validation_service.py)
- [vectorbt_backtest_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/vectorbt_backtest_service.py)
- [factor_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/factor_validation_service.py)

## 8.5 ChampionStrategyService

新文件：

- `backend/services/champion_strategy_service.py`

职责：

- 独立运行搜索任务
- 从候选策略中选出最佳 champion
- 保存 champion 配置和报告
- 提供“查询当前 champion”能力

核心接口：

```python
run_for_stock(code: str, search_space_id: str, start_date: str, end_date: str) -> dict
run_for_pool(codes: list[str], search_space_id: str, start_date: str, end_date: str) -> dict
run_for_cross_sectional(universe: str, search_space_id: str, start_date: str, end_date: str) -> dict
get_champion(scope_type: str, scope_key: str) -> dict | None
apply_champion(scope_type: str, scope_key: str, champion: dict) -> None
```

## 8.6 ChampionRegistryService

新文件：

- `backend/services/champion_registry_service.py`

职责：

- 使用文件方式保存当前 champion
- 不依赖新数据库表

目录建议：

- `data/champions/single_stock/{code}.json`
- `data/champions/stock_group/{group_hash}.json`
- `data/champions/cross_sectional/{universe}.json`


## 9. 对现有服务的改造

## 9.1 temporal_scoring_service 改造

修改文件：

- [temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py)

改造目标：

- 从“固定 8 因子硬编码”改成“profile 驱动”

建议新增能力：

```python
score_one_stock_with_profile(code: str, trade_date: str, profile_id: str) -> dict
score_stock_history_with_profile(code: str, start_date: str, end_date: str, profile_id: str, window: int = 252) -> pd.DataFrame
```

保留：

- 现有 `score_one_stock`
- 现有默认 8 因子 profile，作为 `default_temporal_v1`

这样可以做到：

- 老接口不破
- 新接口支持多套模板

## 9.2 temporal_filter_validation_service 改造

修改文件：

- [temporal_filter_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_filter_validation_service.py)

改造目标：

- 支持 `profile_id`
- 支持 `max_workers`
- 支持池内时序模式的完整验证

新增或修改接口：

```python
build_score_panel(codes, start_date, end_date, profile_id, max_workers=1)
run_temporal_pool_backtest(codes, profile_id, start_date, end_date, top_n, score_threshold, rebalance, max_workers=1)
```

## 9.3 vectorbt_backtest_service 改造

修改文件：

- [vectorbt_backtest_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/vectorbt_backtest_service.py)

改造目标：

- 不只吃单因子
- 支持“多因子组合分”

新增接口建议：

```python
cross_sectional_backtest_composite(df: pd.DataFrame, factor_weights: dict, top_percentile: float, direction: str = "long") -> dict
```

## 9.4 scoring API 改造

修改文件：

- [scoring.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/routers/scoring.py)

改造目标：

- 支持传 `profile_id`
- temporal_pool 读取配置清单后可直接调用

建议新增请求字段：

```json
{
  "codes": ["600519", "000001"],
  "trade_date": "2026-03-21",
  "profile_id": "momentum_v1"
}
```

## 9.5 新增 search API

新增文件：

- `backend/api/routers/search.py`

在 [main.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/main.py) 注册：

- `/api/search`

建议端点：

- `POST /api/search/champion/jobs`
- `GET /api/search/champion/jobs/{task_id}`
- `GET /api/search/champion/jobs/{task_id}/results`
- `GET /api/search/champions/{scope_type}/{scope_key}`
- `GET /api/search/profiles`
- `GET /api/search/search-spaces`


## 10. 搜索策略设计

## 10.1 第一层：模板枚举

先跑所有内置 profile：

- `momentum_v1`
- `mean_reversion_v1`
- `trend_flow_v1`
- `hybrid_v1`
- `low_vol_trend_v1`

目的：

- 快速找到基线
- 结果可解释
- 运行成本最低

## 10.2 第二层：受约束随机搜索

不做无限制随机。

搜索对象：

- 因子子集
- 方向
- 权重
- 风险阈值
- 调仓频率
- 持有期
- `TopN`
- `score_threshold`

约束：

- 总因子数 `4-8`
- 同组最多 `2`
- 必须至少包含 `1` 个风险过滤因子
- 高相关因子不同时入选
- 权重和为 `1`

## 10.3 第三层：逐步精简

在表现最好的前若干个候选上做：

- `forward_selection`
- `backward_pruning`

目的：

- 去掉冗余因子
- 减少过拟合
- 保持可解释性


## 11. 评价体系

## 11.1 单股 champion

单股不适合用截面 IC。

单股评价建议使用：

- 阈值信号回测收益
- 年化收益
- Sharpe
- 最大回撤
- 胜率
- 不同持有期效果
- 不同阶段稳定性

输出一个单股 `stability_score`。

## 11.2 股票组 champion

股票组使用 `temporal_pool`：

1. 每只股票按 profile 独立打分
2. 每日汇总为池内 score
3. 取 `TopN`
4. 做池内策略回测

评价指标：

- `TopN vs Universe` 超额收益
- `TopN vs RandomN`
- IC / IR
- 分层 spread
- Sharpe
- 最大回撤
- 正超额窗口比例

## 11.3 截面 champion

截面模式使用：

- 组合因子打分
- `top_percentile`
- 截面回测结果

评价指标：

- 年化收益
- Sharpe
- 最大回撤
- 截面 IC
- 换手率
- 风格阶段稳定性

## 11.4 Champion 总分

总分不只看收益，建议：

- `30%` OOS Sharpe
- `20%` OOS excess return
- `15%` drawdown penalty
- `10%` IC/IR
- `10%` positive_window_ratio
- `10%` regime_consistency
- `5%` turnover penalty

统一由现有 [comprehensive_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/comprehensive_scoring_service.py) 扩展实现，不新增库。


## 12. Walk-forward 设计

必须用滚动验证，不能只看整段回测。

建议默认：

- 训练期：`12M`
- 验证期：`3M`
- 测试期：`3M`
- 滚动步长：`3M`

每个候选策略都记录：

- train metrics
- valid metrics
- test metrics
- 窗口间波动
- 正收益窗口占比

champion 选择标准：

- 先过滤掉测试期明显失效的候选
- 再按稳定性总分排序


## 13. Champion 独立运行模式

这是本方案的核心要求之一。

`champion_strategy` 不依赖前端页面，可以独立执行。

## 13.1 单股运行

输入：

```json
{
  "scope_type": "single_stock",
  "code": "600519.SH",
  "search_space_id": "temporal_default",
  "start_date": "2023-01-01",
  "end_date": "2026-03-21"
}
```

流程：

1. 加载内置 profile 和 search space
2. 生成候选策略
3. 对该股票做 walk-forward
4. 计算稳定性总分
5. 选出 champion
6. 落盘到：
   - `data/champions/single_stock/600519.SH.json`

## 13.2 股票组运行

输入：

```json
{
  "scope_type": "stock_group",
  "codes": ["600519.SH", "000001.SZ", "..."],
  "search_space_id": "temporal_default",
  "start_date": "2023-01-01",
  "end_date": "2026-03-21"
}
```

流程：

1. 加载候选策略
2. 对每个候选策略执行：
   - 对所有股票逐只打分
   - 汇总池内分数
   - 生成 `Top10`
   - 计算历史表现
3. 选出 champion
4. 落盘到：
   - `data/champions/stock_group/{group_hash}.json`

## 13.3 截面运行

输入：

```json
{
  "scope_type": "cross_sectional_universe",
  "universe": "candidate_pool_100",
  "search_space_id": "cross_sectional_default",
  "start_date": "2023-01-01",
  "end_date": "2026-03-21"
}
```


## 14. 100 -> 10 场景的明确实现方式

这个需求按下面方式实现，不引入新模型：

1. 选定一个 `profile_id`
2. 对 100 支股票逐只调用：
   - `score_one_stock_with_profile`
   - 或 `score_stock_history_with_profile`
3. 得到每只股票的 `score`
4. 对 100 支股票排序
5. 取前 10 支

因此技术上：

- `100 -> 10` 就是“单股评分器循环调用 + 汇总排序”
- 不需要另外开发一套全新模型
- 这正好符合现有 `temporal_scoring_service` 的演进路径


## 15. 与现有截面因子过滤的结合

要兼容当前截面模式，建议做统一的 `mode` 字段：

- `temporal_pool`
- `cross_sectional_filter`

统一入口：

- 候选策略生成
- champion 选择
- 报告输出

不同点只在执行器：

- `temporal_pool`
  - 单股打分后池内排序
- `cross_sectional_filter`
  - 同日截面因子组合打分后排序

这样可以做到：

- 同一套 profile registry
- 同一套 search space
- 同一套 champion runner
- 同一套报告结构


## 16. API 设计

新增 router：

- `backend/api/routers/search.py`

### 16.1 查询配置

- `GET /api/search/profiles?mode=temporal_pool`
- `GET /api/search/search-spaces`

### 16.2 启动 champion 搜索任务

- `POST /api/search/champion/jobs`

输入示例：

```json
{
  "scope_type": "stock_group",
  "codes": ["600519.SH", "000001.SZ"],
  "search_space_id": "temporal_default",
  "start_date": "2023-01-01",
  "end_date": "2026-03-21",
  "max_workers": 1
}
```

### 16.3 查询任务状态

- `GET /api/search/champion/jobs/{task_id}`

### 16.4 查询任务结果

- `GET /api/search/champion/jobs/{task_id}/results`

### 16.5 查询 champion

- `GET /api/search/champions/{scope_type}/{scope_key}`

### 16.6 应用 champion 到时序池

- `POST /api/search/champions/{scope_type}/{scope_key}/apply`


## 17. 落盘与结果文件

每次 champion 搜索任务输出到：

- `data/reports/strategy_search/{task_id}/`

包含：

- `task_status.json`
- `input_scope.json`
- `candidate_leaderboard.csv`
- `candidate_configs.json`
- `walkforward_metrics.parquet`
- `best_strategy.json`
- `selection_report.md`

同时保存 champion：

- `data/champions/...`


## 18. 不引入新库的实现说明

本方案明确不引入新库。

全部基于当前仓库已有依赖和标准库实现：

- `json`
- `pathlib`
- `threading`
- `concurrent.futures`
- `pandas`
- `numpy`
- `scipy`

第一版统一使用：

- `config/*.json`

这样最稳，也最符合“不新增库”的要求。


## 19. 可执行开发拆解

下面是一次性交付的开发拆解，不是“先做一半”。

## A. 配置层

新增：

- `config/temporal_pool_profiles.json`
- `config/cross_sectional_profiles.json`
- `config/strategy_search_space.json`
- `config/champion_runtime.json`

验收：

- 可以列出所有 profile
- 可以按 `profile_id` 正常读取

## B. Profile Registry

新增：

- `backend/services/factor_profile_registry_service.py`

验收：

- profile 校验通过
- 非法配置会报清晰错误

## C. 时序打分 profile 化

修改：

- [temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py)

验收：

- 支持 `profile_id`
- 老的默认接口保持可用

## D. temporal_pool 执行器

新增：

- `backend/services/temporal_pool_service.py`

验收：

- 输入 100 支股票能输出完整 `score/rank`
- 能按 `Top10` 选股

## E. 搜索服务

新增：

- `backend/services/strategy_search_service.py`

验收：

- 能枚举模板
- 能跑受约束随机搜索
- 能输出 candidate leaderboard

## F. 评价服务

新增：

- `backend/services/strategy_evaluation_service.py`

修改：

- [temporal_filter_validation_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_filter_validation_service.py)
- [vectorbt_backtest_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/vectorbt_backtest_service.py)

验收：

- 时序池候选能回测
- 截面候选能回测
- 输出统一指标

## G. Champion 服务

新增：

- `backend/services/champion_strategy_service.py`
- `backend/services/champion_registry_service.py`

验收：

- 单股 champion 可生成
- 股票组 champion 可生成
- champion 可查询可应用

## H. API 与任务层

新增：

- `backend/api/routers/search.py`

修改：

- [main.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/main.py)
- [scoring.py](/Users/fengzhi/Downloads/git/FactorHub/backend/api/routers/scoring.py)

验收：

- 可异步启动搜索任务
- 可查询状态和结果

## I. 已知缺陷修复

必须一并修：

- `max_workers` 透传到底层
- 股票代码读取规范化
- 回测持仓顺序敏感问题
- 缓存 key 加版本信息

验收：

- 显式 `max_workers=1` 时不会走默认 4 线程
- 深市代码不会丢前导零
- 相同持仓集合不会因顺序不同导致净值偏差


## 20. 上线后的运行模式

## 20.1 离线搜索

建议每天或每周离线运行 champion 搜索：

- 单股 champion
- 股票组 champion
- 截面 champion

## 20.2 在线应用

在线只做两件事：

- 读取当前 champion
- 用 champion 做当天打分或筛选

这样可以避免：

- 在线临时重训
- 打分延迟失控
- 搜索与生产混在一起


## 21. 验收标准

满足以下条件即认为方案落地成功：

- `temporal_pool` 可以从配置清单读取 profile 并运行
- `100 -> 10` 可用同一套单股评分器循环完成
- `champion_strategy` 可独立为单股生成最优策略
- `champion_strategy` 可独立为股票组生成最优策略
- 同一框架可支持截面过滤
- 全部基于现有依赖实现，无新增三方库
- 结果可追踪、可复现、可落盘


## 22. 推荐的一期交付口径

如果按“必须一次到位”的要求来做，建议一期直接交付：

- profile registry
- temporal_pool
- champion_strategy for single_stock
- champion_strategy for stock_group
- cross_sectional_filter 接口兼容
- search API
- 结果落盘
- 已知缺陷修复

不建议把第一版只做成：

- 只有模板列表
- 没有 champion 选择
- 没有股票组回测
- 没有截面兼容

那样会造成后续又要重构一次。


## 23. 最终建议

最适合现有 FactorHub 的路线不是“自动随机发明策略公式”，而是：

- 先把策略模板配置化
- 再做受约束搜索
- 再做 champion 自动选择
- 最后统一接到 `temporal_pool` 和 `cross_sectional_filter`

这样既能自动化，也不会把系统做成不可解释的黑箱。
