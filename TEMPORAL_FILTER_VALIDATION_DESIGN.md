# 通用 100 选 10 时序筛选验证设计与代码改造方案

## 1. 目标

本设计文档不再绑定“神奇公式”。

神奇公式现在只作为一个候选池来源示例。真正要验证的问题是：

- 给定任意一组 `100` 支股票
- 使用 [backend/services/temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py) 对这 `100` 支股票打分
- 选出 `Top10`
- 验证筛选后的 `Top10` 是否在未来收益和风险调整后收益上，显著优于筛选前的原始 `100` 支股票池

这里的核心前提是：

- `temporal_scoring_service` 已经视为真实有效、可用于生产的打分引擎

因此本次重点不是重新证明打分器本身成立，而是验证：

- 它作为“股票池内筛选器”是否有增益


## 2. 正确的验证问题

这里要先把目标定义得足够严谨。

不应该验证：

- “筛选后一定各方面都比筛选前好”

因为这在统计上通常不成立。`Top10` 更集中，往往会带来：

- 更高收益的可能
- 也更高波动的可能

更合理的验证目标是：

- `Top10` 的未来收益是否显著高于 `Top100`
- `Top10` 的风险调整收益是否更优
- 这种增益是否稳定，而不是偶然

所以本次结论标准改为：

- 在目标收益指标上优于基准
- 在风险调整后收益上优于基准
- 且具有统计显著性


## 3. 适用场景

本方案适用于两类输入。

### 3.1 固定股票池

输入只有一组 `100` 支股票，例如：

```csv
code
600519
000001
300750
...
```

用途：

- 验证一组固定候选池内，时序筛选是否有效

### 3.2 历史快照股票池

输入按日期给出不同的 `100` 支股票池，例如：

```csv
date,code
2025-01-03,600519
2025-01-03,000001
2025-01-10,600519
2025-01-10,300750
...
```

用途：

- 验证动态股票池内，时序筛选是否有效

建议第一阶段先支持固定池，因为它最容易真实落地。


## 4. 研究假设

本次验证的原假设和备择假设如下。

### 4.1 收益提升假设

```text
H0: Top10 的未来收益均值 <= Top100 的未来收益均值
H1: Top10 的未来收益均值 >  Top100 的未来收益均值
```

### 4.2 风险调整收益假设

```text
H0: Top10 的风险调整后收益 <= Top100
H1: Top10 的风险调整后收益 >  Top100
```

### 4.3 预测能力假设

```text
H0: score 与 future return 无关
H1: score 与 future return 正相关
```


## 5. 回测口径

## 5.1 时间口径

建议默认区间：

- `start_date`: 最近 2 到 3 年
- `end_date`: 最新可得交易日

建议默认调仓频率：

- 预测能力验证：日频观察
- 组合回测：周频调仓

## 5.2 防止未来函数

必须使用下面的时序：

1. 在 `t` 日收盘后拿到价格数据
2. 计算 `score(t)`
3. 在 `t+1` 开始建仓

未来收益定义建议如下：

```text
future_return_5  = close[t+5]  / open[t+1] - 1
future_return_10 = close[t+10] / open[t+1] - 1
future_return_20 = close[t+20] / open[t+1] - 1
```

如果开盘价不可用，可退化为：

```text
future_return_h = close[t+h] / close[t+1] - 1
```

## 5.3 组合定义

对于每个调仓日 `t`：

- `Universe100(t)`: 输入股票池
- `Top10(t)`: 评分最高的前 10 支股票

建议比较四组组合。

### 基准 A：Top100 等权

- 对原始 100 支股票等权持有

### 基准 B：随机 10 支

- 从 Top100 中随机抽 10 支
- 可重复抽样若干次，取均值

### 策略 C：时序评分 Top10

- 按 `score` 从高到低选前 10 支

### 策略 D：时序评分 Top10 + 阈值

- 先筛 `score >= score_threshold`
- 再取前 10 支
- 不足 10 支的剩余权重留现金

策略 D 更接近实盘。


## 6. 需要看的指标

不是所有指标都必须优于基准。

最关键的是下面三层。

## 6.1 预测能力

- `IC_5`
- `IC_10`
- `IC_20`
- `IC > 0` 占比
- 高分组 vs 低分组未来收益差

## 6.2 组合收益

- 平均收益
- 年化收益
- 超额收益
- 胜率

## 6.3 风险调整后收益

- Sharpe
- Information Ratio
- 最大回撤
- 波动率

推荐的判断标准是：

- `Top10` 收益高于 `Top100`
- 风险调整收益不差于 `Top100`
- 统计检验显著


## 7. 验证方法

## 7.1 横截面 IC 验证

对每个调仓日 `t`，在当日股票池内计算：

```text
IC_h(t) = corr(rank(score_t), future_return_h_t)
```

其中 `h ∈ {5, 10, 20}`。

最终统计：

- IC 均值
- IC 标准差
- IR
- 正向比例
- t 值
- p 值

结论标准：

- `IC_mean > 0`
- `p < 0.05`

## 7.2 分层收益验证

对每个调仓日，把 100 支股票按 `score` 分成 5 组或 10 组。

看：

- 高分组未来收益是否高于低分组
- 是否存在明显单调性
- `Top - Bottom` 差值是否显著大于 0

## 7.3 组合回测验证

组合回测直接对比：

- 基准 A：Top100 等权
- 基准 B：随机 10 支
- 策略 C：时序评分 Top10
- 策略 D：时序评分 Top10 + 阈值

然后做：

- 收益曲线比较
- 超额收益比较
- 配对 t 检验
- 风险调整后收益比较


## 8. 输入与输出设计

## 8.1 输入格式

第一阶段最小输入建议只保留：

```csv
code
600519
000001
300750
...
```

第二阶段支持历史快照：

```csv
date,code
2025-01-03,600519
2025-01-03,000001
...
```

## 8.2 输出文件

建议统一落到：

```text
data/reports/temporal_filter_validation/{run_id}/
```

文件建议如下。

### `score_panel.parquet`

```text
date, code, score, rank, trend_break, high_volatility, liquidity_risk
```

### `forward_returns.parquet`

```text
date, code, future_return_5, future_return_10, future_return_20
```

### `ic_summary.csv`

```text
horizon, ic_mean, ic_std, ir, positive_ratio, t_stat, p_value
```

### `quantile_summary.csv`

```text
horizon, quantile, mean_return, annual_return, sharpe, win_rate, sample_count
```

### `portfolio_curves.csv`

```text
date, universe100_eq, random10_avg, top10_score, top10_score_threshold
```

### `strategy_comparison.json`

保存组合对比与统计显著性结果。

### `validation_report.md`

输出可读结论。


## 9. 代码改造总览

当前系统已经有：

- 打分服务：[backend/services/temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py)
- 统计工具：[backend/services/statistics_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/statistics_service.py)
- 策略对比：[backend/services/strategy_comparison_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/strategy_comparison_service.py)

但还没有一条“股票池筛选验证”主链。

建议做如下改造。

### 9.1 改造目标

不是去改打分公式，而是新增一条验证链路：

```text
股票池输入 -> 历史评分面板 -> 未来收益面板 -> 分层验证 -> Top10 组合回测 -> 显著性报告
```

### 9.2 新增文件

建议新增：

- `backend/services/temporal_filter_validation_service.py`
- `scripts/run_temporal_filter_validation.py`
- `tests/test_temporal_filter_validation_service.py`

可选新增：

- `backend/api/routers/validation.py`


## 10. 具体代码改造方案

## 10.1 改造一：增强 temporal_scoring_service

文件：

- [backend/services/temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py)

### 当前问题

现在的服务更适合：

- 给一个股票、一个日期，返回一个点的分数

但不适合做历史验证，因为：

- 如果对每个日期都调用 `score_one_stock`，会重复取数、重复算因子，成本很高
- `_factor_expressions` 只在启动时加载一次，数据库恢复后不会自动刷新

### 建议新增的方法

#### 1. `refresh_factor_expressions(force: bool = False) -> None`

作用：

- 主动刷新预置因子表达式
- 避免服务启动时退回 fallback 后长期不恢复

建议行为：

- 服务初始化时加载一次
- 在批量历史验证开始前强制刷新一次
- 可加一个 `last_refresh_at` 时间戳

#### 2. `score_stock_history(code, start_date, end_date, window=252) -> pd.DataFrame`

作用：

- 一次性返回某只股票在整个历史区间内的每日评分序列

输出建议字段：

```text
date, code, score, trend_break, high_volatility, liquidity_risk
```

实现思路：

1. 一次拉全量价格数据
2. 一次算完整因子时间序列
3. 用 trailing window 计算每个日期的时序百分位
4. 向量化生成 `score`

关键点：

- 必须使用“只看过去”的滚动窗口
- 不能把未来数据算进百分位

#### 3. `score_many_stocks_history(codes, start_date, end_date, max_workers=4) -> pd.DataFrame`

作用：

- 批量生成一个股票池的历史评分面板

实现思路：

- 每只股票独立处理
- 用 `ThreadPoolExecutor(max_workers=4)`
- 每只股票完整产出一个 DataFrame
- 最后再拼接

### 技术细节建议

时序百分位不要用“每天重新切片 + percentileofscore”的暴力循环。

推荐向量化做法：

```text
series.rolling(window=252, min_periods=20).rank(pct=True)
```

它的含义正好是：

- 每个日期的当前值，在过去 252 个观测中的分位位置

这样既避免未来函数，也能明显降低计算量。


## 10.2 改造二：新增验证主服务

文件：

- `backend/services/temporal_filter_validation_service.py`

### 主要职责

#### 1. 加载股票池

建议支持两个入口：

- 固定池：只有 `code`
- 历史快照池：`date, code`

建议方法：

- `load_fixed_pool(csv_path) -> list[str]`
- `load_snapshot_pool(csv_path) -> pd.DataFrame`

#### 2. 构建评分面板

建议方法：

- `build_score_panel(codes, start_date, end_date, rebalance_dates=None) -> pd.DataFrame`

说明：

- 第一版可以先对所有交易日打分
- 如果区间太长，可只保留调仓日评分

#### 3. 构建未来收益面板

建议方法：

- `build_forward_returns(codes, start_date, end_date, horizons=(5,10,20)) -> pd.DataFrame`

输出：

```text
date, code, future_return_5, future_return_10, future_return_20
```

#### 4. 合并为研究面板

建议方法：

- `build_research_panel(score_panel, forward_returns) -> pd.DataFrame`

输出：

```text
date, code, score, rank, future_return_5, future_return_10, future_return_20
```

#### 5. 计算 IC

建议方法：

- `calculate_cross_sectional_ic(panel, horizons=(5,10,20)) -> dict`

注意：

- 每个调仓日都按横截面计算
- 使用 `spearman` 更稳

#### 6. 计算分层收益

建议方法：

- `analyze_quantile_spread(panel, n_quantiles=5, horizons=(5,10,20)) -> dict`

可直接复用：

- [backend/services/statistics_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/statistics_service.py)

但需要补一层：

- 先按日期把股票分层
- 再把各层未来收益序列喂给统计服务

#### 7. 组合回测

建议方法：

- `run_portfolio_backtest(panel, top_n=10, rebalance='W-FRI', score_threshold=None) -> pd.DataFrame`

需要同时输出：

- `universe100_eq`
- `random10_avg`
- `top10_score`
- `top10_score_threshold`

实现方式：

- 第一版不必强依赖 vectorbt
- 可以自己做一个简单等权再平衡模拟器

原因：

- 这里是股票池内横截面筛选，不是单票择时
- 自己写一个等权回测器更可控、更容易和评分面板对齐

#### 8. 统计显著性

建议方法：

- `compare_strategies(curves_or_returns) -> dict`

可部分复用：

- [backend/services/strategy_comparison_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/strategy_comparison_service.py)

但建议不要强依赖其“策略注册器”模式，直接做：

- 配对 t 检验
- 信息比率
- 超额收益均值检验

### 推荐输出结构

建议主服务统一返回：

```text
{
  "score_panel": ...,
  "forward_returns": ...,
  "ic_summary": ...,
  "quantile_summary": ...,
  "portfolio_curves": ...,
  "strategy_comparison": ...,
  "report_path": ...
}
```


## 10.3 改造三：新增脚本入口

文件：

- `scripts/run_temporal_filter_validation.py`

### 作用

提供一个可以直接跑的 CLI。

### 推荐参数

```text
--pool-csv
--start-date
--end-date
--top-n
--score-threshold
--rebalance
--horizons
--max-workers
--output-dir
```

### 示例

```bash
.venv/bin/python scripts/run_temporal_filter_validation.py \
  --pool-csv tests/pool100.csv \
  --start-date 2024-01-01 \
  --end-date 2026-03-20 \
  --top-n 10 \
  --score-threshold 70 \
  --rebalance W-FRI \
  --horizons 5,10,20
```

### 为什么先用脚本而不是 API

因为这个任务本质上是：

- 长耗时
- 研究型
- 文件产出型

先做 CLI 最稳，后续再包 API。


## 10.4 改造四：测试

建议新增：

- `tests/test_temporal_filter_validation_service.py`

至少覆盖以下场景。

### 1. 固定股票池能跑通

- 输入 100 个代码
- 能生成评分面板和 forward returns

### 2. 无未来函数

- `t` 日评分只能使用 `t` 及以前的数据
- `future_return_h` 必须从 `t+1` 开始算

### 3. Top10 组合权重正确

- 每次调仓总权重应为 1
- 使用阈值时不足 10 支，现金权重正确

### 4. 分层和 IC 输出格式正确

- horizon 维度齐全
- p 值在合法区间


## 11. 真实可落地的技术细节

## 11.1 不要按“日期 x 股票”暴力调用 score_one_stock

如果按下面方式做：

```text
500 个交易日 x 100 支股票 = 50,000 次 score_one_stock
```

会很慢，因为每次都会：

- 重复取数
- 重复算因子
- 重复算历史百分位

真实可落地的方案必须改成：

- 每只股票只拉一次全历史数据
- 每只股票只算一次完整因子序列
- 每只股票只生成一次历史评分序列

## 11.2 要有结果缓存

建议缓存两层。

### 价格缓存

沿用已有：

- [backend/services/data_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/data_service.py)

### 历史评分缓存

建议新增缓存文件：

```text
data/cache/temporal_scores/{code}_{start}_{end}.parquet
```

只要：

- 因子定义没变
- 时间范围没变

就复用缓存。

## 11.3 因子表达式刷新

针对你之前提到的问题，建议在历史验证启动前做一次显式刷新。

推荐逻辑：

1. `TemporalScoringService.__init__` 时加载一次
2. `run_temporal_filter_validation.py` 入口调用前执行
   `temporal_scoring_service.refresh_factor_expressions(force=True)`
3. 记录一个 `factor_expression_source` 到报告里

如果 DB 失败：

- 报告里明确标记使用 fallback
- 不允许静默无痕继续

## 11.4 调仓频率不要默认日频实盘化

研究层面可以日频观察。

但组合层建议默认：

- 每周调仓

原因：

- 更接近真实执行
- 换手更低
- 交易噪音更少

## 11.5 随机基准要可复现

随机 10 支组合必须固定随机种子，例如：

```text
seed = 42
```

否则每次验证结果不可比。

## 11.6 失败样本必须保留

不能只丢掉打分失败的股票。

建议额外输出：

```text
date, code, error
```

否则最后很难知道：

- 是停牌
- 是数据不足
- 是因子值为 NaN
- 还是取数失败


## 12. 分阶段实施建议

## Phase 1：固定池验证

目标：

- 快速验证 `Top10` 是否优于 `Top100`

实施内容：

- 固定股票池输入
- 历史评分面板
- forward returns
- IC
- 分层
- Top10 vs Top100 组合回测

这是最小可用方案。

## Phase 2：历史快照股票池

目标：

- 支持不同日期不同的 100 支股票池

新增内容：

- snapshot 输入
- 每个调仓日使用对应的股票池

## Phase 3：API 化

目标：

- 让外部系统可触发验证任务

新增内容：

- 异步任务接口
- 结果查询接口
- 报告下载


## 13. 推荐的最终文档结论口径

最终报告不要写：

- “筛选后一定更好”

要写成：

- “在给定股票池内，时序评分筛选后的 Top10，在未来收益和风险调整收益上是否显著优于原始股票池”

这是统计验证问题，不是确定性承诺。


## 14. 本次建议的直接落地清单

如果现在就开工，建议按下面顺序做。

1. 在 [backend/services/temporal_scoring_service.py](/Users/fengzhi/Downloads/git/FactorHub/backend/services/temporal_scoring_service.py) 增加：
   - `refresh_factor_expressions`
   - `score_stock_history`
   - `score_many_stocks_history`

2. 新增：
   - `backend/services/temporal_filter_validation_service.py`

3. 新增：
   - `scripts/run_temporal_filter_validation.py`

4. 新增：
   - `tests/test_temporal_filter_validation_service.py`

5. 第一版先支持：
   - 固定 100 支股票输入
   - Top10 选择
   - `5/10/20` 日未来收益
   - Top10 vs Top100

这条链路跑通后，再做：

- 阈值版本
- 随机基准
- 历史快照池
- API 封装


## 15. 最终判断

这次更新后的方案，把问题定义成了一个更通用、也更真实的命题：

- 输入一组 100 支股票
- 用 `temporal_scoring_service` 选出 Top10
- 验证 Top10 是否在未来收益和风险调整收益上显著优于原始池

这是一个明确、可回测、可统计检验、可真实落地执行的技术方案。
