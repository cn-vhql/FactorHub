# 需求文档：时序筛选验证模块（temporal-filter-validation）

## 简介

本功能为 FactorHub 新增"时序筛选验证"链路。给定任意一组 100 支股票，使用现有 `temporal_scoring_service` 对股票打分并选出 Top10，通过横截面 IC 验证、分层收益验证、组合回测与统计显著性检验，量化评估 Top10 在未来收益和风险调整后收益上是否显著优于原始 100 支股票池。

本模块不修改打分公式，仅新增验证链路，不侵入现有回测主链。

---

## 词汇表

- **ValidationService**：`temporal_filter_validation_service.py` 中的核心验证服务
- **EnhancedScoringService**：增强后的 `temporal_scoring_service.py`，新增历史批量评分能力
- **score_panel**：历史评分面板，包含每只股票每个交易日的评分与排名
- **forward_returns**：未来收益面板，包含 5/10/20 日未来收益
- **research_panel**：评分面板与未来收益面板合并后的研究面板
- **IC**：信息系数（Information Coefficient），评分与未来收益的 Spearman 相关系数
- **IR**：信息比率（Information Ratio），IC 均值 / IC 标准差
- **universe100_eq**：原始 100 支股票等权持有的基准组合
- **random10_avg**：从 100 支中随机抽 10 支的随机基准（seed=42，重复 20 次取均值）
- **top10_score**：评分最高前 10 支等权持有的策略组合
- **top10_score_threshold**：先筛 score >= 阈值，再取前 10 支的策略组合
- **run_id**：验证任务的唯一标识符（UUID），用于区分不同次验证的输出目录
- **rolling rank**：`series.rolling(window, min_periods=20).rank(pct=True)`，向量化时序百分位，避免未来函数
- **score_cache_dir**：历史评分缓存目录，`data/cache/temporal_scores/`

---

## 需求

### 需求 1：增强 TemporalScoringService —— 因子表达式刷新

**用户故事：** 作为量化研究员，我希望在批量历史验证开始前能主动刷新因子表达式，以便确保使用最新的预置因子定义，而不是服务启动时可能加载的 fallback 版本。

#### 验收标准

1. THE EnhancedScoringService SHALL 新增 `refresh_factor_expressions(force: bool = False) -> None` 方法
2. WHEN `force=True`，THE EnhancedScoringService SHALL 强制重新从数据库加载因子表达式，无论距上次加载多久
3. WHEN `force=False`，THE EnhancedScoringService SHALL 仅在距上次加载超过 1 小时时刷新
4. THE EnhancedScoringService SHALL 记录 `last_refresh_at` 时间戳和 `factor_expression_source`（值为 `'db'` 或 `'fallback'`）
5. IF 数据库连接失败，THEN THE EnhancedScoringService SHALL 使用 fallback 表达式，并将 `factor_expression_source` 设为 `'fallback'`，不静默继续（需在日志中明确记录）

### 需求 2：增强 TemporalScoringService —— 单只股票历史评分

**用户故事：** 作为量化研究员，我希望一次性获取某只股票在整个历史区间内的每日评分序列，以便避免逐日重复取数和计算，提升批量验证效率。

#### 验收标准

1. THE EnhancedScoringService SHALL 新增 `score_stock_history(code, start_date, end_date, window=252) -> pd.DataFrame` 方法
2. THE EnhancedScoringService SHALL 一次拉取 `start_date - window 个自然日` 到 `end_date` 的全量价格数据（仅一次 IO）
3. THE EnhancedScoringService SHALL 使用 `series.rolling(window=window, min_periods=20).rank(pct=True)` 向量化计算时序百分位，**不得**使用逐日 `percentileofscore` 循环
4. THE EnhancedScoringService SHALL 只在输出中包含 `[start_date, end_date]` 区间内的行，预热期数据不输出
5. THE EnhancedScoringService SHALL 返回 DataFrame，列为：`date, code, score, trend_break, high_volatility, liquidity_risk`
6. THE EnhancedScoringService SHALL 将结果缓存到 `data/cache/temporal_scores/{code}_{start_date}_{end_date}.parquet`
7. WHEN 缓存文件存在，THE EnhancedScoringService SHALL 直接读取缓存，不重新计算
8. IF 有效交易日数量不足以支撑计算（< window + 20），THEN THE EnhancedScoringService SHALL 抛出 `ValueError`

### 需求 3：增强 TemporalScoringService —— 批量历史评分

**用户故事：** 作为量化研究员，我希望批量生成一个股票池的历史评分面板，以便高效构建研究所需的横截面数据。

#### 验收标准

1. THE EnhancedScoringService SHALL 新增 `score_many_stocks_history(codes, start_date, end_date, max_workers=4) -> pd.DataFrame` 方法
2. THE EnhancedScoringService SHALL 使用 `ThreadPoolExecutor(max_workers=max_workers)` 并发处理每只股票
3. WHEN 某只股票处理失败，THE EnhancedScoringService SHALL 将失败信息记录到 `self._last_history_errors`（格式：`{"code": ..., "error": ...}`），并继续处理其余股票
4. THE EnhancedScoringService SHALL 返回所有成功股票的评分面板（`pd.concat` 合并），列为：`date, code, score, trend_break, high_volatility, liquidity_risk`
5. FOR ALL 输入股票，`len(成功结果中的唯一 code) + len(_last_history_errors)` SHALL 等于 `len(codes)`（总数守恒）

### 需求 4：加载股票池

**用户故事：** 作为量化研究员，我希望从 CSV 文件加载股票池，支持固定池和历史快照池两种格式，以便灵活输入不同类型的候选股票。

#### 验收标准

1. THE ValidationService SHALL 实现 `load_fixed_pool(csv_path) -> list[str]`，读取含 `code` 列的 CSV，返回去重后的股票代码列表
2. THE ValidationService SHALL 实现 `load_snapshot_pool(csv_path) -> pd.DataFrame`，读取含 `date, code` 列的 CSV，`date` 列转为 `DatetimeIndex`
3. IF CSV 文件不存在，THEN THE ValidationService SHALL 抛出 `FileNotFoundError`
4. IF CSV 文件缺少必需列，THEN THE ValidationService SHALL 抛出 `ValueError` 并说明缺少的列名
5. THE ValidationService SHALL 对股票代码执行去重处理，不抛出异常

### 需求 5：构建评分面板

**用户故事：** 作为量化研究员，我希望构建包含历史评分和截面排名的面板数据，以便后续进行 IC 计算和分层分析。

#### 验收标准

1. THE ValidationService SHALL 实现 `build_score_panel(codes, start_date, end_date) -> pd.DataFrame`
2. THE ValidationService SHALL 调用 `scoring_service.score_many_stocks_history(codes, start_date, end_date)` 获取历史评分
3. THE ValidationService SHALL 按 `date` 分组，计算截面内 `score` 的降序排名，结果存入 `rank` 列（1 = 最高分）
4. THE ValidationService SHALL 返回 DataFrame，列为：`date, code, score, rank, trend_break, high_volatility, liquidity_risk`

### 需求 6：构建未来收益面板

**用户故事：** 作为量化研究员，我希望构建防止未来函数的未来收益面板，以便进行有效的预测能力验证。

#### 验收标准

1. THE ValidationService SHALL 实现 `build_forward_returns(codes, start_date, end_date, horizons=(5,10,20)) -> pd.DataFrame`
2. THE ValidationService SHALL 按以下公式计算未来收益，**不得**使用 `t` 日当日数据作为建仓价格：
   - `future_return_h = close[t+h] / open[t+1] - 1`
   - 若 `open[t+1]` 不可用，退化为 `close[t+h] / close[t+1] - 1`
3. THE ValidationService SHALL 返回 DataFrame，列为：`date, code, future_return_5, future_return_10, future_return_20`（horizons 可配置）
4. WHEN 某只股票价格数据不足以计算某个 horizon 的未来收益，THE ValidationService SHALL 对应行填 `NaN`，不抛出异常

### 需求 7：构建研究面板

**用户故事：** 作为量化研究员，我希望将评分面板与未来收益面板合并，以便在同一数据集上进行所有验证分析。

#### 验收标准

1. THE ValidationService SHALL 实现 `build_research_panel(score_panel, forward_returns) -> pd.DataFrame`
2. THE ValidationService SHALL 按 `(date, code)` 内连接两个面板
3. THE ValidationService SHALL 返回 DataFrame，列为：`date, code, score, rank, future_return_5, future_return_10, future_return_20`

### 需求 8：横截面 IC 验证

**用户故事：** 作为量化研究员，我希望计算评分与未来收益的横截面 IC，以便量化评分的预测能力。

#### 验收标准

1. THE ValidationService SHALL 实现 `calculate_cross_sectional_ic(panel, horizons=(5,10,20)) -> dict`
2. THE ValidationService SHALL 对每个调仓日，在当日股票池内计算 Spearman 相关系数：`IC_h(t) = spearmanr(rank(score_t), future_return_h_t)`
3. THE ValidationService SHALL 对每个 horizon 输出以下统计量：`ic_mean`、`ic_std`、`ir`（= ic_mean / ic_std）、`positive_ratio`（IC > 0 的比例）、`t_stat`、`p_value`
4. THE ValidationService SHALL 返回字典，key 为 `ic_5`、`ic_10`、`ic_20`（对应 horizons）
5. IF 某个 horizon 的有效样本量 < 30，THEN THE ValidationService SHALL 在对应结果中标记 `insufficient_samples: true`，不抛出异常

### 需求 9：分层收益验证

**用户故事：** 作为量化研究员，我希望按评分分层分析各组的未来收益，以便验证评分的单调性预测能力。

#### 验收标准

1. THE ValidationService SHALL 实现 `analyze_quantile_spread(panel, n_quantiles=5, horizons=(5,10,20)) -> dict`
2. THE ValidationService SHALL 对每个调仓日，按 `score` 将股票分成 `n_quantiles` 组（Q1 为最高分组）
3. THE ValidationService SHALL 对每组计算：`mean_return`、`annual_return`、`sharpe`、`win_rate`、`sample_count`
4. THE ValidationService SHALL 计算 `spread`（Q1 - Q_last 的收益差）及其统计显著性
5. THE ValidationService SHALL 返回字典，key 为 `horizon_5`、`horizon_10`、`horizon_20`，每个 key 下包含各分组统计

### 需求 10：组合回测

**用户故事：** 作为量化研究员，我希望同时运行 4 组策略的组合回测，以便直接对比 Top10 筛选策略与基准的表现。

#### 验收标准

1. THE ValidationService SHALL 实现 `run_portfolio_backtest(panel, top_n=10, rebalance='W-FRI', score_threshold=None) -> pd.DataFrame`
2. THE ValidationService SHALL 同时输出以下 4 组策略的净值曲线：
   - `universe100_eq`：原始 100 支等权持有
   - `random10_avg`：随机抽 10 支（`seed=42`，重复 20 次取均值）
   - `top10_score`：评分最高前 `top_n` 支等权持有
   - `top10_score_threshold`：先筛 `score >= score_threshold`，再取前 `top_n` 支等权持有
3. THE ValidationService SHALL 按 `rebalance` 频率（默认 `W-FRI`）调仓
4. WHEN 策略 D（`top10_score_threshold`）可选股票数量 < `top_n`，THE ValidationService SHALL 对不足部分分配现金权重（收益为 0），总权重仍为 1
5. THE ValidationService SHALL 使用固定随机种子 `seed=42` 生成随机基准，保证结果可复现
6. THE ValidationService SHALL 不依赖 vectorbt，使用自行实现的等权再平衡回测器
7. THE ValidationService SHALL 返回 DataFrame，列为：`date, universe100_eq, random10_avg, top10_score, top10_score_threshold`

### 需求 11：统计显著性检验

**用户故事：** 作为量化研究员，我希望对策略间的收益差异进行统计显著性检验，以便判断 Top10 筛选的增益是否具有统计意义。

#### 验收标准

1. THE ValidationService SHALL 实现 `compare_strategies(curves_or_returns) -> dict`
2. THE ValidationService SHALL 计算每个策略的以下指标：`annual_return`、`sharpe`、`max_drawdown`、`volatility`、`win_rate`
3. THE ValidationService SHALL 对以下策略对进行配对 t 检验（`scipy.stats.ttest_rel`）：
   - `top10_score` vs `universe100_eq`
   - `top10_score` vs `random10_avg`
4. THE ValidationService SHALL 计算信息比率（IR = 超额收益均值 / 超额收益标准差）
5. THE ValidationService SHALL 返回字典，包含 `metrics`、`statistical_tests`、`conclusion` 三个 key
6. IF 样本量 < 30，THEN THE ValidationService SHALL 跳过统计检验，在结果中标记 `insufficient_samples: true`

### 需求 12：输出文件

**用户故事：** 作为量化研究员，我希望验证结果以规范的文件格式输出，以便后续分析和存档。

#### 验收标准

1. THE ValidationService SHALL 将所有输出文件写入 `data/reports/temporal_filter_validation/{run_id}/` 目录
2. THE ValidationService SHALL 输出以下文件：
   - `score_panel.parquet`：列 `date, code, score, rank, trend_break, high_volatility, liquidity_risk`
   - `forward_returns.parquet`：列 `date, code, future_return_5, future_return_10, future_return_20`
   - `ic_summary.csv`：列 `horizon, ic_mean, ic_std, ir, positive_ratio, t_stat, p_value`
   - `quantile_summary.csv`：列 `horizon, quantile, mean_return, annual_return, sharpe, win_rate, sample_count`
   - `portfolio_curves.csv`：列 `date, universe100_eq, random10_avg, top10_score, top10_score_threshold`
   - `strategy_comparison.json`：组合对比与统计显著性结果
   - `scoring_errors.csv`：列 `code, error`，记录打分失败的股票（停牌/数据不足/因子NaN等），即使为空也输出
   - `validation_report.md`：可读结论报告
3. THE ValidationService SHALL 在 `validation_report.md` 中明确标注 `factor_expression_source`（`'db'` 或 `'fallback'`）
4. THE ValidationService SHALL 在 `validation_report.md` 中使用统计验证语言，不得写"筛选后一定更好"等确定性结论

### 需求 13：CLI 脚本

**用户故事：** 作为量化研究员，我希望通过命令行脚本触发完整的验证流程，以便在研究环境中快速运行和复现验证。

#### 验收标准

1. THE CLI SHALL 新增 `scripts/run_temporal_filter_validation.py` 脚本
2. THE CLI SHALL 支持以下参数：`--pool-csv`（必填）、`--start-date`（必填）、`--end-date`（必填）、`--top-n`（默认 10）、`--score-threshold`（默认 None）、`--rebalance`（默认 `W-FRI`）、`--horizons`（默认 `5,10,20`）、`--max-workers`（默认 4）、`--output-dir`（默认 `data/reports/temporal_filter_validation/`）、`--n-quantiles`（默认 5）
3. THE CLI SHALL 在执行前调用 `temporal_scoring_service.refresh_factor_expressions(force=True)`
4. THE CLI SHALL 在执行完成后打印输出目录路径
5. IF 任何步骤失败，THEN THE CLI SHALL 打印错误信息并以非零退出码退出

### 需求 14：测试覆盖

**用户故事：** 作为开发人员，我希望有完整的测试覆盖，以便验证模块的正确性，特别是防止未来函数和权重守恒等关键属性。

#### 验收标准

1. THE 测试模块 SHALL 新增 `tests/test_temporal_filter_validation_service.py`
2. THE 测试模块 SHALL 覆盖以下场景：
   - 固定股票池能跑通完整链路
   - `t` 日评分只使用 `t` 及以前的数据（无未来函数）
   - `future_return_h` 从 `t+1` 开始计算
   - Top10 组合每次调仓总权重为 1
   - 策略 D 不足 10 支时现金权重正确
   - IC 输出包含所有 horizon 维度，p 值在 [0, 1]
3. THE 测试模块 SHALL 使用 Hypothesis 库编写属性测试，覆盖设计文档中的属性 1-8
4. THE 测试模块 SHALL 对每个属性测试至少运行 100 次（`@settings(max_examples=100)`）
