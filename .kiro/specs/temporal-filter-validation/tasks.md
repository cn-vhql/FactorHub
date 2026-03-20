# 实施任务清单：时序筛选验证模块（temporal-filter-validation）

## 概述

基于需求文档与技术设计文档，将时序筛选验证模块拆分为以下实施任务。任务 1（增强打分服务）是基础，任务 2（验证主服务）依赖任务 1，任务 3（CLI 脚本）依赖任务 2，任务 4（测试）贯穿全程。

---

## 任务列表

- [x] 1. 增强 TemporalScoringService —— 新增历史批量评分能力
  - [x] 1.1 新增 `refresh_factor_expressions(force: bool = False) -> None` 方法：
    - 记录 `self.last_refresh_at: datetime` 和 `self.factor_expression_source: str`（`'db'` 或 `'fallback'`）
    - `force=True` 时强制重新加载；`force=False` 时仅在距上次加载超过 1 小时时刷新
    - DB 失败时使用 fallback，并在日志中明确记录（不静默继续）
    - _需求：1.1、1.2、1.3、1.4、1.5_

  - [x] 1.2 新增 `score_stock_history(code, start_date, end_date, window=252) -> pd.DataFrame` 方法：
    - 拉取 `start_date - window 个自然日` 到 `end_date` 的全量价格数据（一次 IO）
    - 一次计算完整 8 因子时间序列
    - 使用 `series.rolling(window=window, min_periods=20).rank(pct=True)` 向量化计算时序百分位
    - 向量化生成每日 `score`、`trend_break`、`high_volatility`、`liquidity_risk`
    - 只输出 `[start_date, end_date]` 区间内的行
    - 结果缓存到 `data/cache/temporal_scores/{code}_{start_date}_{end_date}.parquet`
    - 缓存命中时直接读取，不重新计算
    - _需求：2.1、2.2、2.3、2.4、2.5、2.6、2.7、2.8_

  - [x] 1.3 新增 `score_many_stocks_history(codes, start_date, end_date, max_workers=4) -> pd.DataFrame` 方法：
    - 使用 `ThreadPoolExecutor(max_workers=max_workers)` 并发处理每只股票
    - 失败的股票记录到 `self._last_history_errors: list[dict]`，不中断整体
    - 最后 `pd.concat` 所有成功结果并返回
    - _需求：3.1、3.2、3.3、3.4、3.5_

  - [x] 1.4 创建评分缓存目录：
    - 确保 `data/cache/temporal_scores/` 目录在服务初始化时自动创建

- [x] 2. 新增 TemporalFilterValidationService —— 股票池加载与面板构建
  - [x] 2.1 新建 `backend/services/temporal_filter_validation_service.py`，定义 `TemporalFilterValidationService` 类骨架：
    - `__init__` 中初始化 `scoring_service`、`data_service`、`stats_service`
    - 初始化 `output_base_dir = Path("data/reports/temporal_filter_validation")`
    - 初始化 `score_cache_dir = Path("data/cache/temporal_scores")`
    - 自动创建上述目录

  - [x] 2.2 实现 `load_fixed_pool(csv_path) -> list[str]`：
    - 读取含 `code` 列的 CSV，返回去重后的股票代码列表
    - 文件不存在时抛出 `FileNotFoundError`
    - 缺少 `code` 列时抛出 `ValueError`
    - _需求：4.1、4.3、4.4、4.5_

  - [x] 2.3 实现 `load_snapshot_pool(csv_path) -> pd.DataFrame`：
    - 读取含 `date, code` 列的 CSV，`date` 列转为 `DatetimeIndex`
    - 文件不存在时抛出 `FileNotFoundError`
    - 缺少必需列时抛出 `ValueError`
    - _需求：4.2、4.3、4.4_

  - [x] 2.4 实现 `build_score_panel(codes, start_date, end_date) -> pd.DataFrame`：
    - 调用 `scoring_service.score_many_stocks_history(codes, start_date, end_date)`
    - 按 `date` 分组，计算截面内 `score` 的降序排名（`rank` 列，1 = 最高分）
    - 返回列：`date, code, score, rank, trend_break, high_volatility, liquidity_risk`
    - _需求：5.1、5.2、5.3、5.4_

  - [x] 2.5 实现 `build_forward_returns(codes, start_date, end_date, horizons=(5,10,20)) -> pd.DataFrame`：
    - 从 `data_service` 批量获取价格数据
    - 建仓价格使用 `open[t+1]`（不可用时退化为 `close[t+1]`）
    - `future_return_h = close[t+h] / entry_price - 1`
    - 数据不足时对应行填 `NaN`，不抛出异常
    - 返回列：`date, code, future_return_5, future_return_10, future_return_20`
    - _需求：6.1、6.2、6.3、6.4_

  - [x] 2.6 实现 `build_research_panel(score_panel, forward_returns) -> pd.DataFrame`：
    - 按 `(date, code)` 内连接两个面板
    - 返回列：`date, code, score, rank, future_return_5, future_return_10, future_return_20`
    - _需求：7.1、7.2、7.3_

- [x] 3. 新增 TemporalFilterValidationService —— IC 验证与分层分析
  - [x] 3.1 实现 `calculate_cross_sectional_ic(panel, horizons=(5,10,20)) -> dict`：
    - 对每个调仓日，在截面内计算 `spearmanr(rank(score), future_return_h)`
    - 对每个 horizon 输出：`ic_mean`、`ic_std`、`ir`、`positive_ratio`、`t_stat`、`p_value`
    - 样本量 < 30 时标记 `insufficient_samples: true`，不抛出异常
    - 返回字典，key 为 `ic_5`、`ic_10`、`ic_20`
    - _需求：8.1、8.2、8.3、8.4、8.5_

  - [x] 3.2 实现 `analyze_quantile_spread(panel, n_quantiles=5, horizons=(5,10,20)) -> dict`：
    - 对每个调仓日，按 `score` 将股票分成 `n_quantiles` 组（Q1 为最高分组）
    - 对每组计算：`mean_return`、`annual_return`、`sharpe`、`win_rate`、`sample_count`
    - 计算 `spread`（Q1 - Q_last）及其统计显著性
    - 返回字典，key 为 `horizon_5`、`horizon_10`、`horizon_20`
    - _需求：9.1、9.2、9.3、9.4、9.5_

- [x] 4. 新增 TemporalFilterValidationService —— 组合回测
  - [x] 4.1 实现等权再平衡回测器（内部辅助方法 `_equal_weight_backtest`）：
    - 输入：`holdings_by_date: dict[date, list[str]]`，`price_data: dict[str, pd.DataFrame]`
    - 按调仓日等权分配，持有到下次调仓
    - 返回每日净值序列

  - [x] 4.2 实现 `run_portfolio_backtest(panel, top_n=10, rebalance='W-FRI', score_threshold=None) -> pd.DataFrame`：
    - 策略 A（`universe100_eq`）：原始 100 支等权持有
    - 策略 B（`random10_avg`）：`seed=42`，重复 20 次随机抽 `top_n` 支，取均值
    - 策略 C（`top10_score`）：评分最高前 `top_n` 支等权持有
    - 策略 D（`top10_score_threshold`）：先筛 `score >= score_threshold`，再取前 `top_n` 支；不足时剩余权重留现金
    - 按 `rebalance` 频率调仓（默认 `W-FRI`）
    - 不依赖 vectorbt
    - 返回列：`date, universe100_eq, random10_avg, top10_score, top10_score_threshold`
    - _需求：10.1、10.2、10.3、10.4、10.5、10.6、10.7_

- [x] 5. 新增 TemporalFilterValidationService —— 统计显著性检验与输出
  - [x] 5.1 实现 `compare_strategies(curves_or_returns) -> dict`：
    - 计算每个策略的：`annual_return`、`sharpe`、`max_drawdown`、`volatility`、`win_rate`
    - 对 `top10_score vs universe100_eq` 和 `top10_score vs random10_avg` 进行配对 t 检验
    - 计算信息比率（IR）
    - 样本量 < 30 时标记 `insufficient_samples: true`
    - 返回字典，包含 `metrics`、`statistical_tests`、`conclusion`
    - _需求：11.1、11.2、11.3、11.4、11.5、11.6_

  - [x] 5.2 实现输出文件写入逻辑：
    - 生成 `run_id`（UUID）
    - 创建输出目录 `data/reports/temporal_filter_validation/{run_id}/`
    - 写入 `score_panel.parquet`、`forward_returns.parquet`
    - 写入 `ic_summary.csv`、`quantile_summary.csv`、`portfolio_curves.csv`
    - 写入 `strategy_comparison.json`
    - 写入 `scoring_errors.csv`（列：`code, error`，来自 `scoring_service._last_history_errors`，即使为空也输出空文件）
    - 生成 `validation_report.md`（含 `factor_expression_source` 标注，使用统计验证语言）
    - _需求：12.1、12.2、12.3、12.4_

- [ ] 6. 检查点 —— ValidationService 自测
  - 确保所有测试通过，如有疑问请向用户确认。

- [x] 7. 新增 CLI 脚本
  - 新建 `scripts/run_temporal_filter_validation.py`
  - 使用 `argparse` 实现以下参数：
    - `--pool-csv`（必填）
    - `--start-date`（必填）
    - `--end-date`（必填）
    - `--top-n`（默认 10）
    - `--score-threshold`（默认 None）
    - `--rebalance`（默认 `W-FRI`）
    - `--horizons`（默认 `5,10,20`，逗号分隔）
    - `--max-workers`（默认 4）
    - `--output-dir`（默认 `data/reports/temporal_filter_validation/`）
    - `--n-quantiles`（默认 5）
  - 执行前调用 `temporal_scoring_service.refresh_factor_expressions(force=True)`
  - 执行完成后打印输出目录路径
  - 失败时打印错误信息并以非零退出码退出
  - _需求：13.1、13.2、13.3、13.4、13.5_

- [x] 8. 新增测试文件
  - 新建 `tests/test_temporal_filter_validation_service.py`

  - [x] 8.1 编写单元测试 —— 固定股票池完整链路：
    - 使用 mock 数据（5 支股票，2 年历史）跑通完整链路
    - 验证所有输出文件存在且格式正确
    - _需求：14.2_

  - [x] 8.2 编写单元测试 —— 未来收益防未来函数：
    - 验证 `future_return_h` 的建仓价格为 `t+1` 日，不使用 `t` 日数据
    - _需求：14.2_

  - [x] 8.3 编写单元测试 —— Top10 权重守恒：
    - 验证策略 C 每次调仓总权重 = 1
    - _需求：14.2_

  - [x] 8.4 编写单元测试 —— 策略 D 现金权重：
    - 验证不足 `top_n` 支时，现金权重 = `1 - sum(stock_weights)`，总权重 = 1
    - _需求：14.2_

  - [x] 8.5 编写属性测试 —— 无未来函数（属性 1）：
    - **属性 1：无未来函数**
    - **验证：需求 2.3**
    - 使用 `@given` 生成随机价格序列，验证 rolling rank 不使用未来数据
    - `@settings(max_examples=100)`

  - [x] 8.6 编写属性测试 —— 未来收益从 t+1 开始（属性 2）：
    - **属性 2：未来收益从 t+1 开始**
    - **验证：需求 6.2**
    - 使用 `@given` 生成随机价格序列，验证 `entry_price = open[t+1]`
    - `@settings(max_examples=100)`

  - [x] 8.7 编写属性测试 —— Top10 权重守恒（属性 3）：
    - **属性 3：Top10 权重守恒**
    - **验证：需求 10.2**
    - 使用 `@given` 生成随机评分面板（股票数 >= top_n），验证权重之和 = 1
    - `@settings(max_examples=100)`

  - [x] 8.8 编写属性测试 —— 策略 D 现金权重（属性 4）：
    - **属性 4：策略 D 现金权重正确**
    - **验证：需求 10.4**
    - 使用 `@given` 生成少于 top_n 支的评分面板，验证总权重 = 1
    - `@settings(max_examples=100)`

  - [x] 8.9 编写属性测试 —— IC 输出完整性（属性 5）：
    - **属性 5：IC 输出完整性**
    - **验证：需求 8.3、8.4**
    - 使用 `@given` 生成随机 research panel，验证输出包含所有 horizon 维度，p 值在 [0, 1]
    - `@settings(max_examples=100)`

  - [x] 8.10 编写属性测试 —— 随机基准可复现（属性 6）：
    - **属性 6：随机基准可复现**
    - **验证：需求 10.5**
    - 两次调用 `run_portfolio_backtest`，验证 `random10_avg` 列完全一致
    - `@settings(max_examples=50)`

  - [x] 8.11 编写属性测试 —— 失败样本记录完整（属性 8）：
    - **属性 8：失败样本记录完整**
    - **验证：需求 3.5**
    - mock 部分股票抛出异常，验证 `len(成功) + len(失败) == len(输入)`
    - `@settings(max_examples=100)`

  - [x] 8.12 编写属性测试 —— 评分缓存一致性（属性 7）：
    - **属性 7：评分缓存一致性**
    - **验证：需求 2.6、2.7**
    - 使用 `@given` 生成随机 `(code, start_date, end_date)`，删除缓存后重新计算，验证两次结果完全一致
    - `@settings(max_examples=50)`

- [ ] 9. 最终检查点 —— 集成验收测试
  - [x] 9.1 使用 5 支真实股票（mock 数据）跑通完整链路，验证所有输出文件存在
  - [x] 9.2 验证 `portfolio_curves.csv` 的日期范围与输入 `[start_date, end_date]` 一致
  - [x] 9.3 验证 `ic_summary.csv` 包含所有 horizon 行（5、10、20）
  - [x] 9.4 验证 `validation_report.md` 包含 `factor_expression_source` 字段
  - 确保所有测试通过，如有疑问请向用户确认。

---

## 备注

- 任务 1 是基础，必须先完成再开始任务 2
- 任务 2-5 可按顺序开发，每个子任务完成后运行对应测试
- 属性测试使用 Hypothesis 库，每个属性最少运行 100 次
- 属性测试注释格式：`# Feature: temporal-filter-validation, Property N: <属性名>`
- Phase 1 只需支持固定股票池（`load_fixed_pool`），`load_snapshot_pool` 可在 Phase 2 实现
- 随机基准必须固定 `seed=42`，否则每次验证结果不可比
- 失败样本必须保留在 `_last_history_errors` 中，不能静默丢弃
