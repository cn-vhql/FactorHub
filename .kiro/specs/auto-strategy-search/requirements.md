# 需求文档

## 简介

本功能将 FactorHub 从"固定 8 因子打分器"升级为"自动化策略搜索与应用平台"。核心目标是：将策略模板配置化，通过受约束搜索自动评选出最优 Champion 策略，再统一接入 `temporal_pool`（单股时序打分循环汇总）和 `cross_sectional_filter`（截面因子组合打分）两条执行路径。整个方案不引入新三方库，完全基于现有模块扩展实现，搜索结果以 JSON/CSV/Markdown 文件落盘，不新增数据库表。

## 术语表

- **FactorProfileRegistryService**：负责加载、校验、分发 profile 配置和 search space 配置的服务
- **TemporalScoringService**：对单只股票按时序策略打分的服务
- **TemporalPoolService**：对股票池循环调用单股打分并汇总排序的服务
- **StrategySearchService**：生成候选策略的搜索服务（模板枚举 + 受约束随机 + 逐步精简）
- **StrategyEvaluationService**：对候选策略执行 walk-forward 回测并计算稳定性总分的服务
- **ChampionStrategyService**：独立运行搜索任务、选出最优 champion 并落盘的服务
- **ChampionRegistryService**：以文件方式持久化 champion 配置的注册服务
- **Profile**：一套完整的策略配置，包含因子列表、权重、方向、参数等
- **SearchSpace**：定义候选因子集合、参数范围和约束条件的搜索空间配置
- **Champion**：经过 walk-forward 评价后被选出的最优策略配置
- **WalkForwardWindow**：滚动验证中的单个时间窗口，包含训练期、验证期、测试期
- **StabilityScore**：综合多个 walk-forward 窗口指标计算出的策略稳定性总分（0-100）
- **scope_type**：Champion 的作用域类型，取值为 `single_stock`、`stock_group`、`cross_sectional_universe`
- **scope_key**：作用域标识，单股为股票代码，股票组为列表 hash，截面为 universe 名称
- **temporal_pool**：单股时序打分循环汇总模式
- **cross_sectional_filter**：截面因子组合打分排序模式
- **TopN**：从池内评分中按分数降序选出的前 N 只股票


## 需求列表

### 需求 1：Profile 配置注册与校验

**用户故事：** 作为量化研究员，我希望系统能从配置文件加载和校验策略模板，以便我可以通过修改配置文件来管理和扩展策略模板，而无需修改代码。

#### 验收标准

1. THE FactorProfileRegistryService SHALL 从 `config/temporal_pool_profiles.json` 和 `config/cross_sectional_profiles.json` 加载所有 profile 配置
2. WHEN 调用 `load_profiles(mode)` 时，THE FactorProfileRegistryService SHALL 仅返回与指定 mode 匹配的 profile 列表
3. WHEN 调用 `get_profile(profile_id)` 且 profile_id 存在时，THE FactorProfileRegistryService SHALL 返回对应的完整 profile 配置
4. IF `profile_id` 在配置中不存在，THEN THE FactorProfileRegistryService SHALL 抛出 `KeyError`，错误信息包含该 profile_id
5. WHEN 调用 `validate_profile(profile)` 且 profile 缺少必填字段（id、mode、factors、params）时，THE FactorProfileRegistryService SHALL 抛出 `ValueError`，错误信息说明缺少的字段
6. WHEN 调用 `validate_profile(profile)` 且 signal 类因子权重之和不在 [0.999, 1.001] 范围内时，THE FactorProfileRegistryService SHALL 抛出 `ValueError`
7. WHEN 调用 `validate_profile(profile)` 且 `mode` 字段不是 `temporal_pool` 或 `cross_sectional_filter` 时，THE FactorProfileRegistryService SHALL 抛出 `ValueError`
8. WHEN 调用 `validate_profile(profile)` 且 `params.top_n` 不在 [1, 100] 范围内时，THE FactorProfileRegistryService SHALL 抛出 `ValueError`
9. WHEN 调用 `validate_profile(profile)` 且 `params.percentile_window` 不在 [20, 504] 范围内时，THE FactorProfileRegistryService SHALL 抛出 `ValueError`
10. THE FactorProfileRegistryService SHALL 支持热重载，当配置文件变更后重新加载最新配置


### 需求 2：TemporalScoringService Profile 化改造

**用户故事：** 作为量化研究员，我希望时序打分服务支持按 profile 配置动态打分，以便我可以在不修改代码的情况下切换不同的因子组合和参数配置。

#### 验收标准

1. THE TemporalScoringService SHALL 新增 `score_one_stock_with_profile(code, trade_date, profile_id)` 接口，按指定 profile 对单只股票打分
2. WHEN 调用 `score_one_stock_with_profile` 时，THE TemporalScoringService SHALL 返回包含 `date`、`code`、`score`、`factors`、`pcts` 字段的字典
3. WHEN 调用 `score_one_stock_with_profile` 时，THE TemporalScoringService SHALL 返回 `score` 值在 [0, 100] 范围内
4. WHEN 调用 `score_one_stock_with_profile` 时，THE TemporalScoringService SHALL 返回 `pcts` 中每个因子百分位值在 [0, 1] 范围内
5. THE TemporalScoringService SHALL 新增 `score_stock_history_with_profile(code, start_date, end_date, profile_id, window)` 接口，计算单只股票历史评分面板
6. THE TemporalScoringService SHALL 保留现有 `score_one_stock` 和 `score_stock_history` 接口，不改变其行为
7. THE TemporalScoringService SHALL 将现有 8 因子硬编码逻辑提取为 `default_temporal_v1` profile，作为默认配置
8. WHEN 生成缓存 key 时，THE TemporalScoringService SHALL 将 `profile_id` 纳入缓存 key，格式为 `{code}_{start_date}_{end_date}_{profile_id}`，避免不同 profile 缓存冲突
9. IF 股票在指定时间段内有效交易日数少于 `percentile_window + 20`，THEN THE TemporalScoringService SHALL 抛出 `ValueError`


### 需求 3：TemporalPoolService 股票池打分与 TopN 筛选

**用户故事：** 作为量化研究员，我希望系统能对股票池中的每只股票独立打分并汇总排序，以便实现"100 → 10"的池内筛选场景，无需开发新模型。

#### 验收标准

1. THE TemporalPoolService SHALL 实现 `score_pool(codes, profile_id, trade_date, max_workers)` 接口，对股票池逐只打分并返回含 `code`、`score`、`rank` 列的 DataFrame
2. WHEN 调用 `score_pool` 时，THE TemporalPoolService SHALL 返回的 `rank` 列为从 1 开始的连续整数，按 score 降序排列（score 最高者 rank=1）
3. WHEN 调用 `score_pool` 时，THE TemporalPoolService SHALL 按 score 降序排列结果
4. IF 某只股票打分失败，THEN THE TemporalPoolService SHALL 跳过该股票并继续处理其他股票，不影响整体结果
5. THE TemporalPoolService SHALL 实现 `select_top_n(pool_scores, top_n, score_threshold)` 接口，从池内评分中选出 TopN
6. WHEN 调用 `select_top_n` 时，THE TemporalPoolService SHALL 返回数量不超过 `top_n`
7. WHEN 调用 `select_top_n` 且指定 `score_threshold` 时，THE TemporalPoolService SHALL 仅返回 score 大于等于 `score_threshold` 的股票
8. THE TemporalPoolService SHALL 实现 `score_pool_history(codes, profile_id, start_date, end_date, max_workers)` 接口，构建历史池内评分面板，返回含 `date`、`code`、`score`、`rank` 列的 DataFrame
9. WHEN 调用 `score_pool` 且 `max_workers` 大于 1 时，THE TemporalPoolService SHALL 并发调用 `TemporalScoringService.score_one_stock_with_profile`


### 需求 4：StrategySearchService 候选策略生成

**用户故事：** 作为量化研究员，我希望系统能自动生成多套候选策略，以便通过系统化搜索找到比手工调参更优的策略配置。

#### 验收标准

1. THE StrategySearchService SHALL 实现三层搜索：第一层模板枚举、第二层受约束随机搜索、第三层逐步精简
2. WHEN 执行模板枚举时，THE StrategySearchService SHALL 将所有内置 profile 作为候选策略返回
3. WHEN 执行受约束随机搜索时，THE StrategySearchService SHALL 生成的每个候选策略的 signal 因子权重之和在 [0.999, 1.001] 范围内
4. WHEN 执行受约束随机搜索时，THE StrategySearchService SHALL 生成的每个候选策略至少包含 1 个 `risk_filter` 类型的因子
5. WHEN 执行受约束随机搜索时，THE StrategySearchService SHALL 生成的每个候选策略的因子总数在 `[min_factors, max_factors]` 范围内
6. WHEN 执行受约束随机搜索时，THE StrategySearchService SHALL 确保同一因子组（factor_group）内被选中的因子数不超过 `max_per_group`
7. WHEN 执行受约束随机搜索时，THE StrategySearchService SHALL 确保单个 signal 因子权重不超过 0.40
8. THE StrategySearchService SHALL 实现 `search_for_stock`、`search_for_pool`、`search_for_cross_sectional` 三个作用域入口
9. WHEN 受约束随机搜索无法在 `n * 10` 次尝试内生成足够候选时，THE StrategySearchService SHALL 返回已生成的候选（数量可能少于 n），不抛出异常


### 需求 5：StrategyEvaluationService Walk-Forward 评价

**用户故事：** 作为量化研究员，我希望系统能对候选策略执行滚动验证评价，以便通过多时间窗口的稳定性指标选出真正鲁棒的策略，而非过拟合的策略。

#### 验收标准

1. THE StrategyEvaluationService SHALL 实现 `evaluate_temporal_pool_candidate` 和 `evaluate_cross_sectional_candidate` 两个评价入口
2. WHEN 执行 walk-forward 评价时，THE StrategyEvaluationService SHALL 按默认参数（训练期 12 个月、验证期 3 个月、测试期 3 个月、滚动步长 3 个月）生成滚动窗口
3. WHEN 生成 walk-forward 窗口时，THE StrategyEvaluationService SHALL 确保各窗口的训练期、验证期、测试期无重叠
4. WHEN 生成 walk-forward 窗口时，THE StrategyEvaluationService SHALL 确保窗口覆盖完整的指定时间范围
5. WHEN 某个 walk-forward 窗口的测试期 Sharpe 小于训练期 Sharpe 的 0.3 倍时，THE StrategyEvaluationService SHALL 在该窗口结果中标记 `overfitting_flag=True`
6. THE StrategyEvaluationService SHALL 实现 `compute_stability_score(window_metrics)` 方法，按以下权重计算稳定性总分：OOS Sharpe 30%、OOS 超额收益 20%、回撤惩罚 15%、IC/IR 10%、正收益窗口比例 10%、风格阶段一致性 10%、换手率惩罚 5%
7. WHEN 调用 `compute_stability_score` 时，THE StrategyEvaluationService SHALL 返回值在 [0.0, 100.0] 范围内
8. IF 所有 walk-forward 窗口的测试期 Sharpe 均为负，THEN THE StrategyEvaluationService SHALL 返回的稳定性总分小于 30
9. WHEN 调用 `evaluate_temporal_pool_candidate` 时，THE StrategyEvaluationService SHALL 复用 `temporal_filter_validation_service` 和 `vectorbt_backtest_service` 执行回测，不重复实现回测逻辑


### 需求 6：ChampionStrategyService 独立搜索任务执行

**用户故事：** 作为量化研究员，我希望系统能独立运行完整的 champion 搜索流程，以便在不依赖前端页面的情况下自动为单只股票、股票组或截面 universe 选出最优策略。

#### 验收标准

1. THE ChampionStrategyService SHALL 实现 `run_for_stock`、`run_for_pool`、`run_for_cross_sectional` 三个独立运行入口
2. WHEN 执行 champion 搜索时，THE ChampionStrategyService SHALL 按以下流程执行：加载配置 → 生成候选策略 → walk-forward 评价 → 过滤失效候选 → 按稳定性总分排序 → 选出 champion → 落盘
3. WHEN 过滤候选策略时，THE ChampionStrategyService SHALL 优先过滤测试期 Sharpe ≤ 0 或年化收益 < -20% 的候选
4. IF 过滤后无有效候选，THEN THE ChampionStrategyService SHALL 降级为从所有已评价候选中选出稳定性总分最高者
5. WHEN 选出 champion 后，THE ChampionStrategyService SHALL 将 champion 配置写入 `ChampionRegistryService`
6. WHEN 选出 champion 后，THE ChampionStrategyService SHALL 将候选排行榜（leaderboard CSV）和选择报告（selection_report.md）写入 `data/reports/strategy_search/{task_id}/` 目录
7. WHEN 所有候选策略评价均失败时，THE ChampionStrategyService SHALL 抛出 `RuntimeError`，错误信息说明失败原因
8. WHEN 调用 `get_champion(scope_type, scope_key)` 时，THE ChampionStrategyService SHALL 返回当前 champion 配置，不存在时返回 `None`
9. WHEN 调用 `apply_champion` 时，THE ChampionStrategyService SHALL 将指定 champion 写入 registry，供打分接口直接调用
10. WHEN 执行 champion 搜索时，THE ChampionStrategyService SHALL 选出的 champion 的 `stability_score` 大于等于 0


### 需求 7：ChampionRegistryService 文件持久化

**用户故事：** 作为量化研究员，我希望系统能以文件方式持久化 champion 配置，以便在不引入新数据库表的情况下保存和查询历史 champion 结果。

#### 验收标准

1. THE ChampionRegistryService SHALL 将 champion 配置保存到以下路径：单股 `data/champions/single_stock/{code}.json`、股票组 `data/champions/stock_group/{group_hash}.json`、截面 `data/champions/cross_sectional/{universe}.json`
2. WHEN 调用 `save(scope_type, scope_key, champion)` 时，THE ChampionRegistryService SHALL 返回写入文件的 `Path` 对象
3. WHEN 调用 `save` 后立即调用 `load(scope_type, scope_key)` 时，THE ChampionRegistryService SHALL 返回与保存内容完全一致的字典（round-trip 属性）
4. WHEN 调用 `load(scope_type, scope_key)` 且对应文件不存在时，THE ChampionRegistryService SHALL 返回 `None`
5. WHEN 调用 `list_champions(scope_type)` 时，THE ChampionRegistryService SHALL 仅返回指定 scope_type 的 champion 列表
6. WHEN 调用 `delete(scope_type, scope_key)` 后调用 `load(scope_type, scope_key)` 时，THE ChampionRegistryService SHALL 返回 `None`
7. WHEN 调用 `delete` 且文件不存在时，THE ChampionRegistryService SHALL 返回 `False`，不抛出异常
8. WHEN 将股票列表转为 group_hash 时，THE ChampionRegistryService SHALL 对股票列表排序后计算 MD5，确保相同股票集合（不同顺序）产生相同 hash
9. IF `data/champions/` 目录无写权限，THEN THE ChampionRegistryService SHALL 抛出 `IOError`


### 需求 8：现有服务改造

**用户故事：** 作为系统开发者，我希望对现有服务进行最小化改造以支持新功能，以便在不破坏现有接口的前提下扩展系统能力。

#### 验收标准

1. THE temporal_filter_validation_service SHALL 新增 `profile_id` 参数支持，允许按指定 profile 执行时序池验证
2. THE temporal_filter_validation_service SHALL 新增 `max_workers` 参数支持，允许控制并发线程数
3. THE temporal_filter_validation_service SHALL 实现 `build_score_panel(codes, start_date, end_date, profile_id, max_workers)` 接口
4. THE temporal_filter_validation_service SHALL 实现 `run_temporal_pool_backtest(codes, profile_id, start_date, end_date, top_n, score_threshold, rebalance, max_workers)` 接口
5. THE vectorbt_backtest_service SHALL 新增 `cross_sectional_backtest_composite(df, factor_weights, top_percentile, direction)` 接口，支持多因子组合分截面回测
6. WHEN 调用 `cross_sectional_backtest_composite` 时，THE vectorbt_backtest_service SHALL 按 `factor_weights` 字典对各因子加权合成综合分，再执行截面回测
7. WHEN 显式传入 `max_workers=1` 时，THE System SHALL 不启动多线程，在单线程中顺序执行所有打分任务
8. WHEN 处理深市股票代码时，THE System SHALL 保留前导零（如 `000001` 不变为 `1`）


### 需求 9：Search API 与任务管理

**用户故事：** 作为前端开发者，我希望通过 REST API 启动和查询 champion 搜索任务，以便在不直接调用 Python 服务的情况下触发和监控搜索流程。

#### 验收标准

1. THE System SHALL 新增 `backend/api/routers/search.py` 并在 `main.py` 中注册为 `/api/search` 路由
2. WHEN 调用 `POST /api/search/champion/jobs` 时，THE SearchAPI SHALL 异步启动 champion 搜索任务并立即返回 HTTP 202 和包含 `task_id` 的响应体
3. WHEN 调用 `GET /api/search/champion/jobs/{task_id}` 时，THE SearchAPI SHALL 返回任务状态，包含 `status`、`progress`、`total_candidates`、`evaluated_candidates` 字段
4. WHEN 调用 `GET /api/search/champion/jobs/{task_id}/results` 时，THE SearchAPI SHALL 返回已完成任务的 champion 配置和候选排行榜
5. WHEN 调用 `GET /api/search/champions/{scope_type}/{scope_key}` 时，THE SearchAPI SHALL 返回当前 champion 配置，不存在时返回 HTTP 404
6. WHEN 调用 `POST /api/search/champions/{scope_type}/{scope_key}/apply` 时，THE SearchAPI SHALL 将指定 champion 应用到对应作用域的打分接口
7. WHEN 调用 `GET /api/search/profiles` 时，THE SearchAPI SHALL 返回所有可用 profile 列表，支持按 `mode` 参数过滤
8. WHEN 调用 `GET /api/search/search-spaces` 时，THE SearchAPI SHALL 返回所有可用 search space 配置列表
9. WHEN 调用 `POST /api/scoring/temporal-score` 且请求体包含 `profile_id` 时，THE ScoringAPI SHALL 使用指定 profile 执行打分，而非默认 8 因子配置
10. IF `scope_key` 包含路径遍历字符（`/` 或 `..`），THEN THE SearchAPI SHALL 返回 HTTP 400，拒绝请求


### 需求 10：结果落盘与任务追踪

**用户故事：** 作为量化研究员，我希望每次搜索任务的结果都能完整落盘，以便事后审计、复现和比较不同时期的 champion 策略。

#### 验收标准

1. WHEN champion 搜索任务完成时，THE ChampionStrategyService SHALL 在 `data/reports/strategy_search/{task_id}/` 目录下生成以下文件：`task_status.json`、`input_scope.json`、`candidate_leaderboard.csv`、`candidate_configs.json`、`walkforward_metrics.parquet`、`best_strategy.json`、`selection_report.md`
2. WHEN champion 搜索任务开始时，THE ChampionStrategyService SHALL 创建 `task_status.json` 并将 `status` 设为 `running`
3. WHEN champion 搜索任务完成时，THE ChampionStrategyService SHALL 更新 `task_status.json` 中的 `status` 为 `completed`，并记录 `completed_at` 时间戳
4. IF champion 搜索任务失败，THEN THE ChampionStrategyService SHALL 更新 `task_status.json` 中的 `status` 为 `failed`，并记录 `error` 字段
5. THE candidate_leaderboard.csv SHALL 包含所有已评价候选策略的 `candidate_id`、`stability_score`、`test_metrics` 等关键指标，按 `stability_score` 降序排列
6. WHEN 评分缓存 key 生成时，THE System SHALL 使用格式 `{code}_{start_date}_{end_date}_{profile_id}` 避免不同 profile 的缓存冲突

### 需求 11：回测正确性保障

**用户故事：** 作为量化研究员，我希望回测结果不受持仓顺序影响，以便确保策略评价的公平性和可复现性。

#### 验收标准

1. WHEN 对相同持仓集合以不同顺序输入时，THE vectorbt_backtest_service SHALL 产生相同的回测净值曲线
2. WHEN 执行 walk-forward 评价时，THE StrategyEvaluationService SHALL 对每个窗口独立记录 `train_metrics`、`valid_metrics`、`test_metrics`，确保结果可追踪
3. WHEN 时间跨度小于 18 个月时，THE ChampionStrategyService SHALL 拒绝执行搜索任务并返回明确错误信息（因为无法生成完整的 walk-forward 窗口）

