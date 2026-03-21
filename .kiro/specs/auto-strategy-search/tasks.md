# 实现计划：自动化策略搜索与 Champion Strategy

## 概述

将 FactorHub 从"固定 8 因子打分器"升级为"自动化策略搜索与应用平台"。按设计文档中的开发拆解 A-I 顺序，逐步实现配置层、各服务层、API 层，最后补充属性测试和已知缺陷修复。

## 任务列表

- [x] 1. 配置层：补全 config/*.json 文件
  - 在 `config/cross_sectional_profiles.json` 中添加截面模式的内置 profile（至少 2 个）
  - 在 `config/champion_runtime.json` 中添加运行时参数（`default_max_workers`、`search_timeout_seconds`、`min_history_months`）
  - 确保 `config/temporal_pool_profiles.json` 和 `config/strategy_search_space.json` 已存在且结构合法（已有文件，仅验证）
  - _需求：1.1、4.1_

- [x] 2. 实现 FactorProfileRegistryService
  - [x] 2.1 创建 `backend/services/factor_profile_registry_service.py`
    - 实现 `load_profiles(mode)` 从两个 JSON 文件加载并按 mode 过滤
    - 实现 `get_profile(profile_id)` 按 id 查找，不存在时抛 `KeyError`
    - 实现 `load_search_space(space_id)` 从 `strategy_search_space.json` 加载
    - 实现 `validate_profile(profile)` 校验必填字段、权重和、mode、top_n、percentile_window
    - 实现 `list_profile_ids(mode)` 列出所有 profile_id
    - 实现热重载：文件变更后重新加载（基于文件 mtime 检测）
    - _需求：1.1、1.2、1.3、1.4、1.5、1.6、1.7、1.8、1.9、1.10_

  - [x] 2.2 为 FactorProfileRegistryService 编写单元测试
    - 测试合法 profile 加载与过滤
    - 测试 `get_profile` 不存在时抛 `KeyError`
    - 测试 `validate_profile` 各校验规则（缺字段、权重不为 1、mode 非法、top_n 越界、percentile_window 越界）
    - _需求：1.2、1.3、1.4、1.5、1.6、1.7、1.8、1.9_

- [x] 3. 改造 TemporalScoringService：profile 化接口
  - [x] 3.1 在 `backend/services/temporal_scoring_service.py` 中新增 profile 驱动方法
    - 新增 `_compute_factors_from_profile(df, profile)` 按 profile factors 列表动态计算因子
    - 新增 `_compute_score_from_profile(pcts, profile)` 按 profile weight/direction/threshold 合成分数
    - 新增 `score_one_stock_with_profile(code, trade_date, profile_id)` 接口
    - 新增 `score_stock_history_with_profile(code, start_date, end_date, profile_id, window)` 接口
    - 将现有 8 因子硬编码逻辑对应到 `default_temporal_v1` profile（保持老接口不变）
    - 缓存 key 格式改为 `{code}_{start_date}_{end_date}_{profile_id}`（修复需求 10.6）
    - _需求：2.1、2.2、2.3、2.4、2.5、2.6、2.7、2.8、2.9_

  - [x] 3.2 为 score_one_stock_with_profile 编写属性测试（Property 1）
    - **属性 1：单股打分分数范围不变量**
    - 对任意合法 profile 和有足够历史数据的股票，score ∈ [0, 100]，pcts 中每个值 ∈ [0, 1]
    - **验证：需求 2.3、2.4**

- [x] 4. 实现 TemporalPoolService
  - [x] 4.1 创建 `backend/services/temporal_pool_service.py`
    - 实现 `score_pool(codes, profile_id, trade_date, max_workers)` 并发打分并返回含 code/score/rank 的 DataFrame
    - rank 列为 1-based 降序（score 最高者 rank=1），结果按 score 降序排列
    - 单只股票失败时跳过并记录错误，不影响其他股票
    - 实现 `select_top_n(pool_scores, top_n, score_threshold)` 选出 TopN
    - 实现 `score_pool_history(codes, profile_id, start_date, end_date, max_workers)` 构建历史池内评分面板
    - `max_workers=1` 时不启动线程池，顺序执行（修复需求 8.7）
    - _需求：3.1、3.2、3.3、3.4、3.5、3.6、3.7、3.8、3.9_

  - [x] 4.2 为 score_pool 编写属性测试（Property 2）
    - **属性 2：池内排名连续整数不变量**
    - 对任意非空股票列表，rank 列是从 1 开始的连续整数，score 最高者 rank=1
    - **验证：需求 3.2**

  - [x] 4.3 为 select_top_n 编写属性测试（Property 3、Property 4）
    - **属性 3：TopN 数量上界不变量** — 返回行数不超过 top_n
    - **属性 4：score_threshold 过滤正确性** — 返回结果中所有 score ≥ score_threshold
    - **验证：需求 3.6、3.7**

- [x] 5. 实现 StrategySearchService
  - [x] 5.1 创建 `backend/services/strategy_search_service.py`
    - 实现 `_template_enum(base_profiles)` 枚举所有内置 profile 作为候选（第一层）
    - 实现 `_constrained_random(search_space, n)` 受约束随机搜索（第二层）：
      - Dirichlet 采样 signal 因子权重，确保权重和 ∈ [0.999, 1.001]
      - 至少包含 1 个 risk_filter 因子
      - 因子总数在 [min_factors, max_factors] 范围内
      - 同组因子数不超过 max_per_group
      - 单因子权重不超过 0.40
      - 超过 n*10 次尝试后返回已生成候选，不抛异常
    - 实现 `_forward_selection(top_candidates, search_space)` 逐步精简（第三层）
    - 实现 `generate_candidates(mode, base_profiles, search_space, n_candidates)` 组合三层搜索
    - 实现 `search_for_stock`、`search_for_pool`、`search_for_cross_sectional` 三个作用域入口
    - _需求：4.1、4.2、4.3、4.4、4.5、4.6、4.7、4.8、4.9_

  - [x] 5.2 为 _constrained_random 编写属性测试（Property 5、Property 6）
    - **属性 5：受约束随机搜索权重归一化** — signal 因子权重之和 ∈ [0.999, 1.001]
    - **属性 6：受约束随机搜索约束满足性** — 至少 1 个 risk_filter、因子数在范围内、同组不超限、单因子权重 ≤ 0.40
    - **验证：需求 4.3、4.4、4.5、4.6、4.7**

- [x] 6. 实现 StrategyEvaluationService 及现有服务改造
  - [x] 6.1 改造 `backend/services/temporal_filter_validation_service.py`
    - 新增 `profile_id` 参数到 `build_score_panel`，转发给 `TemporalScoringService`
    - 新增 `max_workers` 参数到 `build_score_panel`
    - 新增 `run_temporal_pool_backtest(codes, profile_id, start_date, end_date, top_n, score_threshold, rebalance, max_workers)` 接口
    - _需求：8.1、8.2、8.3、8.4_

  - [x] 6.2 改造 `backend/services/vectorbt_backtest_service.py`
    - 新增 `cross_sectional_backtest_composite(df, factor_weights, top_percentile, direction)` 接口
    - 按 factor_weights 字典对各因子加权合成综合分，再执行截面回测
    - 修复持仓顺序敏感问题：用集合比较替代列表比较（修复需求 11.1）
    - _需求：8.5、8.6、11.1_

  - [x] 6.3 创建 `backend/services/strategy_evaluation_service.py`
    - 实现 `_generate_walk_forward_windows(start_date, end_date, train_months, valid_months, test_months, step_months)` 生成无重叠滚动窗口
    - 实现 `_run_walk_forward_windows(codes, profile, start_date, end_date, ...)` 逐窗口回测
    - 实现 `compute_stability_score(window_metrics)` 按 7 项权重计算稳定性总分（返回值 ∈ [0, 100]）
    - 实现 `evaluate_temporal_pool_candidate(candidate, codes, start_date, end_date, ...)` 复用 temporal_filter_validation_service 和 vectorbt_backtest_service
    - 实现 `evaluate_cross_sectional_candidate(candidate, universe, start_date, end_date, ...)`
    - 过拟合检测：test Sharpe < 0.3 × train Sharpe 时标记 overfitting_flag=True
    - _需求：5.1、5.2、5.3、5.4、5.5、5.6、5.7、5.8、5.9、11.2_

  - [x] 6.4 为 compute_stability_score 编写属性测试（Property 7）
    - **属性 7：稳定性总分范围不变量** — 对任意非空 window_metrics，返回值 ∈ [0.0, 100.0]
    - 测试全负 Sharpe 时返回值 < 30
    - **验证：需求 5.7、5.8_**

  - [x] 6.5 为 walk-forward 窗口生成编写属性测试（Property 14）
    - **属性 14：Walk-Forward 窗口无重叠覆盖完整性**
    - 对任意合法时间范围，生成的窗口各期无重叠，整体覆盖完整时间范围
    - **验证：需求 5.3、5.4**

- [x] 7. 实现 ChampionStrategyService 和 ChampionRegistryService
  - [x] 7.1 创建 `backend/services/champion_registry_service.py`
    - 实现 `save(scope_type, scope_key, champion)` 写入对应路径，返回 Path
    - 实现 `load(scope_type, scope_key)` 读取 JSON，不存在时返回 None
    - 实现 `list_champions(scope_type)` 按 scope_type 过滤列出所有 champion
    - 实现 `delete(scope_type, scope_key)` 删除文件，不存在时返回 False
    - 实现 `_scope_key_to_hash(codes)` 排序后 MD5，确保顺序无关
    - scope_key 路径安全校验：拒绝含 `/` 或 `..` 的 scope_key
    - _需求：7.1、7.2、7.3、7.4、7.5、7.6、7.7、7.8、7.9_

  - [x] 7.2 为 ChampionRegistryService 编写属性测试（Property 8、Property 9、Property 10）
    - **属性 8：save/load round-trip** — save 后立即 load，内容完全一致
    - **属性 9：list_champions 按 scope_type 过滤正确性** — 返回列表中所有元素 scope_type 等于指定值
    - **属性 10：group_hash 顺序无关性** — 任意顺序输入相同股票列表，hash 值相同
    - **验证：需求 7.3、7.5、7.8**

  - [x] 7.3 创建 `backend/services/champion_strategy_service.py`
    - 实现 `run_for_stock(code, search_space_id, start_date, end_date)` 完整搜索流程
    - 实现 `run_for_pool(codes, search_space_id, start_date, end_date, max_workers)` 完整搜索流程
    - 实现 `run_for_cross_sectional(universe, search_space_id, start_date, end_date)` 完整搜索流程
    - 搜索流程：加载配置 → 生成候选 → walk-forward 评价 → 过滤失效候选 → 排序 → 选出 champion → 落盘
    - 过滤规则：优先过滤 test Sharpe ≤ 0 或年化收益 < -20% 的候选；过滤后无有效候选时降级取最优
    - 所有候选评价失败时抛 `RuntimeError`
    - 时间跨度 < 18 个月时拒绝执行并返回明确错误信息（需求 11.3）
    - 实现 `get_champion(scope_type, scope_key)` 和 `apply_champion(scope_type, scope_key, champion)`
    - _需求：6.1、6.2、6.3、6.4、6.5、6.6、6.7、6.8、6.9、6.10、11.3_

  - [x] 7.4 实现任务状态落盘（task_status.json 及报告文件）
    - 任务开始时创建 `data/reports/strategy_search/{task_id}/task_status.json`，status=running
    - 任务完成时更新 status=completed，记录 completed_at
    - 任务失败时更新 status=failed，记录 error 字段
    - 生成 `input_scope.json`、`candidate_leaderboard.csv`（按 stability_score 降序）、`candidate_configs.json`、`walkforward_metrics.parquet`、`best_strategy.json`、`selection_report.md`
    - _需求：10.1、10.2、10.3、10.4、10.5_

  - [x] 7.5 为 ChampionStrategyService 编写属性测试（Property 13）
    - **属性 13：Champion 稳定性总分非负性** — 选出的 champion stability_score ≥ 0
    - **验证：需求 6.10**

- [x] 8. 检查点 — 确保所有已实现服务的测试通过，如有问题请提出

- [x] 9. 实现 Search API 及 scoring.py 改造
  - [x] 9.1 改造 `backend/api/routers/scoring.py`
    - 在 `ScoringRequest` 中新增可选 `profile_id: Optional[str] = None` 字段
    - `POST /api/scoring/temporal-score` 中：若请求体含 `profile_id`，则调用 `score_one_stock_with_profile`，否则保持原有行为
    - _需求：9.9_

  - [x] 9.2 创建 `backend/api/routers/search.py`
    - 实现 `POST /api/search/champion/jobs`：异步启动搜索任务，返回 HTTP 202 + task_id
    - 实现 `GET /api/search/champion/jobs/{task_id}`：返回任务状态（status、progress、total_candidates、evaluated_candidates）
    - 实现 `GET /api/search/champion/jobs/{task_id}/results`：返回已完成任务的 champion 配置和候选排行榜
    - 实现 `GET /api/search/champions/{scope_type}/{scope_key}`：返回 champion 配置，不存在时返回 HTTP 404
    - 实现 `POST /api/search/champions/{scope_type}/{scope_key}/apply`：应用 champion 到对应作用域
    - 实现 `GET /api/search/profiles`：返回所有 profile 列表，支持 mode 参数过滤
    - 实现 `GET /api/search/search-spaces`：返回所有 search space 配置列表
    - scope_key 路径安全校验：含 `/` 或 `..` 时返回 HTTP 400
    - _需求：9.1、9.2、9.3、9.4、9.5、9.6、9.7、9.8、9.10_

  - [x] 9.3 在 `backend/api/main.py` 中注册 search router
    - 导入 `search` router 并注册为 `/api/search` 路由
    - _需求：9.1_

- [ ] 10. 属性测试（PBT）汇总
  - [x] 10.1 为深市股票代码前导零保留编写属性测试（Property 11）
    - **属性 11：深市股票代码前导零保留**
    - 对任意以 `0` 开头的 6 位深市股票代码，经过 `load_fixed_pool` 和 `_normalize_code` 处理后代码字符串不变
    - **验证：需求 8.8**

  - [x] 10.2 为回测持仓顺序无关性编写属性测试（Property 12）
    - **属性 12：回测持仓顺序无关性**
    - 对任意持仓集合，以不同顺序输入 `_equal_weight_backtest`，产生的净值曲线完全相同
    - **验证：需求 11.1**

- [x] 11. 已知缺陷修复
  - [x] 11.1 修复 max_workers 透传问题
    - 检查 `TemporalPoolService`、`TemporalFilterValidationService`、`ChampionStrategyService` 中所有调用链
    - 确保 `max_workers=1` 时不启动 ThreadPoolExecutor，顺序执行
    - _需求：8.7_

  - [x] 11.2 修复股票代码规范化（前导零保留）
    - 在 `TemporalFilterValidationService.load_fixed_pool` 中确认 `_normalize_code` 正确处理深市代码
    - 在 `TemporalPoolService.score_pool` 入口处对 codes 列表做规范化
    - _需求：8.8_

  - [x] 11.3 修复回测持仓顺序敏感问题
    - 在 `temporal_filter_validation_service._equal_weight_backtest` 和 `_equal_weight_backtest_with_cash` 中，将持仓比较从列表比较改为集合比较（已有部分修复，确认完整覆盖）
    - _需求：11.1_

  - [x] 11.4 修复评分缓存 key 版本化
    - 在 `TemporalScoringService.score_stock_history_with_profile` 中，缓存文件名格式改为 `{code}_{start_date}_{end_date}_{profile_id}.parquet`
    - 确保不同 profile 不共享缓存
    - _需求：2.8、10.6_

- [x] 12. 最终检查点 — 确保所有测试通过，如有问题请提出

## 备注

- 标有 `*` 的子任务为可选测试任务，可跳过以加快 MVP 交付
- 每个任务均引用了具体需求条款，便于追溯
- 属性测试使用 `hypothesis` 库（项目已有 `.hypothesis/` 目录）
- Champion 搜索为耗时操作（10-30 分钟），必须通过 API 异步执行
- 生产环境建议离线运行搜索，在线只读取 champion 配置
