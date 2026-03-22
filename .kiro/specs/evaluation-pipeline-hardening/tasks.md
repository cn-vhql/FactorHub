# 实现计划

- [x] 1. 编写 Bug 条件探索测试（P0-2：无效指标仍可产出 champion）
  - **Property 1: Bug Condition** - 全 NaN 窗口指标仍返回非零稳定性分
  - **重要**：在实现修复之前先编写此属性测试
  - **目标**：暴露反例，证明 Bug 存在
  - **范围化 PBT 方法**：将属性范围限定为具体失败场景——所有窗口的 `test_metrics` 均为 NaN 的 `window_metrics` 列表
  - 测试 `compute_stability_score(window_metrics)` 在全 NaN 输入时返回非 NaN 的非零分（来自 bugfix.md Bug 条件 P0-2）
  - 在未修复代码上运行测试——预期**失败**（即当前代码返回约 27.5 的固定分，而非 NaN）
  - 记录发现的反例（例如：`compute_stability_score([{"test_metrics": {"sharpe": nan, ...}}])` 返回 27.5 而非 NaN）
  - _Requirements: 1.3_

- [x] 2. 编写保留属性测试（正常窗口指标下的稳定性计算）
  - **Property 2: Preservation** - 有效窗口数 ≥ min_valid_windows 时按原公式计算
  - **重要**：遵循观察优先方法论
  - 观察：在未修复代码上，`compute_stability_score` 对有效指标（非 NaN Sharpe、annual_return 等）返回 [0.0, 100.0] 范围内的分数
  - 观察：有效窗口数 ≥ 3 时，7 项权重公式正常运行
  - 编写属性测试：对所有有效窗口数 ≥ 3 且指标均为有效数值的输入，结果应在 [0.0, 100.0] 范围内（来自 bugfix.md 保留需求 3.2）
  - 在未修复代码上验证测试通过
  - _Requirements: 3.2_


- [x] 3. 修复 P0-1：数据库 schema 漂移（`formula_type` 列缺失）

  - [x] 3.1 在 `init_db()` 中添加幂等 schema 迁移逻辑
    - 在 `backend/core/database.py` 的 `init_db()` 中，调用 `Base.metadata.create_all` 之后检测 `factors` 表是否缺少 `formula_type` 列
    - 若缺少，执行 `ALTER TABLE factors ADD COLUMN formula_type VARCHAR(20) NOT NULL DEFAULT 'expression'`
    - 为历史数据回填默认值 `expression`
    - 记录 schema 版本号（写入 `schema_migrations` 表或 `settings` 表）
    - 迁移失败时抛出异常，阻止应用启动
    - _Bug_Condition: `NOT column_exists(db, table="factors", column="formula_type")`_
    - _Expected_Behavior: `schema_migrated(result) AND no_crash(result)`_
    - _Preservation: 已迁移的数据库重复启动时跳过迁移，不报错_
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 3.2 验证 Bug 条件探索测试现在通过（P0-1）
    - **Property 1: Expected Behavior** - schema 迁移后应用正常启动
    - **重要**：重新运行步骤 1 中的同一测试，不要编写新测试
    - 运行针对旧版数据库（无 `formula_type` 列）的启动测试
    - **预期结果**：测试通过（确认 Bug 已修复）
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.3 验证保留测试仍然通过（P0-1）
    - **Property 2: Preservation** - 已迁移数据库上的正常 ORM 查询不受影响
    - **重要**：重新运行步骤 2 中的同一测试，不要编写新测试
    - **预期结果**：测试通过（确认无回归）

- [x] 4. 修复 P0-2：无效指标不得产出 champion

  - [x] 4.1 修改 `compute_stability_score` 以检测有效窗口数
    - 在 `backend/services/strategy_evaluation_service.py` 的 `compute_stability_score` 中，统计有效窗口数（`test_metrics` 中 Sharpe 非 NaN 的窗口）
    - 当有效窗口数 < `min_valid_windows`（默认 3）时，返回包含 `stability_score=NaN`、`evaluation_status="invalid"`、`invalid_reason="INSUFFICIENT_VALID_WINDOWS"` 的结构
    - 更新调用方（`evaluate_temporal_pool_candidate`、`evaluate_cross_sectional_candidate`）以处理新返回结构
    - _Bug_Condition: `valid_count < MIN_VALID_WINDOWS`（其中 valid_count 为非全 NaN 指标的窗口数）_
    - _Expected_Behavior: `is_nan(result.stability_score) AND result.evaluation_status = "invalid"`_
    - _Preservation: 有效窗口数 ≥ min_valid_windows 时按原有 7 项权重公式计算，返回值范围 [0.0, 100.0]_
    - _Requirements: 2.4, 3.2_

  - [x] 4.2 修改 `ChampionStrategyService` 过滤逻辑，拒绝无效候选
    - 在 `backend/services/champion_strategy_service.py` 的 `_execute_search` 中，将 `evaluation_status != "valid"` 的候选排除在 `valid_candidates` 之外
    - 当过滤后无 `evaluation_status=valid` 的候选时（包括非 single_stock 场景），将任务状态置为 `failed` 或 `no_valid_candidate`，不向 registry 写入任何 champion
    - 输出包含所有候选失败原因的完整报告
    - _Bug_Condition: 过滤后无 `evaluation_status=valid` 的候选_
    - _Expected_Behavior: 任务状态为 `failed`，registry 无新写入_
    - _Preservation: 存在至少一个有效候选时，按 stability_score 降序排序并选出最优候选写入 registry_
    - _Requirements: 2.5, 3.3, 3.6_

  - [x] 4.3 验证 Bug 条件探索测试现在通过（P0-2）
    - **Property 1: Expected Behavior** - 全 NaN 窗口指标返回 NaN 稳定性分
    - **重要**：重新运行步骤 1 中的同一测试，不要编写新测试
    - **预期结果**：测试通过（确认 Bug 已修复）
    - _Requirements: 2.4, 2.5_

  - [x] 4.4 验证保留测试仍然通过（P0-2）
    - **Property 2: Preservation** - 有效窗口数 ≥ 3 时稳定性分计算不变
    - **重要**：重新运行步骤 2 中的同一测试，不要编写新测试
    - **预期结果**：测试通过（确认无回归）


- [x] 5. 修复 P1-1：数据覆盖率门禁

  - [x] 5.1 在 `DataService` 中新增 `get_multiple_stocks_data_with_report` 方法
    - 在 `backend/services/data_service.py` 中新增方法，返回结构化的 `DataFetchReport`
    - `DataFetchReport` 包含：`requested_codes`、`success_codes`、`failed_codes`、`code_coverage`（成功数/请求数）、`failures`（含错误类型和消息）
    - 原有 `get_multiple_stocks_data` 保持不变，旧调用方不受影响
    - _Bug_Condition: `coverage < EVAL_MIN_CODE_COVERAGE`（默认 0.85）_
    - _Expected_Behavior: 返回 `DataFetchReport`，覆盖率低时调用方可感知_
    - _Preservation: 所有股票拉取成功时，返回与原有接口相同结构的数据字典_
    - _Requirements: 2.6, 3.4_

  - [x] 5.2 在验证层和策略评估层添加覆盖率门禁
    - 在 `TemporalFilterValidationService` 和 `StrategyEvaluationService` 中，调用 `get_multiple_stocks_data_with_report` 后检查覆盖率
    - 当 `code_coverage < EVAL_MIN_CODE_COVERAGE`（默认 0.85）或有效日期数 < `EVAL_MIN_IC_SAMPLES`（默认 30）时，直接返回 `invalid_run`，不进入 champion 选择流程
    - 在报告中记录 `CODE_COVERAGE_TOO_LOW` 或 `INSUFFICIENT_IC_SAMPLES` 原因码
    - 在 `backend/core/settings.py` 中添加 `EVAL_MIN_CODE_COVERAGE` 和 `EVAL_MIN_IC_SAMPLES` 配置项
    - _Requirements: 2.7, 3.4_

  - [x] 5.3 验证覆盖率门禁测试通过
    - **Property 1: Expected Behavior** - 覆盖率低于阈值时评估任务返回 invalid_run
    - 运行针对低覆盖率场景的测试
    - **预期结果**：测试通过（确认门禁生效）
    - _Requirements: 2.6, 2.7_

- [x] 6. 修复 P1-2：缓存版本指纹

  - [x] 6.1 在缓存 key 中纳入内容 hash
    - 在 `backend/services/cache_service.py` 或打分服务中，生成包含因子表达式内容 hash、profile 定义 hash 和打分服务版本号的版本指纹
    - 将版本指纹纳入缓存 key，使内容变更后旧缓存自动失效
    - 打分服务版本号可通过常量或 `__version__` 字段维护
    - _Bug_Condition: 因子表达式、profile 内容或服务版本变更后命中旧缓存_
    - _Expected_Behavior: 生成包含内容 hash 的版本指纹，旧缓存自动失效_
    - _Preservation: 因子表达式和 profile 内容未变更时，仍命中现有缓存，不触发不必要的重新计算_
    - _Requirements: 2.8, 3.5_

  - [x] 6.2 验证缓存版本指纹测试通过
    - 运行针对内容变更场景的缓存失效测试
    - **预期结果**：内容变更后缓存未命中，内容不变时缓存命中
    - _Requirements: 2.8, 3.5_

- [x] 7. 修复 P2-1：统一 run_manifest

  - [x] 7.1 实现 `run_manifest.json` 写出逻辑
    - 新增辅助函数 `_write_run_manifest(task_dir, module, data_coverage, quality_gate, result_status)`
    - `run_manifest.json` 包含：`run_id`、`module`、`generated_at`、`schema_version`、`settings_fingerprint`（settings 关键配置的 hash）、`data_coverage`、`quality_gate`（含 `passed` 布尔值和 `reason_codes` 列表）、`result_status`
    - _Bug_Condition: 评估任务完成后缺少统一的 run_manifest.json_
    - _Expected_Behavior: 任务产物目录下写出 run_manifest.json，包含所有追溯字段_
    - _Preservation: 现有 CSV/JSON/Markdown 报告文件继续生成，run_manifest.json 为新增产物_
    - _Requirements: 2.9, 3.7_

  - [x] 7.2 在三个评估入口写出 run_manifest
    - 在 `ChampionStrategyService._run` 完成时调用 `_write_run_manifest`
    - 在 `TemporalFilterValidationService` 的验证任务完成时调用 `_write_run_manifest`
    - 在 `StrategyEvaluationService` 的评估任务完成时调用 `_write_run_manifest`
    - _Requirements: 2.9, 3.7_

  - [x] 7.3 验证 run_manifest 写出测试通过
    - 运行评估任务后检查产物目录中是否存在 `run_manifest.json` 且包含所有必需字段
    - **预期结果**：测试通过（确认 manifest 正确写出）
    - _Requirements: 2.9_

- [x] 8. 检查点——确保所有测试通过
  - 运行全部属性测试和集成测试，确认无回归
  - 验证五个 Bug（P0-1、P0-2、P1-1、P1-2、P2-1）的修复均已生效
  - 如有疑问，请向用户确认
