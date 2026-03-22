# Bugfix 需求文档：评估链路稳健性修复

## 简介

当前评估链路在数据缺失、schema 漂移、指标无效等异常情况下，仍会输出"看起来完整"的评估结果，导致后续因子扩展和策略决策被误导。本文档描述五个结构性缺陷（P0-1、P0-2、P1-1、P1-2、P2-1）的修复需求，目标是在样本不足或指标无效时让评估任务明确失败，而非产出伪有效结论。

---

## Bug 分析

### 当前行为（缺陷）

**P0-1：数据库 schema 漂移**

1.1 WHEN 应用启动时数据库中 `factors` 表缺少 `formula_type` 列 THEN 系统在执行 ORM 全字段查询时抛出 `no such column: factors.formula_type` 异常，导致评估任务在业务逻辑执行前即失败

1.2 WHEN 旧版数据库（无 `formula_type` 列）与新版 ORM 模型（含 `formula_type` 字段）共存时 THEN 系统静默运行，不报告 schema 版本不一致，直到查询时才崩溃

**P0-2：样本/指标无效时仍可选出 champion**

1.3 WHEN `compute_stability_score` 接收到全部窗口指标均为 NaN 的 `window_metrics` 列表时 THEN 系统对 NaN 使用默认值 `0.0` 进行计算，返回固定的非零分（常见约 27.5），而非标记为无效

1.4 WHEN `champion_strategy_service` 过滤后无任何有效候选（所有候选 Sharpe ≤ 0 或年化收益 < -20%）时 THEN 系统（对非 single_stock 场景）降级为从全部已评价候选中强行选出一个作为 champion，写入 registry

**P1-1：数据拉取失败被弱化处理，覆盖率门禁缺失**

1.5 WHEN `DataService.get_multiple_stocks_data` 中部分股票数据拉取失败时 THEN 系统仅打印 warning 并继续，调用方无法感知实际数据覆盖率，评估流程照常推进

1.6 WHEN 数据覆盖率低于可信阈值（如代码覆盖率 < 85% 或有效日期数 < 60）时 THEN 系统不存在统一的 fail-fast 门禁，验证层和策略评估层仍会产出报告

**P1-2：评估缓存缺少版本指纹**

1.7 WHEN 因子表达式、profile 内容或打分服务实现版本发生变更后重新执行评估时 THEN 系统命中基于旧 `code/date/profile_id` 组合生成的缓存 key，返回过期的历史评分结果，不感知内容变化

**P2-1：评估产物缺少统一 run_manifest**

1.8 WHEN 评估任务（temporal_filter_validation、strategy_evaluation、champion 搜索）完成后 THEN 系统产出 CSV/JSON/Markdown 文件，但缺少统一的 `run_manifest.json`，无法从产物中追溯代码版本、schema 版本、数据覆盖率和门禁判定结果

---

### 期望行为（正确）

**P0-1：数据库 schema 治理**

2.1 WHEN 应用启动时检测到 `factors` 表缺少 `formula_type` 列 THEN 系统 SHALL 自动执行幂等迁移，为该列补充默认值 `expression`，并为历史数据回填，迁移完成后记录 schema 版本号

2.2 WHEN 应用在已迁移的数据库上重复启动时 THEN 系统 SHALL 检测到 schema 版本已是最新，跳过迁移，不报错、不重复执行

2.3 WHEN schema 迁移执行失败时 THEN 系统 SHALL 阻止应用启动，输出明确的错误信息，不进入服务状态

**P0-2：无效指标不得产出 champion**

2.4 WHEN `compute_stability_score` 接收到有效窗口数（非全 NaN 指标的窗口）少于 `min_valid_windows`（默认 3）时 THEN 系统 SHALL 返回 `stability_score=NaN`，并在返回结构中标记 `evaluation_status=invalid`、`invalid_reason=INSUFFICIENT_VALID_WINDOWS`

2.5 WHEN `champion_strategy_service` 过滤后无 `evaluation_status=valid` 的候选时 THEN 系统 SHALL 将任务状态置为 `failed`（或 `no_valid_candidate`），不向 registry 写入任何 champion，并输出包含所有候选失败原因的完整报告

**P1-1：数据覆盖率门禁**

2.6 WHEN `get_multiple_stocks_data_with_report` 执行完成时 THEN 系统 SHALL 返回结构化的 `DataFetchReport`，包含 `requested_codes`、`success_codes`、`failed_codes`、`code_coverage`、`failures`（含错误类型和消息）等字段

2.7 WHEN 数据覆盖率低于 `EVAL_MIN_CODE_COVERAGE`（默认 0.85）或有效日期数低于 `EVAL_MIN_IC_SAMPLES`（默认 30）时 THEN 系统 SHALL 在验证层和策略评估层直接返回 `invalid_run`，不进入 champion 选择流程，并在报告中记录 `CODE_COVERAGE_TOO_LOW` 或 `INSUFFICIENT_IC_SAMPLES` 原因码

**P1-2：缓存版本指纹**

2.8 WHEN 因子表达式内容、profile 定义或打分服务版本号发生变更后执行评估时 THEN 系统 SHALL 生成包含上述内容 hash 的版本指纹，并将其纳入缓存 key，使旧缓存自动失效

**P2-1：统一 run_manifest**

2.9 WHEN 任意评估任务（temporal_filter_validation、strategy_evaluation、champion 搜索）完成时 THEN 系统 SHALL 在任务产物目录下写出 `run_manifest.json`，包含 `run_id`、`module`、`generated_at`、`schema_version`、`settings_fingerprint`、`data_coverage`、`quality_gate`、`result_status` 等字段

---

### 不变行为（回归防护）

**正常数据场景下的评估流程**

3.1 WHEN 数据库 schema 已是最新版本且 `factors` 表包含 `formula_type` 列时 THEN 系统 SHALL CONTINUE TO 正常执行所有 ORM 查询，不受 schema 迁移逻辑影响

3.2 WHEN `compute_stability_score` 接收到有效窗口数 ≥ `min_valid_windows` 且指标均为有效数值时 THEN 系统 SHALL CONTINUE TO 按原有 7 项权重公式计算稳定性总分，返回值范围 [0.0, 100.0]

3.3 WHEN `champion_strategy_service` 存在至少一个 `evaluation_status=valid` 的候选时 THEN 系统 SHALL CONTINUE TO 按 `stability_score` 降序排序并选出最优候选写入 registry

3.4 WHEN `DataService.get_multiple_stocks_data` 中所有股票数据拉取成功时 THEN 系统 SHALL CONTINUE TO 返回与原有接口相同结构的数据字典，旧调用方不受影响

3.5 WHEN 因子表达式和 profile 内容未发生变更时 THEN 系统 SHALL CONTINUE TO 命中现有缓存，不触发不必要的重新计算

3.6 WHEN `single_stock` 场景下过滤后无有效候选时 THEN 系统 SHALL CONTINUE TO 抛出 `RuntimeError`，行为与当前一致（该场景已有明确失败逻辑）

3.7 WHEN 评估任务产出 CSV/JSON/Markdown 报告文件时 THEN 系统 SHALL CONTINUE TO 生成这些文件，`run_manifest.json` 为新增产物，不替换现有文件

---

## Bug 条件伪代码

### P0-1 Bug 条件

```pascal
FUNCTION isBugCondition_P0_1(db_state)
  INPUT: db_state（数据库当前状态）
  OUTPUT: boolean
  RETURN NOT column_exists(db_state, table="factors", column="formula_type")
END FUNCTION

// Fix Checking
FOR ALL db_state WHERE isBugCondition_P0_1(db_state) DO
  result ← startup'(db_state)
  ASSERT schema_migrated(result) AND no_crash(result)
END FOR

// Preservation Checking
FOR ALL db_state WHERE NOT isBugCondition_P0_1(db_state) DO
  ASSERT startup(db_state) = startup'(db_state)
END FOR
```

### P0-2 Bug 条件

```pascal
FUNCTION isBugCondition_P0_2(window_metrics)
  INPUT: window_metrics（窗口指标列表）
  OUTPUT: boolean
  valid_count ← COUNT(w IN window_metrics WHERE NOT all_nan(w.test_metrics))
  RETURN valid_count < MIN_VALID_WINDOWS
END FUNCTION

// Fix Checking
FOR ALL window_metrics WHERE isBugCondition_P0_2(window_metrics) DO
  result ← compute_stability_score'(window_metrics)
  ASSERT is_nan(result.stability_score) AND result.evaluation_status = "invalid"
END FOR

// Preservation Checking
FOR ALL window_metrics WHERE NOT isBugCondition_P0_2(window_metrics) DO
  ASSERT compute_stability_score(window_metrics) = compute_stability_score'(window_metrics)
END FOR
```

### P1-1 Bug 条件

```pascal
FUNCTION isBugCondition_P1_1(fetch_result)
  INPUT: fetch_result（数据拉取结果）
  OUTPUT: boolean
  coverage ← fetch_result.success_codes / fetch_result.requested_codes
  RETURN coverage < EVAL_MIN_CODE_COVERAGE
END FUNCTION

// Fix Checking
FOR ALL fetch_result WHERE isBugCondition_P1_1(fetch_result) DO
  result ← evaluate'(fetch_result)
  ASSERT result.result_status IN {"completed_invalid", "failed"}
    AND result.quality_gate.reason_codes CONTAINS "CODE_COVERAGE_TOO_LOW"
END FOR

// Preservation Checking
FOR ALL fetch_result WHERE NOT isBugCondition_P1_1(fetch_result) DO
  ASSERT evaluate(fetch_result) = evaluate'(fetch_result)
END FOR
```
