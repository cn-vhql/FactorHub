# 需求文档

## 简介

本功能为 FactorHub 新增"日频时序打分模块"（Temporal Scoring Module）。该模块将 FactorHub 扩展为一个批量日频时序打分器：输入股票代码列表与交易日期，系统计算 8 个固定技术因子，通过时序百分位归一化后加权合成综合评分，并以 CSV 文件形式落盘输出。

小批量（≤20 只）请求同步返回结果；大批量（>20 只）请求异步执行，通过任务 ID 查询进度与结果。系统复用现有 `data_service.py` 的数据获取能力和 `factor_service.py` 的因子计算能力，新增 `temporal_scoring_service.py` 与 `scoring.py` 路由。

---

## 词汇表

- **Scoring_Service**：`temporal_scoring_service.py` 中的核心评分服务，负责因子计算、百分位归一化与评分合成
- **Scoring_Router**：`scoring.py` 中的 FastAPI 路由，负责 API 接口与任务编排
- **Task_Manager**：任务管理组件，负责异步任务的创建、状态跟踪与结果落盘
- **Factor_Calculator**：现有 `factor_service.py` 中的因子计算器，被 Scoring_Service 复用
- **Data_Service**：现有 `data_service.py` 中的数据获取服务，被 Scoring_Service 复用
- **trade_date**：交易日期，格式为 `YYYY-MM-DD`，不传时默认为最新交易日
- **task_id**：异步任务的唯一标识符（UUID），用于查询任务状态与结果
- **score**：单只股票在指定交易日的综合评分，取值范围 [0, 100]
- **pct(factor)**：因子值在回望窗口内的历史百分位，取值范围 [0, 1]
- **lookback_days**：计算百分位所需的历史回望天数，默认 372 个自然日
- **SCORING_RESULTS_DIR**：评分结果文件的根目录，默认 `data/reports/scoring_jobs`
- **sync_threshold**：同步/异步模式切换阈值，默认 20 只股票
- **daily_scores.csv**：主输出文件，包含 `date,code,score` 三列
- **status.json**：任务状态文件，记录任务进度与统计信息
- **errors.csv**：错误记录文件，记录评分失败的股票及原因

---

## 需求

### 需求 1：同步批量评分

**用户故事：** 作为量化研究员，我希望对少量股票（≤20 只）发起同步评分请求，以便快速获得当日评分结果，无需轮询任务状态。

#### 验收标准

1. WHEN 请求体中 `codes` 列表长度 ≤ 20，THE Scoring_Router SHALL 在同一 HTTP 响应中返回所有股票的评分结果，响应格式为 `{"success": true, "data": [{"date": "...", "code": "...", "score": ...}]}`
2. WHEN 请求体中 `trade_date` 字段缺失，THE Scoring_Router SHALL 使用最新可用交易日作为评分日期
3. WHEN 请求体中 `trade_date` 字段存在，THE Scoring_Router SHALL 使用该日期作为评分日期
4. IF 某只股票数据获取失败，THEN THE Scoring_Router SHALL 在响应的 `errors` 字段中记录该股票代码与失败原因，并继续返回其余股票的评分结果
5. THE Scoring_Router SHALL 在 POST `/api/scoring/temporal-score` 端点接受如下请求体：`{"codes": ["600519", "000001"], "trade_date": "2026-03-20"}`

### 需求 2：异步批量评分任务

**用户故事：** 作为量化研究员，我希望对大批量股票（>20 只）发起异步评分任务，以便系统在后台处理而不阻塞 API 响应。

#### 验收标准

1. WHEN 请求体中 `codes` 列表长度 > 20，THE Scoring_Router SHALL 立即返回 `task_id` 与初始状态 `pending`，HTTP 状态码为 202
2. WHEN 异步任务创建成功，THE Task_Manager SHALL 在 `{SCORING_RESULTS_DIR}/{task_id}/input_codes.csv` 中写入本次任务的全部股票代码
3. WHEN 异步任务开始执行，THE Task_Manager SHALL 将 `status.json` 中的 `status` 字段更新为 `running`，并记录 `started_at` 时间戳
4. THE Scoring_Router SHALL 在 POST `/api/scoring/temporal-score/jobs` 端点接受与同步接口相同格式的请求体
5. IF 任务创建过程中发生异常，THEN THE Scoring_Router SHALL 返回 HTTP 500 及错误描述

### 需求 3：任务状态查询

**用户故事：** 作为量化研究员，我希望通过 task_id 查询异步任务的执行进度，以便了解任务完成情况。

#### 验收标准

1. WHEN 客户端请求 GET `/api/scoring/temporal-score/jobs/{task_id}`，THE Scoring_Router SHALL 返回该任务的 `status.json` 内容，包含以下字段：`task_id`、`status`、`trade_date`、`total`、`processed`、`success_count`、`failed_count`、`progress`、`started_at`、`updated_at`
2. WHEN 任务状态为 `running`，THE Task_Manager SHALL 保证 `status.json` 中的 `processed` 字段反映已完成处理的股票数量（含成功与失败）
3. WHEN 任务所有股票处理完毕，THE Task_Manager SHALL 将 `status.json` 中的 `status` 字段更新为 `completed`，并记录 `completed_at` 时间戳
4. IF 请求的 `task_id` 不存在，THEN THE Scoring_Router SHALL 返回 HTTP 404 及错误描述
5. THE Scoring_Router SHALL 计算 `progress` 字段为 `processed / total * 100`，精确到小数点后一位

### 需求 4：任务结果查询

**用户故事：** 作为量化研究员，我希望在任务完成后通过 task_id 获取完整的评分结果，以便进行后续分析。

#### 验收标准

1. WHEN 客户端请求 GET `/api/scoring/temporal-score/jobs/{task_id}/results`，THE Scoring_Router SHALL 返回 `daily_scores.csv` 的内容，格式为 JSON 数组 `[{"date": "...", "code": "...", "score": ...}]`
2. WHEN 任务状态不为 `completed`，THE Scoring_Router SHALL 返回 HTTP 400 及提示信息 `任务尚未完成`
3. IF 请求的 `task_id` 不存在，THEN THE Scoring_Router SHALL 返回 HTTP 404 及错误描述
4. WHERE 客户端需要详细因子数据，THE Scoring_Router SHALL 支持通过查询参数 `detail=true` 返回 `daily_scores_detail.csv` 的内容，包含 8 个因子的原始值与百分位值

### 需求 5：评分核心计算

**用户故事：** 作为量化研究员，我希望系统基于 8 个固定技术因子计算时序百分位评分，以便得到标准化的、可横向比较的综合评分。

#### 验收标准

1. THE Scoring_Service SHALL 计算以下 8 个固定因子：`price_vs_sma20`、`momentum_20`、`price_vwma_ratio`、`force_index_ma`、`bollinger_position`、`stochastic_d`、`atr_norm`、`volume_ma_ratio`
2. WHEN 计算单只股票评分，THE Scoring_Service SHALL 获取该股票截至 `trade_date` 的至少 `lookback_days`（默认 372 个自然日）历史数据
3. THE Scoring_Service SHALL 按以下公式计算 `raw_score`：`raw_score = 100 * (0.22 * pct(price_vs_sma20) + 0.18 * pct(momentum_20) + 0.16 * pct(price_vwma_ratio) + 0.14 * pct(force_index_ma) + 0.14 * pct(bollinger_position) + 0.16 * pct(stochastic_d))`
4. THE Scoring_Service SHALL 按以下规则计算惩罚项并得出最终 `score`：
   - `trend_break = 1 if pct(price_vs_sma20) < 0.35 or pct(stochastic_d) < 0.20 else 0`
   - `high_volatility = 1 if pct(atr_norm) > 0.80 else 0`
   - `liquidity_risk = 1 if pct(volume_ma_ratio) < 0.30 else 0`
   - `score = clip(raw_score - 12*trend_break - 10*high_volatility - 8*liquidity_risk, 0, 100)`
5. THE Scoring_Service SHALL 使用 `factor_service.py` 中的 `FactorCalculator` 计算各因子值，复用现有因子定义
6. IF 某只股票的历史数据不足以计算任意一个因子，THEN THE Scoring_Service SHALL 抛出异常，由调用方记录到 `errors.csv`

### 需求 6：因子百分位归一化

**用户故事：** 作为量化研究员，我希望因子值通过历史百分位归一化，以便消除量纲差异，使不同因子可以直接加权合成。

#### 验收标准

1. THE Scoring_Service SHALL 对每个因子，在 `lookback_days` 历史窗口内计算 `trade_date` 当日因子值的历史百分位，结果记为 `pct(factor)`，取值范围为 [0, 1]
2. WHEN 历史窗口内有效数据点少于 20 个，THE Scoring_Service SHALL 抛出数据不足异常
3. THE Scoring_Service SHALL 使用 `scipy.stats.percentileofscore` 或等效方法计算百分位，采用 `rank` 方式处理重复值
4. FOR ALL 有效因子值，THE Scoring_Service SHALL 保证 `pct(factor)` 的计算结果满足：对同一历史窗口内的最小值，百分位趋近于 0；对最大值，百分位趋近于 1（百分位单调性）

### 需求 7：并发控制与任务隔离

**用户故事：** 作为系统运维人员，我希望批量评分任务在受控并发下执行，以便避免因大量并发请求耗尽系统资源。

#### 验收标准

1. THE Task_Manager SHALL 使用 `ThreadPoolExecutor(max_workers=SCORING_MAX_WORKERS)` 执行并发评分，`SCORING_MAX_WORKERS` 默认值为 4
2. WHEN 某只股票的评分过程抛出任何异常，THE Task_Manager SHALL 捕获该异常，将该股票记录到 `errors.csv`，并继续处理其余股票，不中断整个任务
3. THE Task_Manager SHALL 保证不同股票的评分计算相互独立，单只股票的失败不影响其他股票的结果
4. WHILE 任务处于 `running` 状态，THE Task_Manager SHALL 保证同一 `task_id` 的任务只有一个执行实例

### 需求 8：结果增量落盘

**用户故事：** 作为量化研究员，我希望评分结果边计算边写入文件，以便在任务未完成时也能访问已完成的部分结果。

#### 验收标准

1. WHEN 某只股票评分成功，THE Task_Manager SHALL 立即将 `date,code,score` 一行追加写入 `{SCORING_RESULTS_DIR}/{task_id}/daily_scores.csv`
2. WHEN 某只股票评分成功，THE Task_Manager SHALL 立即将该股票的 8 个因子原始值与百分位值追加写入 `{SCORING_RESULTS_DIR}/{task_id}/daily_scores_detail.csv`
3. WHEN 某只股票评分失败，THE Task_Manager SHALL 立即将 `code,error` 一行追加写入 `{SCORING_RESULTS_DIR}/{task_id}/errors.csv`
4. THE Task_Manager SHALL 在每只股票处理完成后（无论成功或失败）更新 `status.json` 中的 `processed`、`success_count`、`failed_count`、`progress`、`updated_at` 字段
5. THE Task_Manager SHALL 保证 `daily_scores.csv` 的首行为表头 `date,code,score`，`daily_scores_detail.csv` 的首行为包含所有列名的表头

### 需求 9：输出文件结构

**用户故事：** 作为量化研究员，我希望评分结果以规范的目录结构存储，以便外部系统可以可靠地读取结果文件。

#### 验收标准

1. THE Task_Manager SHALL 在任务创建时创建目录 `{SCORING_RESULTS_DIR}/{task_id}/`，并在该目录下管理以下文件：`input_codes.csv`、`status.json`、`daily_scores.csv`、`daily_scores_detail.csv`、`errors.csv`
2. THE Task_Manager SHALL 保证 `daily_scores.csv` 的列顺序为 `date,code,score`，其中 `score` 保留一位小数
3. THE Task_Manager SHALL 保证 `input_codes.csv` 包含本次任务的全部输入股票代码，每行一个
4. IF 任务目录已存在，THEN THE Task_Manager SHALL 覆盖已有文件，不抛出异常

### 需求 10：配置管理

**用户故事：** 作为系统运维人员，我希望评分模块的关键参数可通过配置文件或环境变量调整，以便在不修改代码的情况下适配不同部署环境。

#### 验收标准

1. THE Scoring_Service SHALL 从 `settings.py` 读取以下配置项，并在未配置时使用默认值：
   - `SCORING_MAX_WORKERS`（默认 4）：ThreadPoolExecutor 最大并发数
   - `SCORING_SYNC_THRESHOLD`（默认 20）：同步/异步模式切换阈值
   - `SCORING_RESULTS_DIR`（默认 `data/reports/scoring_jobs`）：结果文件根目录
   - `SCORING_LOOKBACK_DAYS`（默认 372）：百分位计算历史回望天数
2. THE Scoring_Service SHALL 支持通过环境变量覆盖上述配置项（遵循 `pydantic-settings` 的现有机制）
3. WHEN 应用启动，THE Scoring_Service SHALL 自动创建 `SCORING_RESULTS_DIR` 目录（若不存在）

### 需求 11：路由注册

**用户故事：** 作为后端开发人员，我希望评分模块的路由在应用启动时自动注册，以便 API 端点可以正常访问。

#### 验收标准

1. THE Scoring_Router SHALL 在 `main.py` 中以 `prefix="/api/scoring"` 注册，tag 为 `"时序打分"`
2. THE Scoring_Router SHALL 暴露以下 4 个端点：
   - `POST /api/scoring/temporal-score`（同步评分）
   - `POST /api/scoring/temporal-score/jobs`（异步任务创建）
   - `GET /api/scoring/temporal-score/jobs/{task_id}`（任务状态查询）
   - `GET /api/scoring/temporal-score/jobs/{task_id}/results`（任务结果查询）
3. THE Scoring_Router SHALL 在 FastAPI 的 `/docs` 页面中正确展示上述端点的请求/响应 schema
