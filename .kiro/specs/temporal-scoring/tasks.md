# 实施任务清单：日频时序打分模块（temporal-scoring）

## 概述

基于需求文档与技术设计文档，将日频时序打分模块拆分为以下实施任务。任务 A（评分核心服务）与任务 B（任务编排与文件落盘）可并行开发，任务 C（API 路由）依赖 A+B 接口约定，任务 D（配置接线）最后完成。属性测试（E）依赖任务 A，集成验收测试（F）依赖全部任务完成。

---

## 任务列表

- [x] 1. 实现评分核心服务（TemporalScoringService）
  - 新建 `backend/services/temporal_scoring_service.py`，定义 `TemporalScoringService` 类骨架
  - 实现 `_get_history_data(code, trade_date) -> pd.DataFrame`：
    - 使用 `data_service.get_stock_data(code, start_date, trade_date)` 获取历史 OHLCV 数据
    - `start_date = trade_date - timedelta(days=settings.SCORING_LOOKBACK_DAYS)`
  - 实现 `_compute_factors(df) -> dict[str, pd.Series]`，计算以下 8 个固定因子：
    - `price_vs_sma20`：`close / SMA(close, timeperiod=20)`（复用 factor_service 表达式）
    - `momentum_20`：`close / close.shift(20) - 1`
    - `price_vwma_ratio`：`close / (SUM(close*volume, 20) / SUM(volume, 20))`（内部实现）
    - `force_index_ma`：`SMA(close.diff(1) * volume, timeperiod=13)`（内部实现）
    - `bollinger_position`：`(close - lower) / (upper - lower)`（复用 factor_service 表达式）
    - `stochastic_d`：`SMA(stochastic_k, timeperiod=3)`（复用 factor_service 表达式）
    - `atr_norm`：`ATR(high, low, close, timeperiod=14) / close`（复用 factor_service 表达式）
    - `volume_ma_ratio`：`volume / SMA(volume, timeperiod=10)`（复用 factor_service 表达式）
    - 使用 `factor_service.calculator.calculate(df, expr)` 计算各因子
  - _需求：5.1、5.5_

- [x] 2. 实现百分位归一化与评分公式
  - [x] 2.1 实现 `_compute_percentile(series, current_value) -> float`：
    - 对 `series` 执行 `dropna()`，有效数据点 < 20 时抛出 `ValueError`
    - 调用 `scipy.stats.percentileofscore(history, current_value, kind='rank')`
    - 返回值除以 100 归一化到 `[0, 1]`
    - _需求：6.1、6.2、6.3_

  - [x] 2.2 为 `_compute_percentile` 编写属性测试（属性 7）
    - **属性 7：百分位正确性（范围 + 单调性）**
    - **验证：需求 6.1、6.4**
    - 使用 `@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=20))`
    - 验证输出始终在 `[0, 1]` 范围内
    - 验证对同一历史窗口，较大因子值对应不小于较小因子值的百分位（单调性）

  - [x] 2.3 实现 `_compute_score(pcts) -> float`：
    - 按公式计算 `raw_score = 100 * (0.22*pct_price_vs_sma20 + 0.18*pct_momentum_20 + 0.16*pct_price_vwma_ratio + 0.14*pct_force_index_ma + 0.14*pct_bollinger_position + 0.16*pct_stochastic_d)`
    - 计算三个惩罚项：`trend_break`、`high_volatility`、`liquidity_risk`
    - 返回 `np.clip(raw_score - 12*trend_break - 10*high_volatility - 8*liquidity_risk, 0, 100)`
    - _需求：5.3、5.4_

  - [x] 2.4 为 `_compute_score` 编写属性测试（属性 6）
    - **属性 6：评分公式正确性**
    - **验证：需求 5.3、5.4**
    - 使用 `@given(st.fixed_dictionaries({factor: st.floats(0, 1) for factor in FACTOR_NAMES}))`，`max_examples=200`
    - 验证输出等于手动计算的 `clip(raw_score - penalties, 0, 100)`
    - 验证输出始终在 `[0, 100]` 范围内

- [x] 3. 实现 score_one_stock 与 score_many_stocks
  - [x] 3.1 实现 `score_one_stock(code, trade_date) -> dict`：
    - 调用 `_get_history_data` → `_compute_factors` → `_compute_percentile`（×8）→ `_compute_score`
    - 返回 `{"date": trade_date, "code": code, "score": round(score, 1), "factors": {...}, "pcts": {...}}`
    - 若任意步骤抛出异常，向上传播（由调用方处理）
    - _需求：5.2、5.6_

  - [x] 3.2 实现 `score_many_stocks(codes, trade_date) -> tuple[list[dict], list[dict]]`：
    - 顺序调用 `score_one_stock`，捕获每只股票的异常
    - 成功结果收集到 `results`，失败记录到 `errors`（`{"code": ..., "error": str(e)}`）
    - 返回 `(results, errors)`
    - _需求：1.4、7.2、7.3_

  - [x] 3.3 为错误隔离编写属性测试（属性 2）
    - **属性 2：错误隔离不中断**
    - **验证：需求 1.4、7.2、7.3**
    - 使用 `@given`，mock `score_one_stock` 使部分股票抛出异常
    - 验证 `len(results) + len(errors) == len(codes)`（总数守恒）
    - 验证有异常的股票出现在 `errors` 中，无异常的出现在 `results` 中

- [x] 4. 检查点 —— 核心评分服务自测
  - 确保所有测试通过，如有疑问请向用户确认。

- [x] 5. 实现 TaskManager —— 任务创建与目录管理
  - 在 `temporal_scoring_service.py` 中新增 `TaskManager` 类
  - 实现 `_task_dir(task_id) -> Path`：返回 `settings.SCORING_RESULTS_DIR / task_id`
  - 实现 `create_task(codes, trade_date) -> str`：
    - 生成 UUID 作为 `task_id`
    - 创建任务目录 `{SCORING_RESULTS_DIR}/{task_id}/`
    - 写入 `input_codes.csv`（每行一个股票代码，无表头）
    - 写入初始 `status.json`（`status=pending`，含所有必需字段）
    - 返回 `task_id`
  - _需求：2.1、2.2、9.1、9.3_

- [x] 6. 实现 TaskManager —— 文件落盘与原子写入
  - [x] 6.1 实现 `_write_status(task_id, status_dict) -> None`：
    - 写入临时文件 `status.json.tmp`，然后 `rename` 到 `status.json`（原子操作）
    - 在 `TaskManager.__init__` 中初始化 `self._file_locks: dict[str, threading.Lock] = {}`
    - _需求：8.4、设计文档错误处理策略_

  - [x] 6.2 实现 `_append_score_row(task_id, result) -> None`：
    - 加锁（`self._file_locks[task_id]`）后追加写入 `daily_scores.csv`（`date,code,score`，score 保留 1 位小数）
    - 同时追加写入 `daily_scores_detail.csv`（含 8 个因子原始值与百分位值）
    - 若文件不存在则先写表头行
    - _需求：8.1、8.2、8.5、9.2_

  - [x] 6.3 实现 `_append_error_row(task_id, code, error) -> None`：
    - 加锁后追加写入 `errors.csv`（`code,error`）
    - 若文件不存在则先写表头行
    - _需求：8.3_

  - [x] 6.4 为结果文件格式编写属性测试（属性 8）
    - **属性 8：结果文件格式正确性**
    - **验证：需求 8.5、9.2**
    - 使用 `@given(st.lists(st.builds(ScoreItem, ...), min_size=1))`
    - 调用 `_append_score_row` 写入随机评分结果后读取文件
    - 验证首行为表头 `date,code,score`
    - 验证每行 `score` 值保留 1 位小数（`str(round(score, 1)) == score_str`）
    - 验证数据行数等于写入的记录数

- [x] 7. 实现 TaskManager —— 后台任务执行
  - 实现 `run_task(task_id, codes, trade_date) -> None`：
    - 更新 `status.json`：`status=running`，`started_at=now`
    - 初始化 `ThreadPoolExecutor(max_workers=settings.SCORING_MAX_WORKERS)`
    - 提交所有股票的 `score_one_stock` 任务，使用 `as_completed` 收集结果
    - 每只股票完成后：
      - 成功：调用 `_append_score_row`，更新 `success_count`
      - 失败：调用 `_append_error_row`，更新 `failed_count`
      - 更新 `processed`、`progress`、`updated_at`，调用 `_write_status`
    - 全部完成后：更新 `status=completed`，`completed_at=now`
    - 任务级异常：更新 `status=failed`，记录错误信息
  - _需求：2.3、3.2、3.3、7.1、7.4、8.1-8.4_

- [x] 8. 实现 TaskManager —— 状态与结果查询
  - 实现 `get_status(task_id) -> dict`：
    - 读取并返回 `status.json` 内容
    - 任务目录不存在时抛出 `FileNotFoundError`
    - _需求：3.1、3.2、3.4、3.5_

  - 实现 `get_results(task_id, detail=False) -> list[dict]`：
    - 先调用 `get_status` 检查状态，非 `completed` 时抛出 `ValueError("任务尚未完成")`
    - `detail=False`：读取 `daily_scores.csv` 返回 JSON 数组
    - `detail=True`：读取 `daily_scores_detail.csv` 返回 JSON 数组
    - _需求：4.1、4.2、4.4_

- [x] 9. 检查点 —— TaskManager 自测
  - 确保所有测试通过，如有疑问请向用户确认。

- [x] 10. 实现 API 路由（scoring.py）
  - 新建 `backend/api/routers/scoring.py`
  - 定义 Pydantic 请求/响应模型：
    - `ScoringRequest`：`codes: list[str]`，`trade_date: Optional[str] = None`
    - `ScoreItem`：`date: str`，`code: str`，`score: float`
    - `SyncScoringResponse`：`success: bool`，`data: list[ScoreItem]`，`errors: list[dict]`
    - `AsyncJobResponse`：`task_id: str`，`status: str`
    - `JobStatus`：含所有状态字段（见设计文档 Pydantic Schema）
  - _需求：1.5、11.3_

- [x] 11. 实现同步与异步评分端点
  - [x] 11.1 实现 `POST /temporal-score`（同步评分）：
    - 调用 `scoring_service.score_many_stocks(codes, trade_date)`
    - 返回 `SyncScoringResponse`
    - _需求：1.1、1.2、1.3、1.4_

  - [x] 11.2 实现 `POST /temporal-score/jobs`（异步任务创建，HTTP 202）：
    - 调用 `task_manager.create_task(codes, trade_date)` 获取 `task_id`
    - 通过 `BackgroundTasks` 触发 `task_manager.run_task(task_id, codes, trade_date)`
    - 返回 `AsyncJobResponse(task_id=task_id, status="pending")`，状态码 202
    - 任务创建异常时返回 HTTP 500
    - _需求：2.1、2.4、2.5_

  - [x] 11.3 实现 `GET /temporal-score/jobs/{task_id}`（任务状态查询）：
    - 调用 `task_manager.get_status(task_id)`
    - `FileNotFoundError` 时返回 HTTP 404
    - 返回 `JobStatus`
    - _需求：3.1、3.4、3.5_

  - [x] 11.4 实现 `GET /temporal-score/jobs/{task_id}/results`（任务结果查询）：
    - 调用 `task_manager.get_results(task_id, detail=detail)`
    - `FileNotFoundError` 时返回 HTTP 404
    - `ValueError("任务尚未完成")` 时返回 HTTP 400
    - 支持查询参数 `?detail=true`
    - _需求：4.1、4.2、4.3、4.4_

- [x] 12. 配置接线
  - 在 `backend/core/settings.py` 的 `Settings` 类中新增 4 个配置项：
    - `SCORING_MAX_WORKERS: int = 4`
    - `SCORING_SYNC_THRESHOLD: int = 20`
    - `SCORING_RESULTS_DIR: Path = REPORTS_DIR / "scoring_jobs"`
    - `SCORING_LOOKBACK_DAYS: int = 372`
  - 在 `Settings.__init__` 中添加 `self.SCORING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)`
  - 在 `backend/api/main.py` 中导入并注册 scoring router：
    - `from .routers import scoring`
    - `app.include_router(scoring.router, prefix="/api/scoring", tags=["时序打分"])`
  - _需求：10.1、10.2、10.3、11.1_

- [x] 13. 最终检查点 —— 集成验收测试
  - [x] 13.1 编写 10 只股票同步评分集成测试：
    - 调用 `POST /api/scoring/temporal-score`，`codes` 长度为 10
    - 验证响应包含 `success=true`，`data` 数组长度 + `errors` 数组长度 = 10
    - 验证每条记录的 `date` 字段与请求的 `trade_date` 一致
    - _需求：1.1、1.3_

  - [x] 13.2 编写 50 只股票异步评分集成测试：
    - 调用 `POST /api/scoring/temporal-score/jobs`，`codes` 长度为 50
    - 验证响应状态码为 202，响应体包含有效 UUID 格式的 `task_id`
    - 轮询 `GET /api/scoring/temporal-score/jobs/{task_id}` 直至 `status=completed`
    - 验证 `daily_scores.csv` 行数等于 `success_count`
    - _需求：2.1、3.3、8.1_

  - [x] 13.3 编写 200 只股票异步评分增量落盘集成测试：
    - 调用 `POST /api/scoring/temporal-score/jobs`，`codes` 长度为 200
    - 任务运行中途（`status=running`）读取 `daily_scores.csv`，验证已有部分结果写入
    - 等待任务完成后验证 `status.json` 中 `processed == total`
    - _需求：8.1、8.4、3.2_

  - 确保所有测试通过，如有疑问请向用户确认。

---

## 备注

- 标注 `*` 的子任务为可选项，可在 MVP 阶段跳过以加快交付
- 每个任务均引用了具体的需求条款，便于追溯
- 属性测试使用 Hypothesis 库，每个属性最少运行 100 次（建议 200 次）
- 属性测试注释格式：`# Feature: temporal-scoring, Property N: <属性名>`
- `price_vwma_ratio` 和 `force_index_ma` 需在 `TemporalScoringService` 内部直接用 pandas 实现，不依赖 `FactorCalculator.calculate`（因其表达式不在预置因子中）
