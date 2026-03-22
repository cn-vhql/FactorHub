# 需求文档：OHLCV 因子扩展（ohlcv-factor-expansion）

## 需求概述

在不引入任何外部数据、只使用日频 OHLCV 的前提下，向现有因子库补充 12 个 P0 正交时序因子，覆盖隔夜/盘中拆分、K 线结构、突破强度、趋势质量、高阶波动率、增强量价流、自适应趋势七个子域。不改动数据库 schema、不修改 API 契约、不影响评分链路。

---

## 需求列表

### 需求 1：P0 因子定义与注册

**用户故事**：作为系统管理员，我希望服务启动时自动加载 12 个 P0 预置因子，以便因子库覆盖更多正交时序域。

#### 验收标准

**1.1** 在 `FactorService._get_default_factors()` 中新增以下 12 个因子条目，分布在对应分类下：

| 因子名 | 分类 |
|--------|------|
| `overnight_return_1` | 价格收益率 |
| `intraday_return_1` | 价格收益率 |
| `body_ratio` | 技术形态 |
| `upper_shadow_ratio` | 技术形态 |
| `close_location_value` | 价格位置 |
| `breakout_strength_20` | 价格位置 |
| `efficiency_ratio_20` | 动量趋势 |
| `kama_ratio_20` | 动量趋势 |
| `parkinson_vol_20` | 波动率风险 |
| `garman_klass_vol_20` | 波动率风险 |
| `cmf_20` | 资金流动 |
| `volume_zscore_20` | 成交量资金流 |

**1.2** 每个因子条目必须包含 `name`、`code`（表达式字符串）、`description` 三个字段。

**1.3** 所有新增因子的 `formula_type` 实际为 `"expression"`（`FactorModel` 字段默认值），`source` 为 `"preset"`，`is_active` 为 `1`（默认值）。字典条目只需包含 `name`、`code`、`description`，与现有内置因子格式一致，无需显式传 `formula_type`。

**1.4** 因子名称使用 snake_case + 主窗口后缀，长度不超过 100 字符。

---

### 需求 2：表达式正确性与防御性编码

**用户故事**：作为量化研究员，我希望每个新因子的表达式在边界条件下不抛出异常，以便在真实数据中稳定运行。

#### 验收标准

**2.1** 所有涉及 `high - low`、`high - low + ...`、`STD(volume, n)` 等可能为零的分母，必须加 `1e-12` 防零除。

**2.2** `breakout_strength_20` 中的历史最高价引用必须使用 `HHV(high, 20).shift(1)`，不得直接使用 `HHV(high, 20)`，以避免未来函数污染。

**2.3** 对于 `high == low`（一字板）的行，`body_ratio`、`upper_shadow_ratio`、`close_location_value`、`cmf_20` 的计算结果必须是有限数（非 inf、非 NaN）。

**2.4** 对于 `volume == 0` 的行，`volume_zscore_20` 和 `cmf_20` 的计算结果必须是有限数。

---

### 需求 3：warmup NaN 一致性

**用户故事**：作为量化研究员，我希望每个因子的 warmup NaN 数量与其主窗口长度一致，以便在回测中正确截断数据。

#### 验收标准

**3.1** 对于 250 日干净 OHLCV 数据，各因子的 warmup NaN 行数（从序列头部起）满足：

| 因子名 | 预期 warmup NaN 行数 |
|--------|---------------------|
| `overnight_return_1` | 1 |
| `intraday_return_1` | 0 |
| `body_ratio` | 0 |
| `upper_shadow_ratio` | 0 |
| `close_location_value` | 0 |
| `breakout_strength_20` | 20 |
| `efficiency_ratio_20` | 20 |
| `parkinson_vol_20` | 19 |
| `garman_klass_vol_20` | 19 |
| `cmf_20` | 19 |
| `volume_zscore_20` | 19 |
| `kama_ratio_20` | 20 |

**3.2** warmup 之后的行不得全部为 NaN（至少有一个有效值）。

---

### 需求 4：值域约束

**用户故事**：作为量化研究员，我希望有界因子的输出值域在理论范围内，以便用于截面排序和打分。

#### 验收标准

**4.1** `body_ratio` 的有效值（非 NaN）必须在 `[-1, 1]` 范围内。

**4.2** `upper_shadow_ratio` 的有效值必须在 `[0, 1]` 范围内。

**4.3** `close_location_value` 的有效值必须在 `[0, 1]` 范围内。

**4.4** `efficiency_ratio_20` 的有效值必须在 `[0, 1]` 范围内。

**4.5** `parkinson_vol_20` 和 `garman_klass_vol_20` 的有效值必须 ≥ 0。

**4.6** `cmf_20` 的有效值必须在 `[-1, 1]` 范围内。

---

### 需求 5：自动加载与幂等性

**用户故事**：作为运维人员，我希望服务重启后因子库状态保持一致，不会重复插入或丢失因子。

#### 验收标准

**5.1** 干净数据库（无任何因子记录）启动后，`/api/factors` 接口返回的因子列表必须包含全部 12 个 P0 因子（按 `name` 字段匹配）。

**5.2** 服务重启后再次调用 `load_preset_factors()`，不得产生重复因子记录（幂等性）。

**5.3** `config/factors.yaml` 为空时，新增因子必须通过 `_get_default_factors()` 内置路径加载，不依赖 YAML 文件。

**5.4** 新增因子不得修改数据库 schema（`factors` 表结构不变）。

---

### 需求 6：不影响现有因子和评分链路

**用户故事**：作为系统管理员，我希望新增因子不破坏现有 90 个因子的计算结果和评分链路，以便系统平滑升级。

#### 验收标准

**6.1** 现有 90 个预置因子的 `name`、`code`、`category`、`description` 字段不得被修改。

**6.2** `/api/scoring` 相关接口的行为不得因新增因子而改变。

**6.3** `run_factor_library_audit.py` 重新运行后，核心域覆盖仍为 8/8，总因子数增加 ≥ 12。

**6.4** 不新增数据库迁移脚本，不修改 `FactorModel`、`FactorRepository`、`FactorLibraryAuditService`。

---

### 需求 7：单因子稳定性验证

**用户故事**：作为量化研究员，我希望通过单因子烟测筛选出稳定因子，以便确定进入下一阶段搜索空间的候选集。

#### 验收标准

**7.1** 对每个 P0 因子，使用至少 1 只股票、不少于 250 个交易日的真实数据，验证：
- 表达式可计算（无异常）
- warmup NaN 数量符合需求 3.1
- 无 inf 值
- 缺失率（NaN 比例）< 20%

**7.2** 对每个 P0 因子，输出分布统计：均值、标准差、5%/25%/50%/75%/95% 分位数、缺失率。

**7.3** 计算每个 P0 因子与现有 baseline 因子的绝对相关系数；若某因子与任意现有因子的绝对相关系数 > 0.95，则将其标记为"高相关"，降为 P1 候选。

**7.4** 至少 4 个 P0 因子通过稳定性验证（满足 7.1 且未被 7.3 降级），进入下一阶段搜索空间。

---

### 需求 8：P1 备选因子预留

**用户故事**：作为量化研究员，我希望 P1 备选因子有明确的定义文档，以便在第二阶段快速实现。

#### 验收标准

**8.1** 以下 6 个 P1 因子的表达式和分类在设计文档中有明确记录（本阶段不实现）：
- `lower_shadow_ratio`
- `channel_width_20`
- `up_day_ratio_20`
- `price_volume_corr_20`
- `willr_14`
- `kama_slope_10`

**8.2** P1 因子不得在本阶段写入 `_get_default_factors()`，不得入库。
