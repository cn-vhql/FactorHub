# 实现任务：OHLCV 因子扩展（ohlcv-factor-expansion）

## 任务列表

- [x] 1 在 `_get_default_factors()` 中新增 12 个 P0 因子
  - [x] 1.1 在"价格收益率"分类下新增 `overnight_return_1` 和 `intraday_return_1`
  - [x] 1.2 在"技术形态"分类下新增 `body_ratio` 和 `upper_shadow_ratio`
  - [x] 1.3 在"价格位置"分类下新增 `close_location_value` 和 `breakout_strength_20`
  - [x] 1.4 在"动量趋势"分类下新增 `efficiency_ratio_20` 和 `kama_ratio_20`
  - [x] 1.5 在"波动率风险"分类下新增 `parkinson_vol_20` 和 `garman_klass_vol_20`
  - [x] 1.6 在"资金流动"分类下新增 `cmf_20`
  - [x] 1.7 在"成交量资金流"分类下新增 `volume_zscore_20`

- [ ] 2 编写单因子单元测试
  - [x] 2.1 为 12 个 P0 因子编写可计算性测试（250 日合成数据，无异常）
  - [x] 2.2 为 12 个 P0 因子编写 warmup NaN 数量验证测试
  - [x] 2.3 为有界因子（body_ratio / upper_shadow_ratio / close_location_value / efficiency_ratio_20 / cmf_20）编写值域约束测试
  - [x] 2.4 为 parkinson_vol_20 / garman_klass_vol_20 编写非负约束测试
  - [x] 2.5 编写边界条件测试：`high == low`（一字板）时 body_ratio / upper_shadow_ratio / close_location_value / cmf_20 返回有限数
  - [x] 2.6 编写边界条件测试：`volume == 0` 时 volume_zscore_20 / cmf_20 返回有限数
  - [x] 2.7 验证 `breakout_strength_20` 使用 `.shift(1)` 不含未来函数（对比 shift 前后结果差异）

- [ ] 3 编写属性测试（Hypothesis）
  - [x] 3.1 属性测试：对任意合法 OHLCV 序列，`body_ratio` ∈ `[-1, 1]`
  - [x] 3.2 属性测试：对任意合法 OHLCV 序列，`efficiency_ratio_20` ∈ `[0, 1]`（序列长度 ≥ 21）
  - [x] 3.3 属性测试：对任意合法 OHLCV 序列，`close_location_value` ∈ `[0, 1]`

- [ ] 4 编写集成测试
  - [x] 4.1 测试干净数据库启动后 `/api/factors` 返回包含全部 12 个 P0 因子
  - [x] 4.2 测试 `load_preset_factors()` 幂等性（重复调用不产生重复记录）
  - [x] 4.3 验证现有 90 个因子的 name/code/category 未被修改

- [ ] 5 单因子烟测与稳定性验证脚本
  - [x] 5.1 编写 `scripts/run_ohlcv_factor_smoke_test.py`：对每个 P0 因子用真实股票数据（≥ 250 日）验证可计算性、warmup、inf/NaN 比例
  - [x] 5.2 在脚本中输出每个因子的分布统计（均值、标准差、分位数、缺失率）
  - [x] 5.3 在脚本中计算 P0 因子与现有 baseline 因子的相关性矩阵，标记绝对相关系数 > 0.95 的因子

- [ ] 6 审计回归验证
  - [x] 6.1 运行 `run_factor_library_audit.py`，确认核心域仍为 8/8，总因子数增加 ≥ 12
  - [x] 6.2 确认新增因子的分类正确映射到审计域（CATEGORY_DOMAIN_MAP 已覆盖所有新分类）
