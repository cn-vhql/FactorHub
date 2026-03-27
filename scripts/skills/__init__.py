"""
FactorHub Claude Code Skills - 可运行的技能脚本集合

这个目录包含7个核心Skills的Python实现:

1. get_stock_data.py     - 获取股票OHLCV数据
2. validate_formula.py   - 验证因子公式
3. factor_analyze.py     - IC/IR/稳定性分析
4. backtest_factor.py    - 单因子回测
5. compare_strategies.py - 多策略对比
6. mine_factors.py       - 遗传算法因子挖掘
7. portfolio_optimize.py - 组合权重优化

每个脚本都可以独立运行，并通过HTTP API与FactorHub后端通信。

快速开始:
  python get_stock_data.py 000001 2024-01-01 2024-03-27
  python factor_analyze.py 000001 "RSI(close, 14)" 2023-01-01 2024-12-31
  python backtest_factor.py 000001 "close/SMA(close, 20)" 2023-01-01 2024-12-31

使用帮助:
  python <skill_name>.py --help
"""

__all__ = [
    'skills_base',
    'get_stock_data',
    'validate_formula',
    'factor_analyze',
    'backtest_factor',
    'compare_strategies',
    'mine_factors',
    'portfolio_optimize',
]
