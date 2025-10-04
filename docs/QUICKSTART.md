# FactorHub 快速开始指南

## 🚀 快速启动

### 1. 环境准备

```bash
# 克隆或下载项目
cd FactorHub

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动应用

#### 方式一：使用启动脚本（推荐）
```bash
python scripts/start_app.py
```

#### 方式二：使用Streamlit
```bash
streamlit run ui/app.py
```

#### 方式三：使用主程序
```bash
python main.py
```

### 3. 访问界面

打开浏览器访问:
- 本地访问: http://localhost:8501
- 网络访问: http://服务器IP地址:8501
- 如果运行在云服务器，使用公有IP地址

## 📋 核心功能使用

### 数据管理
1. 选择市场（沪市/深市/全部）
2. 选择股票池（沪深300/中证500/创业板）
3. 设置时间范围和数据频率
4. 点击"获取数据"

### 因子管理
1. **预置因子**：选择RSI、MACD等技术指标
2. **自定义因子**：编写Python代码或上传CSV文件
3. **批量计算**：选择多个因子进行批量计算

### 因子分析
1. 选择已计算的因子
2. 执行IC分析、分层收益、换手率等分析
3. 查看可视化结果和统计指标

### 策略回测
1. 选择因子组合构建策略
2. 配置回测参数（时间范围、交易成本等）
3. 查看回测结果和绩效报告

### 因子挖掘
1. 选择挖掘方法（组合生成/遗传算法）
2. 设置挖掘参数
3. 自动发现新的有效因子

## 💡 使用示例

### 示例1：基础因子分析
```python
# 获取数据
data_provider = AKShareDataProvider()
data = data_provider.get_multiple_stocks_data(
    ['000001', '000002'], '2020-01-01', '2023-12-31'
)

# 计算因子
calculator = FactorCalculator()
factor_data = calculator.calculate_single_factor('RSI', data)

# 分析因子
analyzer = FactorAnalyzer()
ic_results = analyzer.calculate_ic_analysis(factor_data, 'RSI')
```

### 示例2：策略回测
```python
# 创建策略
strategy = FactorStrategy(
    name='RSI策略',
    factor_names=['RSI'],
    top_percent=0.2,
    long_only=True
)

# 执行回测
backtester = Backtester(initial_capital=1000000)
results = backtester.run_backtest(strategy, data, '2021-01-01', '2023-12-31')
```

## 🔧 常见问题

### Q: 数据获取失败怎么办？
A: 检查网络连接，确认股票代码格式，查看AKShare接口状态。

### Q: 因子计算出错怎么办？
A: 检查数据完整性，确认因子名称正确，查看错误日志。

### Q: 回测结果异常怎么办？
A: 检查时间范围设置，确认策略参数，验证数据格式。

### Q: 如何添加自定义因子？
A: 在"因子管理"页面的"自定义因子"标签页，编写Python代码或上传CSV文件。

### Q: 如何导出结果？
A: 在回测结果页面点击"导出回测结果"，或在分析页面使用导出功能。

## 📖 详细文档

- [完整使用说明](README.md)
- [API文档](docs/api.md)
- [开发指南](docs/development.md)

## 🆘 获取帮助

- 问题反馈：提交GitHub Issues
- 功能建议：提交Feature Requests
- 技术支持：查看FAQ或联系开发团队

---

**提示**: 首次使用建议运行示例代码熟悉系统功能。