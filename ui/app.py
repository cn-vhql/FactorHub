"""
Streamlit主应用
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="FactorHub - 量化因子分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用安全导入
from imports import (
    safe_import, DEFAULT_CONFIG, logger,
    AKShareDataProvider, DataPreprocessor, FactorLibrary, FactorCalculator,
    CustomFactorManager, FactorAnalyzer, PortfolioAnalyzer, Backtester,
    FactorStrategy, FactorGenerator, GeneticFactorMiner
)

def main():
    """主函数"""
    st.title("📈 FactorHub - 量化因子分析平台")
    st.markdown("---")

    # 侧边栏导航
    st.sidebar.title("功能导航")
    page = st.sidebar.selectbox(
        "选择功能模块",
        [
            "📊 数据管理",
            "🔧 因子管理",
            "📈 因子分析",
            "🎯 策略回测",
            "⚡ 因子挖掘",
            "📋 系统概览"
        ]
    )

    # 初始化session state
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'factors' not in st.session_state:
        st.session_state.factors = pd.DataFrame()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = {}

    # 页面路由
    if page == "📊 数据管理":
        data_management_page()
    elif page == "🔧 因子管理":
        factor_management_page()
    elif page == "📈 因子分析":
        factor_analysis_page()
    elif page == "🎯 策略回测":
        backtest_page()
    elif page == "⚡ 因子挖掘":
        factor_mining_page()
    elif page == "📋 系统概览":
        overview_page()

def data_management_page():
    """数据管理页面"""
    st.header("📊 数据管理")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("数据获取配置")

        # 数据获取参数
        market = st.selectbox("选择市场", ["全部", "沪市", "深市"], index=0)
        stock_pool = st.selectbox("选择股票池", ["沪深300", "中证500", "创业板", "自定义"], index=0)

        if stock_pool == "自定义":
            custom_symbols = st.text_area("输入股票代码（每行一个）", "000001\n000002\n600036")
            symbols = [s.strip() for s in custom_symbols.split('\n') if s.strip()]
        else:
            symbols = None

        start_date = st.date_input("开始日期", datetime(2020, 1, 1))
        end_date = st.date_input("结束日期", datetime(2023, 12, 31))

        frequency = st.selectbox("数据频率", ["日线", "周线"], index=0)
        adjust_type = st.selectbox("复权类型", ["前复权", "后复权", "不复权"], index=0)

        # 数据获取按钮
        if st.button("🚀 获取数据", type="primary"):
            with st.spinner("正在获取数据..."):
                try:
                    # 数据提供者
                    data_provider = AKShareDataProvider()

                    if stock_pool == "自定义" and symbols:
                        stock_list = pd.DataFrame({'symbol': symbols})
                    else:
                        pool_map = {"沪深300": "hs300", "中证500": "zz500", "创业板": "cyb"}
                        stock_symbols = data_provider.get_stock_pool(pool_map.get(stock_pool, "hs300"))
                        stock_list = pd.DataFrame({'symbol': stock_symbols})

                    # 获取数据
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def progress_callback(progress, symbol):
                        progress_bar.progress(progress)
                        progress_text.text(f"正在获取: {symbol}")

                    data = data_provider.get_multiple_stocks_data(
                        stock_list['symbol'].tolist(),
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        'qfq',  # 简化处理
                        progress_callback
                    )

                    if not data.empty:
                        # 数据预处理
                        preprocessor = DataPreprocessor()
                        processed_data = preprocessor.preprocess_pipeline(data)

                        st.session_state.data = processed_data
                        st.success(f"✅ 数据获取成功！共 {len(processed_data)} 条记录")
                        progress_text.text("数据获取完成！")
                    else:
                        st.error("❌ 数据获取失败")

                except Exception as e:
                    st.error(f"❌ 获取数据时发生错误: {str(e)}")

    with col2:
        st.subheader("数据概览")
        if not st.session_state.data.empty:
            # 数据摘要
            summary = {
                "记录数": f"{len(st.session_state.data):,}",
                "股票数": st.session_state.data['symbol'].nunique(),
                "日期范围": f"{st.session_state.data['date'].min().date()} 至 {st.session_state.data['date'].max().date()}",
                "缺失值": st.session_state.data.isnull().sum().sum()
            }

            for key, value in summary.items():
                st.metric(key, value)

            # 缓存信息
            if AKShareDataProvider and hasattr(AKShareDataProvider, 'get_cache_info'):
                try:
                    with st.expander("📦 缓存信息"):
                        provider = AKShareDataProvider()

                        # 检查 provider 是否有 get_cache_info 方法
                        if hasattr(provider, 'get_cache_info'):
                            cache_info = provider.get_cache_info()

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("缓存文件数", cache_info["total_files"])
                            with col2:
                                st.metric("缓存大小", f"{cache_info['total_size_mb']:.1f} MB")

                            st.caption(f"缓存目录: {cache_info['cache_dir']}")

                            # 缓存清理按钮
                            if st.button("🗑️ 清理过期缓存", help="清理超过7天的缓存文件"):
                                if hasattr(provider, 'clear_cache'):
                                    provider.clear_cache()
                                    st.success("缓存清理完成！")
                                    st.rerun()
                                else:
                                    st.warning("缓存清理功能不可用")
                        else:
                            st.warning("缓存信息功能不可用，请检查模块导入")

                except Exception as e:
                    st.warning(f"缓存信息获取失败: {str(e)}")
            elif AKShareDataProvider:
                st.warning("缓存功能部分不可用，请检查模块导入")
            else:
                st.info("缓存功能未启用，数据提供商未正确导入")

            # 数据预览
            st.subheader("数据预览")
            st.dataframe(st.session_state.data.head(10))
        else:
            st.info("📝 请先获取数据")

def factor_management_page():
    """因子管理页面"""
    st.header("🔧 因子管理")

    if st.session_state.data.empty:
        st.warning("⚠️ 请先在数据管理页面获取数据")
        return

    tab1, tab2, tab3 = st.tabs(["预置因子库", "自定义因子", "因子计算"])

    with tab1:
        st.subheader("预置因子库")

        factor_lib = FactorLibrary()
        categories = factor_lib.get_factor_categories()

        selected_category = st.selectbox("选择因子类别", ["全部"] + categories)

        if selected_category == "全部":
            factors_info = factor_lib.list_factors()
        else:
            factors_info = factor_lib.list_factors(selected_category)

        if factors_info:
            st.write(f"共有 {len(factors_info)} 个因子:")

            for factor_name, info in factors_info.items():
                with st.expander(f"{factor_name}: {info['description']}"):
                    st.write(f"因子名称: {info['name']}")
                    st.write(f"描述: {info['description']}")

                    if st.button(f"计算 {factor_name}", key=f"calc_{factor_name}"):
                        with st.spinner(f"正在计算因子 {factor_name}..."):
                            try:
                                calculator = FactorCalculator()
                                result = calculator.calculate_single_factor(
                                    factor_name, st.session_state.data
                                )
                                if not result.empty and factor_name in result.columns:
                                    # 添加到session state
                                    if factor_name not in st.session_state.factors.columns:
                                        st.session_state.factors[factor_name] = result[factor_name]
                                    st.success(f"✅ 因子 {factor_name} 计算完成")

                                    # 显示因子统计
                                    factor_stats = calculator.get_factor_summary(result, factor_name)
                                    st.json(factor_stats)
                            except Exception as e:
                                st.error(f"❌ 计算因子失败: {str(e)}")

    with tab2:
        st.subheader("自定义因子")

        custom_manager = CustomFactorManager()

        factor_name = st.text_input("因子名称")
        factor_description = st.text_area("因子描述")
        factor_category = st.text_input("因子类别", value="custom")

        st.subheader("因子代码")
        factor_code = st.text_area(
            "输入Python代码",
            height=200,
            placeholder="""def calculate_factor(data):
    # 计算因子值
    # data: 包含股票数据的DataFrame
    # 返回: 因子值的Series

    # 示例：计算收盘价的5日移动平均
    return data.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())"""
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存因子"):
                if factor_name and factor_code:
                    try:
                        success = custom_manager.add_factor_from_code(
                            factor_name, factor_description, factor_category, factor_code
                        )
                        if success:
                            st.success(f"✅ 因子 {factor_name} 保存成功")
                        else:
                            st.error("❌ 因子保存失败")
                    except Exception as e:
                        st.error(f"❌ 保存因子时发生错误: {str(e)}")
                else:
                    st.error("❌ 请填写因子名称和代码")

        with col2:
            if st.button("📤 上传CSV文件"):
                st.file_uploader("选择CSV文件", type=['csv'])

        # 显示已保存的自定义因子
        st.subheader("已保存的自定义因子")
        custom_factors = custom_manager.list_factors()

        if custom_factors:
            for factor in custom_factors:
                with st.expander(f"{factor.name} ({factor.category})"):
                    st.write(f"描述: {factor.description}")
                    st.write(f"创建时间: {factor.created_at}")
                    st.write(f"版本: {factor.version}")
                    st.code(factor.code, language='python')

    with tab3:
        st.subheader("批量因子计算")

        calculator = FactorCalculator()
        factor_lib = FactorLibrary()
        custom_manager = CustomFactorManager()

        # 获取所有可用因子
        preset_factors = list(factor_lib.factors.keys())
        custom_factors = [f.name for f in custom_manager.list_factors() if f.enabled]
        all_factors = preset_factors + custom_factors

        selected_factors = st.multiselect("选择要计算的因子", all_factors)

        if st.button("🧮 批量计算"):
            if selected_factors:
                with st.spinner("正在批量计算因子..."):
                    try:
                        progress_bar = st.progress(0)

                        for i, factor_name in enumerate(selected_factors):
                            result = calculator.calculate_single_factor(
                                factor_name, st.session_state.data
                            )
                            if not result.empty and factor_name in result.columns:
                                st.session_state.factors[factor_name] = result[factor_name]

                            progress = (i + 1) / len(selected_factors)
                            progress_bar.progress(progress)

                        st.success(f"✅ 批量计算完成，共计算 {len(selected_factors)} 个因子")
                    except Exception as e:
                        st.error(f"❌ 批量计算失败: {str(e)}")
            else:
                st.warning("⚠️ 请选择要计算的因子")

        # 显示已计算的因子
        if not st.session_state.factors.empty:
            st.subheader("已计算的因子")
            st.dataframe(st.session_state.factors.describe())

def factor_analysis_page():
    """因子分析页面"""
    st.header("📈 因子分析")

    if st.session_state.data.empty:
        st.warning("⚠️ 请先在数据管理页面获取数据")
        return

    if st.session_state.factors.empty:
        st.warning("⚠️ 请先在因子管理页面计算因子")
        return

    # 选择要分析的因子
    available_factors = [col for col in st.session_state.factors.columns if col not in st.session_state.data.columns]
    selected_factor = st.selectbox("选择因子", available_factors)

    if selected_factor:
        # 合并数据
        analysis_data = st.session_state.data.copy()
        analysis_data[selected_factor] = st.session_state.factors[selected_factor]

        # 计算收益率
        analysis_data['return_1d'] = analysis_data.groupby('symbol')['close'].pct_change()

        tab1, tab2, tab3, tab4 = st.tabs(["IC分析", "分层收益", "换手率", "稳定性"])

        with tab1:
            st.subheader("IC分析")

            analyzer = FactorAnalyzer()

            method = st.selectbox("IC计算方法", ["spearman", "pearson"])
            rolling_window = st.slider("滚动窗口", 20, 120, 60)

            if st.button("📊 执行IC分析"):
                with st.spinner("正在执行IC分析..."):
                    try:
                        ic_results = analyzer.calculate_ic_analysis(
                            analysis_data, selected_factor, 'return_1d', method, rolling_window
                        )

                        if 'error' not in ic_results:
                            # 显示IC统计
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("IC均值", f"{ic_results['ic_mean']:.4f}")
                            col2.metric("IR值", f"{ic_results['ir']:.4f}")
                            col3.metric("IC胜率", f"{ic_results['ic_win_rate']:.2%}")
                            col4.metric("IC标准差", f"{ic_results['ic_std']:.4f}")

                            # IC序列图
                            if 'ic_series' in ic_results:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=ic_results['ic_series'].index,
                                    y=ic_results['ic_series'].values,
                                    mode='lines+markers',
                                    name='IC值'
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="red")
                                fig.update_layout(
                                    title=f"{selected_factor} IC序列",
                                    xaxis_title="日期",
                                    yaxis_title="IC值"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ IC分析失败")
                    except Exception as e:
                        st.error(f"❌ IC分析时发生错误: {str(e)}")

        with tab2:
            st.subheader("分层收益分析")

            layers = st.slider("分层数量", 3, 10, 5)

            if st.button("📊 执行分层分析"):
                with st.spinner("正在执行分层分析..."):
                    try:
                        layer_results = analyzer.calculate_layered_returns(
                            analysis_data, selected_factor, 'return_1d', layers
                        )

                        if 'error' not in layer_results:
                            # 显示分层收益统计
                            summary = layer_results['summary']

                            # 创建表格显示各层收益
                            layer_data = []
                            for i in range(1, layers + 1):
                                layer_key = f'layer_{i}'
                                if layer_key in summary:
                                    layer_data.append({
                                        '分层': f'第{i}层',
                                        '累计收益': f"{summary[layer_key]['cumulative_return']:.2%}",
                                        '年化收益': f"{summary[layer_key]['mean_return']*252:.2%}",
                                        '夏普比率': f"{summary[layer_key]['sharpe']:.2f}",
                                        '胜率': f"{summary[layer_key]['win_rate']:.2%}"
                                    })

                            df_layers = pd.DataFrame(layer_data)
                            st.dataframe(df_layers)

                            # 分层收益曲线
                            layer_df = layer_results['layer_data']

                            fig = go.Figure()
                            for i in range(1, layers + 1):
                                layer_key = f'layer_{i}'
                                if layer_key in layer_df.columns:
                                    cum_returns = (1 + layer_df[layer_key]).cumprod()
                                    fig.add_trace(go.Scatter(
                                        x=layer_df['date'],
                                        y=cum_returns,
                                        mode='lines',
                                        name=f'第{i}层'
                                    ))

                            fig.update_layout(
                                title=f"{selected_factor} 分层收益曲线",
                                xaxis_title="日期",
                                yaxis_title="累计收益"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ 分层分析失败")
                    except Exception as e:
                        st.error(f"❌ 分层分析时发生错误: {str(e)}")

        with tab3:
            st.subheader("换手率分析")

            top_percent = st.slider("头部比例", 0.05, 0.5, 0.2, 0.05)

            if st.button("📊 执行换手率分析"):
                with st.spinner("正在执行换手率分析..."):
                    try:
                        turnover_results = analyzer.calculate_turnover_analysis(
                            analysis_data, selected_factor, top_percent
                        )

                        if 'error' not in turnover_results:
                            # 显示换手率统计
                            col1, col2, col3 = st.columns(3)
                            col1.metric("平均换手率", f"{turnover_results['avg_turnover']:.2%}")
                            col2.metric("最大换手率", f"{turnover_results['max_turnover']:.2%}")
                            col3.metric("最小换手率", f"{turnover_results['min_turnover']:.2%}")

                            # 换手率时间序列
                            turnover_df = turnover_results['turnover_data']

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=turnover_df['date'],
                                y=turnover_df['turnover'],
                                mode='lines+markers',
                                name='换手率'
                            ))

                            fig.update_layout(
                                title=f"{selected_factor} 换手率时间序列",
                                xaxis_title="日期",
                                yaxis_title="换手率"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ 换手率分析失败")
                    except Exception as e:
                        st.error(f"❌ 换手率分析时发生错误: {str(e)}")

        with tab4:
            st.subheader("稳定性分析")

            window = st.slider("分析窗口", 20, 120, 60)

            if st.button("📊 执行稳定性分析"):
                with st.spinner("正在执行稳定性分析..."):
                    try:
                        stability_results = analyzer.calculate_stability_analysis(
                            analysis_data, selected_factor, window
                        )

                        if 'error' not in stability_results:
                            # 显示稳定性统计
                            stability_metrics = stability_results['stability_metrics']

                            col1, col2, col3 = st.columns(3)
                            col1.metric("均值变异系数", f"{stability_metrics['mean_cv']:.4f}")
                            col2.metric("标准差变异系数", f"{stability_metrics['std_cv']:.4f}")
                            col3.metric("极差变异系数", f"{stability_metrics['range_cv']:.4f}")

                            # 因子值分布时间序列
                            daily_stats = stability_results['daily_stats']

                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=['因子值均值', '因子值标准差'],
                                vertical_spacing=0.1
                            )

                            fig.add_trace(go.Scatter(
                                x=daily_stats['date'],
                                y=daily_stats['mean'],
                                mode='lines',
                                name='均值'
                            ), row=1, col=1)

                            fig.add_trace(go.Scatter(
                                x=daily_stats['date'],
                                y=daily_stats['std'],
                                mode='lines',
                                name='标准差'
                            ), row=2, col=1)

                            fig.update_layout(
                                title=f"{selected_factor} 稳定性分析",
                                height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ 稳定性分析失败")
                    except Exception as e:
                        st.error(f"❌ 稳定性分析时发生错误: {str(e)}")

def backtest_page():
    """策略回测页面"""
    st.header("🎯 策略回测")

    if st.session_state.data.empty:
        st.warning("⚠️ 请先在数据管理页面获取数据")
        return

    if st.session_state.factors.empty:
        st.warning("⚠️ 请先在因子管理页面计算因子")
        return

    # 回测配置
    st.subheader("回测配置")

    col1, col2 = st.columns(2)

    with col1:
        # 策略配置
        available_factors = [col for col in st.session_state.factors.columns if col not in st.session_state.data.columns]
        selected_factors = st.multiselect("选择因子", available_factors, default=available_factors[:1])

        strategy_type = st.selectbox("策略类型", ["因子策略", "等权重策略", "市值策略", "动量策略"])

        if strategy_type == "因子策略":
            top_percent = st.slider("头部比例", 0.05, 0.5, 0.2, 0.05)
            bottom_percent = st.slider("底部比例", 0.05, 0.5, 0.2, 0.05)
            long_only = st.checkbox("仅多头", value=True)
        elif strategy_type == "动量策略":
            lookback_period = st.slider("回看期", 5, 60, 20)

    with col2:
        # 回测参数
        start_date = st.date_input("回测开始日期", datetime(2021, 1, 1))
        end_date = st.date_input("回测结束日期", datetime(2023, 12, 31))

        initial_capital = st.number_input("初始资金", value=1000000, step=100000)
        commission = st.number_input("手续费率", value=0.0003, format="%.4f")
        slippage = st.number_input("滑点率", value=0.001, format="%.4f")

        rebalance_frequency = st.selectbox("调仓频率", ["月度", "周度", "日度"])

    # 执行回测
    if st.button("🚀 开始回测", type="primary"):
        if selected_factors:
            with st.spinner("正在进行回测..."):
                try:
                    # 准备数据
                    backtest_data = st.session_state.data.copy()
                    for factor_name in selected_factors:
                        backtest_data[factor_name] = st.session_state.factors[factor_name]

                    backtest_data['return_1d'] = backtest_data.groupby('symbol')['close'].pct_change()

                    # 创建策略
                    if strategy_type == "因子策略":
                        strategy = FactorStrategy(
                            name=f"因子策略_{'+'.join(selected_factors)}",
                            factor_names=selected_factors,
                            top_percent=top_percent,
                            bottom_percent=bottom_percent,
                            long_only=long_only
                        )
                    elif strategy_type == "等权重策略":
                        # 选择前20%股票
                        symbols = backtest_data['symbol'].unique()[:len(backtest_data['symbol'].unique()) // 5]
                        from backtester.strategy import EqualWeightStrategy
                        strategy = EqualWeightStrategy("等权重策略", symbols.tolist())
                    elif strategy_type == "市值策略":
                        from backtester.strategy import MarketCapStrategy
                        strategy = MarketCapStrategy("市值策略", top_percent=0.3)
                    else:  # 动量策略
                        from backtester.strategy import MomentumStrategy
                        strategy = MomentumStrategy("动量策略", lookback_period, top_percent)

                    # 执行回测
                    backtester = Backtester(
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage
                    )

                    results = backtester.run_backtest(
                        strategy,
                        backtest_data,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        rebalance_frequency.lower().replace("度", "ly")
                    )

                    if 'error' not in results:
                        st.session_state.backtest_results = results

                        # 显示回测结果
                        st.success("✅ 回测完成！")

                        # 绩效指标
                        st.subheader("📊 绩效指标")
                        metrics = results['performance_metrics']

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("总收益率", f"{metrics['total_return']:.2%}")
                        col2.metric("年化收益率", f"{metrics['annual_return']:.2%}")
                        col3.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                        col4.metric("最大回撤", f"{metrics['max_drawdown']:.2%}")

                        # 净值曲线
                        st.subheader("📈 净值曲线")
                        portfolio_values = results['detailed_data']['portfolio_values']

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=portfolio_values['date'],
                            y=portfolio_values['value'],
                            mode='lines',
                            name='策略净值',
                            line=dict(color='blue')
                        ))

                        # 添加基准
                        if 'benchmark_data' in results:
                            benchmark_data = results['benchmark_data']
                            # 对齐时间序列
                            benchmark_aligned = pd.merge(
                                portfolio_values[['date', 'value']],
                                benchmark_data,
                                on='date',
                                how='left'
                            )
                            benchmark_aligned['value_norm'] = benchmark_aligned['value_x'] / benchmark_aligned['value_x'].iloc[0]
                            benchmark_aligned['benchmark_norm'] = benchmark_aligned['value_y'] / benchmark_aligned['value_y'].iloc[0]

                            fig.add_trace(go.Scatter(
                                x=benchmark_aligned['date'],
                                y=benchmark_aligned['benchmark_norm'] * portfolio_values['value'].iloc[0],
                                mode='lines',
                                name='基准净值',
                                line=dict(color='red', dash='dash')
                            ))

                        fig.update_layout(
                            title="策略净值曲线",
                            xaxis_title="日期",
                            yaxis_title="净值"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 交易统计
                        st.subheader("📋 交易统计")
                        trading_stats = results['trading_statistics']

                        col1, col2, col3 = st.columns(3)
                        col1.metric("总交易次数", trading_stats['total_trades'])
                        col2.metric("买入次数", trading_stats['buy_trades'])
                        col3.metric("卖出次数", trading_stats['sell_trades'])

                        # 持仓分析
                        st.subheader("💼 持仓分析")
                        portfolio_stats = results['portfolio_statistics']

                        col1, col2 = st.columns(2)
                        col1.metric("平均持仓数", f"{portfolio_stats['avg_positions']:.1f}")
                        col2.metric("最大持仓数", portfolio_stats['max_positions'])

                        # 导出结果
                        if st.button("📥 导出回测结果"):
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"backtest_results_{timestamp}.xlsx"
                                success = backtester.export_results(results, filename)
                                if success:
                                    st.success(f"✅ 回测结果已导出到 {filename}")
                            except Exception as e:
                                st.error(f"❌ 导出失败: {str(e)}")
                    else:
                        st.error("❌ 回测失败")
                except Exception as e:
                    st.error(f"❌ 回测时发生错误: {str(e)}")
        else:
            st.warning("⚠️ 请选择至少一个因子")

def factor_mining_page():
    """因子挖掘页面"""
    st.header("⚡ 因子挖掘")

    if st.session_state.data.empty:
        st.warning("⚠️ 请先在数据管理页面获取数据")
        return

    # 挖掘配置
    st.subheader("挖掘配置")

    col1, col2 = st.columns(2)

    with col1:
        mining_method = st.selectbox("挖掘方法", ["组合生成", "遗传算法", "随机森林"])

        if mining_method == "组合生成":
            max_complexity = st.slider("最大复杂度", 2, 5, 3)
            max_combinations = st.slider("最大组合数", 50, 500, 200)

            include_simple = st.checkbox("包含简单因子", value=True)
            include_combination = st.checkbox("包含组合因子", value=True)
            include_cross_sectional = st.checkbox("包含截面因子", value=True)
            include_technical = st.checkbox("包含技术因子", value=True)

        elif mining_method == "遗传算法":
            population_size = st.slider("种群大小", 20, 100, 50)
            generations = st.slider("迭代代数", 10, 50, 20)
            mutation_rate = st.slider("变异率", 0.05, 0.3, 0.1)
            crossover_rate = st.slider("交叉率", 0.5, 0.9, 0.7)

    with col2:
        target_count = st.slider("目标因子数量", 5, 30, 10)
        evaluation_method = st.selectbox("评估方法", ["IC值", "夏普比率", "信息比率"])

        # 收益率数据
        st.subheader("收益率配置")
        return_period = st.selectbox("收益率周期", ["1日", "5日", "20日"], index=0)

    # 执行挖掘
    if st.button("🚀 开始因子挖掘", type="primary"):
        with st.spinner("正在进行因子挖掘..."):
            try:
                # 准备数据
                mining_data = st.session_state.data.copy()
                # 安全处理中文周期字符串
                try:
                    # 清理字符串，移除所有非数字字符
                    import re
                    period_str = re.sub(r'[^\d]', '', return_period)
                    if not period_str:
                        raise ValueError(f"无法从收益率周期中提取数字: {return_period}")
                    period_days = int(period_str)
                    st.info(f"解析收益率周期: {return_period} -> {period_days}天")
                except ValueError as e:
                    st.error(f"收益率周期解析错误: {return_period} -> {str(e)}")
                    return
                return_col = f'return_{period_days}d'
                mining_data[return_col] = mining_data.groupby('symbol')['close'].pct_change(period_days)

                if mining_method == "组合生成":
                    generator = FactorGenerator()

                    # 生成因子
                    with st.spinner("正在生成因子..."):
                        generated_factors = generator.generate_all_factors(
                            mining_data,
                            include_simple=include_simple,
                            include_combination=include_combination,
                            include_cross_sectional=include_cross_sectional,
                            include_technical=include_technical,
                            max_complexity=max_complexity,
                            max_combinations=max_combinations
                        )

                    if generated_factors:
                        st.success(f"✅ 生成了 {len(generated_factors)} 个因子")

                        # 评估因子
                        with st.spinner("正在评估因子有效性..."):
                            returns = mining_data[return_col]
                            evaluation_results = generator.evaluate_factors(
                                generated_factors, returns, evaluation_method.lower()
                            )

                        if evaluation_results:
                            # 显示结果
                            st.subheader("📊 挖掘结果")

                            # 结果表格
                            results_df = pd.DataFrame(evaluation_results, columns=['因子名称', '有效性'])
                            st.dataframe(results_df.head(target_count))

                            # 可视化
                            fig = go.Figure(data=[
                                go.Bar(x=results_df['因子名称'][:target_count],
                                     y=results_df['有效性'][:target_count])
                            ])
                            fig.update_layout(
                                title=f"前{target_count}个因子有效性排名",
                                xaxis_title="因子名称",
                                yaxis_title="有效性"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 添加优质因子到session state
                            top_factors = evaluation_results[:target_count]
                            for factor_name, score in top_factors:
                                if factor_name in generated_factors:
                                    st.session_state.factors[factor_name] = generated_factors[factor_name]

                            st.success(f"✅ 已将前{target_count}个因子添加到因子库")
                        else:
                            st.warning("⚠️ 没有找到有效的因子")
                    else:
                        st.warning("⚠️ 没有生成任何因子")

                elif mining_method == "遗传算法":
                    miner = GeneticFactorMiner(
                        population_size=population_size,
                        generations=generations,
                        mutation_rate=mutation_rate,
                        crossover_rate=crossover_rate
                    )

                    # 执行挖掘
                    with st.spinner("正在进行遗传算法挖掘..."):
                        returns = mining_data[return_col]
                        mined_factors = miner.mine_factors(mining_data, returns, target_count)

                    if mined_factors:
                        st.success(f"✅ 挖掘完成，获得 {len(mined_factors)} 个优质因子")

                        # 显示结果
                        st.subheader("📊 挖掘结果")

                        results_data = []
                        for i, factor in enumerate(mined_factors):
                            results_data.append({
                                '排名': i + 1,
                                '因子名称': factor.name,
                                '表达式': factor.expression,
                                '适应度': f"{factor.fitness:.4f}",
                                '代数': factor.generation
                            })

                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df)

                        # 可视化适应度分布
                        fig = go.Figure(data=[
                            go.Bar(x=[f.name for f in mined_factors],
                                 y=[f.fitness for f in mined_factors])
                        ])
                        fig.update_layout(
                            title="挖掘因子适应度分布",
                            xaxis_title="因子名称",
                            yaxis_title="适应度"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 添加因子到session state
                        for factor in mined_factors:
                            st.session_state.factors[factor.name] = factor.factor_values

                        st.success("✅ 已将挖掘的因子添加到因子库")
                    else:
                        st.warning("⚠️ 挖掘失败，没有找到有效因子")

            except Exception as e:
                st.error(f"❌ 因子挖掘时发生错误: {str(e)}")

def overview_page():
    """系统概览页面"""
    st.header("📋 系统概览")

    # 系统状态
    st.subheader("🔧 系统状态")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("数据状态", "✅ 已就绪" if not st.session_state.data.empty else "❌ 未加载")

    with col2:
        factor_count = len([col for col in st.session_state.factors.columns if col not in st.session_state.data.columns])
        st.metric("已计算因子", factor_count)

    with col3:
        analysis_count = len(st.session_state.analysis_results)
        st.metric("分析结果", analysis_count)

    with col4:
        backtest_count = len(st.session_state.backtest_results)
        st.metric("回测结果", backtest_count)

    # 数据概览
    if not st.session_state.data.empty:
        st.subheader("📊 数据概览")

        col1, col2 = st.columns(2)

        with col1:
            # 数据统计
            data_summary = {
                "记录数": f"{len(st.session_state.data):,}",
                "股票数": st.session_state.data['symbol'].nunique(),
                "交易日期数": st.session_state.data['date'].nunique(),
                "数据列数": len(st.session_state.data.columns)
            }

            for key, value in data_summary.items():
                st.metric(key, value)

        with col2:
            # 数据范围
            date_range = {
                "开始日期": st.session_state.data['date'].min().date(),
                "结束日期": st.session_state.data['date'].max().date(),
                "数据天数": (st.session_state.data['date'].max() - st.session_state.data['date'].min()).days
            }

            for key, value in date_range.items():
                st.metric(key, value if isinstance(value, str) else f"{value:,}")

    # 因子概览
    if not st.session_state.factors.empty:
        st.subheader("🔧 因子概览")

        factor_columns = [col for col in st.session_state.factors.columns if col not in st.session_state.data.columns]

        if factor_columns:
            # 因子统计
            factor_stats = st.session_state.factors[factor_columns].describe()
            st.dataframe(factor_stats)

    # 最近活动
    st.subheader("📈 最近活动")

    activities = []

    if not st.session_state.data.empty:
        activities.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "活动": f"加载了 {len(st.session_state.data):,} 条数据记录",
            "状态": "✅ 成功"
        })

    if factor_columns:
        activities.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "活动": f"计算了 {len(factor_columns)} 个因子",
            "状态": "✅ 成功"
        })

    if st.session_state.backtest_results:
        activities.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "活动": "完成了策略回测",
            "状态": "✅ 成功"
        })

    if activities:
        activities_df = pd.DataFrame(activities)
        st.dataframe(activities_df)

    # 使用说明
    st.subheader("📖 使用说明")

    with st.expander("🚀 快速开始"):
        st.markdown("""
        1. **数据管理**: 选择股票池和时间范围，获取市场数据
        2. **因子管理**: 计算预置因子或创建自定义因子
        3. **因子分析**: 分析因子的有效性、稳定性和收益特征
        4. **策略回测**: 基于因子构建投资策略并进行回测
        5. **因子挖掘**: 使用算法自动发现新的有效因子
        """)

    with st.expander("⚙️ 系统配置"):
        st.markdown("""
        - **数据源**: AKShare (实时市场数据)
        - **因子库**: 内置技术指标和基本面因子
        - **回测引擎**: 基于事件驱动的回测框架
        - **挖掘算法**: 遗传算法和组合生成方法
        - **可视化**: Plotly交互式图表
        """)

if __name__ == "__main__":
    main()