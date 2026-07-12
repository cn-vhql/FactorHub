import React, { useState, useRef, useEffect } from 'react'
import BacktestCharts from '../components/BacktestCharts'
import {
  Card,
  Form,
  Select,
  Input,
  InputNumber,
  Button,
  DatePicker,
  Slider,
  Radio,
  Tabs,
  Row,
  Col,
  Tag,
  message,
  Statistic,
  Divider,
  Space,
  Modal,
  List,
  Popconfirm
} from 'antd'
import {
  LineChartOutlined,
  SaveOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  FundOutlined,
  ReloadOutlined,
  FolderOpenOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'
import * as echarts from '@/utils/echarts'
import axios from 'axios'
import './Backtesting.css'

const { RangePicker } = DatePicker
const { Option } = Select
const { TextArea } = Input

// ========== 类型定义 ==========

interface BacktestConfig {
  data_mode: 'single' | 'pool'
  stock_codes: string[]
  start_date: string
  end_date: string
  factor_name?: string
  factor_names?: string[]  // 多因子选择
  strategy_type: 'single_factor' | 'multi_factor'
  initial_capital: number
  commission_rate: number
  slippage: number
  percentile: number
  direction: 'long' | 'short'
  n_quantiles: number
  weight_method?: string
  shares_per_trade: number
}

interface BacktestFormValues {
  data_mode: 'single' | 'pool'
  stock_code?: string
  stock_codes?: string
  dateRange: [Dayjs, Dayjs]
  factor_name?: string
  factor_names?: string[]
  strategy_type: 'single_factor' | 'multi_factor'
  initial_capital: number
  commission_rate: number
  slippage: number
  percentile: number
  direction: 'long' | 'short'
  weight_method?: string
  shares_per_trade: number
}

interface StrategyTemplate {
  id: number
  name: string
  description: string
  config: Omit<BacktestFormValues, 'dateRange'> & {
    dateRange?: [string, string]
  }
  created_at: string
}

interface BacktestResult {
  metrics: {
    total_return: number
    annual_return: number
    volatility: number
    sharpe_ratio: number
    max_drawdown: number
    calmar_ratio?: number
    win_rate?: number
    sortino_ratio?: number
  }
  result: {
    equity_curve?: number[]
    returns?: number[]
    trades?: any[]
  }
}

// ========== 组件 ==========

const Backtesting: React.FC = () => {
  // ========== 状态管理 ==========
  const [form] = Form.useForm<BacktestFormValues>()
  const [loading, setLoading] = useState(false)
  const [factors, setFactors] = useState<any[]>([])

  // 单策略回测
  const [singleBacktestResult, setSingleBacktestResult] = useState<BacktestResult | null>(null)
  const [chartData, setChartData] = useState<any>({})
  const equityChartRef = useRef<HTMLDivElement>(null)
  const equityChartInstance = useRef<echarts.ECharts | null>(null)
  const drawdownChartRef = useRef<HTMLDivElement>(null)
  const drawdownChartInstance = useRef<echarts.ECharts | null>(null)

  // 策略管理
  const [savedStrategies, setSavedStrategies] = useState<StrategyTemplate[]>([])
  const [strategyModalVisible, setStrategyModalVisible] = useState(false)
  const [currentStrategyName, setCurrentStrategyName] = useState('')
  const [activeTab, setActiveTab] = useState<string>('single')

  // ========== 生命周期 ==========
  useEffect(() => {
    loadFactors()
    loadSavedStrategies()

    // 设置默认日期范围
    form.setFieldsValue({
      dateRange: [dayjs().subtract(1, 'year'), dayjs()]
    })

    return () => {
      // 清理图表实例
      if (equityChartInstance.current) {
        equityChartInstance.current.dispose()
      }
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.dispose()
      }
    }
  }, [])

  // ========== 数据加载 ==========
  const loadFactors = async () => {
    try {
      const response = await axios.get('/api/factors')
      if (response.data.success) {
        setFactors(response.data.data || [])
      }
    } catch (error) {
      console.error('加载因子失败:', error)
    }
  }

  const loadSavedStrategies = () => {
    const stored = localStorage.getItem('backtest_strategies')
    if (stored) {
      try {
        setSavedStrategies(JSON.parse(stored))
      } catch (error) {
        console.error('加载策略失败:', error)
        setSavedStrategies([])
      }
    }
  }

  // ========== 单策略回测 ==========
  const runSingleBacktest = async (values: BacktestFormValues) => {
    const [startDate, endDate] = values.dateRange

    // 处理因子选择：单因子或多因子
    const isMultiFactor = values.strategy_type === 'multi_factor'
    let factorName: string | undefined
    let factorNames: string[] | undefined

    if (isMultiFactor) {
      // 多因子策略
      factorNames = values.factor_names || []
      factorName = factorNames[0]  // 使用第一个因子作为主因子（用于向后兼容）
    } else {
      // 单因子策略
      factorName = values.factor_name
      factorNames = factorName ? [factorName] : []
    }

    const config: BacktestConfig = {
      data_mode: values.data_mode,
      stock_codes: values.data_mode === 'single'
        ? (values.stock_code ? [values.stock_code] : [])
        : (values.stock_codes?.split('\n').filter((s: string) => s.trim()) ?? []),
      start_date: startDate.format('YYYY-MM-DD'),
      end_date: endDate.format('YYYY-MM-DD'),
      factor_name: factorName,
      factor_names: factorNames,
      strategy_type: values.strategy_type,
      initial_capital: values.initial_capital,
      commission_rate: values.commission_rate / 100,
      slippage: values.slippage / 100,
      percentile: values.percentile,
      direction: values.direction,
      n_quantiles: 5,
      weight_method: values.weight_method || 'equal_weight',
      shares_per_trade: (values.shares_per_trade || 1) * 100  // 手数转股数（1手=100股）
    }

    try {
      setLoading(true)
      // 清空旧的图表数据，确保新图表完全重新加载
      setChartData({})
      setSingleBacktestResult(null)

      const response = await axios.post('/api/backtest/single', config)

      if (response.data.success) {
        setSingleBacktestResult(response.data.data)

        // 设置图表数据
        if (response.data.data.chart_data) {
          const chartDataForStocks: any = {}

          // 检查是否是单股票模式返回的结构
          if (response.data.data.chart_data.kline) {
            // 单股票模式：直接使用返回的数据
            const firstStockCode = config.stock_codes[0]
            chartDataForStocks[firstStockCode] = response.data.data.chart_data
          } else {
            // 多股票模式：数据结构已经是 {stock_code: chart_data}
            Object.assign(chartDataForStocks, response.data.data.chart_data)
          }

          setChartData(chartDataForStocks)
        }

        message.success('回测完成')

        // 延迟渲染旧图表（保留向后兼容）
        setTimeout(() => {
          renderEquityChart(response.data.data.result.equity_curve || response.data.data.result.returns || [])
          renderDrawdownChart(response.data.data.result.equity_curve || response.data.data.result.returns || [])
        }, 300)
      } else {
        message.error(response.data.message || '回测失败')
      }
    } catch (error: any) {
      console.error('回测失败:', error)
      message.error('回测失败: ' + (error.message || '未知错误'))
    } finally {
      setLoading(false)
    }
  }

  // ========== 图表渲染 ==========
  const renderEquityChart = (data: number[]) => {
    if (!equityChartRef.current) return

    let chart = equityChartInstance.current
    if (!chart) {
      chart = echarts.init(equityChartRef.current)
      equityChartInstance.current = chart
    }

    chart.clear()

    const option = {
      title: {
        text: '净值曲线',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: data.map((_, i) => i)
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      series: [{
        data: data,
        type: 'line',
        smooth: true,
        showSymbol: false,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
            ]
          }
        },
        itemStyle: {
          color: '#3b82f6'
        }
      }]
    }

    chart.setOption(option)
  }

  const renderDrawdownChart = (data: number[]) => {
    if (!drawdownChartRef.current) return

    let chart = drawdownChartInstance.current
    if (!chart) {
      chart = echarts.init(drawdownChartRef.current)
      drawdownChartInstance.current = chart
    }

    chart.clear()

    // 计算回撤
    let max = -Infinity
    const drawdowns = data.map(v => {
      max = Math.max(max, v)
      return ((v - max) / max) * 100
    })

    const option = {
      title: {
        text: '回撤曲线',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          return `${params[0].name}: ${params[0].value.toFixed(2)}%`
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: drawdowns.map((_, i) => i)
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [{
        data: drawdowns,
        type: 'line',
        smooth: true,
        showSymbol: false,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(239, 68, 68, 0.3)' },
              { offset: 1, color: 'rgba(239, 68, 68, 0.05)' }
            ]
          }
        },
        itemStyle: {
          color: '#ef4444'
        }
      }]
    }

    chart.setOption(option)
  }

  // ========== 策略管理 ==========
  const saveStrategy = async () => {
    try {
      const values = await form.validateFields()

      // 将 Dayjs 对象转换为 ISO 字符串以便存储
      const configToSave = {
        ...values
      } as Omit<StrategyTemplate['config'], 'dateRange'> & {
        dateRange?: BacktestFormValues['dateRange'] | [string, string]
      }
      if (configToSave.dateRange && Array.isArray(configToSave.dateRange)) {
        configToSave.dateRange = configToSave.dateRange.map((d: any) =>
          dayjs.isDayjs(d) ? d.toISOString() : String(d)
        ) as [string, string]
      }

      const strategy: StrategyTemplate = {
        id: Date.now(),
        name: currentStrategyName || `策略_${dayjs().format('YYYYMMDD_HHmmss')}`,
        description: `${values.strategy_type === 'single_factor' ? '单因子' : '多因子'}策略`,
        config: configToSave as StrategyTemplate['config'],
        created_at: new Date().toISOString()
      }

      const updatedStrategies = [...savedStrategies, strategy]
      setSavedStrategies(updatedStrategies)

      // 保存到本地存储
      localStorage.setItem('backtest_strategies', JSON.stringify(updatedStrategies))

      message.success('策略已保存')
      setStrategyModalVisible(false)
      setCurrentStrategyName('')
    } catch (error) {
      console.error('保存策略失败:', error)
      message.error('保存策略失败，请检查表单填写')
    }
  }

  const loadStrategy = (strategy: StrategyTemplate) => {
    try {
      console.log('[loadStrategy] 开始加载策略:', strategy.name)
      console.log('[loadStrategy] 策略配置:', strategy.config)

      // 转换日期字段为 Dayjs 对象
      const config = { ...strategy.config }

      // 处理 dateRange 字段（从 localStorage 读取的需要转换）
      if (config.dateRange) {
        if (Array.isArray(config.dateRange) && config.dateRange.length === 2) {
          console.log('[loadStrategy] 转换日期字段:', config.dateRange)
          const [savedStartDate, savedEndDate] = config.dateRange
          const formConfig = {
            ...config,
            dateRange: [dayjs(savedStartDate), dayjs(savedEndDate)]
          }
          console.log('[loadStrategy] 转换后的日期:', formConfig.dateRange)
          console.log('[loadStrategy] 设置表单值:', formConfig)
          form.setFieldsValue(formConfig)
        } else {
          form.setFieldsValue({
            ...config,
            dateRange: [dayjs().subtract(1, 'year'), dayjs()]
          })
        }
      } else {
        // 如果没有 dateRange，设置默认值
        console.log('[loadStrategy] 策略没有 dateRange，设置默认值')
        form.setFieldsValue({
          ...config,
          dateRange: [dayjs().subtract(1, 'year'), dayjs()]
        })
      }

      console.log('[loadStrategy] 切换到单策略回测页面')
      setActiveTab('single')  // 自动切换到单策略回测页面

      console.log('[loadStrategy] 策略加载成功')
      message.success(`已加载策略: ${strategy.name}`)
    } catch (error) {
      console.error('[loadStrategy] 加载策略失败:', error)
      message.error(`加载策略失败: ${error instanceof Error ? error.message : '未知错误'}`)
    }
  }

  const deleteStrategy = (id: number) => {
    const updatedStrategies = savedStrategies.filter(s => s.id !== id)
    setSavedStrategies(updatedStrategies)
    localStorage.setItem('backtest_strategies', JSON.stringify(updatedStrategies))
    message.success('策略已删除')
  }

  const clearAllStrategies = () => {
    setSavedStrategies([])
    localStorage.removeItem('backtest_strategies')
    message.success('已清空所有策略')
  }

  // ========== 渲染 ==========
  return (
    <div className="backtesting-container">
      {/* 背景效果 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      {/* 主内容区域 */}
      <div className="backtesting-content">
        {/* 页面头部 */}
        <div className="page-header">
          <div className="header-content">
            <FundOutlined className="header-icon" />
            <div>
              <h1 className="page-title">策略回测</h1>
              <p className="page-subtitle">基于因子的量化策略回测与性能分析</p>
            </div>
          </div>
        </div>

        {/* 主卡片 */}
        <Card className="main-card">
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
            {
              key: 'single',
              label: (
                <span>
                  <LineChartOutlined />
                  单策略回测
                </span>
              ),
              children: (
                <div>
                  <Row gutter={[24, 24]}>
                    {/* 左侧配置面板 */}
                    <Col xs={24} lg={8}>
                      <Card title="回测配置" className="config-card">
                        <Form
                          form={form}
                          layout="vertical"
                          onFinish={runSingleBacktest}
                          initialValues={{
                            data_mode: 'single',
                            stock_code: '000001',
                            strategy_type: 'single_factor',
                            initial_capital: 1000000,
                            commission_rate: 0.03,
                            slippage: 0,
                            percentile: 50,
                            direction: 'long',
                            shares_per_trade: 1
                          }}
                        >
                          {/* 数据配置 */}
                          <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                            数据配置
                          </Divider>

                          <Form.Item label="数据模式" name="data_mode">
                            <Select size="large">
                              <Option value="single">单股票</Option>
                              <Option value="pool">股票池</Option>
                            </Select>
                          </Form.Item>

                          <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => {
                            return prevValues?.data_mode !== currentValues?.data_mode
                          }}>
                            {({ getFieldValue }) => {
                              const isSingle = getFieldValue('data_mode') === 'single'
                              return isSingle ? (
                                <Form.Item label="股票代码" name="stock_code" rules={[{ required: true }]}>
                                  <Input placeholder="例如: 000001" />
                                </Form.Item>
                              ) : (
                                <Form.Item
                                  label="股票代码列表"
                                  name="stock_codes"
                                  rules={[{ required: true }]}
                                >
                                  <TextArea
                                    rows={3}
                                    placeholder="每行一个股票代码&#10;000001&#10;600000"
                                  />
                                </Form.Item>
                              )
                            }}
                          </Form.Item>

                          <Form.Item label="日期范围" name="dateRange" rules={[{ required: true }]}>
                            <RangePicker style={{ width: '100%' }} />
                          </Form.Item>

                          <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                            因子配置
                          </Divider>

                          <Form.Item label="策略类型" name="strategy_type">
                            <Select size="large">
                              <Option value="single_factor">单因子策略</Option>
                              <Option value="multi_factor">多因子策略</Option>
                            </Select>
                          </Form.Item>

                          <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => {
                            return prevValues?.strategy_type !== currentValues?.strategy_type
                          }}>
                            {({ getFieldValue }) => {
                              const isMultiFactor = getFieldValue('strategy_type') === 'multi_factor'

                              return (
                                <>
                                  <Form.Item
                                    label={isMultiFactor ? "选择因子（可多选）" : "选择因子"}
                                    name={isMultiFactor ? "factor_names" : "factor_name"}
                                    rules={[{ required: true, message: '请选择因子' }]}
                                  >
                                    <Select
                                      showSearch
                                      placeholder={isMultiFactor ? "输入因子名称搜索" : "输入因子名称搜索"}
                                      optionFilterProp="label"
                                      mode={isMultiFactor ? "multiple" : undefined}
                                      maxTagCount="responsive"
                                      size="large"
                                      filterOption={(input, option) => {
                                        const label = String(option?.label ?? '')
                                        const value = String(option?.value ?? '')
                                        return (
                                          label.toLowerCase().includes(input.toLowerCase()) ||
                                          value.toLowerCase().includes(input.toLowerCase())
                                        )
                                      }}
                                      optionLabelProp="label"
                                    >
                                      {factors.map((factor) => (
                                        <Option
                                          key={factor.id}
                                          value={factor.name}
                                          label={factor.name}
                                        >
                                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                              <span style={{ fontWeight: 500 }}>{factor.name}</span>
                                              <Tag color={factor.source === 'preset' ? 'success' : 'warning'}>
                                                {factor.source === 'preset' ? '预置' : '自定义'}
                                              </Tag>
                                              <Tag color="blue">{factor.category}</Tag>
                                            </div>
                                            <div style={{ fontSize: 12, color: '#64748b', fontFamily: 'monospace' }}>
                                              {factor.code}
                                            </div>
                                            {factor.description && (
                                              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                                                {factor.description}
                                              </div>
                                            )}
                                          </div>
                                        </Option>
                                      ))}
                                    </Select>
                                  </Form.Item>

                                  {/* 因子选择提示和操作按钮 */}
                                  {isMultiFactor && (
                                    <Form.Item noStyle shouldUpdate>
                                      {() => {
                                        const selectedCount = form.getFieldValue('factor_names')?.length || 0
                                        return (
                                          <div style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            marginBottom: 16
                                          }}>
                                            <span className="text-hint">
                                              已选择 <strong style={{ color: '#3b82f6' }}>{selectedCount}</strong> 个因子
                                            </span>
                                            <Space size="small">
                                              <Button
                                                type="link"
                                                size="small"
                                                                onClick={() => {
                                                                  form.setFieldsValue({
                                                                    factor_names: factors.map((f) => f.name)
                                                                  })
                                                                }}
                                                              >
                                                全选
                                              </Button>
                                              <Button
                                                type="link"
                                                size="small"
                                                                onClick={() => {
                                                                  form.setFieldsValue({
                                                                    factor_names: []
                                                                  })
                                                                }}
                                                              >
                                                清空
                                              </Button>
                                            </Space>
                                          </div>
                                        )
                                      }}
                                    </Form.Item>
                                  )}
                                </>
                              )
                            }}
                          </Form.Item>

                          <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                            回测参数
                          </Divider>

                          <Form.Item label="初始资金" name="initial_capital">
                            <InputNumber<number>
                              style={{ width: '100%' }}
                              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                              parser={(value) => Number((value ?? '').replace(/\$\s?|(,*)/g, '')) || 0}
                              min={0}
                              step={10000}
                            />
                          </Form.Item>

                          <Row gutter={16}>
                            <Col span={8}>
                              <Form.Item label="费率(%)" name="commission_rate">
                                <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item label="滑点(%)" name="slippage">
                                <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={8}>
                              <Form.Item label="每次手数" name="shares_per_trade">
                                <InputNumber
                                  min={1}
                                  max={100}
                                  step={1}
                                  style={{ width: '100%' }}
                                  addonAfter="手"
                                />
                              </Form.Item>
                            </Col>
                          </Row>

                          <Form.Item label="分位数" name="percentile">
                            <Slider marks={{ 10: '10%', 50: '50%', 90: '90%' }} />
                          </Form.Item>

                          <Form.Item label="交易方向" name="direction">
                            <Radio.Group>
                              <Radio value="long">做多</Radio>
                              <Radio value="short">做空</Radio>
                            </Radio.Group>
                          </Form.Item>

                          <Divider />

                          {/* 操作按钮 */}
                          <Space direction="vertical" style={{ width: '100%' }} size="middle">
                            <Form.Item>
                              <Button
                                type="primary"
                                htmlType="submit"
                                icon={<PlayCircleOutlined />}
                                loading={loading}
                                block
                                size="large"
                              >
                                运行回测
                              </Button>
                            </Form.Item>

                            <Button
                              icon={<SaveOutlined />}
                              onClick={() => {
                                setCurrentStrategyName('')
                                setStrategyModalVisible(true)
                              }}
                              block
                            >
                              保存当前配置
                            </Button>
                          </Space>
                        </Form>
                      </Card>
                    </Col>

                    {/* 右侧结果展示 */}
                    <Col xs={24} lg={16}>
                      <Card title="回测结果" className="result-card">
                        {!singleBacktestResult && (
                          <div className="placeholder-content">
                            <LineChartOutlined className="placeholder-icon" />
                            <p className="placeholder-text">配置回测参数后点击"运行回测"按钮</p>
                            <p className="placeholder-hint">支持单因子和多因子策略回测</p>
                          </div>
                        )}

                        {singleBacktestResult && (
                          <div className="backtest-result">
                            {/* 性能指标 */}
                            <div className="metrics-section">
                              <Row gutter={16}>
                                <Col span={6}>
                                  <Statistic
                                    title="累计收益率"
                                    value={(singleBacktestResult.metrics.total_return * 100).toFixed(2)}
                                    suffix="%"
                                    valueStyle={{
                                      color: singleBacktestResult.metrics.total_return > 0 ? '#ef4444' : '#10b981',
                                      fontWeight: 700
                                    }}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="年化收益率"
                                    value={(singleBacktestResult.metrics.annual_return * 100).toFixed(2)}
                                    suffix="%"
                                    valueStyle={{
                                      color: singleBacktestResult.metrics.annual_return > 0 ? '#ef4444' : '#10b981',
                                      fontWeight: 700
                                    }}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="夏普比率"
                                    value={singleBacktestResult.metrics.sharpe_ratio.toFixed(4)}
                                    valueStyle={{ fontWeight: 700 }}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="最大回撤"
                                    value={(singleBacktestResult.metrics.max_drawdown * 100).toFixed(2)}
                                    suffix="%"
                                    valueStyle={{ color: '#ef4444', fontWeight: 700 }}
                                  />
                                </Col>
                              </Row>
                            </div>

                            {/* 新的多图表联动系统 */}
                            {chartData && Object.keys(chartData).length > 0 ? (
                              <div>
                                <Divider />
                                <div className="chart-section" style={{ paddingBottom: '24px' }}>
                                  <h4 className="chart-title">
                                    <LineChartOutlined style={{ marginRight: 8 }} />
                                    综合分析图表
                                  </h4>
                                  <BacktestCharts data={chartData} loading={loading} />
                                </div>
                              </div>
                            ) : (
                              <>
                                {/* 旧图表（向后兼容） */}
                                <div className="chart-section">
                                  <h4 className="chart-title">
                                    <LineChartOutlined style={{ marginRight: 8 }} />
                                    净值曲线
                                  </h4>
                                  <div ref={equityChartRef} className="chart-container" style={{ height: '350px' }}></div>
                                </div>

                                <div className="chart-section">
                                  <h4 className="chart-title">
                                    <LineChartOutlined style={{ marginRight: 8 }} />
                                    回撤曲线
                                  </h4>
                                  <div ref={drawdownChartRef} className="chart-container" style={{ height: '300px' }}></div>
                                </div>
                              </>
                            )}
                          </div>
                        )}
                      </Card>
                    </Col>
                  </Row>
                </div>
              )
            },
            {
              key: 'saved',
              label: (
                <span>
                  <FolderOpenOutlined />
                  已保存的策略
                </span>
              ),
              children: (
                <div>
                  <Row gutter={[24, 24]}>
                    <Col xs={24} lg={24}>
                      <Card
                        title="策略管理"
                        className="result-card"
                        extra={
                          <Space>
                            <Button
                              icon={<ReloadOutlined />}
                              onClick={() => {
                                loadSavedStrategies()
                                message.success('策略列表已刷新')
                              }}
                            >
                              刷新列表
                            </Button>
                            {savedStrategies.length > 0 && (
                              <Popconfirm
                                title="确定清空所有策略？"
                                description="此操作将删除所有已保存的策略，无法恢复"
                                onConfirm={clearAllStrategies}
                              >
                                <Button danger icon={<DeleteOutlined />}>
                                  清空全部
                                </Button>
                              </Popconfirm>
                            )}
                          </Space>
                        }
                      >
                        {savedStrategies.length === 0 ? (
                          <div className="placeholder-content">
                            <FolderOpenOutlined className="placeholder-icon" />
                            <p className="placeholder-text">暂无保存的策略</p>
                            <p className="placeholder-hint">
                              在"单策略回测"页面配置完成后，点击"保存当前配置"按钮即可保存策略
                            </p>
                          </div>
                        ) : (
                          <List
                            dataSource={savedStrategies}
                            renderItem={(strategy) => (
                              <List.Item
                                style={{
                                  border: '1px solid #e5e7eb',
                                  borderRadius: '8px',
                                  marginBottom: '12px',
                                  padding: '16px',
                                  background: '#fafafa'
                                }}
                                actions={[
                                  <Button
                                    type="primary"
                                    size="small"
                                    icon={<FolderOpenOutlined />}
                                    onClick={() => loadStrategy(strategy)}
                                  >
                                    加载
                                  </Button>,
                                  <Popconfirm
                                    title="确定删除此策略？"
                                    onConfirm={() => deleteStrategy(strategy.id)}
                                  >
                                    <Button danger size="small" icon={<DeleteOutlined />}>
                                      删除
                                    </Button>
                                  </Popconfirm>
                                ]}
                              >
                                <List.Item.Meta
                                  title={
                                    <div>
                                      <span style={{ fontWeight: 500 }}>{strategy.name}</span>
                                      <Tag
                                        color={strategy.config.strategy_type === 'single_factor' ? 'blue' : 'green'}
                                        style={{ marginLeft: '8px' }}
                                      >
                                        {strategy.config.strategy_type === 'single_factor' ? '单因子' : '多因子'}
                                      </Tag>
                                    </div>
                                  }
                                  description={
                                    <div>
                                      <div style={{ marginBottom: '8px' }}>
                                        {strategy.description}
                                      </div>
                                      <div style={{ fontSize: '12px', color: '#64748b' }}>
                                        <Space size="large">
                                          <span>
                                            数据: {strategy.config.data_mode === 'single' ? '单股票' : '股票池'}
                                          </span>
                                          <span>•</span>
                                          <span>
                                            因子: {strategy.config.factor_name || strategy.config.factor_names?.join(', ') || '-'}
                                          </span>
                                          <span>•</span>
                                          <span>
                                            {strategy.config.direction === 'long' ? '做多' : '做空'}
                                          </span>
                                        </Space>
                                      </div>
                                      <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                                        创建于: {dayjs(strategy.created_at).format('YYYY-MM-DD HH:mm:ss')}
                                      </div>
                                    </div>
                                  }
                                />
                              </List.Item>
                            )}
                          />
                        )}
                      </Card>
                    </Col>
                  </Row>
                </div>
              )
            }
          ]}
        />
        </Card>
      </div>

      {/* 保存策略对话框 */}
      <Modal
        title="保存策略配置"
        open={strategyModalVisible}
        onOk={saveStrategy}
        onCancel={() => {
          setStrategyModalVisible(false)
          setCurrentStrategyName('')
        }}
      >
        <Form.Item label="策略名称">
          <Input
            placeholder="输入策略名称，例如：动量20_做多_50分位"
            value={currentStrategyName}
            onChange={(e) => setCurrentStrategyName(e.target.value)}
            maxLength={50}
          />
          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '8px' }}>
            策略名称将显示在"已保存的策略"页面中
          </p>
        </Form.Item>
      </Modal>
    </div>
  )
}

export default Backtesting
