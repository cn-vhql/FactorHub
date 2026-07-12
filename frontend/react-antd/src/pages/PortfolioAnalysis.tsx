import { useState, useEffect, useRef } from 'react'
import {
  Card,
  Form,
  DatePicker,
  Button,
  Select,
  Input,
  InputNumber,
  Row,
  Col,
  message,
  Space,
  Divider,
  Tag,
  Tabs,
  Table,
  Statistic,
  Spin
} from 'antd'
import {
  PieChartOutlined,
  RocketOutlined,
  BarChartOutlined,
  LineChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'
import { AimOutlined, BulbOutlined } from '@ant-design/icons'
import * as echarts from '@/utils/echarts'
import {
  buildCompositeFactorDescription,
  extractErrorMessage,
  FACTOR_COPY,
  formatActionFailure,
  formatActionSuccess
} from '@/utils/factorCopy'
import axios from 'axios'
import dayjs from 'dayjs'
import './PortfolioAnalysis.css'

const { Option } = Select
const { RangePicker } = DatePicker

interface Factor {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
}

interface OptimizationResult {
  weights: Record<string, number>
  method: string
  stock_code?: string
  factors: string[]
  metrics: {
    return: number
    ic: number
    ir: number
  }
  weights_history?: Array<{
    iteration: number
    weights: Record<string, number>
    sharpe_ratio: number
  }>
  composite_score?: {
    dates: string[]
    values: number[]
  }
  composite_stats?: {
    mean: number
    std: number
    min: number
    max: number
  }
}

const METHOD_LABELS: Record<string, string> = {
  equal_weight: '等权重',
  ic_weight: 'IC加权',
  ir_weight: 'IR加权',
  max_sharpe: '最大夏普',
  max_return: '最大收益',
  min_variance: '最小方差'
}

const stripDfPrefixes = (code: string) => code
  .replace(/df\[['"]open['"]\]/g, 'open')
  .replace(/df\[['"]high['"]\]/g, 'high')
  .replace(/df\[['"]low['"]\]/g, 'low')
  .replace(/df\[['"]close['"]\]/g, 'close')
  .replace(/df\[['"]volume['"]\]/g, 'volume')
  .replace(/df\[['"]amount['"]\]/g, 'amount')

const looksLikeMyLanguageProgram = (code: string) =>
  code.includes(':=') || /(^|[;\n])\s*[A-Za-z_][A-Za-z0-9_]*\s*:/.test(code)

const replaceVariable = (expression: string, variableName: string, replacement: string) =>
  expression.replace(
    new RegExp(`(?<![A-Za-z0-9_])${variableName}(?![A-Za-z0-9_])`, 'g'),
    `(${replacement})`
  )

const normalizeMyLanguageExpression = (
  expression: string,
  assignments: Record<string, string>
) => {
  let normalized = expression.trim()
  let changed = true
  let safetyCounter = 0

  while (changed && safetyCounter < 20) {
    changed = false
    safetyCounter += 1
    Object.entries(assignments).forEach(([name, value]) => {
      const replaced = replaceVariable(normalized, name, value)
      if (replaced !== normalized) {
        normalized = replaced
        changed = true
      }
    })
  }

  return normalized
}

const extractMyLanguagePrimaryExpression = (code: string): string | null => {
  if (!looksLikeMyLanguageProgram(code)) {
    return null
  }

  const statements = code
    .split(/[;\n]+/)
    .map((item) => item.trim())
    .filter(Boolean)

  const assignments: Record<string, string> = {}
  const outputs: Array<{ name: string; expression: string }> = []
  let lastExpression: string | null = null

  statements.forEach((statement) => {
    const match = statement.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*(:=|:)\s*(.+)$/)
    if (!match) {
      lastExpression = normalizeMyLanguageExpression(statement, assignments)
      return
    }

    const [, rawName, operator, rawExpression] = match
    const normalizedExpression = normalizeMyLanguageExpression(rawExpression, assignments)
    assignments[rawName] = normalizedExpression
    assignments[rawName.toUpperCase()] = normalizedExpression
    lastExpression = normalizedExpression

    if (operator === ':') {
      outputs.push({ name: rawName, expression: normalizedExpression })
    }
  })

  const xgOutput = outputs.find((item) => item.name.toUpperCase() === 'XG')
  if (xgOutput) {
    return xgOutput.expression
  }
  if (outputs.length > 0) {
    return outputs[outputs.length - 1].expression
  }
  return lastExpression
}

const extractFactorExpression = (code: string): string | null => {
  const trimmed = code.trim()
  if (!trimmed) {
    return null
  }

  if (!trimmed.startsWith('def')) {
    const myLanguageExpression = extractMyLanguagePrimaryExpression(trimmed)
    if (myLanguageExpression) {
      return stripDfPrefixes(myLanguageExpression)
    }
    return stripDfPrefixes(trimmed)
  }

  const expressionMatch = trimmed.match(/^\s*表达式:\s*(.+?)\s*$/m)
  if (expressionMatch?.[1]) {
    return stripDfPrefixes(expressionMatch[1].trim())
  }

  const returnLine = trimmed.split('\n').find(line => line.trim().startsWith('return '))
  if (!returnLine) {
    return null
  }

  const returnExpression = returnLine.trim().replace(/^return\s+/, '').trim()
  if (!returnExpression || /^[A-Za-z_][A-Za-z0-9_]*$/.test(returnExpression)) {
    return null
  }

  return stripDfPrefixes(returnExpression)
}

const buildExpandingZScoreExpression = (expression: string) => {
  const wrapped = `(${expression})`
  return `((${wrapped} - ${wrapped}.expanding(min_periods=1).mean()) / (${wrapped}.expanding(min_periods=1).std() + 1e-8))`
}

const PortfolioAnalysis: React.FC = () => {
  const [optimizeForm] = Form.useForm()
  const [compareForm] = Form.useForm()
  const weightChartRef = useRef<HTMLDivElement>(null)
  const convergenceChartRef = useRef<HTMLDivElement>(null)
  const weightChartInstanceRef = useRef<echarts.ECharts | null>(null)
  const convergenceChartInstanceRef = useRef<echarts.ECharts | null>(null)

  // 综合得分相关ref
  const compositeScoreChartRef = useRef<HTMLDivElement>(null)
  const compositeDistChartRef = useRef<HTMLDivElement>(null)
  const compositeScoreChartInstanceRef = useRef<echarts.ECharts | null>(null)
  const compositeDistChartInstanceRef = useRef<echarts.ECharts | null>(null)

  // 方法对比相关ref
  const compareChartRef = useRef<HTMLDivElement>(null)
  const compareChartInstanceRef = useRef<echarts.ECharts | null>(null)

  const [factors, setFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null)
  const [activeTab, setActiveTab] = useState('weights')
  const [compositeFactorCode, setCompositeFactorCode] = useState<string>('')
  const [savingFactor, setSavingFactor] = useState(false)

  // 方法对比状态
  const [compareResult, setCompareResult] = useState<any>(null)

  // 加载因子列表
  const loadFactors = async () => {
    try {
      const response = await axios.get('/api/factors')
      if (response.data.success) {
        setFactors(response.data.data)
      }
    } catch (error) {
      console.error('加载因子列表失败:', error)
      message.error(formatActionFailure('加载因子列表', extractErrorMessage(error)))
    }
  }

  useEffect(() => {
    loadFactors()

    // 设置默认值
    const endDate = dayjs()
    const startDate = dayjs().subtract(1, 'year')
    optimizeForm.setFieldsValue({
      dateRange: [startDate, endDate],
      method: 'max_sharpe',
      rebalance_frequency: 'monthly',
      risk_free_rate: 0.03,
      min_weight: 0,
      max_weight: 1,
      target_return: undefined
    })

    return () => {
      // 清理所有图表
      const allCharts = [
        weightChartInstanceRef,
        convergenceChartInstanceRef,
        compositeScoreChartInstanceRef,
        compositeDistChartInstanceRef,
        compareChartInstanceRef
      ]
      allCharts.forEach(chartRef => {
        if (chartRef.current) {
          chartRef.current.dispose()
          chartRef.current = null
        }
      })
    }
  }, [])

  // 开始权重优化
  const startOptimization = async (values: any) => {
    const selectedFactors = values.factors || []

    if (selectedFactors.length < 2) {
      message.warning('请至少选择2个因子进行组合优化')
      return
    }

    const [startDate, endDate] = values.dateRange
    const requestData = {
      stock_code: values.stock_code || '000001', // 添加必需的 stock_code 字段
      factors: selectedFactors,
      start_date: startDate.format('YYYY-MM-DD'),
      end_date: endDate.format('YYYY-MM-DD'),
      method: values.method,
      rebalance_freq: values.rebalance_frequency // 使用正确的字段名
    }

    try {
      setLoading(true)
      setOptimizing(true)

      // 清空结果前先销毁图表实例
      if (weightChartInstanceRef.current) {
        weightChartInstanceRef.current.dispose()
        weightChartInstanceRef.current = null
      }
      if (convergenceChartInstanceRef.current) {
        convergenceChartInstanceRef.current.dispose()
        convergenceChartInstanceRef.current = null
      }

      setOptimizationResult(null)

      const response = await axios.post('/api/portfolio/optimize-weights', requestData)

      if (response.data.success) {
        const resultData = response.data.data
        setOptimizationResult(resultData)
        message.success(formatActionSuccess(FACTOR_COPY.portfolio.optimizeAction))

        // 立即生成组合因子代码（使用返回的数据而不是状态）
        console.log('[runOptimization] 开始生成组合因子代码')
        console.log('[runOptimization] 优化结果:', resultData)
        console.log('[runOptimization] 可用因子列表:', factors)

        const code = generateCompositeFactorCode(resultData)
        console.log('[runOptimization] 生成的代码长度:', code.length)
        console.log('[runOptimization] 生成的代码预览:', code.substring(0, 200))

        setCompositeFactorCode(code)

        // 延迟渲染图表，等待DOM更新完成
        setTimeout(() => {
          console.log('[runOptimization] 开始更新图表')
          updateCharts(resultData)

          // 如果有综合得分数据，也更新综合得分图表
          if (resultData.composite_score) {
            updateCompositeScoreChart(resultData.composite_score)
            if (resultData.composite_score.values) {
              updateCompositeDistChart(resultData.composite_score.values)
            }
          }
        }, 300)
      } else {
        message.error(
          formatActionFailure(
            FACTOR_COPY.portfolio.optimizeAction,
            response.data.message || '未知错误'
          )
        )
      }
    } catch (error: any) {
      console.error('权重优化失败:', error)
      message.error(
        formatActionFailure(
          FACTOR_COPY.portfolio.optimizeAction,
          extractErrorMessage(error)
        )
      )
    } finally {
      setLoading(false)
      setOptimizing(false)
    }
  }

  // 更新图表
  const updateCharts = (result: OptimizationResult) => {
    updateWeightChart(result.weights || {})
    if (result.weights_history && result.weights_history.length > 0) {
      updateConvergenceChart(result.weights_history)
    }
  }

  // 更新权重饼图
  const updateWeightChart = (weights: Record<string, number>) => {
    console.log('[updateWeightChart] 开始更新权重图表', weights)

    if (!weightChartRef.current) {
      console.error('[updateWeightChart] weightChartRef.current 不存在')
      return
    }

    // 检查容器尺寸
    const rect = weightChartRef.current.getBoundingClientRect()
    console.log('[updateWeightChart] 容器尺寸:', rect.width, 'x', rect.height)

    if (rect.width === 0 || rect.height === 0) {
      console.error('[updateWeightChart] 容器尺寸为0，无法渲染图表')
      return
    }

    let chart = weightChartInstanceRef.current
    if (!chart) {
      console.log('[updateWeightChart] 创建新的图表实例')
      chart = echarts.init(weightChartRef.current)
      weightChartInstanceRef.current = chart
    } else {
      console.log('[updateWeightChart] 使用已有图表实例')
    }

    // 清空图表数据
    chart.clear()

    const data = Object.entries(weights).map(([name, value]) => ({
      name,
      value: (value * 100).toFixed(2)
    }))

    console.log('[updateWeightChart] 图表数据:', data)

    const option = {
      title: {
        text: '因子权重分布',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c}% ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        top: 'middle'
      },
      series: [
        {
          name: '权重',
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['60%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: false,
            position: 'center'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 20,
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: false
          },
          data: data
        }
      ]
    }

    chart.setOption(option)
    console.log('[updateWeightChart] 图表设置完成')

    // 调整图表大小
    setTimeout(() => {
      if (chart) {
        chart.resize()
        console.log('[updateWeightChart] 图表resize完成')
      }
    }, 100)
  }

  // 更新收敛曲线
  const updateConvergenceChart = (history: Array<{
    iteration: number
    weights: Record<string, number>
    sharpe_ratio: number
  }>) => {
    if (!convergenceChartRef.current) return

    let chart = convergenceChartInstanceRef.current
    if (!chart) {
      chart = echarts.init(convergenceChartRef.current)
      convergenceChartInstanceRef.current = chart
    }

    // 清空图表数据
    chart.clear()

    const iterations = history.map(h => h.iteration)
    const sharpeRatios = history.map(h => h.sharpe_ratio)

    const option = {
      title: {
        text: '优化收敛曲线',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
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
        name: '迭代次数',
        data: iterations
      },
      yAxis: {
        type: 'value',
        name: '夏普比率'
      },
      series: [
        {
          name: '夏普比率',
          type: 'line',
          data: sharpeRatios,
          smooth: true,
          itemStyle: { color: '#3b82f6' },
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
          }
        }
      ]
    }

    chart.setOption(option)
  }

  // 权重数据转换为表格数据
  const getWeightTableData = () => {
    if (!optimizationResult || !optimizationResult.weights) return []
    return Object.entries(optimizationResult.weights).map(([name, weight], index) => ({
      key: index,
      name,
      weight: (weight * 100).toFixed(2) + '%',
      weight_value: weight
    }))
  }

  const weightColumns = [
    {
      title: '排名',
      dataIndex: 'key',
      key: 'rank',
      width: 80,
      render: (_: any, __: any, index: number) => index + 1
    },
    {
      title: '因子名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      render: (text: string, record: any) => (
        <Tag color={record.weight_value > 0.2 ? 'green' : record.weight_value > 0.1 ? 'blue' : 'default'}>
          {text}
        </Tag>
      ),
      sorter: (a: any, b: any) => a.weight_value - b.weight_value
    }
  ]

  // 生成组合因子代码
  const generateCompositeFactorCode = (result?: OptimizationResult) => {
    const data = result || optimizationResult
    if (!data || !data.weights) return ''

    const weights = data.weights
    const factorNames = Object.keys(weights)

    // 从因子列表中获取因子代码
    const getFactorCode = (factorName: string) => {
      const factor = factors.find(f => f.name === factorName)
      if (factor && factor.code) {
        const expression = extractFactorExpression(factor.code)
        if (expression) {
          return expression
        }
      }

      console.warn(`未找到可组合的因子表达式: ${factorName}`)
      return null
    }

    const parts: string[] = []
    factorNames.forEach((name) => {
      const weight = weights[name]
      const factorCode = getFactorCode(name)
      if (!factorCode) {
        return
      }
      parts.push(`${weight.toFixed(6)} * ${buildExpandingZScoreExpression(factorCode)}`)
    })

    if (parts.length !== factorNames.length) {
      return ''
    }

    const code = parts.join(' +\n')

    console.log('[generateCompositeFactorCode] 生成代码长度:', code.length)
    return code
  }

  // 保存组合因子
  const saveCompositeFactor = async () => {
    if (!optimizationResult) {
      message.warning('请先完成权重优化')
      return
    }

    const method = optimizationResult.method || 'equal_weight'
    const factorNames = Object.keys(optimizationResult.weights || {})

    // 生成时间戳：年月日时分秒格式 (YYYYMMDDHHMMSS)
    const now = new Date()
    const year = now.getFullYear()
    const month = String(now.getMonth() + 1).padStart(2, '0')
    const day = String(now.getDate()).padStart(2, '0')
    const hours = String(now.getHours()).padStart(2, '0')
    const minutes = String(now.getMinutes()).padStart(2, '0')
    const seconds = String(now.getSeconds()).padStart(2, '0')
    const timestamp = `${year}${month}${day}${hours}${minutes}${seconds}`

    // 获取股票代码
    const stockCode = optimizationResult.stock_code || '000001'

    // 新命名格式：组合因子_目标函数_年月日时分秒_股票代码
    const factorData = {
      name: `组合因子_${method}_${timestamp}_${stockCode}`,
      category: '组合因子',
      description: buildCompositeFactorDescription(
        factorNames.length,
        METHOD_LABELS[method] || method
      ),
      code: compositeFactorCode,
      formula_type: 'python'
    }

    if (!compositeFactorCode.trim()) {
      message.error(FACTOR_COPY.portfolio.missingExpression)
      return
    }

    try {
      setSavingFactor(true)
      const response = await axios.post('/api/factors', factorData)

      if (response.data.success) {
        message.success(formatActionSuccess(FACTOR_COPY.portfolio.saveAction))
        // 重新加载因子列表
        loadFactors()
      } else {
        message.error(
          formatActionFailure(
            FACTOR_COPY.portfolio.saveAction,
            response.data.message || '未知错误'
          )
        )
      }
    } catch (error: any) {
      console.error('保存组合因子失败:', error)
      message.error(
        formatActionFailure(
          FACTOR_COPY.portfolio.saveAction,
          extractErrorMessage(error)
        )
      )
    } finally {
      setSavingFactor(false)
    }
  }

  // 更新综合得分时序图
  const updateCompositeScoreChart = (data: any) => {
    if (!compositeScoreChartRef.current) return

    let chart = compositeScoreChartInstanceRef.current
    if (!chart) {
      chart = echarts.init(compositeScoreChartRef.current)
      compositeScoreChartInstanceRef.current = chart
    }

    // 清空图表数据
    chart.clear()

    const dates = data.dates || []
    const values = data.values || []

    const option = {
      title: {
        text: '综合得分时序图',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
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
        data: dates,
        axisLabel: {
          rotate: 45,
          interval: Math.floor(dates.length / 20)
        }
      },
      yAxis: {
        type: 'value',
        name: '综合得分'
      },
      series: [
        {
          name: '综合得分',
          type: 'line',
          data: values,
          smooth: true,
          itemStyle: { color: '#3b82f6' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2:1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
              ]
            }
          }
        }
      ]
    }

    chart.setOption(option)
  }

  // 更新综合得分分布图
  const updateCompositeDistChart = (values: number[]) => {
    if (!compositeDistChartRef.current || values.length === 0) return

    let chart = compositeDistChartInstanceRef.current
    if (!chart) {
      chart = echarts.init(compositeDistChartRef.current)
      compositeDistChartInstanceRef.current = chart
    }

    // 清空图表数据
    chart.clear()

    // 计算直方图数据
    const min = Math.min(...values)
    const max = Math.max(...values)
    const binCount = 30
    const binSize = (max - min) / binCount

    const bins = new Array(binCount).fill(0)
    const binLabels: string[] = []

    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize
      const binEnd = min + (i + 1) * binSize
      binLabels.push(`${binStart.toFixed(2)}-${binEnd.toFixed(2)}`)
    }

    values.forEach((value) => {
      const binIndex = Math.min(Math.floor((value - min) / binSize), binCount - 1)
      bins[binIndex]++
    })

    const option = {
      title: {
        text: '得分分布直方图',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
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
        data: binLabels,
        axisLabel: {
          rotate: 45,
          interval: Math.floor(binCount / 10)
        }
      },
      yAxis: {
        type: 'value',
        name: '频数'
      },
      series: [
        {
          name: '频数',
          type: 'bar',
          data: bins,
          itemStyle: {
            color: '#10b981'
          }
        }
      ]
    }

    chart.setOption(option)
  }

  // ========== 方法对比相关函数 ==========

  // 运行方法对比
  const runMethodComparison = async (values: any) => {
    console.log('[runMethodComparison] Form values:', values)

    const selectedFactors = values.compare_factors || []
    const selectedMethods = values.compare_methods || []

    console.log('[runMethodComparison] Selected factors:', selectedFactors)
    console.log('[runMethodComparison] Selected methods:', selectedMethods)
    console.log('[runMethodComparison] Date range:', values.dateRange)

    const [startDate, endDate] = values.dateRange
    const requestData = {
      stock_code: values.stock_code || '000001',
      factors: selectedFactors,
      start_date: startDate.format('YYYY-MM-DD'),
      end_date: endDate.format('YYYY-MM-DD'),
      methods: selectedMethods
    }

    console.log('[runMethodComparison] Request data:', requestData)

    try {
      setLoading(true)

      // 清空结果前先销毁图表实例
      if (compareChartInstanceRef.current) {
        compareChartInstanceRef.current.dispose()
        compareChartInstanceRef.current = null
      }

      setCompareResult(null)

      const response = await axios.post('/api/portfolio/compare-methods', requestData)

      if (response.data.success) {
        setCompareResult(response.data.data.results || response.data.data)
        message.success(formatActionSuccess(FACTOR_COPY.portfolio.compareAction))

        // 延迟渲染图表
        setTimeout(() => {
          updateCompareChart(response.data.data.results || response.data.data)
        }, 300)
      } else {
        message.error(
          formatActionFailure(
            FACTOR_COPY.portfolio.compareAction,
            response.data.message || '未知错误'
          )
        )
      }
    } catch (error: any) {
      console.error('方法对比失败:', error)
      message.error(
        formatActionFailure(
          FACTOR_COPY.portfolio.compareAction,
          extractErrorMessage(error)
        )
      )
    } finally {
      setLoading(false)
    }
  }

  // 更新方法对比图表
  const updateCompareChart = (results: any) => {
    if (!compareChartRef.current || !results) return

    let chart = compareChartInstanceRef.current
    if (!chart) {
      chart = echarts.init(compareChartRef.current)
      compareChartInstanceRef.current = chart
    }

    // 清空图表数据
    chart.clear()

    const methods = Object.keys(results)
    const returnData = methods.map(m => ((results[m].annual_return || 0) * 100).toFixed(2))
    const sharpeData = methods.map(m => (results[m].sharpe_ratio || 0).toFixed(2))

    const option = {
      title: {
        text: '优化方法对比',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: (params: any) => {
          let result = params[0].name + '<br/>'
          params.forEach((item: any) => {
            result += `${item.marker} ${item.seriesName}: ${item.value}<br/>`
          })
          return result
        }
      },
      legend: {
        data: ['年化IC收益率(%)', 'IR'],
        top: 30
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: methods.map(m => {
          const methodMap: Record<string, string> = {
            equal_weight: '等权重',
            ic_weight: 'IC加权',
            ir_weight: 'IR加权',
            max_sharpe: '最大夏普',
            max_return: '最大收益',
            min_variance: '最小方差'
          }
          return methodMap[m] || m
        })
      },
      yAxis: [
        {
          type: 'value',
          name: '年化IC收益率(%)',
          position: 'left'
        },
        {
          type: 'value',
          name: 'IR',
          position: 'right'
        }
      ],
      series: [
        {
          name: '年化IC收益率(%)',
          type: 'bar',
          data: returnData,
          itemStyle: {
            color: '#3b82f6'
          }
        },
        {
          name: 'IR',
          type: 'line',
          yAxisIndex: 1,
          data: sharpeData,
          itemStyle: {
            color: '#10b981'
          }
        }
      ]
    }

    chart.setOption(option)
  }

  // 方法对比表格数据
  const getCompareTableData = () => {
    if (!compareResult) return []
    return Object.entries(compareResult).map(([method, metrics]: [string, any]) => {
      const methodMap: Record<string, string> = {
        equal_weight: '等权重',
        ic_weight: 'IC加权',
        ir_weight: 'IR加权',
        max_sharpe: '最大夏普',
        max_return: '最大收益',
        min_variance: '最小方差'
      }
      const annualReturn = metrics.annual_return || 0
      const icMean = metrics.ic_mean || 0
      const icStd = metrics.ic_std || 0
      const ir = metrics.ir || 0

      return {
        key: method,
        method: methodMap[method] || method,
        return_rate: (annualReturn * 100).toFixed(2) + '%',
        volatility: (icStd * 100).toFixed(4),
        sharpe_ratio: ir.toFixed(4),
        return_value: annualReturn,
        sharpe_value: ir,
        ic_mean: icMean.toFixed(4),
        ic_std: icStd.toFixed(4)
      }
    })
  }

  const compareColumns = [
    {
      title: '优化方法',
      dataIndex: 'method',
      key: 'method'
    },
    {
      title: '年化IC收益率',
      dataIndex: 'return_rate',
      key: 'return_rate',
      render: (text: string, record: any) => (
        <Tag color={record.return_value > 0.05 ? 'green' : record.return_value > 0.02 ? 'blue' : 'default'}>
          {text}
        </Tag>
      ),
      sorter: (a: any, b: any) => a.return_value - b.return_value
    },
    {
      title: 'IC标准差',
      dataIndex: 'ic_std',
      key: 'ic_std',
      sorter: (a: any, b: any) => parseFloat(a.ic_std) - parseFloat(b.ic_std)
    },
    {
      title: 'IR (信息比率)',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      render: (text: string, record: any) => (
        <Tag color={record.sharpe_value > 1 ? 'green' : record.sharpe_value > 0.5 ? 'blue' : 'orange'}>
          {text}
        </Tag>
      ),
      sorter: (a: any, b: any) => a.sharpe_value - b.sharpe_value
    }
  ]

  return (
    <div className="portfolio-analysis-container">
      {/* 背景 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      <div className="portfolio-analysis-content">
        <div className="page-header">
          <div className="header-content">
            <PieChartOutlined className="header-icon" />
            <div>
              <h1 className="page-title">组合分析</h1>
              <p className="page-subtitle">{FACTOR_COPY.portfolio.subtitle}</p>
            </div>
          </div>
        </div>

        <Card className="main-card">
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            tabBarStyle={{ marginBottom: 24 }}
          >
            {/* Tab 1: 权重优化 */}
            <Tabs.TabPane
              tab={
                <span>
                  <ThunderboltOutlined />
                  权重优化
                </span>
              }
              key="weights"
            >
              <Row gutter={[24, 24]}>
                {/* 左侧配置面板 */}
                <Col xs={24} lg={8}>
                  <Card title="优化配置" className="config-card">
                    <Form
                      form={optimizeForm}
                      layout="vertical"
                      onFinish={startOptimization}
                    >
                      {/* 因子选择 */}
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        因子选择
                      </Divider>
                      <p className="text-hint">选择用于构建组合的因子（至少2个）</p>

                      <Form.Item
                        name="factors"
                        rules={[{ required: true, message: '请至少选择2个因子' }]}
                      >
                        <Select
                          mode="multiple"
                          placeholder="输入因子名称搜索"
                          style={{ width: '100%' }}
                          showSearch
                          filterOption={(input, option) => {
                            const label = String(option?.label ?? '')
                            const value = String(option?.value ?? '')
                            return (
                              label.toLowerCase().includes(input.toLowerCase()) ||
                              value.toLowerCase().includes(input.toLowerCase())
                            )
                          }}
                          optionLabelProp="label"
                          maxTagCount="responsive"
                          size="large"
                        >
                          {factors.map((factor) => (
                            <Option
                              key={factor.id}
                              value={factor.name}
                              label={factor.name}
                            >
                              <div
                                style={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: 4
                                }}
                              >
                                <div
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 8
                                  }}
                                >
                                  <span style={{ fontWeight: 500 }}>
                                    {factor.name}
                                  </span>
                                  <Tag
                                    color={
                                      factor.source === 'preset'
                                        ? 'success'
                                        : 'warning'
                                    }
                                  >
                                    {factor.source === 'preset' ? '预置' : '自定义'}
                                  </Tag>
                                  <Tag color="blue">{factor.category}</Tag>
                                </div>
                                <div
                                  style={{
                                    fontSize: 12,
                                    color: '#64748b',
                                    fontFamily: 'monospace'
                                  }}
                                >
                                  {factor.code}
                                </div>
                                {factor.description && (
                                  <div
                                    style={{
                                      fontSize: 12,
                                      color: '#94a3b8'
                                    }}
                                  >
                                    {factor.description}
                                  </div>
                                )}
                              </div>
                            </Option>
                          ))}
                        </Select>
                      </Form.Item>

                      {/* 股票代码 */}
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        股票代码
                      </Divider>
                      <Form.Item
                        name="stock_code"
                        label="股票代码"
                        initialValue="000001"
                        rules={[{ required: true, message: '请输入股票代码' }]}
                      >
                        <Input placeholder="例如: 000001" />
                      </Form.Item>

                      <Form.Item noStyle shouldUpdate>
                        {() => {
                          const selectedCount =
                            optimizeForm.getFieldValue('factors')?.length || 0
                          return (
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: 16
                              }}
                            >
                              <span className="text-hint">
                                已选择{' '}
                                <strong style={{ color: '#3b82f6' }}>
                                  {selectedCount}
                                </strong>{' '}
                                个因子
                              </span>
                              <Space size="small">
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    optimizeForm.setFieldsValue({
                                      factors: factors.map((f) => f.name)
                                    })
                                  }}
                                >
                                  全选
                                </Button>
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    optimizeForm.setFieldsValue({
                                      factors: []
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

                      {/* 日期范围 */}
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        数据范围
                      </Divider>

                      <Form.Item
                        label="日期范围"
                        name="dateRange"
                        rules={[{ required: true, message: '请选择日期范围' }]}
                      >
                        <RangePicker style={{ width: '100%' }} />
                      </Form.Item>

                      {/* 优化方法 */}
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        优化方法
                      </Divider>

                      <Form.Item
                        label="优化目标"
                        name="method"
                        tooltip="选择组合优化的目标函数"
                      >
                        <Select>
                          <Option value="max_sharpe">最大化夏普比率</Option>
                          <Option value="ic_weight">IC加权</Option>
                          <Option value="ir_weight">IR加权</Option>
                          <Option value="max_return">最大化收益</Option>
                          <Option value="min_variance">最小化方差</Option>
                          <Option value="equal_weight">等权重</Option>
                        </Select>
                      </Form.Item>

                      <Form.Item
                        label="再平衡频率"
                        name="rebalance_frequency"
                      >
                        <Select>
                          <Option value="daily">每日</Option>
                          <Option value="weekly">每周</Option>
                          <Option value="monthly">每月</Option>
                          <Option value="quarterly">每季度</Option>
                        </Select>
                      </Form.Item>

                      {/* 参数配置 */}
                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">
                        参数配置
                      </Divider>

                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item
                            label="无风险利率"
                            name="risk_free_rate"
                            tooltip="用于计算夏普比率"
                          >
                            <InputNumber
                              min={0}
                              max={1}
                              step={0.01}
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item
                            label="目标收益率"
                            name="target_return"
                            tooltip="仅在最大化收益时使用"
                          >
                            <InputNumber
                              min={0}
                              max={1}
                              step={0.01}
                              style={{ width: '100%' }}
                              placeholder="可选"
                            />
                          </Form.Item>
                        </Col>
                      </Row>

                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item
                            label="最小权重"
                            name="min_weight"
                            tooltip="单个因子的最小权重限制"
                          >
                            <InputNumber
                              min={0}
                              max={1}
                              step={0.05}
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item
                            label="最大权重"
                            name="max_weight"
                            tooltip="单个因子的最大权重限制"
                          >
                            <InputNumber
                              min={0}
                              max={1}
                              step={0.05}
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Col>
                      </Row>

                      <Form.Item>
                        <Button
                          type="primary"
                          htmlType="submit"
                          icon={<RocketOutlined />}
                          loading={loading}
                          block
                          size="large"
                          disabled={optimizing}
                        >
                          {optimizing ? '优化中...' : '开始优化'}
                        </Button>
                      </Form.Item>
                    </Form>
                  </Card>
                </Col>

                {/* 右侧结果展示 */}
                <Col xs={24} lg={16}>
                  <Card title="优化结果" className="result-card">
                    {/* 等待提示 */}
                    {!optimizing && !optimizationResult && (
                      <div className="placeholder-content">
                        <BarChartOutlined className="placeholder-icon" />
                        <p className="placeholder-text">
                          {FACTOR_COPY.portfolio.emptyStateTitle}
                        </p>
                        <p className="placeholder-hint">
                          {FACTOR_COPY.portfolio.emptyStateHint}
                        </p>
                      </div>
                    )}

                    {/* 优化中 */}
                    {optimizing && (
                      <div className="optimizing-status">
                        <Spin size="large" description={FACTOR_COPY.portfolio.running} />
                      </div>
                    )}

                    {/* 优化结果 */}
                    {optimizationResult && (
                      <div className="optimization-result">
                        {/* 组合因子表达式 */}
                        {compositeFactorCode && (
                          <div style={{ marginBottom: 24 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                              <h4 className="result-title" style={{ margin: 0 }}><AimOutlined /> {FACTOR_COPY.portfolio.expressionTitle}</h4>
                              <Button
                                type="primary"
                                icon={<RocketOutlined />}
                                onClick={saveCompositeFactor}
                                loading={savingFactor}
                                size="small"
                              >
                                保存到因子库
                              </Button>
                            </div>
                            <Card
                              style={{
                                background: 'rgba(248, 250, 252, 0.8)',
                                border: '1px solid rgba(59, 130, 246, 0.1)',
                                borderRadius: '8px'
                              }}
                            >
                              <pre
                                style={{
                                  margin: 0,
                                  padding: '16px',
                                  background: 'transparent',
                                  fontSize: '12px',
                                  lineHeight: '1.6',
                                  color: '#334155',
                                  overflowX: 'auto',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word'
                                }}
                              >
                                {compositeFactorCode}
                              </pre>
                            </Card>
                            <p style={{ marginTop: 12, fontSize: '13px', color: '#64748b' }}>
                              <BulbOutlined style={{ marginRight: 4 }} />{FACTOR_COPY.portfolio.expressionHint}
                            </p>
                          </div>
                        )}

                        <Divider />

                        {/* 权重分布图表 */}
                        <div className="chart-section" style={{ marginBottom: 24 }}>
                          <h4 className="chart-title">因子权重分布</h4>
                          <div
                            ref={weightChartRef}
                            className="chart-container"
                            style={{ height: '350px' }}
                          ></div>
                        </div>

                        <Divider />

                        {/* 权重明细表 */}
                        <h4 className="result-title">权重明细</h4>
                        <Table
                          columns={weightColumns}
                          dataSource={getWeightTableData()}
                          pagination={false}
                          size="small"
                          bordered
                          style={{ marginBottom: 24 }}
                        />

                        {/* 性能指标 */}
                        <div className="metrics-section">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Statistic
                                title="组合收益率"
                                value={optimizationResult?.metrics?.return ? ((optimizationResult.metrics.return * 100).toFixed(2)) : '-'}
                                suffix={optimizationResult?.metrics?.return ? '%' : undefined}
                                valueStyle={{
                                  color: '#3b82f6',
                                  fontWeight: 700
                                }}
                              />
                            </Col>
                            <Col span={8}>
                              <Statistic
                                title="IC值"
                                value={optimizationResult?.metrics?.ic?.toFixed(4) || '-'}
                                valueStyle={{
                                  color: '#ef4444',
                                  fontWeight: 700
                                }}
                              />
                            </Col>
                            <Col span={8}>
                              <Statistic
                                title="IR值"
                                value={optimizationResult?.metrics?.ir?.toFixed(4) || '-'}
                                valueStyle={{
                                  color:
                                    (optimizationResult?.metrics?.ir || 0) > 1
                                      ? '#22c55e'
                                      : '#f59e0b',
                                  fontWeight: 700
                                }}
                              />
                            </Col>
                          </Row>
                        </div>

                        {/* 综合得分评估 */}
                        {optimizationResult.composite_stats && (
                          <>
                            <Divider />
                            <div className="metrics-section" style={{ marginBottom: 24 }}>
                              <h4 className="chart-title" style={{ marginBottom: 16 }}>
                                <LineChartOutlined style={{ marginRight: 8 }} />
                                综合得分评估
                              </h4>
                              <Row gutter={16}>
                                <Col span={6}>
                                  <Statistic
                                    title="得分均值"
                                    value={optimizationResult.composite_stats.mean || 0}
                                    precision={4}
                                    valueStyle={{ color: '#3b82f6', fontWeight: 600 }}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="得分标准差"
                                    value={optimizationResult.composite_stats.std || 0}
                                    precision={4}
                                    valueStyle={{ color: '#ef4444', fontWeight: 600 }}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="最小值"
                                    value={optimizationResult.composite_stats.min || 0}
                                    precision={4}
                                  />
                                </Col>
                                <Col span={6}>
                                  <Statistic
                                    title="最大值"
                                    value={optimizationResult.composite_stats.max || 0}
                                    precision={4}
                                  />
                                </Col>
                              </Row>
                            </div>

                            {/* 综合得分时序图 */}
                            {optimizationResult.composite_score && optimizationResult.composite_score.values && (
                              <>
                                <div className="chart-section" style={{ marginBottom: 24 }}>
                                  <div
                                    ref={compositeScoreChartRef}
                                    className="chart-container"
                                    style={{ height: '300px' }}
                                  ></div>
                                </div>

                                {/* 综合得分分布图 */}
                                <div className="chart-section">
                                  <div
                                    ref={compositeDistChartRef}
                                    className="chart-container"
                                    style={{ height: '250px' }}
                                  ></div>
                                </div>
                              </>
                            )}
                          </>
                        )}

                        {/* 收敛曲线 */}
                        {optimizationResult.weights_history &&
                          optimizationResult.weights_history.length > 0 && (
                            <>
                              <Divider />
                              <div
                                className="chart-section"
                                style={{ marginBottom: 24 }}
                              >
                                <h4 className="chart-title">优化收敛曲线</h4>
                                <div
                                  ref={convergenceChartRef}
                                  className="chart-container"
                                  style={{ height: '300px' }}
                                ></div>
                              </div>
                            </>
                          )}
                      </div>
                    )}
                  </Card>
                </Col>
              </Row>
            </Tabs.TabPane>

            {/* Tab 2: 方法对比 */}
            <Tabs.TabPane
              tab={
                <span>
                  <BarChartOutlined />
                  方法对比
                </span>
              }
              key="compare"
            >
              <Row gutter={[24, 24]}>
                {/* 左侧配置面板 */}
                <Col xs={24} lg={8}>
                  <Card title="方法对比配置" className="config-card">
                    <Form
                      form={compareForm}
                      layout="vertical"
                      onFinish={runMethodComparison}
                    >
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        因子选择
                      </Divider>
                      <p className="text-hint">选择用于对比的因子（至少1个）</p>

                      <Form.Item
                        name="compare_factors"
                        rules={[
                          { required: true, message: '请至少选择1个因子' },
                          {
                            validator: (_, value) => {
                              if (!value || value.length < 1) {
                                return Promise.reject('请至少选择1个因子')
                              }
                              return Promise.resolve()
                            }
                          }
                        ]}
                      >
                        <Select
                          mode="multiple"
                          placeholder="输入因子名称搜索"
                          style={{ width: '100%' }}
                          showSearch
                          filterOption={(input, option) => {
                            const label = String(option?.label ?? '')
                            const value = String(option?.value ?? '')
                            return (
                              label.toLowerCase().includes(input.toLowerCase()) ||
                              value.toLowerCase().includes(input.toLowerCase())
                            )
                          }}
                          optionLabelProp="label"
                          maxTagCount="responsive"
                          size="large"
                        >
                          {factors.map((factor) => (
                            <Option
                              key={factor.id}
                              value={factor.name}
                              label={factor.name}
                            >
                              <div
                                style={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: 4
                                }}
                              >
                                <div
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 8
                                  }}
                                >
                                  <span style={{ fontWeight: 500 }}>
                                    {factor.name}
                                  </span>
                                  <Tag
                                    color={
                                      factor.source === 'preset'
                                        ? 'success'
                                        : 'warning'
                                    }
                                  >
                                    {factor.source === 'preset' ? '预置' : '自定义'}
                                  </Tag>
                                  <Tag color="blue">{factor.category}</Tag>
                                </div>
                                <div
                                  style={{
                                    fontSize: 12,
                                    color: '#64748b',
                                    fontFamily: 'monospace'
                                  }}
                                >
                                  {factor.code}
                                </div>
                                {factor.description && (
                                  <div
                                    style={{
                                      fontSize: 12,
                                      color: '#94a3b8'
                                    }}
                                  >
                                    {factor.description}
                                  </div>
                                )}
                              </div>
                            </Option>
                          ))}
                        </Select>
                      </Form.Item>

                      <Form.Item noStyle shouldUpdate>
                        {() => {
                          const selectedCount =
                            compareForm.getFieldValue('compare_factors')?.length || 0
                          return (
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: 16
                              }}
                            >
                              <span className="text-hint">
                                已选择{' '}
                                <strong style={{ color: '#3b82f6' }}>
                                  {selectedCount}
                                </strong>{' '}
                                个因子
                              </span>
                              <Space size="small">
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    compareForm.setFieldsValue({
                                      compare_factors: factors.map((f) => f.name)
                                    })
                                  }}
                                >
                                  全选
                                </Button>
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    compareForm.setFieldsValue({
                                      compare_factors: []
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

                      {/* 股票代码 */}
                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        股票代码
                      </Divider>
                      <Form.Item
                        name="stock_code"
                        label="股票代码"
                        initialValue="000001"
                        rules={[{ required: true, message: '请输入股票代码' }]}
                      >
                        <Input placeholder="例如: 000001" />
                      </Form.Item>

                      <Divider style={{ fontSize: '13px', fontWeight: 600, color: '#0f172a' }}>
                        数据范围
                      </Divider>

                      <Form.Item
                        label="日期范围"
                        name="dateRange"
                        initialValue={[dayjs().subtract(1, 'year'), dayjs()]}
                        rules={[{ required: true, message: '请选择日期范围' }]}
                      >
                        <RangePicker style={{ width: '100%' }} />
                      </Form.Item>

                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">
                        对比方法
                      </Divider>
                      <p className="text-hint">选择要对比的优化方法（至少2个）</p>

                      <Form.Item
                        name="compare_methods"
                        rules={[
                          { required: true, message: '请至少选择2个方法' },
                          {
                            validator: (_, value) => {
                              if (!value || value.length < 2) {
                                return Promise.reject('请至少选择2个方法')
                              }
                              return Promise.resolve()
                            }
                          }
                        ]}
                      >
                        <Select
                          mode="multiple"
                          placeholder="选择对比方法"
                          style={{ width: '100%' }}
                          size="large"
                        >
                          <Option value="equal_weight">等权重</Option>
                          <Option value="ic_weight">IC加权</Option>
                          <Option value="ir_weight">IR加权</Option>
                          <Option value="max_sharpe">最大夏普</Option>
                          <Option value="max_return">最大收益</Option>
                          <Option value="min_variance">最小方差</Option>
                        </Select>
                      </Form.Item>

                      <Form.Item noStyle shouldUpdate>
                        {() => {
                          const selectedMethods =
                            compareForm.getFieldValue('compare_methods') || []
                          return (
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: 16
                              }}
                            >
                              <span className="text-hint">
                                已选择{' '}
                                <strong style={{ color: '#3b82f6' }}>
                                  {selectedMethods.length}
                                </strong>{' '}
                                个方法
                              </span>
                              <Space size="small">
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    compareForm.setFieldsValue({
                                      compare_methods: ['equal_weight', 'ic_weight', 'ir_weight', 'max_sharpe', 'max_return', 'min_variance']
                                    })
                                  }}
                                >
                                  全选
                                </Button>
                                <Button
                                  type="link"
                                  size="small"
                                  onClick={() => {
                                    compareForm.setFieldsValue({
                                      compare_methods: []
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

                      <Form.Item>
                        <Button
                          type="primary"
                          htmlType="submit"
                          icon={<BarChartOutlined />}
                          loading={loading}
                          block
                          size="large"
                        >
                          开始方法对比
                        </Button>
                      </Form.Item>
                    </Form>
                  </Card>
                </Col>

                {/* 右侧结果展示 */}
                <Col xs={24} lg={16}>
                  <Card title="方法对比结果" className="result-card">
                    {!compareResult && (
                      <div className="placeholder-content">
                        <BarChartOutlined className="placeholder-icon" />
                        <p className="placeholder-text">
                          {FACTOR_COPY.portfolio.compareEmptyStateTitle}
                        </p>
                        <p className="placeholder-hint">
                          {FACTOR_COPY.portfolio.compareEmptyStateHint}
                        </p>
                      </div>
                    )}

                    {compareResult && (
                      <div className="compare-result">
                        {/* 对比图表 */}
                        <div className="chart-section" style={{ marginBottom: 24 }}>
                          <h4 className="chart-title">优化方法对比</h4>
                          <div
                            ref={compareChartRef}
                            className="chart-container"
                          ></div>
                        </div>

                        <Divider />

                        {/* 详细数据表格 */}
                        <h4 className="result-title">详细指标对比</h4>
                        <Table
                          columns={compareColumns}
                          dataSource={getCompareTableData()}
                          pagination={false}
                          size="small"
                          bordered
                          rowKey="method"
                        />
                      </div>
                    )}
                  </Card>
                </Col>
              </Row>
            </Tabs.TabPane>
          </Tabs>
        </Card>
      </div>
    </div>
  )
}

export default PortfolioAnalysis
