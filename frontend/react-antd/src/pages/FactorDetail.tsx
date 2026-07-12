import { useEffect, useState, useRef, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Button,
  Space,
  Tag,
  Spin,
  Alert,
  message,
  Row,
  Col,
  Statistic,
  Select,
  Input,
  DatePicker,
  Modal,
  Form,
  Divider,
  Tabs,
  Progress,
  Tooltip,
  Typography
} from 'antd'
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  RobotOutlined,
  DownloadOutlined,
  StopOutlined,
  LineChartOutlined,
  EditOutlined,
  DeleteOutlined,
  CopyOutlined,
  ExperimentOutlined,
  FundOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons'
import * as echarts from '@/utils/echarts'
import { api } from '@/services/api'
import CodeTextArea from '@/components/CodeTextArea'
import MarkdownReportViewer from '@/components/MarkdownReportViewer'
import {
  FORMULA_EMPTY_MESSAGE,
  FORMULA_FIELD_LABEL,
  FORMULA_TYPE_OPTIONS,
  getFormulaHelpContent,
  getFormulaPlaceholder,
  normalizeFormulaType,
} from '@/utils/formula'
import './FactorDetail.css'
import dayjs from 'dayjs'

const { Option } = Select
const { RangePicker } = DatePicker

interface FactorDetail {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
  formula_type?: string
  is_active?: boolean
  created_at?: string
  updated_at?: string
}

interface AnalysisData {
  ic?: {
    data: {
      ic_stats: Record<string, any>
    }
  }
}

interface ChartData {
  stock: Array<{
    date: string
    open: number
    high: number
    low: number
    close: number
    volume: number
  }>
  factor: {
    dates: string[]
    values: number[]
    plotSeries?: Array<{
      name: string
      dates: string[]
      values: number[]
      renderType?: 'line' | 'bar'
    }>
    formulaType?: string
  }
}

const PLOT_SERIES_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4']

const getPlotSeriesColor = (index: number) => PLOT_SERIES_COLORS[index % PLOT_SERIES_COLORS.length]

const alignValuesToDates = (
  dates: string[],
  series: { dates: string[]; values: number[] }
): Array<number | null> => {
  const valueMap = new Map(series.dates.map((date, index) => [date, series.values[index] ?? null]))
  return dates.map((date) => valueMap.get(date) ?? null)
}

// 信息提示组件
const InfoTooltip: React.FC<{ title: string; content: string }> = ({ title, content }) => (
  <Tooltip title={content} placement="top">
    <span style={{ cursor: 'help', marginLeft: '4px' }}>
      {title}
      <QuestionCircleOutlined style={{ color: '#1890ff', marginLeft: '4px' }} />
    </span>
  </Tooltip>
)

const FactorDetail: React.FC = () => {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const id = searchParams.get('id')

  // 基本状态
  const [factor, setFactor] = useState<FactorDetail | null>(null)
  const [loading, setLoading] = useState(false)

  // 分析相关
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [analyzing, setAnalyzing] = useState(false)

  // 编辑相关
  const [editing, setEditing] = useState(false)
  const [editForm, setEditForm] = useState<any>({})

  // 行情图表相关
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [chartPeriod, setChartPeriod] = useState<string>('1y')
  const [factorChartType, setFactorChartType] = useState<string>('line')
  const [loadingChart, setLoadingChart] = useState(false)
  const [stockCode, setStockCode] = useState<string>('000001.SZ')
  // 用于显示的股票代码（不带后缀）
  const [stockCodeDisplay, setStockCodeDisplay] = useState<string>(stockCode.replace(/\.(SH|SZ)$/, ''))

  // 同步 stockCode 和 stockCodeDisplay 的状态
  useEffect(() => {
    setStockCodeDisplay(stockCode.replace(/\.(SH|SZ)$/, ''))
  }, [stockCode])
  const [customStartDate, setCustomStartDate] = useState<string>('')
  const [customEndDate, setCustomEndDate] = useState<string>('')
  const [showCustomDatePicker, setShowCustomDatePicker] = useState(false)

  // Tab 2-5 数据状态
  const [exposureData, setExposureData] = useState<any>(null)
  const [effectivenessData, setEffectivenessData] = useState<any>(null)
  const [attributionData, setAttributionData] = useState<any>(null)
  const [monitoringData, setMonitoringData] = useState<any>(null)
  const [loadingAnalysisTabs, setLoadingAnalysisTabs] = useState(false)
  const [activeTabKey, setActiveTabKey] = useState<string>('chart')
  const [aiReportMarkdown, setAiReportMarkdown] = useState<string>('')
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false)
  const [aiAnalysisError, setAiAnalysisError] = useState<string>('')
  const [aiAnalysisMeta, setAiAnalysisMeta] = useState<{
    model?: string
    requestUrl?: string
    factorName?: string
    generatedAt?: string
  } | null>(null)

  // 图表容器引用
  const distributionChartRef = useRef<HTMLDivElement>(null)
  const icSeriesChartRef = useRef<HTMLDivElement>(null)
  const icHistogramChartRef = useRef<HTMLDivElement>(null)
  const priceChartRef = useRef<HTMLDivElement>(null)

  // Tab 2-5 图表容器引用
  const exposureHistogramRef = useRef<HTMLDivElement>(null)
  const percentileTimeSeriesRef = useRef<HTMLDivElement>(null)
  const scatterChartRef = useRef<HTMLDivElement>(null)
  const icTimeSeriesChartRef = useRef<HTMLDivElement>(null)
  const eventResponseChartRef = useRef<HTMLDivElement>(null)
  const decayCurveChartRef = useRef<HTMLDivElement>(null)
  const rollingBandChartRef = useRef<HTMLDivElement>(null)
  const transitionMatrixRef = useRef<HTMLDivElement>(null)
  const structuralBreakChartRef = useRef<HTMLDivElement>(null)
  const seasonalityChartRef = useRef<HTMLDivElement>(null)

  // 图表实例
  const chartsRef = useRef<Record<string, echarts.ECharts>>({})
  const aiAnalysisAbortRef = useRef<AbortController | null>(null)
  const aiReportBufferRef = useRef('')
  const aiReportFlushTimerRef = useRef<number | null>(null)

  // 加载因子详情
  const loadFactorDetail = useCallback(async () => {
    if (!id) {
      message.warning('缺少因子ID参数')
      return
    }
    setLoading(true)
    try {
      const response = await api.getFactorDetail(Number(id)) as any
      if (response && response.success) {
        setFactor(response.data)
      } else {
        message.error('因子不存在')
      }
    } catch (error) {
      console.error('Failed to load factor detail:', error)
      message.error('加载因子详情失败')
    } finally {
      setLoading(false)
    }
  }, [id])

  // 分析因子
  const analyzeFactor = useCallback(async () => {
    if (!id || !factor) return

    // 使用用户选择的时间范围和股票代码
    let startDate: string
    let endDate: string

    if (chartPeriod === 'custom') {
      if (!customStartDate || !customEndDate) {
        message.warning('请先选择自定义日期范围')
        return
      }
      startDate = customStartDate
      endDate = customEndDate
    } else {
      endDate = new Date().toISOString().split('T')[0]
      startDate = getStartDateByPeriod(chartPeriod)
    }

    setAnalyzing(true)
    try {
      const response = await api.calculateIC({
        factor_name: factor.name,
        stock_codes: [stockCode],
        start_date: startDate,
        end_date: endDate
      } as any) as any

      if (response.success && response.data) {
        let icStats = response.data.ic_stats || response.data?.metadata?.ic_stats || {}

        if (icStats.ic_stats) {
          icStats = icStats.ic_stats
        }

        const factorNames = Object.keys(icStats)
        if (factorNames.length === 0) {
          message.warning('未获取到IC统计数据')
          setAnalysisData(null)
          return
        }

        const firstFactor = factorNames[0]
        const stats = icStats[firstFactor]

        if (!stats['IC序列'] || Object.keys(stats['IC序列']).length === 0) {
          message.warning('IC序列为空')
          setAnalysisData(null)
          return
        }

        setAnalysisData({
          ic: {
            data: {
              ic_stats: icStats
            }
          }
        })
        message.success('因子分析完成')
        // 同时加载 Tab 2-5 的数据
        loadAnalysisTabsData()
      } else {
        message.error('因子分析失败：' + (response.message || '未知错误'))
      }
    } catch (error: any) {
      console.error('因子分析失败:', error)
      message.error('因子分析失败')
    } finally {
      setAnalyzing(false)
    }
  }, [id, factor, chartPeriod, customStartDate, customEndDate, stockCode])

  // 编辑相关函数
  const handleEdit = () => {
    if (!factor) return
    setEditForm({
      name: factor.name,
      category: factor.category,
      description: factor.description || '',
      code: factor.code,
      formula_type: normalizeFormulaType(factor.formula_type)
    })
    setEditing(true)
  }

  const handleSaveEdit = async () => {
    if (!factor || !editForm.name || !editForm.category || !editForm.code) {
      message.error('请填写所有必填字段')
      return
    }

    try {
      const validateResponse = await api.validateFactor({
        code: editForm.code,
        formula_type: normalizeFormulaType(editForm.formula_type)
      } as any) as any

      if (!validateResponse.success) {
        message.error(validateResponse.message || '因子表达式验证失败')
        return
      }

      const updateResponse = await api.updateFactor(factor.id, {
        name: editForm.name,
        category: editForm.category,
        description: editForm.description,
        code: editForm.code,
        formula_type: normalizeFormulaType(editForm.formula_type),
      } as any) as any

      if (updateResponse.success) {
        message.success('因子更新成功')
        setEditing(false)
        loadFactorDetail()
      } else {
        message.error('因子更新失败')
      }
    } catch (error: any) {
      message.error('操作失败')
    }
  }

  const handleCancelEdit = () => {
    setEditing(false)
    setEditForm({})
  }

  const handleValidateFormula = async () => {
    if (!editForm.code) {
      message.warning(FORMULA_EMPTY_MESSAGE)
      return
    }

    try {
      const response = await api.validateFactor({
        code: editForm.code,
        formula_type: normalizeFormulaType(editForm.formula_type)
      } as any) as any
      if (response.success) {
        message.success(response.message || '验证通过')
      } else {
        message.error(response.message || '验证失败')
      }
    } catch (error) {
      message.error('验证失败')
    }
  }

  // 删除因子
  const handleDeleteFactor = async () => {
    if (!factor || factor.source === 'preset') return

    Modal.confirm({
      title: '确认删除',
      content: `确定要删除因子 "${factor.name}" 吗？`,
      onOk: async () => {
        try {
          const response = await api.deleteFactor(factor.id) as any
          if (response.success) {
            message.success('删除成功')
            navigate('/factor-management')
          } else {
            message.error(response.message || '删除失败')
          }
        } catch (error) {
          message.error('删除失败')
        }
      }
    })
  }

  // 复制因子
  const handleCopyFactor = async () => {
    if (!factor) return

    try {
      const response = await api.copyFactor(factor.id) as any
      if (response.success) {
        message.success(`因子已复制为 "${response.data.name}"`)
        // 可选：跳转到新复制的因子详情页
        // navigate(`/factor-detail?id=${response.data.id}`)
      } else {
        message.error(response.message || '复制失败')
      }
    } catch (error) {
      message.error('复制失败')
    }
  }

  // 格式化时间显示
  const formatDateTime = (dateStr?: string) => {
    if (!dateStr) return '-'
    return dayjs(dateStr).format('YYYY-MM-DD HH:mm:ss')
  }

  // 图表初始化函数
  const initChart = (chartDom: HTMLDivElement | null, chartKey: string) => {
    if (!chartDom) return null

    const rect = chartDom.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) {
      return null
    }

    // 如果图表实例已存在，直接返回
    if (chartsRef.current[chartKey]) {
      return chartsRef.current[chartKey]
    }

    // 否则创建新的图表实例
    const myChart = echarts.init(chartDom)
    chartsRef.current[chartKey] = myChart
    return myChart
  }

  // 绘制因子分布图
  const drawDistributionChart = useCallback(() => {
    const chartDom = distributionChartRef.current
    const myChart = initChart(chartDom, 'distribution')
    if (!myChart) return

    const data = Array.from({ length: 100 }, () => (Math.random() - 0.5) * 4)
    const binCount = 20
    const min = Math.min(...data)
    const max = Math.max(...data)
    const binWidth = (max - min) / binCount

    const bins = Array(binCount).fill(0)
    const labels: string[] = []

    for (let i = 0; i < binCount; i++) {
      labels.push((min + i * binWidth).toFixed(2))
    }

    data.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1)
      bins[binIndex]++
    })

    const option: echarts.EChartsOption = {
      title: {
        text: '因子值分布直方图',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        name: '因子值',
        type: 'category',
        data: labels,
        axisLabel: { fontSize: 10 }
      },
      yAxis: {
        name: '频次',
        type: 'value'
      },
      series: [
        {
          name: '频次',
          type: 'bar',
          data: bins,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#83bff6' },
              { offset: 0.5, color: '#188df0' },
              { offset: 1, color: '#188df0' }
            ])
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [])

  // 绘制IC序列图
  const drawICSeriesChart = useCallback(() => {
    const chartDom = icSeriesChartRef.current
    if (!chartDom || !analysisData?.ic) return

    const myChart = initChart(chartDom, 'icSeries')
    if (!myChart) return

    const stats = analysisData.ic.data.ic_stats || {}
    const factorName = Object.keys(stats)[0]
    const factorStats = factorName ? stats[factorName] : {}
    const icSeries = factorStats['IC序列'] || {}
    const icArray = Object.values(icSeries) as number[]

    const option: echarts.EChartsOption = {
      title: {
        text: 'IC序列',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: { trigger: 'axis' },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: icArray.map((_, i) => i)
      },
      yAxis: {
        type: 'value',
        name: 'IC值'
      },
      series: [
        {
          name: 'IC值',
          type: 'line',
          data: icArray,
          smooth: true,
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
            ])
          },
          lineStyle: {
            color: '#3b82f6',
            width: 2
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [analysisData])

  // 绘制IC直方图
  const drawICHistogramChart = useCallback(() => {
    const chartDom = icHistogramChartRef.current
    if (!chartDom || !analysisData?.ic) return

    const myChart = initChart(chartDom, 'icHistogram')
    if (!myChart) return

    const stats = analysisData.ic.data.ic_stats || {}
    const factorName = Object.keys(stats)[0]
    const factorStats = factorName ? stats[factorName] : {}
    const icSeries = factorStats['IC序列'] || {}
    const icArray = Object.values(icSeries) as number[]

    const colors = icArray.map((v: number) =>
      v > 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)'
    )

    const option: echarts.EChartsOption = {
      title: {
        text: 'IC分布直方图',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: icArray.map((_, i) => i)
      },
      yAxis: {
        type: 'value',
        name: 'IC值'
      },
      series: [
        {
          name: 'IC分布',
          type: 'bar',
          data: icArray,
          itemStyle: {
            color: (params: any) => colors[params.dataIndex]
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [analysisData])

  // 绘制因子暴露度直方图
  const drawExposureHistogram = useCallback(() => {
    const chartDom = exposureHistogramRef.current
    if (!chartDom || !exposureData) return

    const myChart = initChart(chartDom, 'exposureHistogram')
    if (!myChart) return

    const histogram = exposureData.histogram || {}
    const bins = histogram.bins || []
    const counts = histogram.counts || []

    // 计算区间中心点
    const binCenters = bins.slice(0, -1).map((bin: number, i: number) => (bin + bins[i + 1]) / 2)

    const option: echarts.EChartsOption = {
      title: {
        text: '因子值历史分布',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          const bin1 = parseFloat(bins[param.dataIndex])
          const bin2 = parseFloat(bins[param.dataIndex + 1])
          const range = `${isNaN(bin1) ? '-' : bin1.toFixed(4)} - ${isNaN(bin2) ? '-' : bin2.toFixed(4)}`
          return `区间: [${range}]<br/>频次: ${param.value}`
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
        data: binCenters,
        name: '因子值',
        axisLabel: {
          formatter: (value: any) => {
            const num = typeof value === 'number' ? value : parseFloat(value)
            return isNaN(num) ? '-' : num.toFixed(2)
          }
        }
      },
      yAxis: {
        type: 'value',
        name: '频次'
      },
      series: [
        {
          name: '频次',
          type: 'bar',
          data: counts,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.8)' }
              ]
            },
            borderRadius: [4, 4, 0, 0]
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [exposureData])

  // 绘制分位数时间序列曲线
  const drawPercentileTimeSeries = useCallback(() => {
    const chartDom = percentileTimeSeriesRef.current
    if (!chartDom || !exposureData) return

    const myChart = initChart(chartDom, 'percentileTimeSeries')
    if (!myChart) return

    const percentileSeries = exposureData.percentile_time_series || {}
    const dates = percentileSeries.dates || []
    const percentiles = percentileSeries.percentiles || []
    const values = percentileSeries.values || []

    const option: echarts.EChartsOption = {
      title: {
        text: '因子暴露度分位数变化',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          const date = dates[param.dataIndex] || '-'
          const percentile = percentiles[param.dataIndex] || 0
          const value = values[param.dataIndex] || 0
          return `日期: ${date}<br/>分位数: ${percentile.toFixed(2)}%<br/>因子值: ${value}`
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        name: '日期',
        axisLabel: {
          formatter: (value: any) => {
            // 显示年-月-日
            const dateStr = String(value)
            const parts = dateStr.split(' ')
            if (parts.length > 0) {
              const dateParts = parts[0].split('-')
              if (dateParts.length >= 3) {
                return `${dateParts[0]}-${dateParts[1]}-${dateParts[2]}`
              }
            }
            return value
          }
        }
      },
      yAxis: {
        type: 'value',
        name: '分位数 (%)',
        min: 0,
        max: 100,
        axisLabel: {
          formatter: '{value}%'
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(59, 130, 246, 0.1)'
          }
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: '分位数',
          type: 'line',
          data: percentiles,
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            width: 2,
            color: '#3b82f6'
          },
          itemStyle: {
            color: '#3b82f6'
          },
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
          markLine: {
            symbol: 'none',
            label: {
              show: true,
              position: 'end',
              formatter: '{b}: {c}%'
            },
            lineStyle: {
              type: 'dashed',
              color: '#ef4444'
            },
            data: [
              { yAxis: 25, name: '低暴露' },
              { yAxis: 50, name: '中位数', lineStyle: { color: '#eab308' } },
              { yAxis: 75, name: '高暴露', lineStyle: { color: '#22c55e' } }
            ]
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [exposureData])

  // 绘制因子-收益散点图
  const drawScatterChart = useCallback(() => {
    const chartDom = scatterChartRef.current
    if (!chartDom || !effectivenessData) {
      console.log('drawScatterChart: chartDom or effectivenessData is null', { chartDom, effectivenessData })
      return
    }

    const myChart = initChart(chartDom, 'scatterChart')
    if (!myChart) return

    console.log('drawScatterChart: effectivenessData', effectivenessData)
    const scatterData = effectivenessData.scatter_plot || {}
    console.log('drawScatterChart: scatterData', scatterData)
    const x = scatterData.x || []
    const y = scatterData.y || []
    console.log('drawScatterChart: x, y lengths', { xLength: x.length, yLength: y.length })

    const option: echarts.EChartsOption = {
      title: {
        text: '因子-收益散点图',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (!params.data) return ''
          return `因子值: ${params.data[0].toFixed(4)}<br/>收益率: ${(params.data[1] * 100).toFixed(2)}%`
        }
      },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '10%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: '因子值',
        scale: true,
        splitLine: {
          lineStyle: { color: 'rgba(59, 130, 246, 0.1)' }
        }
      },
      yAxis: {
        type: 'value',
        name: '收益率',
        scale: true,
        axisLabel: {
          formatter: (value: any) => `${(value * 100).toFixed(1)}%`
        },
        splitLine: {
          lineStyle: { color: 'rgba(59, 130, 246, 0.1)' }
        }
      },
      series: [
        {
          name: '数据点',
          type: 'scatter',
          data: x.map((xi: number, i: number) => [xi, y[i]]),
          symbolSize: 6,
          itemStyle: {
            color: 'rgba(59, 130, 246, 0.6)',
            borderColor: '#3b82f6',
            borderWidth: 1
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [effectivenessData])

  // 绘制IC时序图
  const drawICTimeSeriesChart = useCallback(() => {
    const chartDom = icTimeSeriesChartRef.current
    if (!chartDom || !effectivenessData) {
      console.log('drawICTimeSeriesChart: chartDom or effectivenessData is null', { chartDom, effectivenessData })
      return
    }

    const myChart = initChart(chartDom, 'icTimeSeries')
    if (!myChart) return

    console.log('drawICTimeSeriesChart: effectivenessData', effectivenessData)
    const icData = effectivenessData.ic_time_series || {}
    console.log('drawICTimeSeriesChart: icData', icData)
    const dates = icData.dates || []
    const icValues = icData.ic_values || []
    const icMean = icData.ic_mean || 0
    console.log('drawICTimeSeriesChart: dates, icValues lengths', { datesLength: dates.length, icValuesLength: icValues.length, icMean })

    const option: echarts.EChartsOption = {
      title: {
        text: 'IC时序分析',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          return `日期: ${param.axisValue}<br/>IC: ${param.value.toFixed(4)}`
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        name: '日期',
        axisLabel: {
          formatter: (value: any) => {
            const dateStr = String(value)
            const parts = dateStr.split(' ')
            if (parts.length > 0) {
              const dateParts = parts[0].split('-')
              if (dateParts.length >= 3) {
                return `${dateParts[1]}-${dateParts[2]}`
              }
            }
            return value
          }
        }
      },
      yAxis: {
        type: 'value',
        name: 'IC值',
        axisLabel: {
          formatter: (value: any) => value.toFixed(3)
        },
        splitLine: {
          lineStyle: { color: 'rgba(59, 130, 246, 0.1)' }
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: 'IC',
          type: 'line',
          data: icValues,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { width: 2, color: '#3b82f6' },
          itemStyle: { color: '#3b82f6' },
          markLine: {
            symbol: 'none',
            label: { show: true, position: 'end', formatter: `均值: ${icMean.toFixed(4)}` },
            lineStyle: { type: 'dashed', color: '#ef4444' },
            data: [{ yAxis: icMean }]
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [effectivenessData])

  // 绘制事件响应图
  const drawEventResponseChart = useCallback(() => {
    const chartDom = eventResponseChartRef.current
    if (!chartDom || !effectivenessData) return

    const myChart = initChart(chartDom, 'eventResponse')
    if (!myChart) return

    const eventData = effectivenessData.event_response || {}
    const highReturns = eventData.high_exposure_returns || {}
    const lowReturns = eventData.low_exposure_returns || {}
    const periods = Object.keys(highReturns)

    const option: echarts.EChartsOption = {
      title: {
        text: '事件响应分析（高/低暴露后收益）',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          let result = `${params[0].axisValue}<br/>`
          params.forEach((param: any) => {
            result += `${param.seriesName}: ${(param.value * 100).toFixed(2)}%<br/>`
          })
          return result
        }
      },
      legend: {
        data: ['高暴露收益', '低暴露收益', '超额收益'],
        top: '8%'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '20%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: periods,
        name: '持有期'
      },
      yAxis: {
        type: 'value',
        name: '收益率',
        axisLabel: {
          formatter: (value: any) => `${(value * 100).toFixed(1)}%`
        },
        splitLine: {
          lineStyle: { color: 'rgba(59, 130, 246, 0.1)' }
        }
      },
      series: [
        {
          name: '高暴露收益',
          type: 'bar',
          data: periods.map(p => highReturns[p] || 0),
          itemStyle: { color: '#ef4444' }
        },
        {
          name: '低暴露收益',
          type: 'bar',
          data: periods.map(p => lowReturns[p] || 0),
          itemStyle: { color: '#22c55e' }
        },
        {
          name: '超额收益',
          type: 'line',
          data: periods.map(p => (highReturns[p] || 0) - (lowReturns[p] || 0)),
          itemStyle: { color: '#3b82f6' },
          lineStyle: { width: 2 },
          symbol: 'circle',
          symbolSize: 6
        }
      ]
    }

    myChart.setOption(option)
  }, [effectivenessData])

  // 绘制因子衰减曲线
  const drawDecayCurveChart = useCallback(() => {
    const chartDom = decayCurveChartRef.current
    if (!chartDom || !effectivenessData) return

    const myChart = initChart(chartDom, 'decayCurve')
    if (!myChart) return

    const decayData = effectivenessData.decay_analysis || {}
    const curve = decayData.decay_curve || []
    const periods = curve.map((c: any) => c.period)
    const icValues = curve.map((c: any) => c.ic)

    const option: echarts.EChartsOption = {
      title: {
        text: '因子衰减曲线（IC vs 持有期）',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          return `持有期: ${param.name}<br/>IC: ${param.value.toFixed(4)}`
        }
      },
      grid: {
        left: '8%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: periods,
        name: '持有期'
      },
      yAxis: {
        type: 'value',
        name: 'IC值',
        axisLabel: {
          formatter: (value: any) => value.toFixed(3)
        },
        splitLine: {
          lineStyle: { color: 'rgba(59, 130, 246, 0.1)' }
        }
      },
      series: [
        {
          name: 'IC',
          type: 'line',
          data: icValues,
          smooth: true,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: { width: 3, color: '#3b82f6' },
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

    myChart.setOption(option)
  }, [effectivenessData])

  // 绘制滚动窗口带状图
  const drawRollingBandChart = useCallback(() => {
    const chartDom = rollingBandChartRef.current
    if (!chartDom || !monitoringData?.rolling_chart) return

    const myChart = initChart(chartDom, 'rollingBand')
    if (!myChart) return

    const data = monitoringData.rolling_chart
    const dates = data.dates || []
    const values = data.values || []
    const rollingMean = data.rolling_mean || []
    const upperBand = data.upper_band || []
    const lowerBand = data.lower_band || []

    // 构建置信区间区域数据（使用多边形）
    const areaData: (number | string)[][] = []
    // 从左到右：上界线（从右到左）+ 下界线（从左到右）
    for (let i = dates.length - 1; i >= 0; i--) {
      if (upperBand[i] !== undefined && upperBand[i] !== null) {
        areaData.push([dates[i], upperBand[i]])
      }
    }
    for (let i = 0; i < dates.length; i++) {
      if (lowerBand[i] !== undefined && lowerBand[i] !== null) {
        areaData.push([dates[i], lowerBand[i]])
      }
    }

    const option: echarts.EChartsOption = {
      title: {
        text: '因子值滚动窗口带状图',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const date = params[0].axisValue
          let result = `日期: ${date}<br/>`
          params.forEach((param: any) => {
            if (param.seriesName !== '置信区间') {
              result += `${param.seriesName}: ${param.value?.toFixed(4) ?? '-'}<br/>`
            }
          })
          // 添加置信区间信息
          const idx = dates.indexOf(params[0].axisValue)
          if (idx >= 0) {
            const lower = lowerBand[idx]
            const upper = upperBand[idx]
            if (lower !== undefined && upper !== undefined && lower !== null && upper !== null) {
              result += `置信区间: [${lower.toFixed(4)}, ${upper.toFixed(4)}]`
            }
          }
          return result
        }
      },
      legend: {
        data: ['因子值', '滚动均值', '上界', '下界'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          formatter: (value: string) => value.substring(0, 10)
        },
        boundaryGap: false
      },
      yAxis: {
        type: 'value',
        name: '因子值',
        scale: true  // 自适应高度，不从0开始
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 }
      ],
      series: [
        {
          name: '置信区间',
          type: 'line',
          data: areaData,
          lineStyle: { width: 0 },
          areaStyle: {
            color: 'rgba(239, 68, 68, 0.12)'
          },
          showSymbol: false,
          tooltip: { show: false },
          z: 0
        },
        {
          name: '因子值',
          type: 'line',
          data: values,
          lineStyle: { width: 1.5, color: '#64748b' },
          symbol: 'none',
          z: 1
        },
        {
          name: '滚动均值',
          type: 'line',
          data: rollingMean,
          lineStyle: { width: 2.5, color: '#3b82f6' },
          symbol: 'none',
          z: 2
        },
        {
          name: '上界',
          type: 'line',
          data: upperBand,
          lineStyle: { width: 1, color: '#ef4444', type: 'dashed' },
          symbol: 'none',
          z: 1
        },
        {
          name: '下界',
          type: 'line',
          data: lowerBand,
          lineStyle: { width: 1, color: '#ef4444', type: 'dashed' },
          symbol: 'none',
          z: 1
        }
      ]
    }

    myChart.setOption(option)
  }, [monitoringData])

  // 绘制暴露度转移矩阵热力图
  const drawTransitionMatrix = useCallback(() => {
    const chartDom = transitionMatrixRef.current
    if (!chartDom || !monitoringData?.transition_matrix) return

    const myChart = initChart(chartDom, 'transitionMatrix')
    if (!myChart) return

    const data = monitoringData.transition_matrix
    const matrix = data.matrix || []
    const binLabels = data.bin_labels || []

    // 转换矩阵数据为热力图格式
    const heatmapData: [number, number, number][] = []
    matrix.forEach((row: number[], i: number) => {
      row.forEach((value: number, j: number) => {
        heatmapData.push([i, j, value])
      })
    })

    const option: echarts.EChartsOption = {
      title: {
        text: '暴露度转移概率矩阵',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 600 }
      },
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          if (!params || params.data === undefined) return ''
          const [from, to, prob] = params.data as [number, number, number]
          return `从 ${binLabels[from]} 到 ${binLabels[to]}<br/>转移概率: ${(prob * 100).toFixed(2)}%`
        }
      },
      grid: {
        height: '70%',
        top: '15%'
      },
      xAxis: {
        type: 'category',
        data: binLabels,
        splitArea: { show: true }
      },
      yAxis: {
        type: 'category',
        data: binLabels,
        splitArea: { show: true }
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: ['#e0f2fe', '#0369a1']
        }
      },
      series: [
        {
          name: '转移概率',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: (params: any) => {
              return (params.data[2] * 100).toFixed(1) + '%'
            }
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [monitoringData])

  // 绘制结构断点检测图
  const drawStructuralBreakChart = useCallback(() => {
    const chartDom = structuralBreakChartRef.current
    if (!chartDom || !monitoringData?.structural_break || !monitoringData?.rolling_chart) return

    const myChart = initChart(chartDom, 'structuralBreak')
    if (!myChart) return

    const rollingData = monitoringData.rolling_chart
    const breakData = monitoringData.structural_break

    const dates = rollingData.dates || []
    const values = rollingData.values || []
    const breakpoints = breakData.breakpoints || []

    // 标记断点位置
    const markPointData = breakpoints.map((bp: string) => {
      const idx = dates.indexOf(bp)
      if (idx >= 0) {
        return {
          coord: [bp, values[idx]],
          value: '断点',
          itemStyle: { color: '#ef4444' }
        }
      }
      return null
    }).filter(Boolean)

    const option: echarts.EChartsOption = {
      title: {
        text: '因子值与结构断点',
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
        data: dates,
        axisLabel: {
          formatter: (value: string) => value.substring(0, 10)
        }
      },
      yAxis: {
        type: 'value',
        name: '因子值',
        scale: true  // 自适应高度，不从0开始
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 }
      ],
      series: [
        {
          name: '因子值',
          type: 'line',
          data: values,
          lineStyle: { width: 2 },
          symbol: 'none',
          markPoint: {
            data: markPointData,
            symbolSize: 40
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [monitoringData])

  // 绘制周期性分析图
  const drawSeasonalityChart = useCallback(() => {
    const chartDom = seasonalityChartRef.current
    if (!chartDom || !monitoringData?.seasonality) return

    const myChart = initChart(chartDom, 'seasonality')
    if (!myChart) return

    const data = monitoringData.seasonality
    const frequencies = data.frequencies || []
    const powers = data.powers || []

    // 只显示前半部分的频率（正频率）
    const halfLen = Math.ceil(frequencies.length / 2)
    const displayFreqs = frequencies.slice(0, halfLen)
    const displayPowers = powers.slice(0, halfLen)

    // 转换频率为周期（天数）
    const option: echarts.EChartsOption = {
      title: {
        text: '功率谱（频域分析）',
        left: 'center',
        textStyle: { fontSize: 14, fontWeight: 600 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const param = params[0]
          const freq = parseFloat(param.axisValue)
          const freqStr = isNaN(freq) ? '-' : freq.toFixed(4)
          const period = freq > 0 ? (1 / freq).toFixed(1) : '∞'
          const power = param.value !== undefined && param.value !== null ? parseFloat(param.value).toFixed(2) : '-'
          return `频率: ${freqStr}<br/>周期: ${period} 天<br/>功率: ${power}`
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
        data: displayFreqs,
        name: '频率',
        axisLabel: {
          formatter: (value: any) => {
            const num = parseFloat(value)
            return isNaN(num) ? '-' : num.toFixed(4)
          }
        }
      },
      yAxis: {
        type: 'value',
        name: '功率'
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 50 }
      ],
      series: [
        {
          name: '功率',
          type: 'bar',
          data: displayPowers,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(99, 102, 241, 0.3)' },
                { offset: 1, color: 'rgba(99, 102, 241, 0.9)' }
              ]
            },
            borderRadius: [2, 2, 0, 0]
          }
        }
      ]
    }

    myChart.setOption(option)
  }, [monitoringData])

  // 获取开始日期
  const getStartDateByPeriod = (period: string): string => {
    const now = new Date()
    let startDate: Date

    switch (period) {
      case '3m':
        startDate = new Date(now.setMonth(now.getMonth() - 3))
        break
      case '6m':
        startDate = new Date(now.setMonth(now.getMonth() - 6))
        break
      case '1y':
        startDate = new Date(now.setFullYear(now.getFullYear() - 1))
        break
      case '3y':
        startDate = new Date(now.setFullYear(now.getFullYear() - 3))
        break
      case 'all':
        startDate = new Date('2020-01-01')
        break
      default:
        startDate = new Date(now.setFullYear(now.getFullYear() - 1))
    }

    return startDate.toISOString().split('T')[0]
  }

  const resolveCurrentDateRange = useCallback(() => {
    if (chartPeriod === 'custom') {
      if (!customStartDate || !customEndDate) {
        throw new Error('请先选择自定义日期范围')
      }
      return {
        startDate: customStartDate,
        endDate: customEndDate,
      }
    }

    return {
      startDate: getStartDateByPeriod(chartPeriod),
      endDate: new Date().toISOString().split('T')[0],
    }
  }, [chartPeriod, customStartDate, customEndDate])

  const buildAiAnalysisContext = useCallback(() => {
    const plotSeries = chartData?.factor.plotSeries ?? []
    return {
      ic_analysis: analysisData,
      exposure_analysis: exposureData,
      effectiveness_analysis: effectivenessData,
      attribution_analysis: attributionData,
      monitoring_analysis: monitoringData,
      chart_summary: chartData ? {
        chart_start_date: chartData.stock[0]?.date ?? null,
        chart_end_date: chartData.stock[chartData.stock.length - 1]?.date ?? null,
        stock_points: chartData.stock.length,
        latest_close: chartData.stock[chartData.stock.length - 1]?.close ?? null,
        factor_points: chartData.factor.values.length,
        latest_factor_value: chartData.factor.values[chartData.factor.values.length - 1] ?? null,
        formula_type: chartData.factor.formulaType ?? factor?.formula_type ?? null,
        plot_series_names: plotSeries.map((series) => series.name),
      } : null,
    }
  }, [analysisData, exposureData, effectivenessData, attributionData, monitoringData, chartData, factor?.formula_type])

  const stopAIInterpretation = useCallback(() => {
    if (aiAnalysisAbortRef.current) {
      aiAnalysisAbortRef.current.abort()
      aiAnalysisAbortRef.current = null
    }
    if (aiReportFlushTimerRef.current !== null) {
      window.clearTimeout(aiReportFlushTimerRef.current)
      aiReportFlushTimerRef.current = null
    }
    setAiAnalysisLoading(false)
  }, [])

  const flushAiReportBuffer = useCallback((force = false) => {
    if (!force && !aiReportBufferRef.current) {
      return
    }

    if (aiReportFlushTimerRef.current !== null) {
      window.clearTimeout(aiReportFlushTimerRef.current)
      aiReportFlushTimerRef.current = null
    }

    const pendingContent = aiReportBufferRef.current
    if (!pendingContent) {
      return
    }

    aiReportBufferRef.current = ''
    setAiReportMarkdown((previous) => previous + pendingContent)
  }, [])

  const enqueueAiReportChunk = useCallback((chunk: string) => {
    if (!chunk) {
      return
    }

    aiReportBufferRef.current += chunk
    if (aiReportFlushTimerRef.current !== null) {
      return
    }

    aiReportFlushTimerRef.current = window.setTimeout(() => {
      flushAiReportBuffer(true)
    }, 80)
  }, [flushAiReportBuffer])

  const exportAIReport = useCallback(() => {
    if (!aiReportMarkdown.trim()) {
      message.warning('当前没有可导出的 AI 分析报告')
      return
    }

    const timestamp = dayjs().format('YYYYMMDD_HHmmss')
    const safeName = (factor?.name || 'factor_report').replace(/[\\/:*?"<>|]/g, '_')
    const header = [
      `# ${factor?.name || '因子'} AI 分析报告`,
      '',
      `- 生成时间：${dayjs(aiAnalysisMeta?.generatedAt || new Date().toISOString()).format('YYYY-MM-DD HH:mm:ss')}`,
      `- 因子分类：${factor?.category || '-'}`,
      `- 股票代码：${stockCode}`,
      '',
    ].join('\n')
    const content = `${header}\n${aiReportMarkdown}`.trim()
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${safeName}_ai_analysis_${timestamp}.md`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [aiReportMarkdown, aiAnalysisMeta?.generatedAt, factor?.category, factor?.name, stockCode])

  const startAIInterpretation = useCallback(async () => {
    if (!factor) {
      message.warning('因子详情尚未加载完成')
      return
    }

    if (!analysisData && !exposureData && !effectivenessData && !attributionData && !monitoringData) {
      message.warning('请先点击“分析因子”获取分析结果后，再进行 AI 解读')
      return
    }

    let dateRange: { startDate: string; endDate: string }
    try {
      dateRange = resolveCurrentDateRange()
    } catch (error) {
      message.warning((error as Error).message)
      return
    }

    stopAIInterpretation()
    const controller = new AbortController()
    aiAnalysisAbortRef.current = controller
    aiReportBufferRef.current = ''
    if (aiReportFlushTimerRef.current !== null) {
      window.clearTimeout(aiReportFlushTimerRef.current)
      aiReportFlushTimerRef.current = null
    }
    setAiAnalysisLoading(true)
    setAiAnalysisError('')
    setAiReportMarkdown('')
    setAiAnalysisMeta(null)

    try {
      const response = await fetch('/api/ai/interpret-factor/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          factor: {
            id: factor.id,
            name: factor.name,
            category: factor.category,
            description: factor.description,
            formula_type: factor.formula_type,
            code: factor.code,
          },
          stock_code: stockCode,
          start_date: dateRange.startDate,
          end_date: dateRange.endDate,
          chart_period: chartPeriod,
          analysis_context: buildAiAnalysisContext(),
        }),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        const errorText = await response.text()
        let detail = errorText || `AI 解读请求失败 (${response.status})`
        try {
          const parsed = JSON.parse(errorText)
          detail = parsed.detail || parsed.message || detail
        } catch {
          // ignore JSON parse error
        }
        throw new Error(detail)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let shouldStop = false

      const processSseEvent = (rawEvent: string) => {
        const dataLine = rawEvent
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('\n')

        if (!dataLine) {
          return
        }

        let eventPayload: any
        try {
          eventPayload = JSON.parse(dataLine)
        } catch {
          return
        }

        if (eventPayload.type === 'metadata') {
          setAiAnalysisMeta({
            model: eventPayload.model,
            requestUrl: eventPayload.request_url,
            factorName: eventPayload.factor_name,
            generatedAt: new Date().toISOString(),
          })
          return
        }

        if (eventPayload.type === 'chunk') {
          enqueueAiReportChunk(eventPayload.content || '')
          return
        }

        if (eventPayload.type === 'error') {
          throw new Error(eventPayload.message || 'AI 解读流式输出失败')
        }

        if (eventPayload.type === 'done') {
          flushAiReportBuffer(true)
          shouldStop = true
        }
      }

      const processBuffer = (isFinal = false) => {
        while (true) {
          const separatorMatch = buffer.match(/\r?\n\r?\n/)
          if (!separatorMatch || separatorMatch.index === undefined) {
            break
          }

          const separatorIndex = separatorMatch.index
          const rawEvent = buffer.slice(0, separatorIndex)
          buffer = buffer.slice(separatorIndex + separatorMatch[0].length)

          if (!rawEvent.trim()) {
            continue
          }

          processSseEvent(rawEvent)
          if (shouldStop) {
            return
          }
        }

        if (isFinal && buffer.trim()) {
          processSseEvent(buffer)
          buffer = ''
        }
      }

      while (!shouldStop) {
        const { value, done } = await reader.read()
        if (done) {
          buffer += decoder.decode()
          processBuffer(true)
          break
        }

        buffer += decoder.decode(value, { stream: true })
        processBuffer()
      }
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        flushAiReportBuffer(true)
        setAiAnalysisError('AI 解读已停止')
        return
      }
      flushAiReportBuffer(true)
      const errorMessage = (error as Error).message || 'AI 解读失败'
      setAiAnalysisError(errorMessage)
      message.error(errorMessage)
    } finally {
      flushAiReportBuffer(true)
      aiAnalysisAbortRef.current = null
      setAiAnalysisLoading(false)
    }
  }, [
    factor,
    analysisData,
    exposureData,
    effectivenessData,
    attributionData,
    monitoringData,
    resolveCurrentDateRange,
    stopAIInterpretation,
    stockCode,
    chartPeriod,
    buildAiAnalysisContext,
    enqueueAiReportChunk,
    flushAiReportBuffer,
  ])

  // 处理股票代码变化
  const handleStockCodeChange = (value: string) => {
    // 移除可能存在的后缀
    const cleanCode = value.replace(/\.(SH|SZ)$/, '')

    // 自动补全后缀
    let fullCode = cleanCode
    if (cleanCode && !cleanCode.includes('.')) {
      if (cleanCode.startsWith('6')) {
        fullCode = cleanCode + '.SH'
      } else if (cleanCode.startsWith('0') || cleanCode.startsWith('3')) {
        fullCode = cleanCode + '.SZ'
      } else {
        fullCode = cleanCode
      }
    }

    setStockCode(fullCode)
    setStockCodeDisplay(cleanCode)
  }

  // 处理自定义日期变化
  const handleCustomDateChange = (dates: any, dateStrings: [string, string]) => {
    if (dates && dates.length === 2) {
      setCustomStartDate(dateStrings[0])
      setCustomEndDate(dateStrings[1])
      setTimeout(() => {
        loadChartData()
      }, 100)
    }
  }

  // 加载行情数据
  const loadChartData = useCallback(async () => {
    if (!factor) return

    let startDate: string
    let endDate: string

    if (chartPeriod === 'custom') {
      if (!customStartDate || !customEndDate) {
        message.warning('请选择自定义日期范围')
        return
      }
      startDate = customStartDate
      endDate = customEndDate
    } else {
      endDate = new Date().toISOString().split('T')[0]
      startDate = getStartDateByPeriod(chartPeriod)
    }

    setLoadingChart(true)
    try {
      const stockResponse = await api.getStockData(stockCode, startDate, endDate) as any

      if (!stockResponse || !stockResponse.data) {
        message.warning('未获取到股票数据')
        return
      }

      const rawData = stockResponse.data
      if (!rawData.data || rawData.data.length === 0) {
        message.warning('股票数据为空')
        return
      }

      const stockData = rawData.data.map((row: any, i: number) => ({
        date: rawData.index[i],
        open: row[rawData.columns.indexOf('open')],
        high: row[rawData.columns.indexOf('high')],
        low: row[rawData.columns.indexOf('low')],
        close: row[rawData.columns.indexOf('close')],
        volume: row[rawData.columns.indexOf('volume')]
      }))

      const factorResponse = await api.calculateFactor({
        factor_name: factor.name,
        stock_codes: [stockCode],
        start_date: startDate,
        end_date: endDate
      } as any) as any

      if (!factorResponse || !factorResponse.success || !factorResponse.data) {
        message.warning('因子计算失败')
        return
      }

      const factorDataMap = factorResponse.data[stockCode]
      if (!factorDataMap) {
        message.warning('因子数据为空')
        return
      }

      const factorData = {
        dates: factorDataMap.dates,
        values: factorDataMap.factor_values,
        plotSeries: Array.isArray(factorDataMap.plot_series)
          ? factorDataMap.plot_series.map((series: any) => ({
            name: series.name,
            dates: Array.isArray(series.dates) ? series.dates : [],
            values: Array.isArray(series.values) ? series.values : [],
            renderType: series.render_type === 'bar' ? 'bar' : 'line'
          }))
          : undefined,
        formulaType: factorDataMap.formula_type
      }

      setChartData({
        stock: stockData,
        factor: factorData
      })
    } catch (error) {
      console.error('加载行情数据失败:', error)
      message.error('加载行情数据失败')
    } finally {
      setLoadingChart(false)
    }
  }, [factor, chartPeriod, stockCode, customStartDate, customEndDate])

  // 加载所有分析 Tab 数据（Tab 2-5）
  const loadAnalysisTabsData = useCallback(async () => {
    if (!factor) return

    let startDate: string
    let endDate: string

    if (chartPeriod === 'custom') {
      if (!customStartDate || !customEndDate) {
        message.warning('请选择自定义日期范围')
        return
      }
      startDate = customStartDate
      endDate = customEndDate
    } else {
      endDate = new Date().toISOString().split('T')[0]
      startDate = getStartDateByPeriod(chartPeriod)
    }

    setLoadingAnalysisTabs(true)
    try {
      // 并行加载所有分析数据
      const [exposure, effectiveness, attribution, monitoring] = await Promise.all([
        api.analyzeExposure({
          factor_name: factor.name,
          stock_codes: [stockCode],
          start_date: startDate,
          end_date: endDate
        } as any),
        api.analyzeEffectiveness({
          factor_name: factor.name,
          stock_codes: [stockCode],
          start_date: startDate,
          end_date: endDate
        } as any),
        api.analyzeAttribution({
          factor_name: factor.name,
          stock_codes: [stockCode],
          start_date: startDate,
          end_date: endDate
        } as any),
        api.analyzeMonitoring({
          factor_name: factor.name,
          stock_codes: [stockCode],
          start_date: startDate,
          end_date: endDate
        } as any)
      ])

      console.log('API responses:', { exposure, effectiveness, attribution, monitoring })

      if (exposure && (exposure as any).success) {
        setExposureData((exposure as any).data)
        console.log('exposureData set:', (exposure as any).data)
      }
      if (effectiveness && (effectiveness as any).success) {
        setEffectivenessData((effectiveness as any).data)
        console.log('effectivenessData set:', (effectiveness as any).data)
      }
      if (attribution && (attribution as any).success) {
        setAttributionData((attribution as any).data)
        console.log('attributionData set:', (attribution as any).data)
      }
      if (monitoring && (monitoring as any).success) {
        setMonitoringData((monitoring as any).data)
        console.log('monitoringData set:', (monitoring as any).data)
      }

      message.success('分析数据加载完成')
    } catch (error) {
      console.error('加载分析数据失败:', error)
      message.error('加载分析数据失败')
    } finally {
      setLoadingAnalysisTabs(false)
    }
  }, [factor, chartPeriod, stockCode, customStartDate, customEndDate])

  // 绘制行情图表
  const drawPriceChart = useCallback(() => {
    const chartDom = priceChartRef.current
    const myChart = initChart(chartDom, 'price')
    if (!myChart || !chartData) return

    // 清除之前的图表配置，防止图表类型切换时出现异常
    myChart.clear()

    const { stock, factor } = chartData

    const displayDates = stock.map((item) => item.date)
    const displayStock = stock
    const primaryFactorValues = alignValuesToDates(displayDates, factor)
    const visualSeries = factor.plotSeries && factor.plotSeries.length > 0
      ? factor.plotSeries
      : [
        {
          name: '因子值',
          dates: factor.dates,
          values: factor.values,
          renderType: 'line' as const
        }
      ]
    const alignedPlotSeries = visualSeries.map((series, index) => ({
      ...series,
      color: getPlotSeriesColor(index),
      alignedValues: alignValuesToDates(displayDates, series)
    }))
    const hasMultiplePlotLines = alignedPlotSeries.length > 1

    const klineData = displayStock.map(d => [d.open, d.close, d.low, d.high])

    // 双轴同图模式
    if (factorChartType === 'overlay') {
      const overlaySeries = [
        {
          type: 'candlestick',
          name: '日K线',
          data: klineData,
          yAxisIndex: 0,
          itemStyle: {
            color: '#ef4444',
            color0: '#22c55e',
            borderColor: '#ef4444',
            borderColor0: '#22c55e'
          }
        },
        ...alignedPlotSeries.map((series) => ({
          type: series.renderType === 'bar' ? 'bar' : 'line',
          name: series.name,
          data: series.alignedValues,
          yAxisIndex: 1,
          smooth: series.renderType !== 'bar',
          showSymbol: false,
          barMaxWidth: series.renderType === 'bar' ? 12 : undefined,
          lineStyle: series.renderType !== 'bar' ? {
            color: series.color,
            width: 2
          } : undefined,
          itemStyle: {
            color: series.color
          }
        }))
      ]

      const option: echarts.EChartsOption = {
        animation: false,
        grid: {
          left: '8%',
          right: '10%',
          top: '10%',
          bottom: '15%'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params: any) => {
            if (!params || params.length === 0) return ''
            const date = params[0].axisValue
            let result = `<div style="font-weight: bold; margin-bottom: 5px;">${date}</div>`

            params.forEach((param: any) => {
              if (param.seriesName === '日K线') {
                const data = param.data
                result += `<div style="margin: 2px 0;">
                  <span style="display: inline-block; width: 10px; height: 10px; background: ${param.color}; border-radius: 50%; margin-right: 5px;"></span>
                  <span style="font-weight: bold;">日K线:</span>
                  开:${data[0]?.toFixed(2)} 收:${data[1]?.toFixed(2)}
                  低:${data[2]?.toFixed(2)} 高:${data[3]?.toFixed(2)}
                </div>`
              } else if (param.value !== null && param.value !== undefined && param.value !== '-') {
                result += `<div style="margin: 2px 0;">
                  <span style="display: inline-block; width: 10px; height: 10px; background: ${param.color}; border-radius: 50%; margin-right: 5px;"></span>
                  <span style="font-weight: bold; color: ${param.color};">${param.seriesName}:</span>
                  ${Number(param.value).toFixed(4)}
                </div>`
              }
            })
            return result
          }
        },
        xAxis: {
          type: 'category',
          data: displayDates,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          }
        },
        yAxis: [
          {
            type: 'value',
            scale: true,
            position: 'left',
            axisLabel: {
              fontSize: 10,
              color: '#64748b'
            },
            splitLine: {
              lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
            }
          },
          {
            type: 'value',
            scale: true,
            position: 'right',
            axisLabel: {
              fontSize: 10,
              color: '#3b82f6'
            },
            splitLine: { show: false }
          }
        ],
        dataZoom: [
          {
            type: 'inside',
            start: 0,
            end: 100
          }
        ],
        series: overlaySeries
      }

      myChart.setOption(option, true)
      return
    }

    // 单轴归一化模式
    if (factorChartType === 'normalized') {
      const baseFactor = primaryFactorValues.find((value) => value !== null && value !== undefined) ?? 1
      const basePrice = displayStock[0]?.close || 1

      const normalizedPrices = displayStock.map((d) =>
        Number(((d.close - basePrice) / basePrice * 100).toFixed(2))
      )
      const factorDenominator = Math.abs(baseFactor) > 1e-12 ? Math.abs(baseFactor) : 1
      const normalizedFactors = primaryFactorValues.map((value) =>
        value === null || value === undefined
          ? null
          : Number((((value - baseFactor) / factorDenominator) * 100).toFixed(2))
      )

      const option: echarts.EChartsOption = {
        animation: false,
        grid: {
          left: '8%',
          right: '8%',
          top: '10%',
          bottom: '15%'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params: any) => {
            if (!params || params.length === 0) return ''
            const date = params[0].axisValue
            let result = `<div style="font-weight: bold; margin-bottom: 5px;">${date}</div>`

            params.forEach((param: any) => {
              if (param.value === null || param.value === undefined || param.value === '-') {
                return
              }
              const value = Number(param.value).toFixed(2)
              result += `<div style="margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px; background: ${param.color}; border-radius: 50%; margin-right: 5px;"></span>
                <span style="font-weight: bold; color: ${param.color === '#ef4444' ? '#ef4444' : '#3b82f6'};">${param.seriesName}:</span>
                ${value}%
              </div>`
            })

            // 添加原始值信息
            const idx = params[0].dataIndex
            result += `<div style="margin-top: 5px; padding-top: 5px; border-top: 1px solid rgba(148, 163, 184, 0.2); font-size: 11px; color: #64748b;">
              原始价格: ${displayStock[idx]?.close?.toFixed(2)}<br/>
              主因子: ${primaryFactorValues[idx] == null ? '-' : Number(primaryFactorValues[idx]).toFixed(4)}
            </div>`

            return result
          }
        },
        xAxis: {
          type: 'category',
          data: displayDates,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          }
        },
        yAxis: {
          type: 'value',
          name: '变化率 (%)',
          axisLabel: {
            formatter: '{value}%'
          },
          axisLine: { lineStyle: { color: '#94a3b8' } },
          splitLine: {
            lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
          }
        },
        dataZoom: [
          {
            type: 'inside',
            start: 0,
            end: 100
          }
        ],
        series: [
          {
            type: 'line',
            name: '价格(归一化)',
            data: normalizedPrices,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              color: '#ef4444',
              width: 2
            },
            itemStyle: {
              color: '#ef4444'
            },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(239, 68, 68, 0.2)' },
                { offset: 1, color: 'rgba(239, 68, 68, 0.02)' }
              ])
            }
          },
          {
            type: 'line',
            name: '主因子(归一化)',
            data: normalizedFactors,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              color: '#3b82f6',
              width: 2
            },
            itemStyle: {
              color: '#3b82f6'
            }
          }
        ]
      }

      myChart.setOption(option)
      return
    }

    // 分屏模式（折线图、柱状图、面积图）
    const splitSeries = [
      {
        type: 'candlestick',
        name: '日K线',
        data: klineData,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: '#ef4444',
          color0: '#22c55e',
          borderColor: '#ef4444',
          borderColor0: '#22c55e'
        }
      },
      ...alignedPlotSeries.map((series) => {
        const seriesType = hasMultiplePlotLines
          ? (series.renderType === 'bar' ? 'bar' : 'line')
          : (factorChartType === 'bar' ? 'bar' : 'line')

        return {
          type: seriesType,
          name: series.name,
          data: series.alignedValues,
          xAxisIndex: 1,
          yAxisIndex: 1,
          smooth: seriesType !== 'bar',
          showSymbol: false,
          barMaxWidth: seriesType === 'bar' ? 12 : undefined,
          lineStyle: seriesType !== 'bar' ? {
            color: series.color,
            width: 2
          } : undefined,
          itemStyle: {
            color: series.color
          },
          areaStyle: !hasMultiplePlotLines && factorChartType === 'area' && seriesType !== 'bar' ? {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
            ])
          } : undefined
        }
      })
    ]

    const option: echarts.EChartsOption = {
      animation: false,
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
        label: {
          backgroundColor: '#777'
        }
      },
      grid: [
        { left: '8%', right: '8%', top: '10%', height: '40%' },
        { left: '8%', right: '8%', top: '60%', height: '30%' }
      ],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const date = params[0].axisValue
          let result = `<div style="font-weight: bold; margin-bottom: 5px;">${date}</div>`

            params.forEach((param: any) => {
              if (param.seriesName === '日K线') {
                const data = param.data
                result += `<div style="margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px; background: ${param.color}; border-radius: 50%; margin-right: 5px;"></span>
                <span style="font-weight: bold;">日K线:</span>
                开:${data[0]?.toFixed(2)} 收:${data[1]?.toFixed(2)}
                低:${data[2]?.toFixed(2)} 高:${data[3]?.toFixed(2)}
              </div>`
            } else if (param.value !== null && param.value !== undefined && param.value !== '-') {
              result += `<div style="margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px; background: ${param.color}; border-radius: 50%; margin-right: 5px;"></span>
                <span style="font-weight: bold; color: ${param.color};">${param.seriesName}:</span>
                ${Number(param.value).toFixed(4)}
              </div>`
            }
          })
          return result
        }
      },
      xAxis: [
        {
          type: 'category',
          data: displayDates,
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          axisLabel: { show: false },
          axisPointer: {
            type: 'shadow',
            z: 100
          }
        },
        {
          type: 'category',
          data: displayDates,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          },
          axisPointer: {
            type: 'shadow',
            z: 100
          }
        }
      ],
      yAxis: [
        {
          type: 'value',
          scale: true,
          gridIndex: 0,
          splitLine: {
            show: true,
            lineStyle: {
              color: 'rgba(148, 163, 184, 0.1)'
            }
          }
        },
        {
          type: 'value',
          scale: true,
          gridIndex: 1,
          splitLine: {
            show: true,
            lineStyle: {
              color: 'rgba(148, 163, 184, 0.1)'
            }
          }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 0,
          end: 100
        }
      ],
      series: splitSeries
    }

    myChart.setOption(option, true)
  }, [chartData, factorChartType])

  // 初始加载
  useEffect(() => {
    loadFactorDetail()
  }, [loadFactorDetail])

  // 页面加载完成后滚动到顶部，并自动加载行情图表
  useEffect(() => {
    if (factor) {
      window.scrollTo({ top: 0, behavior: 'smooth' })
      // 清理旧的行情图表实例，确保因子变化时重新创建干净的图表
      if (chartsRef.current['price']) {
        chartsRef.current['price'].dispose()
        delete chartsRef.current['price']
      }
      // 清空旧的图表数据
      setChartData(null)
      // 延迟加载新的图表数据
      setTimeout(() => {
        loadChartData()
      }, 50)
    }
  }, [factor, loadChartData])

  // 图表绘制
  useEffect(() => {
    if (analysisData) {
      setTimeout(() => {
        drawDistributionChart()
        drawICSeriesChart()
        drawICHistogramChart()
      }, 100)
    }
  }, [analysisData, drawDistributionChart, drawICSeriesChart, drawICHistogramChart])

  // 监控effectivenessData变化（调试用）
  useEffect(() => {
    console.log('effectivenessData changed:', effectivenessData)
  }, [effectivenessData])

  // 分析 Tab 图表绘制（Tab 2-5）
  useEffect(() => {
    if (exposureData) {
      setTimeout(() => {
        drawExposureHistogram()
        drawPercentileTimeSeries()
      }, 100)
    }
    if (effectivenessData) {
      setTimeout(() => {
        drawScatterChart()
        drawICTimeSeriesChart()
        drawEventResponseChart()
        drawDecayCurveChart()
      }, 100)
    }
    if (monitoringData) {
      setTimeout(() => {
        drawRollingBandChart()
        drawTransitionMatrix()
        drawStructuralBreakChart()
        drawSeasonalityChart()
      }, 100)
    }
  }, [exposureData, effectivenessData, monitoringData, drawExposureHistogram, drawPercentileTimeSeries, drawScatterChart, drawICTimeSeriesChart, drawEventResponseChart, drawDecayCurveChart, drawRollingBandChart, drawTransitionMatrix, drawStructuralBreakChart, drawSeasonalityChart])

  // Tab 切换时重新绘制图表（确保图表容器已渲染）
  useEffect(() => {
    const redrawTimer = setTimeout(() => {
      if (activeTabKey === 'exposure' && exposureData) {
        drawExposureHistogram()
        drawPercentileTimeSeries()
      } else if (activeTabKey === 'effectiveness' && effectivenessData) {
        // 更长的延迟确保DOM完全渲染
        setTimeout(() => {
          drawScatterChart()
          drawICTimeSeriesChart()
          drawEventResponseChart()
          drawDecayCurveChart()
        }, 100)
      } else if (activeTabKey === 'attribution' && attributionData) {
        // Tab 4 是纯数据展示，不需要重绘图表
      } else if (activeTabKey === 'monitoring' && monitoringData) {
        drawRollingBandChart()
        drawTransitionMatrix()
        drawStructuralBreakChart()
        drawSeasonalityChart()
      } else if (activeTabKey === 'chart' && chartData) {
        drawPriceChart()
      }
    }, 200) // 增加延迟确保 Tab 切换动画完成

    return () => clearTimeout(redrawTimer)
  }, [activeTabKey, exposureData, monitoringData, effectivenessData, attributionData, chartData, drawExposureHistogram, drawPercentileTimeSeries, drawScatterChart, drawICTimeSeriesChart, drawEventResponseChart, drawDecayCurveChart, drawRollingBandChart, drawTransitionMatrix, drawStructuralBreakChart, drawSeasonalityChart, drawPriceChart])

  // 行情图表数据变化时重绘
  useEffect(() => {
    if (chartData) {
      setTimeout(() => {
        drawPriceChart()
      }, 100)
    }
  }, [chartData, drawPriceChart])

  // 时间范围或图表类型变化时重新绘制
  useEffect(() => {
    if (chartData) {
      setTimeout(() => {
        drawPriceChart()
      }, 50)
    }
  }, [chartPeriod, factorChartType, stockCode, customStartDate, customEndDate, chartData, drawPriceChart])

  // 窗口大小变化时调整图表
  useEffect(() => {
    const handleResize = () => {
      Object.values(chartsRef.current).forEach(chart => {
        chart && chart.resize()
      })
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      Object.values(chartsRef.current).forEach(chart => {
        chart && chart.dispose()
      })
    }
  }, [])

  useEffect(() => {
    return () => {
      if (aiAnalysisAbortRef.current) {
        aiAnalysisAbortRef.current.abort()
        aiAnalysisAbortRef.current = null
      }
    }
  }, [])

  return (
    <div className="factor-detail-container">
      {/* 背景装饰 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      {/* 主内容区域 - 左右分栏布局 */}
      <Row gutter={[24, 24]} className="factor-detail-content">
        {loading ? (
          <Col span={24}>
            <div className="loading-container">
              <Spin size="large" description="加载中..." />
            </div>
          </Col>
        ) : !factor ? (
          <Col span={24}>
            <Card className="empty-card" variant="borderless">
              <p>{id ? '因子不存在或已被删除' : '未指定因子ID'}</p>
              <Button type="primary" onClick={() => navigate('/factor-management')} style={{ marginTop: '16px' }}>
                返回因子列表
              </Button>
            </Card>
          </Col>
        ) : (
          <>
            {/* 左栏 - 基本信息和因子公式 */}
            <Col xs={24} lg={6} className="left-column">
              {/* 基本信息卡片 */}
              <Card className="basic-info-card" variant="borderless">
                <div className="info-header">
                  <h2 className="factor-title">{factor.name}</h2>
                  <Space orientation="vertical" size="small" style={{ width: '100%' }}>
                    <Button
                      icon={<ArrowLeftOutlined />}
                      onClick={() => navigate('/factor-management')}
                      block
                    >
                      返回
                    </Button>
                    <Button
                      icon={<CopyOutlined />}
                      onClick={handleCopyFactor}
                      block
                    >
                      复制
                    </Button>
                    {factor.source === 'user' && (
                      <>
                        <Button
                          icon={<EditOutlined />}
                          onClick={handleEdit}
                          block
                        >
                          编辑
                        </Button>
                        <Button
                          danger
                          icon={<DeleteOutlined />}
                          onClick={handleDeleteFactor}
                          block
                        >
                          删除
                        </Button>
                      </>
                    )}
                  </Space>
                </div>
                <Divider />
                <div className="info-content">
                  <div className="info-item">
                    <span className="label">因子描述:</span>
                    <span className="value">{factor.description || '-'}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">分类标签:</span>
                    <div className="value"><Tag color="blue">{factor.category}</Tag></div>
                  </div>
                  <div className="info-item">
                    <span className="label">来源:</span>
                    <div className="value">
                      <Tag color={factor.source === 'preset' ? 'success' : 'warning'}>
                        {factor.source === 'preset' ? '预置' : '自定义'}
                      </Tag>
                    </div>
                  </div>
                  <div className="info-item">
                    <span className="label">创建时间:</span>
                    <span className="value">{formatDateTime(factor.created_at)}</span>
                  </div>
                  {factor.updated_at && factor.updated_at !== factor.created_at && (
                    <div className="info-item">
                      <span className="label">更新时间:</span>
                      <span className="value">{formatDateTime(factor.updated_at)}</span>
                    </div>
                  )}
                </div>
              </Card>

              {/* 因子公式卡片 */}
              <Card className="formula-card" variant="borderless" style={{ marginTop: '16px' }}>
                <div className="formula-header">
                  <h3>因子公式</h3>
                  {factor.source === 'user' && (
                    <Button
                      type="link"
                      icon={<EditOutlined />}
                      onClick={handleEdit}
                    >
                      编辑
                    </Button>
                  )}
                </div>
                <pre className="formula-code">{factor.code}</pre>
              </Card>
            </Col>

            {/* 右栏 - 数据筛选和分析Tab */}
            <Col xs={24} lg={18} className="right-column">
              {/* 数据筛选卡片 */}
              <Card className="filter-card" variant="borderless">
                <Row gutter={[16, 16]} align="middle">
                  <Col xs={24} sm={8}>
                    <Input
                      placeholder="股票代码"
                      value={stockCodeDisplay}
                      onChange={(e) => handleStockCodeChange(e.target.value)}
                      onPressEnter={analyzeFactor}
                    />
                  </Col>
                  <Col xs={24} sm={8}>
                    <Select
                      value={chartPeriod}
                      onChange={(value) => {
                        setChartPeriod(value)
                        if (value === 'custom') {
                          setShowCustomDatePicker(true)
                        }
                      }}
                      style={{ width: '100%' }}
                    >
                      <Option value="1y">近1年</Option>
                      <Option value="3y">近3年</Option>
                      <Option value="custom">自定义</Option>
                    </Select>
                  </Col>
                  <Col xs={24} sm={8}>
                    {showCustomDatePicker ? (
                      <RangePicker
                        value={customStartDate && customEndDate ? [dayjs(customStartDate), dayjs(customEndDate)] : null}
                        onChange={handleCustomDateChange}
                        format="YYYY-MM-DD"
                        style={{ width: '100%' }}
                      />
                    ) : (
                      <Button
                        type="primary"
                        block
                        icon={<LineChartOutlined />}
                        onClick={analyzeFactor}
                        loading={analyzing}
                      >
                        分析因子
                      </Button>
                    )}
                  </Col>
                </Row>
              </Card>

              {/* 分析Tab页 */}
              <Card className="analysis-tabs-card" variant="borderless" style={{ marginTop: '16px' }}>
                <Tabs
                  activeKey={activeTabKey}
                  onChange={setActiveTabKey}
                  items={[
                    {
                      key: 'chart',
                      label: '行情图表',
                      children: (
                        <>
                          {/* 图表类型选择 */}
                          <div style={{ marginBottom: '16px' }}>
                            <Space>
                              <Select
                                value={factorChartType}
                                onChange={(value) => setFactorChartType(value)}
                                style={{ width: 140 }}
                              >
                                <Option value="overlay">双轴同图</Option>
                                <Option value="normalized">单轴归一化</Option>
                                <Option value="line">折线图</Option>
                                <Option value="bar">柱状图</Option>
                                <Option value="area">面积图</Option>
                              </Select>
                              <Button
                                icon={<ReloadOutlined />}
                                size="small"
                                onClick={loadChartData}
                                loading={loadingChart}
                              >
                                刷新
                              </Button>
                            </Space>
                          </div>
                          <div ref={priceChartRef} className="chart-container large"></div>
                        </>
                      )
                    },
                    {
                      key: 'exposure',
                      label: '因子暴露度',
                      children: (
                        <>
                          {!exposureData ? (
                            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                              <LineChartOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                              <p>{loadingAnalysisTabs ? '加载中...' : '请先点击"分析因子"按钮进行因子分析'}</p>
                            </div>
                          ) : (
                            <>
                              {/* 迷你数据卡片 */}
                              <Card className="stats-card" variant="borderless" style={{ marginBottom: '16px' }}>
                                <Row gutter={[16, 16]}>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="当前因子值"
                                          content="最新一天计算的因子值，反映了当前时点的因子暴露水平。用于判断当前因子处于高值还是低值状态。"
                                        />
                                      }
                                      value={exposureData.current_value ?? '-'}
                                      precision={4}
                                      styles={{ content: { color: '#3b82f6', fontSize: '18px', fontWeight: 'bold' } }}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="分位数"
                                          content="当前因子值在历史分布中的百分位。例如：80%表示当前值高于历史上80%的时间。用于判断因子暴露相对历史的高低水平。"
                                        />
                                      }
                                      value={exposureData.percentile ?? '-'}
                                      precision={1}
                                      suffix="%"
                                      styles={{
                                        content: {
                                          color: (exposureData.percentile ?? 50) > 50 ? '#ef4444' : '#22c55e',
                                          fontSize: '18px',
                                          fontWeight: 'bold'
                                        }
                                      }}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="滚动标准差"
                                          content="过去20个交易日的因子值标准差，反映因子的波动性。标准差越大说明因子值波动越剧烈，稳定性越差。"
                                        />
                                      }
                                      value={exposureData.latest_std ?? '-'}
                                      precision={4}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="变异系数"
                                          content="标准差与均值的比值（无量纲），用于比较不同波动水平。CV越大说明相对波动越大。一般CV<0.3为低波动，0.3-0.7为中等，>0.7为高波动。"
                                        />
                                      }
                                      value={exposureData.cv ?? '-'}
                                      precision={4}
                                    />
                                  </Col>
                                </Row>
                              </Card>

                              {/* 分位数指示器 */}
                              <Card
                                title={
                                  <InfoTooltip
                                    title="因子暴露度分位数"
                                    content="显示当前因子值在历史分布中的位置，用于判断因子暴露相对水平。分位数越低表示因子值越低（低暴露），越高表示因子值越高（高暴露）。"
                                  />
                                }
                                variant="borderless"
                                style={{ marginBottom: '16px' }}
                              >
                                <div style={{ marginBottom: '8px' }}>
                                  <span style={{ fontWeight: 500 }}>当前分位数：</span>
                                  <span style={{ marginLeft: '8px', color: '#64748b' }}>
                                    {exposureData.percentile?.toFixed(1)}%
                                  </span>
                                </div>
                                <div style={{ position: 'relative' }}>
                                  <Progress
                                    percent={exposureData.percentile ?? 0}
                                    status="active"
                                    strokeColor={{
                                      '0%': '#22c55e',
                                      '50%': '#eab308',
                                      '100%': '#ef4444',
                                    }}
                                    showInfo={false}
                                  />
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '12px', color: '#64748b' }}>
                                  <span>低暴露</span>
                                  <span>中等</span>
                                  <span>高暴露</span>
                                </div>
                              </Card>

                              {/* 分位数时间序列曲线 */}
                              <Card
                                title={
                                  <InfoTooltip
                                    title="分位数变化曲线"
                                    content="展示因子值分位数随时间的变化趋势，可以观察因子暴露的周期性和趋势性。适用于判断因子是否处于周期性波动，以及识别极端暴露时期。"
                                  />
                                }
                                variant="borderless"
                                style={{ marginBottom: '16px' }}
                              >
                                <div ref={percentileTimeSeriesRef} className="chart-container" style={{ height: '350px' }}></div>
                              </Card>

                              {/* 历史分布直方图 */}
                              <Card
                                title={
                                  <InfoTooltip
                                    title="历史分布直方图"
                                    content="展示因子值的历史频率分布，帮助判断因子是否符合正态分布、是否存在肥尾效应。用于识别极端值和评估因子的统计特性。"
                                  />
                                }
                                variant="borderless"
                              >
                                <div ref={exposureHistogramRef} className="chart-container" style={{ height: '350px' }}></div>
                              </Card>
                            </>
                          )}
                        </>
                      )
                    },
                    {
                      key: 'effectiveness',
                      label: '因子有效性',
                      children: (
                        <>
                          {!effectivenessData ? (
                            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                              <ExperimentOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                              <p>{loadingAnalysisTabs ? '加载中...' : '请先点击"分析因子"按钮进行因子分析'}</p>
                            </div>
                          ) : (
                            <>
                              {/* IC统计卡片 */}
                              <Card className="stats-card" variant="borderless" style={{ marginBottom: '16px' }}>
                                <Row gutter={[16, 16]}>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="IC均值"
                                          content="Information Coefficient均值，衡量因子值与未来收益的相关性。IC>0表示因子值越高收益越高，IC<0则相反。绝对值>0.03为有效，>0.05为优秀。"
                                        />
                                      }
                                      value={effectivenessData.ic_time_series?.ic_mean ?? '-'}
                                      precision={effectivenessData.ic_time_series?.ic_mean !== undefined ? 4 : undefined}
                                      styles={{
                                        content: {
                                          color: (effectivenessData.ic_time_series?.ic_mean || 0) > 0 ? '#ef4444' : '#22c55e',
                                          fontSize: '20px',
                                          fontWeight: 'bold'
                                        }
                                      }}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="IC标准差"
                                          content="IC的波动性度量，标准差越小IC越稳定。稳定性是因子有效性的重要指标，低标准差意味着因子在不同时期表现一致。"
                                        />
                                      }
                                      value={effectivenessData.ic_time_series?.ic_std ?? '-'}
                                      precision={effectivenessData.ic_time_series?.ic_std !== undefined ? 4 : undefined}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="IR比率"
                                          content="Information Ratio = IC均值/IC标准差，综合衡量因子有效性和稳定性。IR>0.5为良好，>1.0为优秀。IR越高说明因子既有效又稳定。"
                                        />
                                      }
                                      value={effectivenessData.ic_time_series?.ir ?? '-'}
                                      precision={effectivenessData.ic_time_series?.ir !== undefined ? 4 : undefined}
                                      styles={{ content: { color: '#22c55e' } }}
                                    />
                                  </Col>
                                  <Col xs={12} sm={6}>
                                    <Statistic
                                      title={
                                        <InfoTooltip
                                          title="IC>0比例"
                                          content="IC为正的天数占比，衡量因子胜率。>50%说明因子多数时期有效，>60%为优秀。高胜率意味着因子可靠性更强。"
                                        />
                                      }
                                      value={effectivenessData.ic_time_series?.ic_positive_ratio ?? '-'}
                                      precision={effectivenessData.ic_time_series?.ic_positive_ratio !== undefined ? 2 : undefined}
                                      suffix={effectivenessData.ic_time_series?.ic_positive_ratio !== undefined ? '%' : ''}
                                    />
                                  </Col>
                                </Row>
                              </Card>

                              {/* 因子有效性图表 */}
                              <Row gutter={[16, 16]}>
                                <Col xs={24} lg={24}>
                                  <Card
                                    title={
                                      <InfoTooltip
                                        title="因子-收益散点图"
                                        content="横轴为因子值，纵轴为未来收益，散点分布反映因子与收益的相关性。若呈现明显向上（下）趋势，说明正（负）相关。点的分布越集中说明相关性越稳定。用于直观判断因子有效性。"
                                      />
                                    }
                                    variant="borderless"
                                  >
                                    {effectivenessData.scatter_plot?.error ? (
                                      <div style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
                                        {effectivenessData.scatter_plot.error}
                                      </div>
                                    ) : (
                                      <div ref={scatterChartRef} className="chart-container" style={{ height: '400px' }}></div>
                                    )}
                                  </Card>
                                </Col>
                                <Col xs={24} lg={24}>
                                  <Card
                                    title={
                                      <InfoTooltip
                                        title="IC时序分析"
                                        content="展示每日IC值的时间序列，包含IC曲线、零线和±1标准差带。用于识别因子在不同时期的有效性变化，识别失效时期。IC持续在零线以上说明因子稳定有效。"
                                      />
                                    }
                                    variant="borderless"
                                  >
                                    {effectivenessData.ic_time_series?.error ? (
                                      <div style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
                                        {effectivenessData.ic_time_series.error}
                                      </div>
                                    ) : (
                                      <div ref={icTimeSeriesChartRef} className="chart-container" style={{ height: '400px' }}></div>
                                    )}
                                  </Card>
                                </Col>
                                <Col xs={24} lg={24}>
                                  <Card
                                    title={
                                      <InfoTooltip
                                        title="事件响应分析（高/低暴露后收益）"
                                        content="当因子值突破高/低阈值后N天的累计收益分布。用于识别因子极端暴露的预测能力，红色（高暴露）高于绿色（低暴露）说明因子能够预测未来收益。"
                                      />
                                    }
                                    variant="borderless"
                                  >
                                    {effectivenessData.event_response?.error ? (
                                      <div style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
                                        {effectivenessData.event_response.error}
                                      </div>
                                    ) : (
                                      <div ref={eventResponseChartRef} className="chart-container" style={{ height: '400px' }}></div>
                                    )}
                                  </Card>
                                </Col>
                                <Col xs={24} lg={24}>
                                  <Card
                                    title={
                                      <InfoTooltip
                                        title="因子衰减曲线（IC vs 持有期）"
                                        content="展示因子预测能力随持有期延长而衰减的曲线。IC随持有期下降越快，说明因子信息越短暂。用于确定最佳持有周期，通常选择IC衰减前的持仓周期。"
                                      />
                                    }
                                    variant="borderless"
                                  >
                                    {effectivenessData.decay_analysis?.error ? (
                                      <div style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
                                        {effectivenessData.decay_analysis.error}
                                      </div>
                                    ) : (
                                      <div ref={decayCurveChartRef} className="chart-container" style={{ height: '400px' }}></div>
                                    )}
                                  </Card>
                                </Col>
                              </Row>
                            </>
                          )}
                        </>
                      )
                    },
                    {
                      key: 'attribution',
                      label: '因子贡献度分解',
                      children: (
                        <>
                          {!attributionData ? (
                            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                              <FundOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                              <p>{loadingAnalysisTabs ? '加载中...' : '请先点击"分析因子"按钮进行因子分析'}</p>
                            </div>
                          ) : (
                            <Row gutter={[16, 16]}>
                              {/* Alpha-Beta 分解 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="Alpha-Beta 分析"
                                      content="将收益分解为Alpha（超额收益）和Beta（市场风险暴露）。Alpha衡量因子的选股能力，Beta衡量对市场的敏感度。高Alpha说明因子能够获得超额收益，R²>0.7说明拟合良好。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {attributionData.alpha_beta?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {attributionData.alpha_beta.error}
                                    </div>
                                  ) : attributionData.alpha_beta?.has_benchmark === false ? (
                                    <>
                                      <div style={{ marginBottom: '16px', padding: '12px', background: '#fff7ed', borderRadius: '8px', fontSize: '13px', color: '#c2410c' }}>
                                        {attributionData.alpha_beta.message || '未提供基准数据'}
                                      </div>
                                      {attributionData.alpha_beta.portfolio_return && (
                                        <Row gutter={[16, 16]}>
                                          <Col xs={12} sm={6}>
                                            <Statistic
                                              title={
                                                <InfoTooltip
                                                  title="年化收益"
                                                  content="投资组合的年化收益率，衡量投资回报水平。正值表示盈利，负值表示亏损。年化收益>10%为良好，>20%为优秀。"
                                                />
                                              }
                                              value={attributionData.alpha_beta.portfolio_return.annual_return ?? '-'}
                                              precision={4}
                                              suffix="%"
                                              styles={{
                                                content: {
                                                  color: (attributionData.alpha_beta.portfolio_return.annual_return ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                                }
                                              }}
                                            />
                                          </Col>
                                          <Col xs={12} sm={6}>
                                            <Statistic
                                              title={
                                                <InfoTooltip
                                                  title="年化波动率"
                                                  content="收益的标准差，衡量投资风险。波动率越大风险越高。一般<15%为低风险，15-30%为中等，>30%为高风险。"
                                                />
                                              }
                                              value={attributionData.alpha_beta.portfolio_return.volatility ?? '-'}
                                              precision={4}
                                              suffix="%"
                                            />
                                          </Col>
                                          <Col xs={12} sm={6}>
                                            <Statistic
                                              title={
                                                <InfoTooltip
                                                  title="夏普比率"
                                                  content="风险调整后收益指标 = (收益-无风险利率)/波动率。夏普>1为良好，>2为优秀。数值越高说明单位风险的收益越高。"
                                                />
                                              }
                                              value={attributionData.alpha_beta.portfolio_return.sharpe ?? '-'}
                                              precision={4}
                                            />
                                          </Col>
                                          <Col xs={12} sm={6}>
                                            <Statistic
                                              title={
                                                <InfoTooltip
                                                  title="日均收益"
                                                  content="每日平均收益率，反映日常盈利水平。用于评估短期收益能力。"
                                                />
                                              }
                                              value={attributionData.alpha_beta.portfolio_return.daily_mean ?? '-'}
                                              precision={6}
                                            />
                                          </Col>
                                        </Row>
                                      )}
                                    </>
                                  ) : (
                                    <>
                                      <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title={
                                              <InfoTooltip
                                                title="年化 Alpha"
                                                content="超额收益，衡量因子超越市场的能力。Alpha>0说明因子能获得超额收益，>5%为优秀，<0说明跑输市场。"
                                              />
                                            }
                                            value={attributionData.alpha_beta?.alpha ?? '-'}
                                            precision={4}
                                            suffix="%"
                                            styles={{
                                              content: {
                                                color: (attributionData.alpha_beta?.alpha ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                              }
                                            }}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title={
                                              <InfoTooltip
                                                title="Beta"
                                                content="市场风险暴露，衡量因子对市场的敏感度。Beta=1表示与市场同步，>1表示高弹性（涨跌幅大于市场），<1表示低弹性。"
                                              />
                                            }
                                            value={attributionData.alpha_beta?.beta ?? '-'}
                                            precision={4}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title={
                                              <InfoTooltip
                                                title="拟合度 (R²)"
                                                content="模型解释力，0-1之间，越接近1说明市场风险对收益解释越强。R²>0.7为良好拟合，>0.9为优秀拟合。"
                                              />
                                            }
                                            value={attributionData.alpha_beta?.r_squared ?? '-'}
                                            precision={4}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title={
                                              <InfoTooltip
                                                title="日均 Alpha"
                                                content="每日平均超额收益，反映日常超越市场的能力。用于评估短期超额收益稳定性。"
                                              />
                                            }
                                            value={attributionData.alpha_beta?.daily_alpha ?? '-'}
                                            precision={6}
                                          />
                                        </Col>
                                      </Row>
                                      {attributionData.alpha_beta?.interpretation && (
                                        <div style={{ marginTop: '16px', padding: '12px', background: '#f8fafc', borderRadius: '8px', fontSize: '13px', color: '#475569' }}>
                                          {attributionData.alpha_beta.interpretation}
                                        </div>
                                      )}
                                    </>
                                  )}
                                </Card>
                              </Col>

                              {/* 收益分解 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="收益分解"
                                      content="将总收益分解为因子收益和残差收益，展示因子的贡献度。因子收益越高说明因子对总收益贡献越大，残差收益是未被因子解释的部分。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {attributionData.return_decomposition?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {attributionData.return_decomposition.error}
                                    </div>
                                  ) : attributionData.return_decomposition?.overall_stats ? (
                                    <>
                                      <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="年化收益"
                                            value={attributionData.return_decomposition.overall_stats.annual_return ?? '-'}
                                            precision={4}
                                            suffix="%"
                                            styles={{
                                              content: {
                                                color: (attributionData.return_decomposition.overall_stats.annual_return ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                              }
                                            }}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="累计收益"
                                            value={attributionData.return_decomposition.overall_stats.cumulative_return ?? '-'}
                                            precision={4}
                                            suffix="%"
                                            styles={{
                                              content: {
                                                color: (attributionData.return_decomposition.overall_stats.cumulative_return ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                              }
                                            }}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="年化波动率"
                                            value={attributionData.return_decomposition.overall_stats.volatility_annual ?? '-'}
                                            precision={4}
                                            suffix="%"
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="夏普比率"
                                            value={attributionData.return_decomposition.overall_stats.sharpe_ratio ?? '-'}
                                            precision={4}
                                          />
                                        </Col>
                                      </Row>
                                      <Row gutter={[16, 16]}>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="日均收益"
                                            value={attributionData.return_decomposition.overall_stats.avg_daily_return ?? '-'}
                                            precision={6}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="日均波动率"
                                            value={attributionData.return_decomposition.overall_stats.daily_volatility ?? '-'}
                                            precision={6}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="胜率"
                                            value={attributionData.return_decomposition.overall_stats.win_rate ?? '-'}
                                            precision={2}
                                            suffix="%"
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="股票数量"
                                            value={attributionData.return_decomposition.stock_count ?? '-'}
                                          />
                                        </Col>
                                      </Row>
                                    </>
                                  ) : (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
                                      暂无数据
                                    </div>
                                  )}
                                </Card>
                              </Col>

                              {/* 因子收益贡献 */}
                              <Col xs={24} lg={24}>
                                <Card title="因子收益贡献" variant="borderless">
                                  {attributionData.factor_contribution?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {attributionData.factor_contribution.error}
                                    </div>
                                  ) : attributionData.factor_contribution?.ic !== undefined ? (
                                    <>
                                      <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="IC (信息系数)"
                                            value={attributionData.factor_contribution.ic ?? '-'}
                                            precision={4}
                                            styles={{
                                              content: {
                                                color: (attributionData.factor_contribution.ic ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                              }
                                            }}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="IC P值"
                                            value={attributionData.factor_contribution.ic_pvalue ?? '-'}
                                            precision={4}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="高暴露组收益"
                                            value={attributionData.factor_contribution.high_exposure_return ?? '-'}
                                            precision={6}
                                            suffix="%"
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="低暴露组收益"
                                            value={attributionData.factor_contribution.low_exposure_return ?? '-'}
                                            precision={6}
                                            suffix="%"
                                          />
                                        </Col>
                                      </Row>
                                      <Row gutter={[16, 16]}>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="多空收益"
                                            value={attributionData.factor_contribution.long_short_return ?? '-'}
                                            precision={6}
                                            suffix="%"
                                            styles={{
                                              content: {
                                                color: (attributionData.factor_contribution.long_short_return ?? 0) > 0 ? '#ef4444' : '#22c55e'
                                              }
                                            }}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="因子贡献比例"
                                            value={attributionData.factor_contribution.contribution_ratio ?? '-'}
                                            precision={4}
                                            suffix="%"
                                            formatter={(value) => `${((Number(value) || 0) * 100).toFixed(2)}%`}
                                          />
                                        </Col>
                                        <Col xs={12} sm={6}>
                                          <Statistic
                                            title="样本数量"
                                            value={attributionData.factor_contribution.sample_size ?? '-'}
                                          />
                                        </Col>
                                      </Row>
                                    </>
                                  ) : (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
                                      暂无数据
                                    </div>
                                  )}
                                </Card>
                              </Col>
                            </Row>
                          )}
                        </>
                      )
                    },
                    {
                      key: 'monitoring',
                      label: '时间序列动态监测',
                      children: (
                        <>
                          {!monitoringData ? (
                            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                              <LineChartOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                              <p>{loadingAnalysisTabs ? '加载中...' : '请先点击"分析因子"按钮进行因子分析'}</p>
                            </div>
                          ) : (
                            <Row gutter={[16, 16]}>
                              {/* 滚动窗口带状图 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="滚动窗口图（均值 ± 2倍标准差）"
                                      content="展示因子值、滚动均值和置信区间（均值±2倍标准差）。用于识别异常值（超出置信区间）和判断因子稳定性。置信区间窄说明因子波动小，宽说明波动大。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {monitoringData.rolling_chart?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {monitoringData.rolling_chart.error}
                                    </div>
                                  ) : (
                                    <div ref={rollingBandChartRef} className="chart-container" style={{ height: '350px' }}></div>
                                  )}
                                </Card>
                              </Col>

                              {/* 暴露度转移矩阵 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="暴露度转移矩阵（马尔可夫转移概率）"
                                      content="展示因子暴露状态（低/中/高）之间的转移概率，颜色越深表示转移概率越高。用于预测因子未来状态，判断因子的持续性。对角线元素高说明状态稳定。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {monitoringData.transition_matrix?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {monitoringData.transition_matrix.error}
                                    </div>
                                  ) : (
                                    <div ref={transitionMatrixRef} className="chart-container" style={{ height: '350px' }}></div>
                                  )}
                                </Card>
                              </Col>

                              {/* 结构断点检测 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="结构性断点检测"
                                      content="识别因子序列中的结构性变化点（统计显著的位置）。用于发现市场制度变化、因子失效时期。断点越多说明因子稳定性越差，需谨慎使用。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {monitoringData.structural_break?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {monitoringData.structural_break.error}
                                    </div>
                                  ) : (
                                    <>
                                      <div ref={structuralBreakChartRef} className="chart-container" style={{ height: '280px' }}></div>
                                      <div style={{ marginTop: '12px', padding: '12px', background: '#f8fafc', borderRadius: '8px' }}>
                                        <div style={{ fontSize: '13px', color: '#475569', marginBottom: '4px' }}>
                                          <strong>检测方法:</strong> {monitoringData.structural_break?.method || '-'}
                                        </div>
                                        <div style={{ fontSize: '13px', color: '#475569', marginBottom: '4px' }}>
                                          <strong>断点数量:</strong> {monitoringData.structural_break?.num_breaks ?? 0}
                                        </div>
                                        <div style={{ fontSize: '13px', color: '#475569' }}>
                                          {monitoringData.structural_break?.interpretation || '-'}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </Card>
                              </Col>

                              {/* 周期性分析 */}
                              <Col xs={24} lg={24}>
                                <Card
                                  title={
                                    <InfoTooltip
                                      title="周期性分析（FFT 傅里叶变换）"
                                      content="通过傅里叶变换识别因子序列中的周期成分。横轴为周期（天数），纵轴为功率（强度）。用于发现因子的季节性规律，如月度效应、季度效应等。峰值越突出说明周期性越明显。"
                                    />
                                  }
                                  variant="borderless"
                                >
                                  {monitoringData.seasonality?.error ? (
                                    <div style={{ textAlign: 'center', padding: '20px', color: '#ef4444' }}>
                                      {monitoringData.seasonality.error}
                                    </div>
                                  ) : (
                                    <>
                                      <div ref={seasonalityChartRef} className="chart-container" style={{ height: '300px' }}></div>
                                      {monitoringData.seasonality?.dominant_periods && monitoringData.seasonality.dominant_periods.length > 0 && (
                                        <div style={{ marginTop: '12px' }}>
                                          <div style={{ fontSize: '13px', fontWeight: 500, marginBottom: '8px', color: '#0f172a' }}>
                                            主要周期成分:
                                          </div>
                                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                            {monitoringData.seasonality.dominant_periods.slice(0, 5).map((period: any, idx: number) => (
                                              <Tag key={idx} color="blue" style={{ fontSize: '12px' }}>
                                                {period.period_days?.toFixed(1)} 天 (功率: {period.power?.toFixed(2)})
                                              </Tag>
                                            ))}
                                          </div>
                                        </div>
                                      )}
                                    </>
                                  )}
                                </Card>
                              </Col>
                            </Row>
                          )}
                        </>
                      )
                    },
                    {
                      key: 'ai',
                      label: 'AI分析',
                      children: (
                        <Row gutter={[16, 16]}>
                          <Col xs={24}>
                            <Card
                              title="AI 解读"
                              variant="borderless"
                              extra={(
                                <Space wrap>
                                  {aiAnalysisLoading ? (
                                    <Button icon={<StopOutlined />} onClick={stopAIInterpretation}>
                                      停止生成
                                    </Button>
                                  ) : (
                                    <Button type="primary" icon={<RobotOutlined />} onClick={startAIInterpretation}>
                                      AI解读
                                    </Button>
                                  )}
                                  <Button
                                    icon={<DownloadOutlined />}
                                    onClick={exportAIReport}
                                    disabled={!aiReportMarkdown.trim()}
                                  >
                                    导出 MD 报告
                                  </Button>
                                </Space>
                              )}
                            >
                              {!analysisData && !exposureData && !effectivenessData && !attributionData && !monitoringData ? (
                                <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                                  <RobotOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                                  <p>请先点击“分析因子”获取分析结果，再进入 AI 分析。</p>
                                </div>
                              ) : (
                                <>
                                  <Alert
                                    type="info"
                                    showIcon
                                    style={{ marginBottom: 16 }}
                                    message="AI 将基于当前因子背景、所选股票与已加载的分析结果生成研究报告"
                                    description="报告以 Markdown 实时流式输出，可直接导出为 .md 文件。"
                                  />

                                  {aiAnalysisMeta && (
                                    <div style={{ marginBottom: 12, display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                      <Tag color="blue">{aiAnalysisMeta.model || '已连接模型'}</Tag>
                                      <Tag color="cyan">{aiAnalysisMeta.factorName || factor?.name}</Tag>
                                      <Tag color="processing">{stockCode}</Tag>
                                      {aiAnalysisMeta.generatedAt && (
                                        <Tag color="default">{dayjs(aiAnalysisMeta.generatedAt).format('YYYY-MM-DD HH:mm:ss')}</Tag>
                                      )}
                                    </div>
                                  )}

                                  {aiAnalysisLoading && (
                                    <div style={{ marginBottom: 16 }}>
                                      <Spin size="small" />
                                      <Typography.Text style={{ marginLeft: 8, color: '#475569' }}>
                                        正在接收模型流式输出...
                                      </Typography.Text>
                                    </div>
                                  )}

                                  {aiAnalysisError && (
                                    <Alert
                                      type="error"
                                      showIcon
                                      style={{ marginBottom: 16 }}
                                      message={aiAnalysisError}
                                    />
                                  )}

                                  {aiReportMarkdown.trim() ? (
                                    <div className="ai-analysis-report">
                                      <MarkdownReportViewer content={aiReportMarkdown} />
                                    </div>
                                  ) : (
                                    <div style={{ textAlign: 'center', padding: '48px 20px', color: '#64748b' }}>
                                      {aiAnalysisLoading ? '模型正在生成报告，请稍候...' : '点击 “AI解读” 开始生成因子研究报告'}
                                    </div>
                                  )}
                                </>
                              )}
                            </Card>
                          </Col>
                        </Row>
                      )
                    }
                  ]}
                />
              </Card>
            </Col>
          </>
        )}
      </Row>

      {/* 编辑因子弹窗 */}
      <Modal
        title="编辑因子"
        open={editing}
        onCancel={handleCancelEdit}
        footer={null}
        width={600}
        destroyOnHidden
      >
        <Form layout="vertical">
          <Form.Item
            label="因子名称"
            required
            style={{ marginBottom: 16 }}
          >
            <Input
              value={editForm.name}
              onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
              placeholder="例如：RSI指标"
            />
          </Form.Item>

          <Form.Item
            label="分类"
            required
            style={{ marginBottom: 16 }}
          >
            <Select
              value={editForm.category}
              onChange={(value) => setEditForm({ ...editForm, category: value })}
              placeholder="请选择"
            >
              <Option value="技术指标">技术指标</Option>
              <Option value="价格动量">价格动量</Option>
              <Option value="成交量">成交量</Option>
              <Option value="波动率">波动率</Option>
              <Option value="自定义">自定义</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="说明"
            style={{ marginBottom: 16 }}
          >
            <Input.TextArea
              value={editForm.description}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              placeholder="简要描述因子的含义和用途"
              rows={3}
            />
          </Form.Item>

          <Form.Item
            label="公式类型"
            style={{ marginBottom: 16 }}
          >
            <Select
              value={normalizeFormulaType(editForm.formula_type)}
              onChange={(value) => setEditForm({ ...editForm, formula_type: normalizeFormulaType(value) })}
            >
              {FORMULA_TYPE_OPTIONS.map((option) => (
                <Option key={option.value} value={option.value}>{option.label}</Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            label={
              <Space>
                <span>{FORMULA_FIELD_LABEL}</span>
                <Tooltip title={getFormulaHelpContent(normalizeFormulaType(editForm.formula_type))} placement="right" styles={{ root: { maxWidth: '600px' } }}>
                  <QuestionCircleOutlined style={{ color: '#1890ff', cursor: 'help' }} />
                </Tooltip>
              </Space>
            }
            required
            style={{ marginBottom: 16 }}
          >
            <CodeTextArea
              value={editForm.code}
              onChange={(e) => setEditForm({ ...editForm, code: e.target.value })}
              placeholder={getFormulaPlaceholder(normalizeFormulaType(editForm.formula_type))}
              rows={6}
              className="font-mono"
            />
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Button type="primary" onClick={handleSaveEdit} style={{ flex: 1 }}>
                保存修改
              </Button>
              <Button onClick={handleValidateFormula}>
                验证表达式
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default FactorDetail
