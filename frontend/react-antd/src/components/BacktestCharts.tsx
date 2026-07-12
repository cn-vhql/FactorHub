import React, { useRef, useEffect, useState } from 'react'
import * as echarts from '@/utils/echarts'
import { Tabs, Spin, Empty, Radio, Space } from 'antd'

interface ChartData {
  dates: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
}

interface SingleFactor {
  name: string
  values: number[]
}

interface FactorData {
  dates: string[]
  // 新格式：多因子
  factors?: SingleFactor[]
  // 旧格式：单因子（向后兼容）
  name?: string
  values?: number[]
}

interface SignalTypeData {
  buy: {
    dates: string[]
    prices: number[]
  }
  sell: {
    dates: string[]
    prices: number[]
  }
}

type SignalData =
  | {
      strategy: SignalTypeData  // 策略信号（所有满足条件的信号）
      actual: SignalTypeData    // 实际交易信号（VectorBT执行的交易）
    }
  | SignalTypeData

interface EquityData {
  dates: string[]
  values: number[]
}

interface BacktestChartData {
  kline: ChartData
  factor: FactorData
  signals: SignalData
  equity: EquityData
}

interface BacktestChartsProps {
  data: Record<string, BacktestChartData>
  loading?: boolean
}

type SignalDisplayType = 'actual' | 'strategy'

const isSignalBucketData = (
  signals: SignalData
): signals is { strategy: SignalTypeData; actual: SignalTypeData } => {
  return 'strategy' in signals && 'actual' in signals
}

const BacktestCharts: React.FC<BacktestChartsProps> = ({ data, loading = false }) => {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)
  const [activeStock, setActiveStock] = useState<string>(
    Object.keys(data)[0] || ''
  )
  const [signalDisplayType, setSignalDisplayType] = useState<SignalDisplayType>('actual')

  useEffect(() => {
    if (!chartRef.current) return

    // 初始化图表
    chartInstance.current = echarts.init(chartRef.current)

    // 响应式
    const handleResize = () => {
      chartInstance.current?.resize()
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartInstance.current) {
        chartInstance.current.dispose()
      }
    }
  }, [])

  useEffect(() => {
    if (!chartInstance.current || !data[activeStock]) return

    renderCharts(data[activeStock])
  }, [data, activeStock, signalDisplayType])

  const renderCharts = (chartData: BacktestChartData) => {
    if (!chartInstance.current) return

    console.log("Rendering charts with data:", chartData)

    const { kline, factor, signals, equity } = chartData

    // 向后兼容：处理旧数据格式（只有单一信号类型）
    const normalizedSignals = isSignalBucketData(signals)
      ? signals
      : {
          strategy: signals,
          actual: signals
        }
    if (!isSignalBucketData(signals)) {
      // 旧格式：signals直接包含buy/sell
    }

    // 处理因子数据：支持新旧两种格式
    let factorsList: SingleFactor[] = []
    if (factor.factors && factor.factors.length > 0) {
      // 新格式：多因子
      factorsList = factor.factors
    } else if (factor.name && factor.values) {
      // 旧格式：单因子（向后兼容）
      factorsList = [{ name: factor.name, values: factor.values }]
    }

    // 归一化因子数据（除以第一个值）
    const normalizedFactors = factorsList.map(factor => {
      // 找到第一个有效且非零的值作为基准
      const firstValidValue = factor.values.find(v =>
        v !== null &&
        v !== undefined &&
        !isNaN(v) &&
        v !== 0
      )

      if (firstValidValue !== undefined) {
        return {
          name: factor.name,
          values: factor.values.map(v => {
            // 只对有效值进行归一化
            if (v !== null && v !== undefined && !isNaN(v)) {
              return v / firstValidValue
            }
            return null
          })
        }
      }

      // 如果找不到合适的基准值，返回原始数据
      return { name: factor.name, values: factor.values }
    })

    // 为每个因子分配颜色
    const factorColors = [
      '#3b82f6', // 蓝色
      '#10b981', // 绿色
      '#f59e0b', // 橙色
      '#ef4444', // 红色
      '#8b5cf6', // 紫色
      '#06b6d4', // 青色
      '#ec4899', // 粉色
      '#84cc16', // 黄绿色
    ]

    // 根据选择的信号类型获取对应的信号数据
    const displaySignals = signalDisplayType === 'actual' ? normalizedSignals.actual : normalizedSignals.strategy

    // 创建日期到索引的映射，提高查找效率
    const dateToIndexMap = new Map<string, number>()
    kline.dates.forEach((date, index) => {
      dateToIndexMap.set(date, index)
    })

    // 计算回撤
    let max = -Infinity
    const drawdowns = equity.values.map((v) => {
      max = Math.max(max, v)
      return ((v - max) / max) * 100
    })

    const option = {
      animation: false,
      legend: {
        show: normalizedFactors.length > 1,
        data: normalizedFactors.map(f => f.name),
        top: 0,
        left: 'center',
        itemWidth: 20,
        itemHeight: 10,
        textStyle: {
          fontSize: 11,
          color: '#64748b'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          animation: false,
          link: [{ xAxisIndex: 'all' }]
        },
        formatter: (params: any) => {
          if (!params || params.length === 0) return ''
          const date = params[0].axisValue
          let result = `<div style="font-weight: bold; margin-bottom: 5px;">${date}</div>`

          params.forEach((item: any) => {
            if (item.seriesName && item.value) {
              let valueDisplay = item.value
              if (typeof item.value === 'number') {
                if (item.seriesName === '净值') {
                  valueDisplay = (item.value / 10000).toFixed(2) + '万'
                } else if (item.seriesName === '回撤') {
                  valueDisplay = item.value.toFixed(2) + '%'
                } else {
                  valueDisplay = item.value.toFixed(2)
                }
              } else if (Array.isArray(item.value)) {
                if (item.seriesName === '日K线') {
                  valueDisplay = `开:${item.value[0]?.toFixed(2)} 收:${item.value[1]?.toFixed(2)} 低:${item.value[2]?.toFixed(2)} 高:${item.value[3]?.toFixed(2)}`
                }
              }

              result += `<div style="margin: 2px 0;">
                  <span style="display: inline-block; width: 10px; height: 10px; background: ${item.color}; border-radius: 50%; margin-right: 5px;"></span>
                  <span style="font-weight: bold;">${item.seriesName}:</span>
                  ${valueDisplay}
                </div>`
            }
          })
          return result
        }
      },
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
        label: {
          backgroundColor: '#777'
        }
      },
      grid: [
        // 多因子时需要更多顶部空间给图例
        { left: '8%', right: '8%', top: normalizedFactors.length > 1 ? '10%' : '6%', height: '24%' },
        { left: '8%', right: '8%', top: normalizedFactors.length > 1 ? '36%' : '32%', height: '18%' },
        { left: '8%', right: '8%', top: normalizedFactors.length > 1 ? '56%' : '52%', height: '18%' },
        { left: '8%', right: '8%', top: normalizedFactors.length > 1 ? '76%' : '72%', height: '18%' }
      ],
      xAxis: [
        {
          type: 'category',
          data: kline.dates,
          boundaryGap: false,
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          splitLine: { show: true, lineStyle: { color: 'rgba(148, 163, 184, 0.1)' } },
          axisLabel: { show: false },
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: { type: 'shadow', z: 100 }
        },
        {
          type: 'category',
          data: kline.dates,
          boundaryGap: false,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          splitLine: { show: true, lineStyle: { color: 'rgba(148, 163, 184, 0.1)' } },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          },
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: { type: 'shadow', z: 100 }
        },
        {
          type: 'category',
          data: kline.dates,
          boundaryGap: false,
          gridIndex: 2,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          splitLine: { show: true, lineStyle: { color: 'rgba(148, 163, 184, 0.1)' } },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          },
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: { type: 'shadow', z: 100 }
        },
        {
          type: 'category',
          data: kline.dates,
          boundaryGap: false,
          gridIndex: 3,
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { show: false },
          splitLine: { show: true, lineStyle: { color: 'rgba(148, 163, 184, 0.1)' } },
          axisLabel: {
            fontSize: 10,
            color: '#64748b'
          },
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: { type: 'shadow', z: 100 }
        }
      ],
      yAxis: [
        {
          type: 'value',
          scale: true,
          gridIndex: 0,
          splitLine: {
            show: true,
            lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { lineStyle: { color: '#94a3b8' } },
          axisLabel: {
            color: '#64748b',
            fontSize: 10
          },
          name: '价格',
          nameTextStyle: {
            color: '#6b7280',
            fontSize: 11,
            fontWeight: 500
          }
        },
        {
          type: 'value',
          scale: true,
          gridIndex: 1,
          splitLine: {
            show: true,
            lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { lineStyle: { color: '#94a3b8' } },
          axisLabel: {
            color: '#64748b',
            fontSize: 10,
            formatter: (value: number) => value.toFixed(2)
          },
          name: '因子',
          nameTextStyle: {
            color: '#6b7280',
            fontSize: 11,
            fontWeight: 500
          }
        },
        {
          type: 'value',
          scale: true,
          gridIndex: 2,
          splitLine: {
            show: true,
            lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { lineStyle: { color: '#94a3b8' } },
          axisLabel: {
            color: '#64748b',
            fontSize: 10,
            formatter: (value: number) => (value / 10000).toFixed(1) + '万'
          },
          name: '净值',
          nameTextStyle: {
            color: '#6b7280',
            fontSize: 11,
            fontWeight: 500
          }
        },
        {
          type: 'value',
          scale: true,
          gridIndex: 3,
          splitLine: {
            show: true,
            lineStyle: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          axisLine: { lineStyle: { color: '#94a3b8' } },
          axisTick: { lineStyle: { color: '#94a3b8' } },
          axisLabel: {
            color: '#64748b',
            fontSize: 10,
            formatter: '{value}%'
          },
          name: '回撤',
          nameTextStyle: {
            color: '#6b7280',
            fontSize: 11,
            fontWeight: 500
          }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1, 2, 3],
          start: 0,
          end: 100
        }
      ],
      graphic: [
        // 图表分割线
        {
          type: 'line',
          left: 'center',
          top: '31%',
          shape: {
            x1: 0,
            y1: 0,
            x2: '90%',
            y2: 0
          },
          style: {
            stroke: 'rgba(148, 163, 184, 0.1)',
            lineWidth: 1
          },
          z: 10
        },
        {
          type: 'line',
          left: 'center',
          top: '51%',
          shape: {
            x1: 0,
            y1: 0,
            x2: '90%',
            y2: 0
          },
          style: {
            stroke: 'rgba(148, 163, 184, 0.1)',
            lineWidth: 1
          },
          z: 10
        },
        {
          type: 'line',
          left: 'center',
          top: '71%',
          shape: {
            x1: 0,
            y1: 0,
            x2: '90%',
            y2: 0
          },
          style: {
            stroke: 'rgba(148, 163, 184, 0.1)',
            lineWidth: 1
          },
          z: 10
        }
      ],
      series: [
        // 1. K线图
        {
          name: '日K线',
          type: 'candlestick',
          data: kline.dates.map((_, i) => [
            kline.open[i],
            kline.close[i],
            kline.low[i],
            kline.high[i]
          ]),
          xAxisIndex: 0,
          yAxisIndex: 0,
          itemStyle: {
            color: '#ef4444',
            color0: '#22c55e',
            borderColor: '#ef4444',
            borderColor0: '#22c55e',
            borderWidth: 1
          }
        },
        // 买入信号
        {
          name: '买入',
          type: 'scatter',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: displaySignals.buy.dates
            .map((date, i) => {
              const index = dateToIndexMap.get(date)
              // 只添加在K线数据中能找到的点
              if (index !== undefined) {
                return [index, displaySignals.buy.prices[i]]
              }
              return null
            })
            .filter(point => point !== null),
          symbol: 'circle',
          symbolSize: 10,
          itemStyle: {
            color: '#ef4444',
            borderColor: '#fff',
            borderWidth: 2,
            shadowColor: 'rgba(239, 68, 68, 0.5)',
            shadowBlur: 8
          },
          label: {
            show: true,
            formatter: '买',
            position: 'top',
            color: '#ef4444',
            fontSize: 12,
            fontWeight: 700,
            backgroundColor: 'rgba(255,255,255,0.9)',
            padding: [2, 4],
            borderRadius: 3,
            borderColor: '#ef4444',
            borderWidth: 1
          }
        },
        // 卖出信号
        {
          name: '卖出',
          type: 'scatter',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: displaySignals.sell.dates
            .map((date, i) => {
              const index = dateToIndexMap.get(date)
              // 只添加在K线数据中能找到的点
              if (index !== undefined) {
                return [index, displaySignals.sell.prices[i]]
              }
              return null
            })
            .filter(point => point !== null),
          symbol: 'circle',
          symbolSize: 10,
          itemStyle: {
            color: '#10b981',
            borderColor: '#fff',
            borderWidth: 2,
            shadowColor: 'rgba(16, 185, 129, 0.5)',
            shadowBlur: 8
          },
          label: {
            show: true,
            formatter: '卖',
            position: 'bottom',
            color: '#10b981',
            fontSize: 12,
            fontWeight: 700,
            backgroundColor: 'rgba(255,255,255,0.9)',
            padding: [2, 4],
            borderRadius: 3,
            borderColor: '#10b981',
            borderWidth: 1
          }
        },
        // 2. 因子图（支持多因子）
        ...normalizedFactors.map((factor, index) => ({
          name: factor.name,
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: factor.values,
          smooth: true,
          showSymbol: false,
          itemStyle: { color: factorColors[index % factorColors.length] },
          lineStyle: { width: 2 },
          areaStyle: index === 0 ? {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: `rgba(59, 130, 246, 0.3)` },
                { offset: 1, color: `rgba(59, 130, 246, 0.05)` }
              ]
            }
          } : undefined
        })),
        // 3. 净值曲线
        {
          name: '净值',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: equity.values,
          smooth: true,
          showSymbol: false,
          itemStyle: { color: '#8b5cf6' },
          lineStyle: { width: 2 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(139, 92, 246, 0.3)' },
                { offset: 1, color: 'rgba(139, 92, 246, 0.05)' }
              ]
            }
          }
        },
        // 4. 回撤曲线
        {
          name: '回撤',
          type: 'line',
          xAxisIndex: 3,
          yAxisIndex: 3,
          data: drawdowns,
          smooth: true,
          showSymbol: false,
          itemStyle: { color: '#f59e0b' },
          lineStyle: { width: 2 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(245, 158, 11, 0.3)' },
                { offset: 1, color: 'rgba(245, 158, 11, 0.05)' }
              ]
            }
          }
        }
      ]
    }

    chartInstance.current.setOption(option, true)
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 0' }}>
        <Spin size="large" />
        <p style={{ color: '#64748b', marginTop: '16px' }}>加载图表数据...</p>
      </div>
    )
  }

  if (!data || Object.keys(data).length === 0) {
    return (
      <Empty
        description="暂无图表数据"
        style={{ padding: '60px 0' }}
      />
    )
  }

  const stockItems = Object.keys(data).map(stockCode => ({
    key: stockCode,
    label: stockCode
  }))

  return (
    <div>
      {/* 多股票切换 */}
      {Object.keys(data).length > 1 && (
        <Tabs
          activeKey={activeStock}
          onChange={setActiveStock}
          items={stockItems}
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* 信号类型切换 */}
      <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Space>
          <span style={{ color: '#64748b', fontSize: '14px' }}>信号类型：</span>
          <Radio.Group
            value={signalDisplayType}
            onChange={(e) => setSignalDisplayType(e.target.value)}
            buttonStyle="solid"
          >
            <Radio.Button value="actual">实际交易信号</Radio.Button>
            <Radio.Button value="strategy">策略信号</Radio.Button>
          </Radio.Group>
        </Space>
        <div style={{ color: '#94a3b8', fontSize: '12px' }}>
          {signalDisplayType === 'actual'
            ? '显示VectorBT实际执行的交易（不重复增仓）'
            : '显示所有满足策略条件的信号点'}
        </div>
      </div>

      {/* 图表容器 */}
      <div
        ref={chartRef}
        style={{
          width: '100%',
          height: '850px',
          background: 'rgba(255, 255, 255, 0.5)',
          borderRadius: '12px',
          padding: '12px'
        }}
      />
    </div>
  )
}

export default BacktestCharts
