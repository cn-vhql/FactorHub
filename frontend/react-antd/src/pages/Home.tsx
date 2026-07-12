import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Row, Col, Card, Badge, Button, Space } from 'antd'
import {
  ArrowUpOutlined,
  RocketOutlined,
  FundOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  AppstoreOutlined,
  CloudServerOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons'
import { api } from '@/services/api'
import AiModelConfigModal from '@/components/AiModelConfigModal'
import './Home.css'

interface Stats {
  totalCount: number
  presetCount: number
  userCount: number
  strategyCount: number
  stockCacheCount: number
}

interface SystemHealth {
  backendConnected: boolean
  akshareHealthy: boolean
  lastCheck: string
}

const Home: React.FC = () => {
  const navigate = useNavigate()
  const [, setLastUpdate] = useState(new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }))
  const [stats, setStats] = useState<Stats>({
    totalCount: 0,
    presetCount: 0,
    userCount: 0,
    strategyCount: 0,
    stockCacheCount: 0
  })
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    backendConnected: false,
    akshareHealthy: false,
    lastCheck: '检查中...'
  })
  const [modelConfigVisible, setModelConfigVisible] = useState(false)

  const modules = [
    {
      id: 1,
      title: '因子管理',
      description: '创建、编辑和分析量化因子，支持技术指标、基本面因子等多种类型',
      category: '因子工具',
      complexity: '中级',
      status: '可用',
      statusType: 'success' as const,
      url: '/factor-management',
      icon: <DatabaseOutlined />,
      gradient: 'from-blue-500 to-cyan-400',
      bgColor: 'rgba(59, 130, 246, 0.12)'
    },
    {
      id: 2,
      title: '因子挖掘',
      description: '基于遗传算法自动发现Alpha因子，从历史数据中挖掘超额收益来源',
      category: '智能算法',
      complexity: '高级',
      status: '可用',
      statusType: 'success' as const,
      url: '/factor-mining',
      icon: <ExperimentOutlined />,
      gradient: 'from-cyan-500 to-blue-500',
      bgColor: 'rgba(6, 182, 212, 0.12)'
    },
    {
      id: 3,
      title: '组合分析',
      description: '多因子组合构建与优化，计算风险调整后收益，评估因子有效性',
      category: '组合管理',
      complexity: '中级',
      status: '可用',
      statusType: 'success' as const,
      url: '/portfolio-analysis',
      icon: <FundOutlined />,
      gradient: 'from-indigo-500 to-blue-500',
      bgColor: 'rgba(99, 102, 241, 0.12)'
    },
    {
      id: 4,
      title: '策略回测',
      description: '完整的历史数据回测框架，提供收益曲线、最大回撤、夏普比率等指标',
      category: '绩效评估',
      complexity: '高级',
      status: '可用',
      statusType: 'success' as const,
      url: '/backtesting',
      icon: <LineChartOutlined />,
      gradient: 'from-violet-500 to-cyan-500',
      bgColor: 'rgba(139, 92, 246, 0.12)'
    }
  ]

  useEffect(() => {
    loadStats()
    const interval = setInterval(() => {
      setLastUpdate(new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }))
    }, 60000)

    return () => clearInterval(interval)
  }, [])

  const loadStats = async () => {
    try {
      const response: any = await api.getFactorStats()
      if (response.success) {
        setStats({
          totalCount: response.data.total_count || 0,
          presetCount: response.data.preset_count || 0,
          userCount: response.data.user_count || 0,
          strategyCount: response.data.strategy_count || 0,
          stockCacheCount: response.data.stock_cache_count || 0
        })
        setSystemHealth({
          backendConnected: true,
          akshareHealthy: response.data.akshare_healthy || false,
          lastCheck: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
        })
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
      setSystemHealth({
        backendConnected: false,
        akshareHealthy: false,
        lastCheck: '连接失败'
      })
    }
  }

  const handleModuleClick = (url: string) => {
    navigate(url)
  }

  return (
    <div className="home-container">
      {/* 背景装饰 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>
      <div className="bg-orbs">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      {/* 欢迎横幅 */}
      <div className="welcome-banner">
        <div className="banner-content">
          <h1 className="banner-title">
            <span className="title-gradient">FactorHub</span>
          </h1>
          <p className="banner-subtitle">专业的量化因子分析与策略回测平台</p>
        </div>
      </div>

      {/* 快速开始卡片 */}
      <div className="stats-section">
        <Row gutter={[20, 20]}>
          <Col xs={24}>
            <Card className="quick-start-card" variant="borderless">
              <Row gutter={[40, 40]} align="middle">
                <Col xs={24} md={12}>
                  <div className="quick-start-content">
                    <div className="quick-start-badge">
                      <RocketOutlined />
                      <span>快速开始</span>
                    </div>
                    <h2 className="quick-start-title">开始您的量化研究之旅</h2>
                    <p className="quick-start-desc">
                      创建因子、启动挖掘任务或运行回测，几分钟内即可获得专业的量化分析结果
                    </p>
                    <div className="button-row">
                      <Space size="middle" wrap>
                        <Button
                          type="primary"
                          size="large"
                          icon={<PlusOutlined />}
                          onClick={() => navigate('/factor-management')}
                        >
                          新建因子
                        </Button>
                        <Button
                          size="large"
                          icon={<ExperimentOutlined />}
                          onClick={() => navigate('/factor-mining')}
                        >
                          启动挖掘
                        </Button>
                        <Button
                          size="large"
                          icon={<ApiOutlined />}
                          onClick={() => setModelConfigVisible(true)}
                        >
                          配置模型
                        </Button>
                      </Space>
                    </div>
                  </div>
                </Col>
                <Col xs={24} md={12}>
                  <div className="quick-start-stats">
                    <Row gutter={[20, 20]}>
                      <Col xs={12} sm={6}>
                        <div className="mini-stat">
                          <DatabaseOutlined />
                          <div className="mini-stat-value">{stats.totalCount}</div>
                          <div className="mini-stat-label">因子总数</div>
                        </div>
                      </Col>
                      <Col xs={12} sm={6}>
                        <div className="mini-stat">
                          <AppstoreOutlined />
                          <div className="mini-stat-value">{stats.userCount}</div>
                          <div className="mini-stat-label">自定义因子</div>
                        </div>
                      </Col>
                      <Col xs={12} sm={6}>
                        <div className="mini-stat">
                          <FundOutlined />
                          <div className="mini-stat-value">{stats.strategyCount}</div>
                          <div className="mini-stat-label">策略数量</div>
                        </div>
                      </Col>
                      <Col xs={12} sm={6}>
                        <div className="mini-stat">
                          <CloudServerOutlined />
                          <div className="mini-stat-value">{stats.stockCacheCount}</div>
                          <div className="mini-stat-label">股票缓存</div>
                        </div>
                      </Col>
                    </Row>

                    {/* 系统状态 */}
                    <div className="health-status">
                      <div className="health-item">
                        {systemHealth.backendConnected ? (
                          <CheckCircleOutlined style={{ color: '#10b981', fontSize: '14px' }} />
                        ) : (
                          <CloseCircleOutlined style={{ color: '#ef4444', fontSize: '14px' }} />
                        )}
                        <span className="health-label">后端</span>
                        <span className="health-status-text">
                          {systemHealth.backendConnected ? '已连接' : '未连接'}
                        </span>
                      </div>
                      <div className="health-item">
                        {systemHealth.akshareHealthy ? (
                          <CheckCircleOutlined style={{ color: '#10b981', fontSize: '14px' }} />
                        ) : (
                          <CloseCircleOutlined style={{ color: '#ef4444', fontSize: '14px' }} />
                        )}
                        <span className="health-label">AKShare</span>
                        <span className="health-status-text">
                          {systemHealth.akshareHealthy ? '正常' : '异常'}
                        </span>
                      </div>
                      <div className="health-item">
                        <ClockCircleOutlined style={{ color: '#64748b', fontSize: '14px' }} />
                        <span className="health-time">{systemHealth.lastCheck}</span>
                      </div>
                      <Button
                        type="text"
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={loadStats}
                        style={{
                          color: '#64748b',
                          fontSize: '12px',
                          padding: '2px 8px',
                          height: 'auto'
                        }}
                      >
                        刷新
                      </Button>
                    </div>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      </div>

      {/* 功能模块 */}
      <div className="modules-section">
        <div className="section-header">
          <h2 className="section-title">功能模块</h2>
          <p className="section-subtitle">探索完整的量化因子研究与策略开发流程</p>
        </div>

        <Row gutter={[20, 20]}>
          {modules.map(module => (
            <Col xs={24} lg={12} key={module.id}>
              <Card
                className="module-card"
                variant="borderless"
                hoverable
                onClick={() => handleModuleClick(module.url)}
              >
                <div className="module-bg" style={{ background: module.bgColor }}></div>
                <div className="module-gradient" style={{ backgroundImage: `linear-gradient(135deg, ${module.gradient})` }}></div>

                <div className="module-icon-wrapper" style={{ background: module.bgColor }}>
                  <div className="module-icon">{module.icon}</div>
                </div>

                <div className="module-content">
                  <div className="module-header">
                    <Badge status={module.statusType} text={module.status} />
                    <div className="module-number">0{module.id}</div>
                  </div>

                  <h3 className="module-title">{module.title}</h3>
                  <p className="module-desc">{module.description}</p>

                  <div className="module-footer">
                    <Space size="middle">
                      <span className="module-tag">{module.category}</span>
                      <span className="module-tag">难度：{module.complexity}</span>
                    </Space>
                    <Button type="primary" size="small" icon={<ArrowUpOutlined />}>
                      进入
                    </Button>
                  </div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </div>

      <AiModelConfigModal
        open={modelConfigVisible}
        onClose={() => setModelConfigVisible(false)}
      />

    </div>
  )
}

const PlusOutlined = () => <span style={{ fontSize: '16px' }}>+</span>

export default Home
