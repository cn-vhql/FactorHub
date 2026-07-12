import { useState } from 'react'
import { Alert, Button, Card, Form, Input, Modal, Select, Space, Tag, Typography, message } from 'antd'
import { ApiOutlined, RobotOutlined } from '@ant-design/icons'
import { api } from '@/services/api'
import CodeTextArea from '@/components/CodeTextArea'
import { normalizeFormulaType } from '@/utils/formula'

const { TextArea } = Input

interface GeneratedFactorResult {
  name: string
  description: string
  formula_type: string
  code: string
  validation_message: string
  attempts: number
  request_url?: string
}

interface AiFactorGeneratorModalProps {
  open: boolean
  onClose: () => void
  onApply: (result: GeneratedFactorResult) => void
  onOpenConfig: () => void
  initialFormulaType?: string
}

const AiFactorGeneratorModal: React.FC<AiFactorGeneratorModalProps> = ({
  open,
  onClose,
  onApply,
  onOpenConfig,
  initialFormulaType = 'mylanguage',
}) => {
  const [form] = Form.useForm()
  const [generating, setGenerating] = useState(false)
  const [generatedFactor, setGeneratedFactor] = useState<GeneratedFactorResult | null>(null)

  const handleClose = () => {
    setGeneratedFactor(null)
    form.resetFields()
    onClose()
  }

  const handleGenerate = async () => {
    try {
      const values = await form.validateFields()
      setGenerating(true)
      const response = await api.generateAiFactor(values) as any
      if (response.success) {
        setGeneratedFactor(response.data)
        message.success(response.message || 'AI 已生成可执行因子')
      }
    } catch (error) {
      message.error((error as Error).message || 'AI 生成因子失败')
    } finally {
      setGenerating(false)
    }
  }

  const handleApply = () => {
    if (!generatedFactor) {
      message.warning('请先生成可执行因子')
      return
    }
    onApply(generatedFactor)
    message.success('AI 生成结果已回填到新增因子表单')
    handleClose()
  }

  return (
    <Modal
      title={(
        <Space>
          <RobotOutlined />
          <span>AI 生成因子</span>
        </Space>
      )}
      open={open}
      onCancel={handleClose}
      footer={null}
      width={860}
      destroyOnHidden
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          formula_type: normalizeFormulaType(initialFormulaType),
          max_rounds: 4,
        }}
      >
        <Form.Item
          label="因子需求描述"
          name="requirement"
          rules={[{ required: true, message: '请描述希望生成的因子逻辑' }]}
          extra="建议描述业务意图、目标市场行为、观察窗口和希望使用的指标。"
        >
          <TextArea
            rows={5}
            placeholder="例如：生成一个衡量股价距离20日低点远近的因子，越接近20日低点时值越低，越偏离时值越高，适合后续做选股和组合。"
          />
        </Form.Item>

        <Space size="middle" style={{ display: 'flex', marginBottom: 8 }} wrap>
          <Form.Item
            label="公式类型偏好"
            name="formula_type"
            style={{ minWidth: 180, marginBottom: 0 }}
          >
            <Select
              options={[
                { value: 'mylanguage', label: '麦语言' },
                { value: 'python', label: 'Python 因子' },
                { value: 'auto', label: '自动选择' },
              ]}
            />
          </Form.Item>

          <Form.Item
            label="建议名称"
            name="suggested_name"
            style={{ flex: 1, minWidth: 220, marginBottom: 0 }}
          >
            <Input placeholder="例如：distance_to_low_20" />
          </Form.Item>
        </Space>

        <Form.Item
          label="补充说明"
          name="description_hint"
          extra="可补充偏好的实现方式、是否需要多输出绘图线、是否优先用于因子挖掘/组合/回测。"
        >
          <TextArea
            rows={3}
            placeholder="例如：优先生成可用于因子挖掘和组合分析的主输出序列；如果适合绘图，可额外返回副图线。"
          />
        </Form.Item>

        <Space style={{ marginBottom: 16 }} wrap>
          <Button icon={<ApiOutlined />} onClick={onOpenConfig}>
            配置模型
          </Button>
          <Button type="primary" icon={<RobotOutlined />} onClick={handleGenerate} loading={generating}>
            生成并自动纠错
          </Button>
        </Space>

        {generatedFactor && (
          <Card
            title="生成结果"
            variant="borderless"
            style={{ background: '#f8fafc', border: '1px solid #e2e8f0' }}
          >
            <Space style={{ marginBottom: 12 }} wrap>
              <Tag color="blue">{generatedFactor.formula_type === 'python' ? 'Python 因子' : '麦语言'}</Tag>
              <Tag color="green">第 {generatedFactor.attempts} 轮通过校验</Tag>
              <Tag color="cyan">{generatedFactor.name}</Tag>
            </Space>

            <Typography.Paragraph style={{ marginBottom: 12, color: '#475569' }}>
              {generatedFactor.description || '未提供额外说明'}
            </Typography.Paragraph>

            <Alert
              type="success"
              showIcon
              style={{ marginBottom: 12 }}
              message={generatedFactor.validation_message}
              description={generatedFactor.request_url ? `请求地址：${generatedFactor.request_url}` : undefined}
            />

            <CodeTextArea
              value={generatedFactor.code}
              rows={8}
              readOnly
            />

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
              <Button type="primary" onClick={handleApply}>
                应用到新增因子表单
              </Button>
            </div>
          </Card>
        )}
      </Form>
    </Modal>
  )
}

export default AiFactorGeneratorModal
