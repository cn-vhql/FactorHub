import { useEffect, useState } from 'react'
import { Alert, Button, Form, Input, Modal, Space, Typography, message } from 'antd'
import { ApiOutlined, CheckCircleOutlined } from '@ant-design/icons'
import { api } from '@/services/api'

interface AiModelConfigModalProps {
  open: boolean
  onClose: () => void
  onSaved?: () => void
}

interface AiModelConfigState {
  base_url: string
  model: string
  api_key: string
  request_path: string
  has_api_key?: boolean
  api_key_masked?: string
  configured?: boolean
}

const DEFAULT_VALUES: AiModelConfigState = {
  base_url: '',
  model: '',
  api_key: '',
  request_path: '/chat/completions',
  has_api_key: false,
  api_key_masked: '',
  configured: false,
}

const AiModelConfigModal: React.FC<AiModelConfigModalProps> = ({
  open,
  onClose,
  onSaved,
}) => {
  const [form] = Form.useForm<AiModelConfigState>()
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [configState, setConfigState] = useState<AiModelConfigState>(DEFAULT_VALUES)
  const [validationResult, setValidationResult] = useState<{
    model: string
    request_url: string
    reply_preview: string
  } | null>(null)

  const loadConfig = async () => {
    setLoading(true)
    try {
      const response = await api.getAiModelConfig() as any
      if (response.success) {
        const data = { ...DEFAULT_VALUES, ...response.data }
        setConfigState(data)
        form.setFieldsValue({
          base_url: data.base_url,
          model: data.model,
          api_key: '',
          request_path: data.request_path || '/chat/completions',
        })
      }
    } catch (error) {
      message.error((error as Error).message || '加载模型配置失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!open) {
      return
    }
    setValidationResult(null)
    loadConfig()
  }, [open])

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      setSaving(true)
      const response = await api.saveAiModelConfig(values) as any
      if (response.success) {
        const data = { ...DEFAULT_VALUES, ...response.data }
        setConfigState(data)
        form.setFieldValue('api_key', '')
        message.success(response.message || '模型配置已保存')
        onSaved?.()
      }
    } catch (error) {
      message.error((error as Error).message || '保存模型配置失败')
    } finally {
      setSaving(false)
    }
  }

  const handleValidate = async () => {
    try {
      const values = await form.validateFields()
      setValidating(true)
      const response = await api.validateAiModelConfig(values) as any
      if (response.success) {
        setValidationResult(response.data)
        message.success(response.message || '模型配置验证通过')
      }
    } catch (error) {
      message.error((error as Error).message || '模型配置验证失败')
    } finally {
      setValidating(false)
    }
  }

  return (
    <Modal
      title={(
        <Space>
          <ApiOutlined />
          <span>配置 AI 模型</span>
        </Space>
      )}
      open={open}
      onCancel={onClose}
      footer={null}
      width={680}
      destroyOnHidden
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={DEFAULT_VALUES}
      >
        <Form.Item
          label="模型服务地址"
          name="base_url"
          rules={[{ required: true, message: '请输入 OpenAI 协议服务地址' }]}
        >
          <Input placeholder="例如：https://api.openai.com/v1" disabled={loading} />
        </Form.Item>

        <Form.Item
          label="模型 ID"
          name="model"
          rules={[{ required: true, message: '请输入模型 ID' }]}
        >
          <Input placeholder="例如：gpt-4.1-mini" disabled={loading} />
        </Form.Item>

        <Form.Item
          label="模型密钥"
          name="api_key"
          extra={configState.has_api_key ? `当前已保存密钥：${configState.api_key_masked}。若不更换，可保持空白。` : '首次配置时必须填写有效密钥。'}
        >
          <Input.Password
            placeholder="请输入 API Key"
            disabled={loading}
            autoComplete="new-password"
          />
        </Form.Item>

        <Form.Item
          label="模型请求地址"
          name="request_path"
          rules={[{ required: true, message: '请输入模型请求地址' }]}
          extra="可填写相对路径（如 /chat/completions）或完整请求 URL。"
        >
          <Input placeholder="/chat/completions" disabled={loading} />
        </Form.Item>

        {configState.configured && (
          <Alert
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
            message="当前已存在一套可保存的模型配置"
            description="你可以直接验证当前配置，也可以修改后重新保存。"
          />
        )}

        {validationResult && (
          <Alert
            type="success"
            showIcon
            icon={<CheckCircleOutlined />}
            style={{ marginBottom: 16 }}
            message="模型连通性验证通过"
            description={(
              <div>
                <div>模型：{validationResult.model}</div>
                <div>请求地址：{validationResult.request_url}</div>
                <div>返回预览：{validationResult.reply_preview}</div>
              </div>
            )}
          />
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            保存后，新增因子里的 “AI 生成因子” 会复用这套配置。
          </Typography.Text>
          <Space>
            <Button onClick={handleValidate} loading={validating}>
              验证是否生效
            </Button>
            <Button type="primary" onClick={handleSave} loading={saving}>
              保存配置
            </Button>
          </Space>
        </div>
      </Form>
    </Modal>
  )
}

export default AiModelConfigModal
