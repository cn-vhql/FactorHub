import type { ReactNode } from 'react'
import { WarningOutlined } from '@ant-design/icons'

export type FormulaType = 'mylanguage' | 'python'

export const FORMULA_TYPE_OPTIONS: Array<{ value: FormulaType; label: string }> = [
  { value: 'mylanguage', label: '麦语言' },
  { value: 'python', label: 'Python 因子' },
]

export const FORMULA_FIELD_LABEL = '因子表达式'
export const FORMULA_EMPTY_MESSAGE = '请先输入因子表达式'
export const FORMULA_REQUIRED_MESSAGE = '请输入因子表达式'

export const normalizeFormulaType = (formulaType?: string | null): FormulaType => {
  if (formulaType === 'python' || formulaType === 'function') {
    return 'python'
  }
  return 'mylanguage'
}

export const getFormulaPlaceholder = (formulaType: string): string => {
  const normalized = normalizeFormulaType(formulaType)
  if (normalized === 'python') {
    return '例如：np.log(close / close.shift(1))'
  }
  return '例如：CLOSE / MA(CLOSE, 20)'
}

export const getFormulaHelpContent = (formulaType: string): ReactNode => {
  const normalized = normalizeFormulaType(formulaType)

  if (normalized === 'python') {
    return (
      <div style={{ maxWidth: '600px', fontSize: '12px', color: '#fff' }}>
      <div style={{ marginBottom: '12px' }}>
          <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>Python 因子</div>
          <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>适合 pandas / numpy 向量化表达式，也支持受限的 <code style={codeStyle}>def calculate_factor(df)</code> 写法。</p>
        </div>
        <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
          <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>可执行示例</div>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>np.log(close / close.shift(1))</code></li>
            <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>close.rolling(window=20, min_periods=1).mean()</code></li>
            <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0]</code></li>
            <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>RANGEPOS(close, 20)</code>，可以直接混用系统函数</li>
            <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>IF(close &gt; open, close / open, 0)</code></li>
          </ul>
        </div>
        <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
          <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>手写函数</div>
          <p style={{ margin: '0 0 8px 0', color: '#ccc', lineHeight: '1.6' }}>
            仍支持 <code style={codeStyle}>def calculate_factor(df):</code>，但自动生成因子已不再使用这种旧包装。
          </p>
          <p style={{ margin: '0 0 8px 0', color: '#ccc', lineHeight: '1.6' }}>
            函数体建议只使用 <code style={codeStyle}>import pandas</code>、<code style={codeStyle}>import numpy</code>、简单赋值、<code style={codeStyle}>if</code>、<code style={codeStyle}>try</code> 和 <code style={codeStyle}>return</code>。
          </p>
          <p style={{ margin: 0, color: '#fbbf24', lineHeight: '1.6' }}>
            <WarningOutlined style={{ marginRight: '4px' }} />
            不支持 <code style={codeStyle}>lambda</code>、<code style={codeStyle}>rolling(...).apply(...)</code>、推导式、<code style={codeStyle}>for / while</code>、<code style={codeStyle}>from ... import ...</code>。
          </p>
        </div>
        <div>
          <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>可用字段与函数</div>
          <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>
            <code style={codeStyle}>close, open, high, low, volume, amount, turnover, amplitude, pct_change, date, year, month, day, weekday, np, pd</code>，以及
            <code style={codeStyle}>MA / EMA / SMA / REF / HHV / LLV / RANGEPOS / CROSS / IF / RSI / MACD / ATR / BBANDS</code> 等系统函数。
          </p>
          <p style={{ margin: '8px 0 0 0', color: '#ccc', lineHeight: '1.6' }}>
            序列方法主要支持 <code style={codeStyle}>shift / diff / pct_change / abs / round / fillna / clip / astype / mean / std / max / min / median / sum / skew / kurt / quantile / rank / rolling / expanding / ffill / bfill / replace</code>。
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ maxWidth: '600px', fontSize: '12px', color: '#fff' }}>
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>麦语言因子</div>
        <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>推荐作为因子库默认写法，适合指标、选股条件和组合公式。</p>
      </div>
        <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
          <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>常用字段</div>
          <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>
          <code style={codeStyle}>OPEN, HIGH, LOW, CLOSE, VOL, AMOUNT, TURNOVER, AMPLITUDE, PCT_CHG, DATE, TIME, YEAR, MONTH, DAY, WEEKDAY</code>
          </p>
        </div>
      <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
        <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>常用函数</div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>MA(CLOSE, 20)</code> / <code style={codeStyle}>EMA(CLOSE, 20)</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>HHV(HIGH, 20)</code> / <code style={codeStyle}>LLV(LOW, 20)</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>REF(CLOSE, 1)</code> / <code style={codeStyle}>RANGEPOS(CLOSE, 20)</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>IF(CLOSE &gt; OPEN, 1, 0)</code> / <code style={codeStyle}>CROSS(MA(CLOSE, 5), MA(CLOSE, 20))</code></li>
        </ul>
      </div>
      <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
        <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>注意事项</div>
        <p style={{ margin: 0, color: '#fbbf24', lineHeight: '1.6' }}>
          <WarningOutlined style={{ marginRight: '4px' }} />
          麦语言不支持 <code style={codeStyle}>close.rolling(...)</code> 这类点号链式语法；如果需要链式窗口计算，请改用 Python 因子。
        </p>
      </div>
      <div>
        <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>示例</div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>CLOSE / MA(CLOSE, 20)</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>X1:=RANGEPOS(CLOSE, 20); XG: X1</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>DIFF: EMA(CLOSE, 12) - EMA(CLOSE, 26); DEA: EMA(DIFF, 9); MACD: (DIFF - DEA) * 2; XG: DIFF</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>CROSS(MA(CLOSE, 5), MA(CLOSE, 20))</code></li>
          <li style={{ padding: '2px 0', color: '#fff', lineHeight: '1.6' }}>• <code style={codeStyle}>IF(CLOSE &gt; REF(CLOSE, 1), 1, 0)</code></li>
        </ul>
        <p style={{ margin: '8px 0 0 0', color: '#ccc', lineHeight: '1.6' }}>
          主输出优先使用 <code style={codeStyle}>XG:</code>；没有 <code style={codeStyle}>XG:</code> 时，系统会取最后一个 <code style={codeStyle}>:</code> 输出作为主因子序列。所有 <code style={codeStyle}>:</code> 输出都可用于图表绘制。
        </p>
      </div>
    </div>
  )
}

const codeStyle = {
  background: 'rgba(255, 255, 255, 0.1)',
  color: '#4dabf7',
  padding: '2px 4px',
  borderRadius: '3px',
  fontFamily: 'monospace',
  fontSize: '12px',
} as const
