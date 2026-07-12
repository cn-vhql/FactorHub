import { useMemo, useRef } from 'react'
import type { CSSProperties, ChangeEventHandler, TextareaHTMLAttributes } from 'react'

type NativeTextareaProps = Omit<
  TextareaHTMLAttributes<HTMLTextAreaElement>,
  'value' | 'onChange' | 'style'
>

interface CodeTextAreaProps extends NativeTextareaProps {
  value?: string
  onChange?: ChangeEventHandler<HTMLTextAreaElement>
  rows?: number
  style?: CSSProperties
  className?: string
}

const CodeTextArea: React.FC<CodeTextAreaProps> = ({
  value = '',
  onChange,
  rows = 6,
  style,
  className,
  ...rest
}) => {
  const gutterRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const lineCount = useMemo(() => {
    const contentLines = value ? value.split('\n').length : 1
    return Math.max(rows, contentLines)
  }, [rows, value])

  const lineNumbers = useMemo(
    () => Array.from({ length: lineCount }, (_, index) => index + 1),
    [lineCount]
  )

  const handleScroll = () => {
    if (!gutterRef.current || !textareaRef.current) {
      return
    }
    gutterRef.current.scrollTop = textareaRef.current.scrollTop
  }

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        border: '1px solid #d9d9d9',
        borderRadius: '6px',
        backgroundColor: '#f6f8fa',
        overflow: 'hidden',
        transition: 'all 0.2s',
        ...style,
      }}
    >
      <div
        ref={gutterRef}
        aria-hidden="true"
        style={{
          flex: '0 0 52px',
          padding: '12px 8px 12px 12px',
          background: '#eef2f7',
          borderRight: '1px solid #d9d9d9',
          color: '#64748b',
          fontSize: '14px',
          fontFamily: 'Consolas, Monaco, monospace',
          lineHeight: '22px',
          textAlign: 'right',
          userSelect: 'none',
          overflow: 'hidden',
          whiteSpace: 'pre',
        }}
      >
        {lineNumbers.map((lineNumber) => (
          <div key={lineNumber}>{lineNumber}</div>
        ))}
      </div>

      <textarea
        {...rest}
        ref={textareaRef}
        value={value}
        onChange={onChange}
        rows={rows}
        wrap="off"
        spellCheck={false}
        onScroll={handleScroll}
        style={{
          flex: 1,
          border: 'none',
          outline: 'none',
          resize: 'vertical',
          backgroundColor: 'transparent',
          padding: '12px',
          fontSize: '14px',
          fontFamily: 'Consolas, Monaco, monospace',
          lineHeight: '22px',
          minHeight: '150px',
          maxHeight: '300px',
          overflow: 'auto',
          whiteSpace: 'pre',
          tabSize: 2,
        }}
      />
    </div>
  )
}

export default CodeTextArea
