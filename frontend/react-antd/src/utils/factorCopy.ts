export const FACTOR_TERMINOLOGY = {
  factorLibrary: '因子库',
  factorExpression: '因子表达式',
  factorCode: '因子代码',
  formulaType: '公式类型',
  mylanguage: '麦语言',
  python: 'Python 因子',
  miningFactor: '候选因子',
  compositeFactorExpression: '组合因子表达式',
  importedFactor: '外部导入因子',
} as const

export const FACTOR_COPY_GUIDELINES = [
  '公式类内容统一使用“因子表达式”或“组合因子表达式”；仅在展示数据库原始内容时使用“因子代码”。',
  '保存动作统一使用“保存到因子库”，避免“保存为自定义因子”“保存到自定义因子库”等混用。',
  '成功提示统一采用“动作成功：结果”结构；失败提示统一采用“动作失败：原因”结构。',
  '涉及执行能力时必须与真实引擎一致，明确区分麦语言和 Python 因子，不再使用旧版 def 包装能力做默认说明。',
  '导入工具仅能宣传真实已接入的能力；未接入运行时的外部因子值不得显示为可直接执行。'
] as const

export const FACTOR_COPY = {
  mining: {
    subtitle: '使用遗传算法自动搜索可执行的因子表达式',
    emptyStateTitle: '配置参数后点击“开始挖掘”',
    emptyStateHint: '系统将返回可执行的候选因子表达式；保存到因子库时会自动识别为麦语言或 Python 因子。',
    running: '因子挖掘进行中...',
    loading: '正在执行因子挖掘任务...',
    loadingResults: '正在加载挖掘结果...',
    resultSectionTitle: '候选因子',
    resultSectionHint: '以下结果均为候选因子表达式，不再生成旧版 def calculate_factor 包装。',
    noResult: '本次挖掘未发现符合条件的候选因子',
    saveSingle: '保存到因子库',
    saveAll: '全部保存到因子库',
    startAction: '启动因子挖掘任务',
    progressAction: '获取挖掘进度',
    resultAction: '加载挖掘结果',
    saveAction: '保存候选因子到因子库',
  },
  portfolio: {
    subtitle: '多因子组合优化与性能评估',
    emptyStateTitle: '配置参数后点击“开始优化”',
    emptyStateHint: '系统将基于所选因子计算权重，并生成可直接保存的组合因子表达式。',
    running: '正在优化组合权重...',
    compareEmptyStateTitle: '配置参数后点击“开始方法对比”',
    compareEmptyStateHint: '系统将基于同一组因子评估不同优化方法的表现差异。',
    expressionTitle: '组合因子表达式',
    expressionHint: '该组合因子会以 Python 因子表达式保存到因子库，保存前不会再生成旧版 def calculate_factor 包装。',
    missingExpression: '无法生成组合因子表达式。请先清理旧版 def calculate_factor 包装因子，并确认所选因子都能提取为可执行表达式。',
    optimizeAction: '执行组合优化',
    compareAction: '执行方法对比',
    saveAction: '保存组合因子到因子库',
  },
  importTool: {
    disabled: '导入工具当前未开放“外部因子值直接入库执行”能力。系统仅支持可直接执行的麦语言或 Python 因子表达式；为避免生成不可执行条目，CSV / DataFrame 导入已禁用。',
  },
} as const

export const formatActionSuccess = (action: string, result?: string): string => {
  if (!result) {
    return `${action}成功`
  }
  return `${action}成功：${result}`
}

export const formatActionFailure = (action: string, reason?: string): string => {
  if (!reason) {
    return `${action}失败`
  }
  return `${action}失败：${reason}`
}

export const formatSavedToLibrary = (factorName: string): string =>
  `已保存到因子库：${factorName}`

export const formatBatchSaveSummary = (successCount: number, failCount: number): string => {
  if (failCount === 0) {
    return `已保存到因子库：共 ${successCount} 个因子`
  }
  return `批量保存已完成：成功 ${successCount} 个，失败 ${failCount} 个`
}

export const buildMinedFactorDescription = (params: {
  expression: string
  ic?: number
  ir?: number
  fitness?: number
}): string => {
  const { expression, ic, ir, fitness } = params
  return [
    '遗传算法挖掘结果',
    `因子表达式: ${expression}`,
    `IC: ${Number.isFinite(ic) ? ic?.toFixed(4) : '-'}`,
    `IR: ${Number.isFinite(ir) ? ir?.toFixed(4) : '-'}`,
    `适应度: ${Number.isFinite(fitness) ? fitness?.toFixed(4) : '-'}`,
  ].join(' | ')
}

export const buildCompositeFactorDescription = (factorCount: number, methodLabel: string): string =>
  `基于 ${factorCount} 个因子的${methodLabel}组合因子表达式`

export const extractErrorMessage = (error: any, fallback = '未知错误'): string =>
  error?.response?.data?.detail ||
  error?.response?.data?.message ||
  error?.message ||
  fallback
