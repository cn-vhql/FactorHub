import {
  graphic,
  init,
  use,
  type ECharts,
  type EChartsCoreOption
} from 'echarts/core'
import {
  LineChart,
  BarChart,
  PieChart,
  ScatterChart,
  CandlestickChart,
  HeatmapChart
} from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
  GraphicComponent,
  MarkPointComponent,
  MarkLineComponent,
  ToolboxComponent,
  DatasetComponent,
  TransformComponent
} from 'echarts/components'
import { LabelLayout, UniversalTransition } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'

use([
  LineChart,
  BarChart,
  PieChart,
  ScatterChart,
  CandlestickChart,
  HeatmapChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
  GraphicComponent,
  MarkPointComponent,
  MarkLineComponent,
  ToolboxComponent,
  DatasetComponent,
  TransformComponent,
  LabelLayout,
  UniversalTransition,
  CanvasRenderer
])

export { graphic, init }
export type EChartsOption = EChartsCoreOption
export type { ECharts }
