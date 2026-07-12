import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined
          }

          const normalizedId = id.replace(/\\/g, '/')

          if (
            normalizedId.includes('/echarts/') ||
            normalizedId.includes('/zrender/')
          ) {
            return 'echarts-vendor'
          }

          if (
            normalizedId.includes('/react/') ||
            normalizedId.includes('/react-dom/') ||
            normalizedId.includes('/scheduler/')
          ) {
            return 'react-vendor'
          }

          if (
            normalizedId.includes('/react-router/') ||
            normalizedId.includes('/react-router-dom/') ||
            normalizedId.includes('/history/')
          ) {
            return 'router-vendor'
          }

          if (
            normalizedId.includes('/rc-table/') ||
            normalizedId.includes('/rc-pagination/') ||
            normalizedId.includes('/rc-virtual-list/') ||
            normalizedId.includes('/rc-resize-observer/')
          ) {
            return 'antd-table-vendor'
          }

          if (
            normalizedId.includes('/rc-field-form/') ||
            normalizedId.includes('/async-validator/')
          ) {
            return 'antd-form-vendor'
          }

          if (
            normalizedId.includes('/antd/') ||
            normalizedId.includes('/@ant-design/') ||
            normalizedId.includes('/rc-') ||
            normalizedId.includes('/@rc-component/')
          ) {
            return 'antd-vendor'
          }

          if (
            normalizedId.includes('/dayjs/') ||
            normalizedId.includes('/axios/')
          ) {
            return 'data-utils-vendor'
          }

          return 'vendor'
        }
      }
    }
  },
  server: {
    port: 5173, // 使用 Vite 默认端口，避免 Windows 端口权限问题
    host: 'localhost', // 使用 localhost 而不是 127.0.0.1
    strictPort: false, // 如果端口被占用，自动尝试下一个端口
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
