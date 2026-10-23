import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],

  // 开发时把 API / WebSocket 代理给 FastAPI（默认 8000 端口）
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },

  // 构建产物输出到 Python 后端的静态目录
  build: {
    outDir: fileURLToPath(new URL('../web/dist', import.meta.url)),
    emptyOutDir: true,
  },
})
