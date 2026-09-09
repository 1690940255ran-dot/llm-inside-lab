import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base 用相对路径，方便直接部署到 GitHub Pages 的任意子路径下
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
})
