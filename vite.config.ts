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
  // transformers.js 体积大且含 wasm，交给它自己懒加载，别让 Vite 预打包
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  build: {
    chunkSizeWarningLimit: 800,
    /**
     * 不做「先清空 outDir」这一步。
     *
     * 为什么：Vite 的 emptyDir() 会递归删 dist/，而本机开发环境的 safe-delete
     * 守卫会把这次批量删除拦下来（SAFE_DELETE_BULK_REJECTED），构建直接失败。
     * 更麻烦的是失败发生在清理阶段，Vite 自己那份 vite.config.ts.timestamp-*.mjs
     * 临时文件也没被删掉，于是**下一次构建会撞上更多待删文件，再次失败** ——
     * 一个自我延续的死循环。
     *
     * 关掉它不影响 CI：Actions 每次都是干净检出，dist/ 本来就不存在。
     * 本地想要干净产物，手动把 dist 挪走再构建即可（见 skill 里的做法）。
     */
    emptyOutDir: false,
  },
  test: {
    // 只跑 tests/ 下的正式用例：项目里的临时脚本会放在 tmp/，不该被当成测试收集
    include: ['tests/**/*.test.ts'],
    /**
     * 默认的 5 秒超时对这套用例太短了。
     *
     * 模块⑩ 是「真的在训练」，其中几条断言要跑满 1200 步（约 19 秒）才拿得到结论 ——
     * 注意力熵从 1.00 塌到 0.00 这件事，只有在训练真的跑完之后才成立。
     * 这些用例不是卡住了，是真的在算；所以把超时放宽到 90 秒。
     * 注意：超时只是「兜底」——如果哪天真跑挂了，90 秒后一样会失败。
     */
    testTimeout: 90000,
  },
})
