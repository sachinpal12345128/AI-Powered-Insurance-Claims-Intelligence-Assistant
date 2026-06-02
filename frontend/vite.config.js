import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        // Use 127.0.0.1 (not localhost) to force IPv4.
        // Node 17+ resolves "localhost" to ::1 first; uvicorn on 0.0.0.0
        // listens only on IPv4 -> ECONNRESET on every proxied request.
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
