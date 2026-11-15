import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the React app runs on :5173 and forwards /api calls to FastAPI on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
    // .taste-skill holds the design checks' headless Chrome profile and screenshots; watching its
    // locked session files crashes the dev server on Windows (EBUSY)
    watch: { ignored: ['**/.taste-skill/**'] },
  },
})
