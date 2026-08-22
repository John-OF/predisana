import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // AUD-19: Vitest reusa esta misma config (mismos alias y mismo plugin de React que
  // el build, asi que un test no puede pasar por una resolucion distinta a la real).
  test: {
    environment: 'jsdom',          // los componentes necesitan DOM
    globals: true,                 // describe/it/expect sin importarlos en cada archivo
    setupFiles: './src/test/setup.js',
    css: false,                    // el CSS no aporta nada a estos tests y ralentiza
    include: ['src/**/*.test.{js,jsx}'],
  },
})
