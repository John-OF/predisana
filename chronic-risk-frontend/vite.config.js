import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // recharts reexporta sus componentes desde su index, y con 4 paginas lazy que
        // lo importan (AUD-20) Rollup lo partia entre chunks que se importaban en
        // ciclo: avisaba de un orden de ejecucion roto en Simulacion.jsx y Admin.jsx.
        // Entero en un chunk propio no hay ciclo, y ademas se cachea una sola vez.
        // React va aparte a proposito: un chunk manual se queda con las dependencias
        // que nadie reclama, y recharts se llevaba react y react-dom, con lo que la
        // entrada importaba los ~400 kB de graficos en la primera pintura (AUD-20).
        manualChunks(id) {
          if (/\/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'react';
          if (id.includes('/node_modules/recharts/')) return 'recharts';
        },
      },
    },
  },
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
