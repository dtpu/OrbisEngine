import { defineConfig, loadEnv } from 'vite';
import { sharedAssets } from './server/shared-assets.mjs';
import path from 'node:path';

export default defineConfig(({ mode }) => ({
  plugins: [sharedAssets({
    ...loadEnv(mode, process.cwd(), 'WANDER_'),
    ...Object.fromEntries(Object.entries(process.env).filter(([key]) => key.startsWith('WANDER_'))),
  })],
  server: {
    host: '127.0.0.1',
    fs: { deny: ['.env', '.env.*', '**/.env*', '**/.context/**', '**/*.{crt,pem,key,p12,pfx}', '**/.git/**'] },
    hmr: process.env.RECORD ? false : undefined,
    watch: { ignored: ['worker', 'assets', 'scripts/.frames'].map(dir => path.resolve(import.meta.dirname, dir) + '/**') },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  build: {
    target: 'es2022',
    copyPublicDir: false,
    chunkSizeWarningLimit: 4000,
    rolldownOptions: { input: { index: 'index.html', demo: 'demo.html', fourd: 'fourd.html' } },
  },
}));
