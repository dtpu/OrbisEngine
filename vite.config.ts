import { defineConfig, loadEnv } from 'vite';
import { sharedAssets } from './server/shared-assets.ts';
import path from 'node:path';

export default defineConfig(({ mode }) => ({
  plugins: [
    sharedAssets({
      ...loadEnv(mode, process.cwd(), 'WANDER_'),
      ...Object.fromEntries(
        Object.entries(process.env).filter(([key]) => key.startsWith('WANDER_')),
      ),
    }),
  ],
  server: {
    host: '127.0.0.1',
    fs: {
      deny: [
        '.env',
        '.env.*',
        '**/.env*',
        '**/.context/**',
        '**/*.{crt,pem,key,p12,pfx}',
        '**/.git/**',
      ],
    },
    hmr: process.env.RECORD ? false : undefined,
    watch: {
      // .venv and .context hold tens of thousands of files and no code; watching them exhausts
      // the file-descriptor limit (EMFILE). public/ must stay watched: Vite only serves files it
      // has seen there, so a world staged after start-up would otherwise answer with index.html.
      ignored: ['worker', 'assets', 'scripts/.frames', '.venv', '.context'].map(
        (dir) => path.resolve(import.meta.dirname, dir) + '/**',
      ),
    },
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
