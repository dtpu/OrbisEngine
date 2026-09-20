import path from 'node:path';

// The dashboard is a standalone page: it imports nothing outside src/dashboard, so it deliberately
// skips the viewer's shared-assets plugin (and its @aws-sdk dependency). The config exports a plain
// object rather than calling defineConfig so it loads in a worktree with no node_modules installed.
// WANDER_PIPELINE_API points the /api/pipeline proxy at the orchestrator API.
const target = process.env.WANDER_PIPELINE_API;

export default {
  server: {
    host: '127.0.0.1',
    proxy: target ? { '/api/pipeline': { target, changeOrigin: false } } : undefined,
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
    watch: {
      ignored: ['worker', 'assets', 'scripts/.frames', '.venv', '.context'].map(
        (dir) => path.resolve(import.meta.dirname, dir) + '/**',
      ),
    },
  },
  build: {
    target: 'es2022',
    copyPublicDir: false,
    rolldownOptions: { input: { dashboard: 'dashboard.html' } },
  },
};
