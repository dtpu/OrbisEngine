import { existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Rebuild after installing optional media. Never advertise files that are absent on disk.
function mediaFiles(directory: string, prefix = '/media'): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${directory}/${entry.name}`;
    return entry.isDirectory()
      ? mediaFiles(path, `${prefix}/${entry.name}`)
      : entry.isFile()
        ? [`${prefix}/${entry.name}`]
        : [];
  });
}
export default defineConfig(({ mode }) => ({
  base: loadEnv(mode, process.cwd(), 'VITE_').VITE_SITE_BASE || '/about/',
  plugins: [react()],
  define: {
    __MEDIA_FILES__: JSON.stringify(
      mediaFiles(fileURLToPath(new URL('./public/media', import.meta.url))),
    ),
  },
}));
