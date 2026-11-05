declare const __MEDIA_FILES__: string[];
const available = new Set(__MEDIA_FILES__);
export function mediaUrl(path: string): string | undefined {
  return available.has(path) ? `${import.meta.env.BASE_URL}${path.replace(/^\//, '')}` : undefined;
}
