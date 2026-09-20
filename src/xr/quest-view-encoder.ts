// Main-thread canvas encoding waits for idle time during immersive XR on Quest. A worker
// avoids that multi-second delay and keeps JPEG compression out of the headset render loop.
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext('2d');
let image: ImageData | null = null;
self.onmessage = async ({
  data,
}: MessageEvent<{ pixels: Uint8Array; width: number; height: number }>) => {
  try {
    const { pixels, width, height } = data;
    if (!context) throw new Error('Quest view encoder unavailable');
    if (!image || canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      image = context.createImageData(width, height);
    }
    const row = width * 4;
    for (let y = 0; y < height; y++)
      image.data.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    context.putImageData(image, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
    self.postMessage({ blob, pixels }, { transfer: [pixels.buffer] });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
