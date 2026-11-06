import { createServer, type Server } from 'node:net';
import type { ChildProcess } from 'node:child_process';

/** Check every listening address before migrations or service startup can change local state. */
export async function assertPortsAvailable(ports: number[]): Promise<void> {
  const reservations: Server[] = [];
  try {
    for (const port of ports) {
      const server = createServer();
      reservations.push(server);
      await new Promise<void>((resolve, reject) => {
        server.once('error', (error: NodeJS.ErrnoException) => {
          reject(
            new Error(
              error.code === 'EADDRINUSE'
                ? `Port ${port} is already in use. Stop its owner or select a different API/admin port.`
                : `Cannot listen on port ${port}: ${error.message}`,
            ),
          );
        });
        server.listen(port, '127.0.0.1', () => resolve());
      });
    }
  } finally {
    await Promise.all(
      reservations
        .filter((server) => server.listening)
        .map((server) => new Promise<void>((resolve) => server.close(() => resolve()))),
    );
  }
}

type ServiceProcess = Pick<ChildProcess, 'exitCode' | 'signalCode'>;

function assertRunning(child: ServiceProcess, stopping: () => boolean): void {
  if (stopping()) throw new Error('Local app is stopping.');
  if (child.exitCode !== null || child.signalCode !== null)
    throw new Error('Service exited before becoming ready.');
}

/** A successful response is insufficient if shutdown or the launched service's exit won the race. */
export async function waitForService(
  url: string,
  child: ServiceProcess,
  stopping: () => boolean,
  headers: Record<string, string> = {},
): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    assertRunning(child, stopping);
    let response: Response | undefined;
    try {
      response = await fetch(url, { headers, signal: AbortSignal.timeout(2000) });
    } catch {}
    assertRunning(child, stopping);
    if (response) {
      await response.body?.cancel();
      if (response.ok) {
        // Let queued child exit/error events run before allowing the next service or worker.
        await new Promise<void>((resolve) => setImmediate(resolve));
        assertRunning(child, stopping);
        return;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Service did not become ready at ${new URL(url).origin}.`);
}
