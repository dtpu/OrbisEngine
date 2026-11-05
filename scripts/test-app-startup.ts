import { expect, test } from 'bun:test';
import { createServer } from 'node:http';
import { createServer as createSocketServer } from 'node:net';
import { once } from 'node:events';
import { assertPortsAvailable, waitForService } from '../server/app-startup.ts';

test('port preflight refuses an occupied service port and releases earlier reservations', async () => {
  const occupied = createSocketServer().listen(0, '127.0.0.1');
  const available = createSocketServer().listen(0, '127.0.0.1');
  await Promise.all([once(occupied, 'listening'), once(available, 'listening')]);
  const occupiedPort = (occupied.address() as { port: number }).port;
  const availablePort = (available.address() as { port: number }).port;
  await new Promise<void>((resolve) => available.close(() => resolve()));
  try {
    await expect(assertPortsAvailable([availablePort, occupiedPort])).rejects.toThrow(
      `Port ${occupiedPort} is already in use`,
    );
    // The first reservation must be released even when a later port fails.
    await assertPortsAvailable([availablePort]);
    expect(occupied.listening).toBe(true);
  } finally {
    await new Promise<void>((resolve) => occupied.close(() => resolve()));
  }
});

test('duplicate configured service ports fail preflight before any service starts', async () => {
  const available = createSocketServer().listen(0, '127.0.0.1');
  await once(available, 'listening');
  const port = (available.address() as { port: number }).port;
  await new Promise<void>((resolve) => available.close(() => resolve()));
  await expect(assertPortsAvailable([port, port])).rejects.toThrow('already in use');
  await assertPortsAvailable([port]);
});

for (const reason of ['shutdown', 'child exit'] as const) {
  test(`an in-flight readiness success cannot bypass ${reason}`, async () => {
    const server = createServer().listen(0, '127.0.0.1');
    await once(server, 'listening');
    const port = (server.address() as { port: number }).port;
    let stopping = false;
    const child: { exitCode: number | null; signalCode: NodeJS.Signals | null } = {
      exitCode: null,
      signalCode: null,
    };
    try {
      const incoming = once(server, 'request');
      const ready = waitForService(`http://127.0.0.1:${port}`, child, () => stopping);
      const result = ready.then(
        () => null,
        (error: Error) => error,
      );
      const [, response] = await incoming;
      if (reason === 'shutdown') stopping = true;
      else child.exitCode = 1;
      response.end('ready');
      const error = await result;
      expect(error).toBeInstanceOf(Error);
      expect(error?.message).toContain(reason === 'shutdown' ? 'stopping' : 'exited');
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
}

test('readiness still accepts a healthy launched service', async () => {
  const server = createServer((_request, response) => response.end('ready')).listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    await waitForService(
      `http://127.0.0.1:${(server.address() as { port: number }).port}`,
      { exitCode: null, signalCode: null },
      () => false,
    );
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
