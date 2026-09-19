import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { S3Client } from '@aws-sdk/client-s3';
import { uploadBlob } from './lib/publish-transport.ts';

test(
  'Node streams checksum-verified publisher requests to loopback',
  { timeout: 15000 },
  async (t) => {
    assert.equal(process.versions.bun, undefined, 'Run this fixture with Node, as production does');
    const directory = await mkdtemp(path.join(tmpdir(), 'wander-publish-transport-'));
    const requests: {
      headers: IncomingHttpHeaders;
      body: Buffer;
      method?: string;
      url?: string;
    }[] = [];
    let responseStatus = 200;
    const server = createServer(async (request, response) => {
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      requests.push({
        headers: request.headers,
        body: Buffer.concat(chunks),
        method: request.method,
        url: request.url,
      });
      response.statusCode = responseStatus;
      response.setHeader('Content-Type', 'application/xml');
      response.end(
        responseStatus === 200
          ? ''
          : `<Error><Code>${responseStatus === 412 ? 'PreconditionFailed' : 'AccessDenied'}</Code></Error>`,
      );
    });
    let s3: S3Client | undefined;
    try {
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const address = server.address();
      assert.ok(address && typeof address !== 'string');
      s3 = new S3Client({
        endpoint: `http://127.0.0.1:${address.port}`,
        region: 'us-east-1',
        forcePathStyle: true,
        credentials: { accessKeyId: 'fixture-access-key', secretAccessKey: 'fixture-secret-key' },
        maxAttempts: 1,
      });
      const client = s3;
      for (const size of [0, 37, 2 * 1024 * 1024 + 17]) {
        await t.test(`preserves ${size} body bytes and signed length/checksum`, async () => {
          const bytes = Buffer.alloc(size);
          for (let i = 0; i < bytes.length; i++) bytes[i] = i % 251;
          const file = path.join(directory, `body-${size}`);
          await writeFile(file, bytes);
          const sha256 = createHash('sha256').update(bytes).digest('hex');
          const key = `viewer/blobs/${sha256}`;
          await uploadBlob(client, {
            bucket: 'fixture-bucket',
            key,
            file,
            size,
            contentType: 'application/octet-stream',
            sha256,
          });
          const received = requests.at(-1)!;
          assert.equal(received.method, 'PUT');
          assert.equal(received.url?.split('?')[0], `/fixture-bucket/${key}`);
          assert.deepEqual(received.body, bytes);
          assert.equal(received.headers['content-length'], String(size));
          assert.equal(received.headers['transfer-encoding'], undefined);
          assert.equal(received.headers['content-type'], 'application/octet-stream');
          assert.equal(received.headers['if-none-match'], '*');
          assert.equal(received.headers['x-amz-meta-sha256'], sha256);
          assert.equal(
            received.headers['x-amz-checksum-sha256'],
            createHash('sha256').update(received.body).digest('base64'),
          );
          const signedHeaders =
            received.headers.authorization?.match(/SignedHeaders=([^, ]+)/)?.[1];
          assert.ok(signedHeaders?.split(';').includes('content-length'));
          assert.ok(signedHeaders?.split(';').includes('x-amz-checksum-sha256'));
        });
      }
      for (const status of [403, 412]) {
        await t.test(`propagates HTTP ${status} for publisher failure handling`, async () => {
          responseStatus = status;
          const count = requests.length;
          const bytes = Buffer.from('rejected upload');
          const file = path.join(directory, `rejected-${status}`);
          await writeFile(file, bytes);
          await assert.rejects(
            uploadBlob(client, {
              bucket: 'fixture-bucket',
              key: 'archive/blobs/rejected',
              file,
              size: bytes.length,
              contentType: 'application/octet-stream',
              sha256: createHash('sha256').update(bytes).digest('hex'),
            }),
            (error: unknown) => {
              assert.equal(
                (error as { $metadata?: { httpStatusCode?: number } }).$metadata?.httpStatusCode,
                status,
              );
              return true;
            },
          );
          assert.equal(requests.length, count + 1);
          assert.deepEqual(requests.at(-1)!.body, bytes);
        });
      }
    } finally {
      s3?.destroy();
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
      await rm(directory, { recursive: true, force: true });
    }
  },
);
