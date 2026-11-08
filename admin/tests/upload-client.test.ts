import assert from 'node:assert/strict';
import { test, type TestContext } from 'node:test';
import { uploadRun, type UploadRunResult } from '../lib/client';
import type { GraphOptions } from '../lib/types';

const options: GraphOptions = {
  marble: 'none',
  people: 1,
  all_people: false,
  objects: false,
  finetune: false,
  hands_off: true,
};
const file = new File(['clip bytes'], 'clip.mp4', { type: 'video/mp4' });

function mockUpload(t: TestContext, status: number, body: unknown) {
  const sent = { method: '', url: '', form: new FormData() };
  class UploadRequest extends EventTarget {
    readonly upload = new EventTarget();
    readonly status = status;
    readonly responseText = JSON.stringify(body);

    open(method: string, url: string) {
      sent.method = method;
      sent.url = url;
    }

    send(form: FormData) {
      sent.form = form;
      this.dispatchEvent(new Event('load'));
    }
  }

  const previous = Object.getOwnPropertyDescriptor(globalThis, 'XMLHttpRequest');
  Object.defineProperty(globalThis, 'XMLHttpRequest', {
    configurable: true,
    value: UploadRequest,
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'XMLHttpRequest', previous);
    else Reflect.deleteProperty(globalThis, 'XMLHttpRequest');
  });
  return sent;
}

test('upload opts into reuse and returns the existing run and its recorded status', async (t) => {
  const existing: UploadRunResult = {
    id: 'existing-run',
    status: 'paused',
    sourceArtifactId: 'source-artifact',
    sourceSha256: 'a'.repeat(64),
    reused: true,
  };
  const sent = mockUpload(t, 200, existing);

  assert.deepEqual(await uploadRun({ file, options, requestedId: 'new-name' }), existing);
  assert.equal(sent.method, 'POST');
  assert.equal(sent.url, '/api/pipeline/runs/upload');
  assert.equal(sent.form.get('reuse_existing'), 'true');
  assert.equal(sent.form.get('requested_id'), 'new-name');
  assert.equal(sent.form.get('options'), JSON.stringify(options));
  assert.equal(sent.form.get('budget'), '{}');
  const source = sent.form.get('source');
  assert.ok(source instanceof File);
  assert.equal(source.name, 'clip.mp4');
  assert.equal(await source.text(), 'clip bytes');
});

test('upload surfaces the API error detail', async (t) => {
  mockUpload(t, 409, { detail: 'run name already exists' });
  await assert.rejects(uploadRun({ file, options }), /run name already exists/);
});
