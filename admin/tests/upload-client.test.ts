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

function mockUpload(t: TestContext, status: number, body: unknown, raw = false) {
  const sent = { method: '', url: '', form: new FormData(), submissions: 0 };
  class UploadRequest extends EventTarget {
    readonly upload = new EventTarget();
    readonly status = status;
    readonly responseText = raw ? String(body) : JSON.stringify(body);

    open(method: string, url: string) {
      sent.method = method;
      sent.url = url;
    }

    send(form: FormData) {
      sent.form = form;
      sent.submissions += 1;
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

const validUpload: UploadRunResult = {
  id: 'new-run',
  status: 'queued',
  sourceArtifactId: 'source-artifact',
  sourceSha256: 'b'.repeat(64),
  reused: false,
};

test('a new upload returns its validated receipt', async (t) => {
  const sent = mockUpload(t, 201, validUpload);
  assert.deepEqual(await uploadRun({ file, options }), validUpload);
  assert.equal(sent.submissions, 1);
});

for (const [name, body, raw] of [
  ['HTML', '<html>private proxy diagnostics</html>', true],
  ['empty body', '', true],
  ['null', null, false],
  ['array', [], false],
  ['missing receipt', {}, false],
  ['missing ID', { ...validUpload, id: undefined }, false],
  ['invalid ID', { ...validUpload, id: '../another-run' }, false],
  ['unknown status', { ...validUpload, status: 'complete' }, false],
  ['missing artifact', { ...validUpload, sourceArtifactId: '' }, false],
  ['invalid source hash', { ...validUpload, sourceSha256: 'bad' }, false],
  ['missing reuse flag', { ...validUpload, reused: undefined }, false],
] as const) {
  test(`upload refuses ${name} on success without resubmitting`, async (t) => {
    const sent = mockUpload(t, 200, body, raw);
    await assert.rejects(uploadRun({ file, options }), {
      message:
        'The upload response was incomplete or invalid. The run may have been created; check the runs list before uploading again.',
    });
    assert.equal(sent.submissions, 1);
  });
}
