import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  automaticUploadOptions,
  optionFields,
  optionsFromSchema,
  setupBlocked,
} from '../lib/options';
import type { OptionsSchema, SetupReport } from '../lib/types';

function schema(): OptionsSchema {
  // Deliberately differ from today's defaults so a copied UI preset fails this contract.
  return {
    additionalProperties: false,
    properties: {
      marble: { type: 'string', default: 'none', enum: ['image', 'none'] },
      people: { type: 'integer', default: 4, minimum: 1, maximum: 8 },
      all_people: { type: 'boolean', default: true },
      objects: { type: 'boolean', default: false },
      finetune: { type: 'boolean', default: true },
    },
  };
}

test('submission follows API defaults and only offers published world modes and people limits', () => {
  const description = schema();
  assert.deepEqual(optionsFromSchema(description), {
    marble: 'none',
    people: 4,
    all_people: true,
    objects: false,
    finetune: true,
  });
  const controls = optionFields(description);
  assert.deepEqual(controls.find((field) => field.name === 'marble')?.enum, ['image', 'none']);
  assert.equal(controls.find((field) => field.name === 'people')?.maximum, 8);
});

test('unsupported historical audio fields cannot be sent by the form', () => {
  const description = schema();
  description.properties.reviewed_audio = { type: 'boolean', default: false };
  assert.equal(
    optionFields(description).some((field) => String(field.name) === 'reviewed_audio'),
    false,
  );
  assert.equal('reviewed_audio' in optionsFromSchema(description), false);
});

test('missing defaults are surfaced instead of silently guessing an expensive choice', () => {
  const description = schema();
  delete description.properties.finetune.default;
  assert.throws(() => optionsFromSchema(description), /did not provide a default/);
});

test('new uploads explicitly opt into hands-off processing without changing model choices', () => {
  const description = schema();
  description.properties.hands_off = { type: 'boolean', default: false };
  assert.deepEqual(automaticUploadOptions(description), {
    ...optionsFromSchema(description),
    hands_off: true,
  });
});

test('automatic uploads refuse an older server instead of silently requiring operator steps', () => {
  assert.throws(() => automaticUploadOptions(schema()), /support automatic uploads/);
});

test('unverified provider readiness permits queuing and blocked checks require an explicit queue choice', () => {
  const unverified: SetupReport = {
    ready: false,
    status: 'unverified',
    checks: [{ name: 'Provider access', status: 'unverified', detail: 'No remote request made.' }],
  };
  assert.equal(setupBlocked(unverified), false);
  assert.equal(setupBlocked({ ...unverified, status: 'blocked' }), true);
  assert.equal(
    setupBlocked({
      ...unverified,
      checks: [
        { name: 'FFmpeg', status: 'blocked', detail: 'Missing.', action: 'Install ffmpeg.' },
      ],
    }),
    true,
  );
});
