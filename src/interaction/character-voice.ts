/** Preset voices for fictional characters; these do not reproduce recorded speakers. */
export const CHARACTER_VOICES = ['ash', 'echo'] as const;

export type CharacterVoice = (typeof CHARACTER_VOICES)[number];

export function characterVoice(index: number): CharacterVoice {
  const castIndex = Number.isFinite(index) ? Math.max(0, Math.trunc(index)) : 0;
  return CHARACTER_VOICES[castIndex % CHARACTER_VOICES.length]!;
}

export function isCharacterVoice(value: unknown): value is CharacterVoice {
  return CHARACTER_VOICES.some((voice) => voice === value);
}
