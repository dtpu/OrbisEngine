export type Scene = {
  id: string;
  title: string;
  caption: string;
  label?: string;
  /** Scene id the demo picker accepts (`demo.html?clip=`); absent when the demo has no preset. */
  viewer?: string;
  poster: string;
  wander: string;
};

// Media paths under public/media/, filled by site/media-manifest.json. Each card plays the
// reviewed Wander loop for the scene; the poster is a still from that loop.
const media = (id: string) => ({
  poster: `/media/${id}/poster.jpg`,
  wander: `/media/${id}/wander.mp4`,
});

type Row = [
  id: string,
  title: string,
  caption: string,
  extra?: { label?: string; viewer?: string },
];

const rows: Row[] = [
  [
    'elevator',
    'Two people, one handheld take',
    'Lift doors, railings and floor give the reconstruction landmarks.',
    { viewer: 'elevator' },
  ],
  [
    'lobby',
    'A lobby with a waving subject',
    'Wide space, one subject, clear motion.',
    { viewer: 'lobby' },
  ],
  ['stairs2', 'A short stair ascent', 'Vertical motion; the stairs must support walking.'],
  [
    'atrium',
    'Night atrium, partial view',
    'Dark, glass, and a moving camera on a short clip.',
    { label: 'lighting mismatch' },
  ],
  [
    'tos31',
    'Tears of Steel · shot 31',
    'Cinematic footage with a reflective floor.',
    { viewer: 'tos31' },
  ],
  [
    'gym',
    'Gym — the hardest pose',
    'Body partly occluded by equipment; the dumbbells are inferred.',
    { label: 'inferred dumbbells', viewer: 'gym-accepted' },
  ],
  [
    'kitchen',
    'Kitchen — a full cooking clip',
    'Fifty seconds of one cook, with moving cookware and appliance doors. The inset is the recording.',
    { label: 'partial room', viewer: 'kitchen-repair' },
  ],
];

export const scenes: Scene[] = rows.map(([id, title, caption, extra]) => ({
  id,
  title,
  caption,
  ...extra,
  ...media(id),
}));
