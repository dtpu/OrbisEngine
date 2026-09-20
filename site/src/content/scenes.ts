export type Scene = {
  id: string;
  title: string;
  caption: string;
  label?: string;
  poster: string;
  source: string;
  wander: string;
};

const media = (id: string) => ({
  poster: `/media/${id}/poster.jpg`,
  source: `/media/${id}/source.mp4`,
  wander: `/media/${id}/wander.mp4`,
});

const rows: [string, string, string, string?][] = [
  [
    'elevator',
    'Two people, one handheld take',
    'Lift doors, railings and floor give the reconstruction landmarks.',
  ],
  ['lobby', 'A lobby with a waving subject', 'Wide space, one subject, clear motion.'],
  ['stairs2', 'A short stair ascent', 'Vertical motion; the stairs must support walking.'],
  [
    'atrium',
    'Night atrium, partial view',
    'Dark, glass, and a moving camera on a short clip.',
    'lighting mismatch',
  ],
  ['tos31', 'Tears of Steel · shot 31', 'Cinematic footage with a reflective floor.'],
  ['gym', 'Gym — the hardest pose', 'Body partly occluded by equipment.'],
  ['hp-walk', 'Person and place, separated', 'Shows the person and room branches side by side.'],
  [
    'tos',
    'Tears of Steel — the older shot',
    'An earlier attempt kept for comparison.',
    'older attempt',
  ],
  [
    'living',
    'Living room',
    'Furniture occlusion and a sharper alternate reconstruction.',
    'lower body inferred',
  ],
];

export const scenes: Scene[] = rows.map(([id, title, caption, label]) => ({
  id,
  title,
  caption,
  label,
  ...media(id),
}));
