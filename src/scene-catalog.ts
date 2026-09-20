// Shared by the desktop scene rail and the in-headset sidebar.
export const CLIPS = [
  {
    id: 'elevator',
    title: 'Two people, one handheld take',
    place: 'Fourth-floor lift lobby · phone · 10.0 s',
    src: '/clips/elevator.mp4',
    poster: 1.4,
    sub: 'Two men, a lift lobby, and a bottle moving between them',
    free: 'Explore close to the recorded path; the railings set the boundary',
    lead: 'Two men are reconstructed from one phone clip in the lift lobby where they were filmed. The source shows several bottle throws and catches; the viewer includes the bottle and backpack alongside the people.',
    proof: [
      'Two principal people remain visible through the source clip',
      'The lift doors, railings, and floor provide the scene landmarks',
      'The source contains throws around 0.9–1.3 s, 5.6–6.5 s, and 7.7–8.2 s',
    ],
    caveat:
      'The floor and door headers soften away from the recorded views. The bottle and backpack are present, while their fine timing and the wider walking boundary remain uncertain.',
    srctxt:
      'One 10-second phone clip of two men talking by a lift. Nothing else went in: no rig, no scan, no second angle.',
  },
  {
    id: 'kitchen',
    title: 'Kitchen · fixture repair',
    place: 'Kitchen · phone · 50.73 s',
    src: '/reviews/gym-kitchen/kitchen/source.mp4',
    poster: 40,
    sub: 'Full recording; moving doors and small pot lid',
    free: 'WASD to walk; drag to look',
    lead: 'James cooks through the full recording, with a solid freezer door, animated microwave door and small pot lid. Walk around the counter to reach him.',
    proof: [
      'One cook and original audio across all 50.73 seconds',
      'Moving freezer and microwave doors, pan and small pot lid',
      'Repaired wall beside the fridge and supported kitchen aisle',
    ],
    caveat:
      'Brita and carried bag motion remain absent. Interiors, unseen surfaces and parts of door/lid motion are inferred. Small items remain blurry; seams and imperfect contact remain.',
    srctxt: 'Original 50.73-second kitchen recording, preserving audio and timing.',
  },
  {
    id: 'gym-accepted',
    title: 'Gym · accepted scene',
    place: 'Gym · phone · 18.07 s',
    src: '/reviews/gym-repair/source.mp4',
    poster: 8,
    sub: 'Accepted person, two dumbbells and bench colliders',
    free: 'Walk around the bench',
    lead: 'Austin’s accepted gym reconstruction, preserving its original assets and placement.',
    proof: ['Animated person and both dumbbells', 'Bench collision surfaces retained'],
    caveat:
      'Broad body shape, imperfect foot/back contact and late hand/dumbbell separation remain. Dumbbell geometry is inferred.',
    srctxt: 'Original phone clip of a man lifting dumbbells on an incline bench.',
  },
];

// The rail shows only the accepted scenes above. Older clips still open directly through their
// ?demo= presets in fourd.html; they are not listed here.
export const SPARES: typeof CLIPS = [];

export const ALL = [...CLIPS, ...SPARES];
