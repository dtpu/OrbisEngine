// Shared by the desktop scene rail and the in-headset sidebar.
export interface SceneClip {
  id: string;
  title: string;
  place: string;
  src: string;
  poster: number;
  sub: string;
  free: string;
  lead: string;
  proof: string[];
  caveat: string;
  srctxt: string;
  sceneManifest?: string;
}

export const CLIPS: SceneClip[] = [
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
    id: 'tos31',
    title: 'Tears of Steel · shot 31',
    place: 'Tears of Steel, shot 31 · CC BY 3.0 · 9.9 s',
    src: '/clips/tos31d.mp4',
    poster: 4,
    sub: 'A film shot with a rear-view walk through a machinery hall',
    free: 'Walk behind or beside the actor',
    lead: 'Ten seconds from Tears of Steel, a Blender Foundation film released under CC BY 3.0. The actor turns, briefly shows part of his face and shoulder, then walks away through a machinery hall.',
    proof: [
      'The source follows one older man turning and walking away',
      'The hall, reflective floor, rope, and machinery are visible checks on scene layout',
      'This viewer scene uses the 9.9-second trim shown in the source card',
    ],
    caveat:
      'Only part of the face and front shoulder appear at the opening; most of the walk is seen from behind. A rope is absent from the reconstructed foreground and machinery placement differs, so wider moves are exploratory.',
    srctxt:
      'Ten seconds of Tears of Steel (Blender Foundation, CC BY 3.0) — a released film, treated like any other clip.',
  },
  {
    id: 'lobby',
    title: 'A lobby with a waving subject',
    place: 'Office lobby · phone · 9.6 s',
    src: '/clips/lobby.mp4',
    poster: 1,
    sub: 'One phone clip, a tracked walk, and a turn toward camera',
    free: 'Try small moves around the recorded path; inspect the doors and furniture',
    lead: 'A man walks across an office lobby, turns, and waves. The reconstruction keeps the lobby layout and the person together so you can compare the same action with the source clip.',
    proof: [
      'The source turns and waves around 3.9–4.8 s and again near the end',
      'The elevator doors, notices, and open doorway are useful alignment landmarks',
    ],
    caveat:
      'The chair geometry is visibly distorted and late views lean against the room layout. The avatar has a light patch absent from the source; a matching pose is not a promise of matching appearance.',
    srctxt: 'One 9.6-second phone clip: a man walks toward the camera and waves.',
  },
  {
    id: 'gym-accepted',
    title: 'Gym · dumbbell press',
    place: 'Gym · phone · 18.07 s',
    src: '/reviews/gym-repair/source.mp4',
    poster: 8,
    sub: 'A man lifting two dumbbells on an incline bench',
    free: 'Walk around the bench',
    lead: 'Austin’s accepted gym reconstruction, preserving its original assets and placement.',
    proof: ['Animated person and both dumbbells', 'Bench collision surfaces retained'],
    caveat:
      'Broad body shape, imperfect foot/back contact and late hand/dumbbell separation remain. Dumbbell geometry is inferred.',
    srctxt: 'Original phone clip of a man lifting dumbbells on an incline bench.',
    sceneManifest: '/reviews/gym-kitchen/selected-scenes/gym-accepted.json',
  },
  {
    id: 'kitchen-repair',
    title: 'Kitchen · cooking',
    place: 'Kitchen · phone · 50.73 s',
    src: '/reviews/gym-kitchen/kitchen/source.mp4',
    poster: 40,
    sub: 'James cooking, with moving cookware and appliance doors',
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
    sceneManifest: '/reviews/gym-kitchen/selected-scenes/kitchen-repair.json',
  },
];

export const ALL = CLIPS;
