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
    id: 'lobby',
    title: 'A lobby with a waving subject',
    place: 'Office lobby · phone · 9.6 s',
    src: '/clips/lobby.mp4',
    poster: 1.0,
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
    id: 'stairs2',
    title: 'A short stair ascent',
    place: 'Red stairwell · phone · 4.1 s',
    src: '/clips/stairs2.mp4',
    poster: 1.6,
    sub: 'A man climbs, turns toward the rail, and leans across it',
    free: 'Watch stair contact',
    lead: 'A four-second phone clip follows a man up a red stairwell. He turns toward the chrome rail and leans across it while the treads and mezzanine stay in view.',
    proof: [
      'The source shows the man turning toward the right rail around 2.2 s',
      'The stair edges, handrail, and upper offices remain visible landmarks',
      'The source ends with him leaning toward the rail rather than completing a clean stair-contact check',
    ],
    caveat:
      'Rendered contact with the treads and stair depth diverges by the end. Geometry outside the right balustrade streaks; keep the camera toward the rail and treads.',
    srctxt: 'One 4.1-second phone clip, tracking alongside a man climbing stairs.',
  },
  {
    id: 'atrium',
    title: 'Night atrium, partial view',
    place: 'Atrium · phone · 5.4 s · night',
    src: '/clips/atrium.mp4',
    poster: 2.0,
    sub: 'A man circles a landing while the camera turns',
    free: 'Stay near the landing and railing; the open void is not a walkable floor',
    lead: 'A soft handheld night clip follows a man around an atrium landing. Glass, red benches, balconies, and the stair opening give the viewer a difficult scene with a short, moving source.',
    proof: [
      'The subject turns through the landing and faces the camera near the end',
      'A partial extra person appears at the left opening in the source',
      'The viewer follows the main subject; the partial extra person is missing',
    ],
    caveat:
      'The late wall and floor stretch, and the middle avatar appears over the stair opening where the source shows support on the landing. The face is an appearance approximation, and the backpack is not represented; the void beyond the railing is not a walkable floor.',
    srctxt: 'One 5.4-second handheld phone clip, shot at night, of a man walking around a landing.',
  },
  {
    id: 'tos31',
    title: 'Tears of Steel · shot 31',
    place: 'Tears of Steel, shot 31 · CC BY 3.0 · 9.9 s',
    src: '/clips/tos31d.mp4',
    poster: 4.0,
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
];

export const SPARES = [
  {
    id: 'gym',
    title: 'Gym — the hardest pose',
    place: 'Gym · phone · 18 s',
    src: '/clips/gym.mp4',
    poster: 8,
    sub: 'A man on an incline bench with dumbbells in hand',
    free: 'Watch the bench and held weights',
    lead: 'A phone clip circles a man lying on an incline bench in a gym. The bench and dumbbells make this a close-occlusion test for pose and held objects.',
    proof: [
      'The source keeps the man, bench, and dumbbells in the same action',
      'Mirrors and a second person at the left add background movement',
    ],
    caveat:
      'His forearms and the dumbbells still dissolve into a pale column of smear during occlusion. The side wall closes the useful camera path.',
    srctxt: 'One 18-second phone clip, camera arcing around a man on a bench.',
  },
  {
    id: 'hp-walk',
    title: 'Person and place, separated',
    place: 'Lobby man · generated street',
    src: '/clips/lobby.mp4',
    poster: 1.0,
    sub: 'The man from the lobby clip, walking down a different world',
    free: 'Experiment: person and place are separate',
    lead: 'The man from the lobby clip is shown walking through a street generated from another video. It demonstrates the separation between an animated person and a generated place.',
    proof: ['The picture-in-picture shows the source of the person, not the street'],
    caveat:
      "His light is wrong. He is lit by his lobby's overhead fluorescents in a street lit by warm lanterns, and he casts no shadow into the scene.",
    srctxt:
      'The person came from this lobby clip. The street came from a different video entirely.',
  },
  {
    id: 'tos',
    title: 'Tears of Steel — the older shot',
    place: 'Tears of Steel · CC BY 3.0',
    src: '/clips/tos-hall-walk.mp4',
    poster: 3,
    sub: 'An earlier hall walk from the same film',
    free: 'Stay with the rear-view walk',
    lead: 'An earlier Tears of Steel hall shot, released by the Blender Foundation under CC BY 3.0. The actor walks away through the machinery hall while the camera follows.',
    proof: ['The source keeps the actor mostly in rear view'],
    caveat:
      'The hall is softer and the camera stays closer than the main shot. The actor’s front appearance remains unseen.',
    srctxt: 'Tears of Steel (Blender Foundation, CC BY 3.0), the earlier hall shot.',
  },
  {
    id: 'living',
    title: 'Living room',
    place: 'Living room · phone',
    src: '/clips/living.mp4',
    poster: 2,
    sub: 'A man waves, then walks around a sofa and table',
    free: 'Watch the furniture occlusions',
    lead: 'A phone clip follows a man waving and then walking around a living room. The sofa, table, and bright windows shape what the camera can see of his lower body.',
    proof: ['The source shows a wave followed by a walk around the sofa and table'],
    caveat:
      'The lower body is hidden by furniture in parts of the source. The room has a sharper alternate reconstruction with extra desk and chair geometry; this card uses the selected scene.',
    srctxt: 'One phone clip of a man waving, then walking between a sofa and a table.',
  },
];

export const ALL = [...CLIPS, ...SPARES];
