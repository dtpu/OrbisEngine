export const steps: { n: number; title: string; body: string; image: string }[] = [
  {
    n: 1,
    title: 'Original recording',
    body: 'The clip and the interval used.',
    image: '/media/steps/1.jpg',
  },
  {
    n: 2,
    title: 'Understand motion',
    body: 'Camera path, point cloud, tracked person.',
    image: '/media/steps/2.jpg',
  },
  {
    n: 3,
    title: 'Separate people and room',
    body: 'Mask and cleaned video; removed regions get generated fill.',
    image: '/media/steps/3.jpg',
  },
  {
    n: 4,
    title: 'Build room and people',
    body: 'Generated environment beside the inferred body.',
    image: '/media/steps/4.jpg',
  },
  {
    n: 5,
    title: 'Fit and animate',
    body: 'Placement, object tracks, remaining contact errors shown.',
    image: '/media/steps/5.jpg',
  },
  {
    n: 6,
    title: 'Explore the replay',
    body: 'Packaged scene, original audio, walk controls.',
    image: '/media/steps/6.jpg',
  },
];
