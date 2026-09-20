import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import CompareSlider from '../components/CompareSlider';
import type { SyntheticEvent } from 'react';
import styles from './Compare.module.css';

const hideBroken = (e: SyntheticEvent<HTMLImageElement>) => {
  e.currentTarget.style.visibility = 'hidden';
};

type CompareScene = { id: string; caption: string };

// Each pair is the recorded frame at time t next to the viewer's render at the same t from a
// standing position the recording never had. Offsets are quoted in body-heights (the scene's
// mean character stature), measured from the viewer's start pose by scripts/capture-media.ts;
// they are the numbers that script prints, not estimates.
const compareScenes: CompareScene[] = [
  {
    id: 'elevator',
    caption:
      'Elevator at 6.0 s, a bottle in the air. Viewpoint: 1.0 body-heights right and 0.6 forward of where the walk starts.',
  },
  {
    id: 'lobby',
    caption:
      'Lobby at 4.7 s, mid-wave. Viewpoint: 0.2 body-heights right and 1.2 behind where the walk starts.',
  },
  {
    id: 'gym',
    caption:
      'Gym at 8.0 s, dumbbells raised. Viewpoint: 1.0 body-heights right and 0.4 forward of where the walk starts.',
  },
];

export default function Compare() {
  return (
    <Section id="compare">
      <div className={styles.header}>
        <Eyebrow>Compare</Eyebrow>
        <H2>Same second, new angle</H2>
        <p className={styles.lede}>
          Drag to compare. Left is the recorded frame. Right is the same instant from a place the
          camera never stood.
        </p>
      </div>
      <ul className={styles.grid}>
        {compareScenes.map((scene) => (
          <li key={scene.id} className={styles.item}>
            <CompareSlider
              before={
                <img
                  src={`/media/${scene.id}/frame.jpg`}
                  alt="Recorded frame"
                  onError={hideBroken}
                />
              }
              after={
                <img
                  src={`/media/${scene.id}/render.jpg`}
                  alt="Wander render"
                  onError={hideBroken}
                />
              }
              beforeLabel="Recorded"
              afterLabel="Wander"
            />
            <p className={styles.caption}>{scene.caption}</p>
          </li>
        ))}
      </ul>
    </Section>
  );
}
