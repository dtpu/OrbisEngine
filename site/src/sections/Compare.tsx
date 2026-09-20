import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import CompareSlider from '../components/CompareSlider';
import type { SyntheticEvent } from 'react';
import styles from './Compare.module.css';

const hideBroken = (e: SyntheticEvent<HTMLImageElement>) => {
  e.currentTarget.style.visibility = 'hidden';
};

type CompareScene = { id: string; caption: string };

const compareScenes: CompareScene[] = [
  { id: 'elevator', caption: 'Viewpoint moved ~1.5 body-heights left of the camera.' },
  { id: 'lobby', caption: 'Viewpoint moved ~2 body-heights forward, half a body-height up.' },
  { id: 'gym', caption: 'Viewpoint moved ~1 body-height right and behind the camera.' },
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
