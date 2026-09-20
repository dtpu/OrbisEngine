import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import CompareSlider from '../components/CompareSlider';
import type { SyntheticEvent } from 'react';
import styles from './Compare.module.css';

const hideBroken = (e: SyntheticEvent<HTMLImageElement>) => {
  e.currentTarget.style.visibility = 'hidden';
};

type CompareScene = { id: string; caption: string };

// Offsets are quoted in body-heights once each render pair is exported and measured against its
// source frame. Until then the captions say so rather than stating a number nobody measured.
const compareScenes: CompareScene[] = [
  { id: 'elevator', caption: 'Elevator. Viewpoint offset in body-heights: pending measurement.' },
  { id: 'lobby', caption: 'Lobby. Viewpoint offset in body-heights: pending measurement.' },
  { id: 'gym', caption: 'Gym. Viewpoint offset in body-heights: pending measurement.' },
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
