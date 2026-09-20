import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import VideoLoop from '../components/VideoLoop';
import styles from './Viewer.module.css';

const ITEMS = [
  {
    key: 'walk',
    title: 'Walk',
    body: 'W/A/S/D and mouse look; floors and stairs support you, walls stop you.',
    src: '/media/viewer/walk.mp4',
    poster: '/media/viewer/walk.jpg',
  },
  {
    key: 'rewind',
    title: 'Rewind the moment',
    body: 'The source clip owns the timeline; scrub and the whole scene follows.',
    src: '/media/viewer/rewind.mp4',
    poster: '/media/viewer/rewind.jpg',
  },
  {
    key: 'audio',
    title: 'Hear it where it happened',
    body: 'Original audio in sync; spatial tracks where reviewed.',
    src: '/media/viewer/audio.mp4',
    poster: '/media/viewer/audio.jpg',
  },
] as const;

const CONTROLS = [
  { keys: ['W', 'A', 'S', 'D'], label: 'walk' },
  { keys: ['Space'], label: 'play / pause' },
  { keys: ['Shift'], label: 'faster' },
  { keys: ['R'], label: 'reset view' },
  { keys: ['M'], label: 'overhead map' },
] as const;

export default function Viewer() {
  return (
    <Section id="viewer">
      <Eyebrow>Viewer</Eyebrow>
      <H2 className={styles.heading}>What you can do in the viewer</H2>

      <ul className={styles.grid}>
        {ITEMS.map((item) => (
          <li key={item.key} className={styles.item}>
            <div className={styles.media}>
              <VideoLoop
                src={item.src}
                poster={item.poster}
                label={item.title}
                className={styles.video}
              />
            </div>
            <h3 className={styles.title}>{item.title}</h3>
            <p className={styles.body}>{item.body}</p>
          </li>
        ))}
      </ul>

      <div className={styles.footer}>
        <ul className={styles.controls} aria-label="Keyboard controls">
          {CONTROLS.map((c) => (
            <li key={c.label} className={styles.control}>
              <span className={styles.caps}>
                {c.keys.map((k) => (
                  <kbd key={k} className={styles.kbd}>
                    {k}
                  </kbd>
                ))}
              </span>
              <span className={styles.controlLabel}>{c.label}</span>
            </li>
          ))}
        </ul>
        <p className={styles.headset}>
          <strong>Works in a headset</strong> — Quest via WebXR, teleport or smooth locomotion.
        </p>
      </div>
    </Section>
  );
}
