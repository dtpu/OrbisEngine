import Section from '../components/Section';
import Button from '../components/Button';
import VideoLoop from '../components/VideoLoop';
import { Eyebrow, H1 } from '../components/Heading';
import styles from './Hero.module.css';

const POSTER = '/media/elevator/poster.jpg';

export default function Hero() {
  return (
    <Section id="hero" theme="dark">
      <div className={styles.copy}>
        <Eyebrow>WANDER</Eyebrow>
        <H1 className={styles.title}>Step inside a video.</H1>
        <p className={styles.sub}>
          One clip becomes a room you can walk through, with its people, objects, and sound
          replaying exactly as recorded.
        </p>
        <div className={styles.actions}>
          <Button href="/demo.html">Try the demo</Button>
          <Button href="#how-it-works" variant="secondary">
            See how it works
          </Button>
        </div>
      </div>

      <div className={styles.split}>
        <figure className={styles.pane}>
          <div className={styles.media}>
            <VideoLoop
              src="/media/elevator/source.mp4"
              poster={POSTER}
              label="Recorded elevator footage"
              className={styles.video}
            />
          </div>
          <figcaption className={styles.label}>Recorded</figcaption>
        </figure>
        <figure className={styles.pane}>
          <div className={styles.media}>
            <VideoLoop
              src="/media/elevator/wander.mp4"
              poster={POSTER}
              label="Wander viewpoint moving through the same moment"
              className={styles.video}
            />
          </div>
          <figcaption className={styles.label}>Wander</figcaption>
        </figure>
      </div>
    </Section>
  );
}
