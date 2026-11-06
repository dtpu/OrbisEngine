import Section from '../components/Section';
import Button from '../components/Button';
import VideoLoop from '../components/VideoLoop';
import { Eyebrow, H1 } from '../components/Heading';
import { APP_URL, DEMO_URL } from '../content/links';
import { mediaUrl } from '../content/media';
import styles from './Hero.module.css';

import heroPoster from '../../../docs/images/hero.jpg';

const POSTER = '/media/elevator/poster.jpg';
const hasPreviewPair = Boolean(
  mediaUrl('/media/elevator/source.mp4') &&
  mediaUrl('/media/elevator/wander.mp4') &&
  mediaUrl(POSTER),
);

export default function Hero() {
  return (
    <Section id="hero" theme="dark">
      <div className={styles.copy}>
        <Eyebrow>ORBIS ENGINE</Eyebrow>
        <H1 className={styles.title}>Step inside a video.</H1>
        <p className={styles.sub}>
          One clip becomes a room you can walk through, with its people, objects, and sound
          replaying alongside the original recording.
        </p>
        <div className={styles.actions}>
          <Button href={APP_URL}>Upload a video</Button>
          <Button href={DEMO_URL} variant="secondary">
            Try the demo
          </Button>
          <Button href="#how-it-works" variant="secondary">
            See how it works
          </Button>
        </div>
      </div>

      <p className={styles.sub}>
        Scene media is installed separately. Generated rooms and inferred bodies can differ from the
        recording; unobserved geometry is invented.
      </p>
      {hasPreviewPair ? (
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
                label="Orbis Engine viewpoint moving through the same moment"
                className={styles.video}
              />
            </div>
            <figcaption className={styles.label}>Orbis Engine</figcaption>
          </figure>
        </div>
      ) : (
        <figure className={styles.still} data-hero-still>
          <div className={styles.media}>
            <img className={styles.video} src={heroPoster} alt="Orbis Engine project still" />
          </div>
          <figcaption className={styles.label}>
            Project still · source and reconstruction previews not installed
          </figcaption>
        </figure>
      )}
    </Section>
  );
}
