import Section from '../components/Section';
import Card from '../components/Card';
import VideoLoop from '../components/VideoLoop';
import { Eyebrow, H2 } from '../components/Heading';
import { scenes } from '../content/scenes';
import { demoSceneUrl } from '../content/links';
import styles from './Gallery.module.css';

export default function Gallery() {
  return (
    <Section id="gallery" wide>
      <header className={styles.header}>
        <Eyebrow>Examples</Eyebrow>
        <H2>Walk through these</H2>
        <p className={styles.intro}>
          Nine clips from the demo picker, each rebuilt as a room you can move through. Cards with
          known flaws keep an honest label instead of being hidden.
        </p>
      </header>
      <ul className={styles.grid}>
        {scenes.map((scene) => {
          const href = demoSceneUrl(scene.id);
          return (
            <li key={scene.id} className={styles.item}>
              <Card
                title={scene.title}
                caption={scene.caption}
                label={scene.label}
                href={href}
                media={
                  <VideoLoop
                    src={scene.wander}
                    poster={scene.poster}
                    label={`${scene.title} — Wander loop`}
                    className={styles.video}
                  />
                }
              />
              <a className={styles.open} href={href}>
                Open in viewer →
              </a>
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
