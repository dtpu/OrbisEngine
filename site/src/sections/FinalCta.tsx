import Section from '../components/Section';
import Button from '../components/Button';
import { H2 } from '../components/Heading';
import { DEMO_URL, REPO_URL } from '../content/links';
import styles from './FinalCta.module.css';

export default function FinalCta() {
  return (
    <Section id="cta" theme="dark">
      <div className={styles.wrap}>
        <H2 className={styles.title}>Go walk around.</H2>
        <div className={styles.actions}>
          <Button href={DEMO_URL}>Try the demo</Button>
          <Button href={REPO_URL} variant="secondary">
            Read the code
          </Button>
        </div>
      </div>
    </Section>
  );
}
