import Section from '../components/Section';
import Button from '../components/Button';
import { H2 } from '../components/Heading';
import styles from './FinalCta.module.css';

export default function FinalCta() {
  return (
    <Section id="cta" theme="dark">
      <div className={styles.wrap}>
        <H2 className={styles.title}>Go walk around.</H2>
        <div className={styles.actions}>
          <Button href="/demo.html">Try the demo</Button>
          <Button href="https://github.com/dtpu/htn2026" variant="secondary">
            Read the code
          </Button>
        </div>
      </div>
    </Section>
  );
}
