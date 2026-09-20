import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import { steps } from '../content/steps';
import styles from './HowItWorks.module.css';

export default function HowItWorks() {
  return (
    <Section id="how-it-works">
      <div className={styles.header}>
        <Eyebrow>Pipeline</Eyebrow>
        <H2>How it works</H2>
      </div>
      <ol className={styles.strip} aria-label="Pipeline steps" tabIndex={0}>
        {steps.map((step) => (
          <li key={step.n} className={styles.step}>
            <div className={styles.numberRow}>
              <span className={styles.number}>{String(step.n).padStart(2, '0')}</span>
            </div>
            <div className={styles.media}>
              <img src={step.image} alt={step.title} loading="lazy" />
            </div>
            <h3 className={styles.title}>{step.title}</h3>
            <p className={styles.body}>{step.body}</p>
          </li>
        ))}
      </ol>
      <p className={styles.footer}>
        Completed processing is not accepted quality. Every scene above passed a manual visual
        review against its source.
      </p>
    </Section>
  );
}
