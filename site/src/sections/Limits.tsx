import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import { limits } from '../content/limits';
import styles from './Limits.module.css';

export default function Limits() {
  return (
    <Section id="limits">
      <div className={styles.wrap}>
        <Eyebrow>Limits</Eyebrow>
        <H2 className={styles.title}>What Wander can&rsquo;t do yet.</H2>
        <p className={styles.lede}>We&rsquo;d rather you find out here than in the viewer.</p>
        <ul className={styles.list}>
          {limits.map((limit) => (
            <li key={limit} className={styles.item}>
              {limit}
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}
