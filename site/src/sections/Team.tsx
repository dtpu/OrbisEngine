import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import styles from './Team.module.css';

type Member = { name: string; role: string; href: string };

const REPO = 'https://github.com/dtpu/htn2026';

const members: Member[] = [
  { name: 'Team member', role: 'Role', href: REPO },
  { name: 'Team member', role: 'Role', href: REPO },
  { name: 'Team member', role: 'Role', href: REPO },
  { name: 'Team member', role: 'Role', href: REPO },
];

export default function Team() {
  return (
    <Section id="team">
      <Eyebrow>Team</Eyebrow>
      <H2 className={styles.title}>Built at Hack the North 2026</H2>
      <ul className={styles.grid}>
        {members.map((m, i) => (
          <li key={i} className={styles.member}>
            <p className={styles.name}>{m.name}</p>
            <p className={styles.role}>{m.role}</p>
            <a className={styles.link} href={m.href} target="_blank" rel="noreferrer">
              GitHub &rarr;
            </a>
          </li>
        ))}
      </ul>
      <p className={styles.credits}>
        Built with Three.js and Spark. Reconstruction and generation models credited in the
        repository README.
      </p>
    </Section>
  );
}
