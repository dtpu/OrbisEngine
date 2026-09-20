import Section from '../components/Section';
import { Eyebrow, H2 } from '../components/Heading';
import { REPO_URL } from '../content/links';
import styles from './Team.module.css';

type Member = { name: string; login: string; role?: string };

// Names from the repository's contributor credits; GitHub logins from the repository's
// collaborators. Add a `role` line per person once the team has agreed on wording.
const members: Member[] = [
  { name: 'Aayan Karmali', login: 'StockerMC' },
  { name: 'Austin Jian', login: 'austinjiann' },
  { name: 'Daniel Pu', login: 'dtpu' },
  { name: 'James Li', login: 'jli2007' },
];

export default function Team() {
  return (
    <Section id="team">
      <Eyebrow>Team</Eyebrow>
      <H2 className={styles.title}>Built at Hack the North 2026</H2>
      <ul className={styles.grid}>
        {members.map((m) => (
          <li key={m.login} className={styles.member}>
            <p className={styles.name}>{m.name}</p>
            {m.role && <p className={styles.role}>{m.role}</p>}
            <a
              className={styles.link}
              href={`https://github.com/${m.login}`}
              target="_blank"
              rel="noreferrer"
            >
              @{m.login} &rarr;
            </a>
          </li>
        ))}
      </ul>
      <p className={styles.credits}>
        Built with Three.js and Spark. Tears of Steel footage is (CC) Blender Foundation, CC BY 3.0.
        Reconstruction and generation models are credited in the{' '}
        <a className={styles.link} href={REPO_URL}>
          repository README
        </a>
        .
      </p>
    </Section>
  );
}
