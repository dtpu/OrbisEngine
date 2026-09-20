import logo from '../brand/logo-black.png';
import { DEMO_URL, REPO_URL } from '../content/links';
import styles from './TopBar.module.css';

const LINKS = [
  { href: '#gallery', label: 'Examples' },
  { href: '#how-it-works', label: 'How it works' },
  { href: REPO_URL, label: 'GitHub' },
] as const;

export default function TopBar() {
  return (
    <header className={styles.bar}>
      <div className={styles.inner}>
        <a href="#hero" className={styles.home}>
          <img src={logo} alt="Orbis Engine" className={styles.logo} />
        </a>
        <nav className={styles.nav} aria-label="Page">
          {LINKS.map((link) => (
            <a key={link.href} href={link.href} className={styles.link}>
              {link.label}
            </a>
          ))}
          <a href={DEMO_URL} className={styles.demo}>
            Try the demo
          </a>
        </nav>
      </div>
    </header>
  );
}
