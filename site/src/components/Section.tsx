import type { ReactNode } from 'react';
import styles from './Section.module.css';

export default function Section({
  id,
  theme,
  wide,
  children,
}: {
  id: string;
  theme?: 'light' | 'dark';
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      id={id}
      data-theme={theme}
      className={`${styles.section} ${theme === 'dark' ? styles.dark : ''}`}
    >
      <div className={`${styles.inner} ${wide ? styles.wide : ''}`}>{children}</div>
    </section>
  );
}
