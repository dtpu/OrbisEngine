import type { ReactNode } from 'react';
import styles from './Heading.module.css';

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={`${styles.eyebrow} ${className ?? ''}`}>{children}</p>;
}

export function H1({ children, className }: { children: ReactNode; className?: string }) {
  return <h1 className={`${styles.h1} ${className ?? ''}`}>{children}</h1>;
}

export function H2({ children, className }: { children: ReactNode; className?: string }) {
  return <h2 className={`${styles.h2} ${className ?? ''}`}>{children}</h2>;
}
