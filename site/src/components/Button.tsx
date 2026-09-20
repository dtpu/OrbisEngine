import type { ReactNode } from 'react';
import styles from './Button.module.css';

export default function Button({
  href,
  variant = 'primary',
  children,
}: {
  href: string;
  variant?: 'primary' | 'secondary';
  children: ReactNode;
}) {
  return (
    <a href={href} className={`${styles.button} ${styles[variant]}`}>
      {children}
    </a>
  );
}
