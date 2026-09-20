import type { ReactNode } from 'react';
import styles from './Container.module.css';

export default function Container({ wide, children }: { wide?: boolean; children: ReactNode }) {
  return <div className={`${styles.container} ${wide ? styles.wide : ''}`}>{children}</div>;
}
