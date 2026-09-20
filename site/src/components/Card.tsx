import type { ReactNode } from 'react';
import styles from './Card.module.css';

export default function Card({
  title,
  caption,
  label,
  href,
  media,
}: {
  title: string;
  caption: string;
  label?: string;
  href?: string;
  media: ReactNode;
}) {
  return (
    <article className={styles.card}>
      <div className={styles.media}>
        {media}
        {label && <span className={styles.label}>{label}</span>}
      </div>
      <h3 className={styles.title}>{href ? <a href={href}>{title}</a> : title}</h3>
      <p className={styles.caption}>{caption}</p>
    </article>
  );
}
