import { useRef, useState, type PointerEvent, type ReactNode } from 'react';
import styles from './CompareSlider.module.css';

export default function CompareSlider({
  before,
  after,
  beforeLabel = 'Before',
  afterLabel = 'After',
}: {
  before: ReactNode;
  after: ReactNode;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [pos, setPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const posFromClientX = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return pos;
    return Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setPos(posFromClientX(e.clientX));
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (e.buttons === 0) return;
    setPos(posFromClientX(e.clientX));
  };

  return (
    <div
      ref={containerRef}
      className={styles.slider}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={(e) => e.currentTarget.releasePointerCapture(e.pointerId)}
    >
      <div className={styles.layer}>{before}</div>
      <div className={styles.layer} style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        {after}
      </div>
      <div className={styles.divider} style={{ left: `${pos}%` }}>
        <div className={styles.handle} />
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Compare"
        className={styles.range}
      />
      {beforeLabel && <span className={`${styles.pill} ${styles.left}`}>{beforeLabel}</span>}
      {afterLabel && <span className={`${styles.pill} ${styles.right}`}>{afterLabel}</span>}
    </div>
  );
}
