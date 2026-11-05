import { useEffect, useRef } from 'react';
import { mediaUrl } from '../content/media';
import { MissingMedia } from './MediaImage';

export default function VideoLoop({
  src,
  poster,
  label,
  className,
}: {
  src: string;
  poster: string;
  label: string;
  className?: string;
}) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            video.play().catch(() => {});
          } else {
            video.pause();
          }
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  const videoUrl = mediaUrl(src);
  const posterUrl = mediaUrl(poster);
  if (!videoUrl || !posterUrl) {
    return <MissingMedia label={label} className={className} />;
  }
  return (
    <video
      ref={ref}
      className={className}
      src={videoUrl}
      poster={posterUrl}
      muted
      loop
      playsInline
      preload="none"
      aria-label={label}
    />
  );
}
