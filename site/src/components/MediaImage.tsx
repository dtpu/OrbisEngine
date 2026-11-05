import { mediaUrl } from '../content/media';
export function MissingMedia({ label, className }: { label: string; className?: string }) {
  return (
    <div
      className={`missing-media ${className || ''}`}
      role="img"
      aria-label={label}
      data-media-unavailable
    >
      <span>
        {label}
        <small>Preview not installed</small>
      </span>
    </div>
  );
}
export default function MediaImage({ src, alt }: { src: string; alt: string }) {
  const url = mediaUrl(src);
  return url ? <img src={url} alt={alt} loading="lazy" /> : <MissingMedia label={alt} />;
}
