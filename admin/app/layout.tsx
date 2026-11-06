import type { Metadata } from 'next';
import { IBM_Plex_Mono, IBM_Plex_Sans } from 'next/font/google';
import Link from 'next/link';
import { ViewerOriginProvider } from '@/components/viewer-origin';
import { parseViewerOrigin, viewerUrl } from '@/lib/viewer-url';
import './globals.css';

// One family in two cuts: identifiers, hashes and sizes sit in the mono, everything an operator
// reads as prose sits in the sans, and the two share a skeleton so mixed lines stay level.
const sans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-sans',
  display: 'swap',
});
const mono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Orbis Engine Admin',
  description: 'Operate Orbis Engine generation pipeline runs: create, steer, approve.',
};

// Read the public viewer origin at request time, including with `next start`.
export const dynamic = 'force-dynamic';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const viewerOrigin = parseViewerOrigin(process.env.WANDER_VIEWER_ORIGIN);
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <header className="top">
          <Link className="brand" href="/">
            <svg className="brand__mark" viewBox="0 0 20 20" aria-hidden="true">
              <rect x="1.5" y="3.5" width="17" height="13" rx="1.5" />
              <circle cx="10" cy="10" r="3" />
            </svg>
            <span>Orbis Engine</span>
            <span className="brand__area">admin</span>
          </Link>
          <nav className="top__links" aria-label="Related tools">
            <a href={viewerUrl('/demo.html', viewerOrigin)} target="_blank" rel="noreferrer">
              Viewer
            </a>
            <a href="/about/">About</a>
          </nav>
        </header>
        <ViewerOriginProvider origin={viewerOrigin}>{children}</ViewerOriginProvider>
      </body>
    </html>
  );
}
