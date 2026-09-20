import { Suspense } from 'react';
import RunConsole from './console';

export default async function Page({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  return (
    <Suspense fallback={<main className="page">Loading…</main>}>
      <RunConsole runId={decodeURIComponent(runId)} />
    </Suspense>
  );
}
