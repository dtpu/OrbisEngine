import { CodexTail } from '@/components/codex-tail';

// The Codex log belongs to a run: it follows the review the agent is writing for one of this
// run's stages, and answering it sends a message about that stage. The runs index has no run
// selected, so the rail lives here rather than in the root layout.
export default async function RunLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  return (
    <div className="shell">
      <div className="shell__main">{children}</div>
      <CodexTail runId={decodeURIComponent(runId)} />
    </div>
  );
}
