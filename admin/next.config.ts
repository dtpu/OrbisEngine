import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // The repo root holds the durable AGENTS.md; a generated one here would shadow it.
  agentRules: false,
  // The repo root also has a bun.lock, so pin the workspace root instead of letting Next guess.
  turbopack: { root: path.resolve(import.meta.dirname) },
};

export default nextConfig;
