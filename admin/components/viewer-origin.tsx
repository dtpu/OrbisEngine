'use client';

import { createContext, useContext } from 'react';

const ViewerOrigin = createContext('');

export function ViewerOriginProvider({
  origin,
  children,
}: {
  origin: string;
  children: React.ReactNode;
}) {
  return <ViewerOrigin.Provider value={origin}>{children}</ViewerOrigin.Provider>;
}

export function useViewerOrigin(): string {
  return useContext(ViewerOrigin);
}
