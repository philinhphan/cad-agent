"use client";

import { Component, type ReactNode } from "react";
import dynamic from "next/dynamic";

// ssr:false requires a Client Component (Next 16). The R3F scene is browser-only.
const StlScene = dynamic(() => import("./StlScene"), {
  ssr: false,
  loading: () => <ViewerNote>loading geometry…</ViewerNote>,
});

export function StlViewer({ url }: { url: string }) {
  return (
    <div className="relative h-full w-full">
      <GeometryBoundary>
        <StlScene url={url} />
      </GeometryBoundary>
      <span className="pointer-events-none absolute bottom-2 right-3 tech-label opacity-70">
        drag · orbit / scroll · zoom
      </span>
    </div>
  );
}

function ViewerNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full w-full items-center justify-center tech-label">
      {children}
    </div>
  );
}

/** STL fetch failures (404 / CORS) surface here instead of crashing the page. */
class GeometryBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return <ViewerNote>could not load STL</ViewerNote>;
    }
    return this.props.children;
  }
}
