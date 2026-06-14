"use client";

import { use } from "react";
import { RunView } from "@/components/RunView";
import { useRunStream } from "@/lib/useRunStream";

export default function RunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return <RunPageContent key={id} runId={id} />;
}

function RunPageContent({ runId }: { runId: string }) {
  const state = useRunStream(runId);
  const threshold = state.config?.score_threshold ?? 8;

  return <RunView state={state} threshold={threshold} />;
}
