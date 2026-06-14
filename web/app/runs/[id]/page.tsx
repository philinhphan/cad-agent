"use client";

import { useParams } from "next/navigation";
import { RunPage } from "@/components/RunPage";

export default function RunDetailPage() {
  const params = useParams<{ id: string }>();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (!id) return null;
  return <RunPage runId={id} />;
}
