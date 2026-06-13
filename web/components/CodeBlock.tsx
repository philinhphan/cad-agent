"use client";

import { useEffect, useState } from "react";
import { codeToHtml } from "shiki";

/** Syntax-highlighted CadQuery source (shiki, lazily highlighted client-side). */
export function CodeBlock({ code, lang = "python" }: { code: string; lang?: string }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    codeToHtml(code, { lang, theme: "github-dark-default" })
      .then((out) => alive && setHtml(out))
      .catch(() => alive && setHtml(null));
    return () => {
      alive = false;
    };
  }, [code, lang]);

  if (!html) {
    return (
      <pre className="code-shiki overflow-x-auto p-4 text-[0.78rem] leading-relaxed text-ink-dim">
        {code}
      </pre>
    );
  }
  return (
    // Safe: shiki HTML-escapes the source into tokenized <span>s — the code is
    // rendered as inert, escaped text, never as live markup.
    <div
      className="code-shiki overflow-x-auto text-[0.78rem] leading-relaxed"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
