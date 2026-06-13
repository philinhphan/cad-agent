import type { Metadata } from "next";
import { Saira_Condensed, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const saira = Saira_Condensed({
  variable: "--font-saira",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "cad-gen · text-to-CAD self-refine",
  description:
    "Generate CAD geometry from natural language and watch an AI self-refine loop critique and improve it, iteration by iteration.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${saira.variable} ${jetbrains.variable} h-full`}>
      <body className="min-h-full flex flex-col">
        <SiteHeader />
        <main className="flex-1">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}

function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-base/85 backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-5">
        <Link href="/" className="group flex items-baseline gap-2.5">
          <span className="tech-head text-xl text-ink group-hover:text-accent transition-colors">
            CAD<span className="text-accent">/</span>GEN
          </span>
          <span className="tech-label hidden sm:block">text-to-cad · self-refine</span>
        </Link>
        <nav className="flex items-center gap-1 text-[0.8rem]">
          <NavLink href="/">New&nbsp;run</NavLink>
          <NavLink href="/runs">History</NavLink>
        </nav>
      </div>
    </header>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded-[var(--radius-tech)] px-3 py-1.5 uppercase tracking-wider text-ink-dim hover:bg-panel-2 hover:text-ink transition-colors"
    >
      {children}
    </Link>
  );
}

function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2 px-5 py-3">
        <span className="tech-label">cad-gen · pydantic-ai + cadquery</span>
        <span className="tech-label">
          subprocess sandbox — not a security boundary
        </span>
      </div>
    </footer>
  );
}
