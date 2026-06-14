import type { Metadata } from "next";
import { Poppins, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Kyrall — AI-powered 3D modeling",
  description:
    "Automating design for the physical world. Describe a part or drop an engineering drawing and generate real, self-refining CAD geometry.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${poppins.variable} ${jetbrains.variable} h-full`}
      suppressHydrationWarning
    >
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
    <header className="sticky top-0 z-40 border-b border-line bg-base/70 backdrop-blur-md">
      <div className="flex h-16 w-full items-center justify-end px-6 lg:px-10">
        <nav className="flex items-center gap-2 text-[0.85rem]">
          <NavLink href="/runs">History</NavLink>
          <Link
            href="/"
            className="rounded-full bg-accent px-4 py-2 font-display text-[0.85rem] font-semibold text-white transition-opacity hover:opacity-90"
          >
            Start building now
          </Link>
        </nav>
      </div>
    </header>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded-full px-3.5 py-2 text-ink-dim transition-colors hover:bg-panel-2 hover:text-ink"
    >
      {children}
    </Link>
  );
}

function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="flex w-full items-center justify-between px-6 py-5 lg:px-10">
        <span className="font-display text-sm font-semibold text-ink-dim">Kyrall</span>
        <span className="tech-label">automating design for the physical world</span>
      </div>
    </footer>
  );
}
