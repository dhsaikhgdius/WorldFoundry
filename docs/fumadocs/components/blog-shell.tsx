import Link from 'next/link';
import type { ReactNode } from 'react';

import { SiteHeader } from '@/components/site-header';
import { WORLDFOUNDRY_GITHUB_REPO } from '@/lib/site-links';

type BlogShellProps = {
  children: ReactNode;
  footerLabel?: string;
};

export function BlogShell({ children, footerLabel = 'Blog' }: BlogShellProps) {
  return (
    <main className="pi-home-shell wf-home-shell">
      <SiteHeader
        variant="solid"
        active="blog"
        languageLinks={[
          { href: '/blog', label: 'English', current: true },
          { href: '/zh/docs', label: '中文' },
        ]}
      />

      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        {children}

        <footer className="pi-footer">
          <p>{footerLabel}</p>
          <div>
            <Link href="/docs">Docs</Link>
            <a href={WORLDFOUNDRY_GITHUB_REPO} rel="noreferrer" target="_blank">
              Community
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
