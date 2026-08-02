import Image from 'next/image';
import Link from 'next/link';

import { SiteHeader } from '@/components/site-header';
import { withBasePath } from '@/lib/site-path';
import type { Metadata } from 'next';

const openEnvisionLogoSrc = withBasePath('/openenvision-logo.png') ?? '/openenvision-logo.png';

export const metadata: Metadata = {
  title: 'OpenEnvision',
  description: 'OpenEnvision Lab, GitHub organization, and WorldFoundry repository address.',
};

export default function OpenEnvisionPage() {
  return (
    <main className="pi-home-shell wf-home-shell">
      <SiteHeader
        variant="solid"
        active="openenvision"
        languageLinks={[
          { href: '/openenvision', label: 'English', current: true },
          { href: '/zh/docs', label: '中文' },
        ]}
      />

      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <section className="pi-open-hero" aria-labelledby="openenvision-title">
          <Image
            src={openEnvisionLogoSrc}
            alt="OpenEnvision logo"
            className="pi-open-logo"
            width={148}
            height={148}
            priority
          />
          <div>
            <p className="pi-label">GitHub organization</p>
            <h1 id="openenvision-title">OpenEnvision</h1>
            <p>
              OpenEnvision Lab is a joint research lab advancing open vision intelligence through
              academia-industry collaboration.
            </p>
            <p>
              OpenEnvision Lab 是一个通过产学协作推进 open vision intelligence 的联合研究实验室。
            </p>
          </div>
        </section>

        <section className="pi-open-section" aria-labelledby="openenvision-links">
          <h2 id="openenvision-links">Project Links</h2>
          <table className="pi-open-table">
            <tbody>
              <tr>
                <th>Organization</th>
                <td>
                  <a href="https://github.com/OpenEnvision">https://github.com/OpenEnvision</a>
                </td>
              </tr>
              <tr>
                <th>WorldFoundry repo</th>
                <td>
                  <a href="https://github.com/OpenEnvision/WorldFoundry">
                    https://github.com/OpenEnvision/WorldFoundry
                  </a>
                </td>
              </tr>
              <tr>
                <th>Gaia repo</th>
                <td>
                  <a
                    href="https://github.com/OpenEnvision/Gaia"
                    target="_blank"
                    rel="noreferrer"
                  >
                    https://github.com/OpenEnvision/Gaia
                  </a>
                </td>
              </tr>
              <tr>
                <th>Clone URL</th>
                <td>
                  <code>https://github.com/OpenEnvision/WorldFoundry.git</code>
                </td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="pi-open-section" aria-labelledby="clone-command">
          <h2 id="clone-command">Clone</h2>
          <div className="pi-command" aria-label="WorldFoundry OpenEnvision clone command">
            <code>git clone https://github.com/OpenEnvision/WorldFoundry.git</code>
          </div>
        </section>

        <footer className="pi-footer">
          <p>OpenEnvision</p>
          <div>
            <Link href="/docs">Docs</Link>
            <Link href="/docs/guides/supported-models">Supported Models</Link>
          </div>
        </footer>
      </div>
    </main>
  );
}
