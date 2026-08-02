import { Provider } from '@/components/provider';
import { brandDisplayFont } from '@/lib/brand-font';
import { docsSansFont } from '@/lib/docs-font';
import { withBasePath } from '@/lib/site-path';
import { THEME_BOOTSTRAP_SCRIPT } from '@/lib/theme-bootstrap';
import 'fumadocs-ui/components/image-zoom2.css';
import './styles/tokens.css';
import './global.css';
import './fd-skin.css';
import type { Metadata } from 'next';

const faviconPath = withBasePath('/favicon.svg') ?? '/favicon.svg';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000'),
  title: {
    default: 'WorldFoundry Docs',
    template: '%s | WorldFoundry',
  },
  description:
    'Open-source infrastructure to discover, run, inspect, and evaluate video, 3D/4D, interactive-world, and embodied models.',
  icons: {
    icon: [{ url: faviconPath, type: 'image/svg+xml' }],
    shortcut: faviconPath,
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${brandDisplayFont.variable} ${docsSansFont.variable}`}
    >
      <body className="flex flex-col min-h-screen" suppressHydrationWarning>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }} />
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
