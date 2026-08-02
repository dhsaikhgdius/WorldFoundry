'use client';

import { SiteGitHubLink } from '@/components/site-github-link';
import { SiteThemeSwitch } from '@/components/site-theme-switch';

export function DocsSidebarFooter() {
  return (
    <div className="pi-doc-sidebar-footer">
      <SiteGitHubLink />
      <SiteThemeSwitch />
    </div>
  );
}
