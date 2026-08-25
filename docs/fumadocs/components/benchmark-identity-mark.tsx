import { withBasePath } from '@/lib/site-path';

import logoMap from '@/lib/model-logo-map.json';

type BenchmarkIdentityMarkProps = {
  id: string;
  name: string;
  category: string;
  logoKey?: string;
  size?: 'small' | 'medium' | 'large';
};

type OrgIdentity = {
  key: string;
  name: string;
  abbr: string;
  src?: string;
};

const orgs = logoMap.orgs as Record<string, OrgIdentity>;
const benchmarkLogos = (logoMap.benchmarkLogos ?? {}) as Record<string, string>;

function initials(name: string) {
  const cleaned = name.replace(/[^a-zA-Z0-9]+/g, ' ').trim();
  const parts = cleaned.split(/\s+/).filter(Boolean);

  if (parts.length > 1) {
    return parts
      .slice(0, 2)
      .map((part) => part[0])
      .join('')
      .toUpperCase();
  }

  const word = parts[0] ?? 'BM';
  const caps = word.match(/[A-Z]/g);
  if (caps && caps.length >= 2) {
    return caps.slice(0, 2).join('');
  }

  return word.slice(0, 2).toUpperCase();
}

function orgFor(id: string, logoKey?: string) {
  const key = logoKey || benchmarkLogos[id];
  return key ? orgs[key] : undefined;
}

export function BenchmarkIdentityMark({
  id,
  name,
  category,
  logoKey,
  size = 'medium',
}: BenchmarkIdentityMarkProps) {
  const org = orgFor(id, logoKey);
  const hasLogo = Boolean(org?.src);

  return (
    <span
      className={`wf-model-mark wf-model-mark-${size} wf-benchmark-mark${hasLogo ? ' has-logo' : ''}`}
      data-category={category}
      data-logo={org?.key}
      title={org?.name ?? name}
      aria-hidden="true"
    >
      {org?.src ? (
        <img
          className="wf-model-mark-image"
          src={withBasePath(org.src)}
          alt=""
          width={200}
          height={200}
          draggable={false}
        />
      ) : (
        <span>{org?.abbr ?? initials(name)}</span>
      )}
    </span>
  );
}
