import { withBasePath } from '@/lib/site-path';

import logoMap from '@/lib/model-logo-map.json';

type ModelIdentityMarkProps = {
  id: string;
  name: string;
  provider: string;
  category: string;
  size?: 'small' | 'medium' | 'large';
};

type OrgIdentity = {
  key: string;
  name: string;
  abbr: string;
  src?: string;
};

const orgs = logoMap.orgs as Record<string, OrgIdentity>;
const modelLogos = logoMap.modelLogos as Record<string, string>;

function initials(value: string) {
  const normalized = value
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (normalized.length > 1) {
    return normalized
      .slice(0, 2)
      .map((part) => part[0])
      .join('')
      .toUpperCase();
  }

  return (normalized[0] ?? 'M').slice(0, 2).toUpperCase();
}

function orgFor(id: string) {
  const key = modelLogos[id];
  return key ? orgs[key] : undefined;
}

export function ModelIdentityMark({
  id,
  name,
  category,
  size = 'medium',
}: ModelIdentityMarkProps) {
  const org = orgFor(id);
  const hasLogo = Boolean(org?.src);

  return (
    <span
      className={`wf-model-mark wf-model-mark-${size}${hasLogo ? ' has-logo' : ''}`}
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
