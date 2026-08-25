# PR: Replace GitHub avatar icons with publishing-institution logos

> Draft PR body for branch `cursor/org-logos-8cd6` (base: `main`).
> The docs site previously used personal GitHub avatars as model/benchmark
> icons; this PR replaces every icon with the publishing institution's mark.

## Summary

Every model and benchmark icon on the documentation site now shows the
**publishing institution** (company first, then university, then lab), never a
personal GitHub avatar. Nothing is fetched from `avatars.githubusercontent.com`
anymore, and the 194 GitHub avatar files previously committed under
`public/model-logos/` are deleted.

## What changed

### Data

- `docs/fumadocs/lib/model-logo-map.json` — rewritten. New schema:
  - `orgs`: 114 institutions keyed by org id (`tencent`, `alibaba`, `nvidia`,
    `pku`, `stanford`, …) with formal `name`, `abbr`, optional `src`
    (logo path), `source` URL and `license`.
  - `modelLogos`: model id → org id (263 of 283 models mapped).
  - `benchmarkLogos`: benchmark id → org id (63 of 71 benchmarks mapped).
- `docs/fumadocs/lib/benchmark-catalog-status.json` — every `logoKey`
  rewritten from a GitHub login to an org id (64 values).

### Assets

- New `docs/fumadocs/public/org-logos/` (85 files, 1.9 MB total): official
  logos sourced from Wikimedia Commons (public domain / CC), Simple Icons
  (CC0) and official GitHub organization brand avatars. Provenance, license
  and usage of every file is documented in
  `docs/fumadocs/public/org-logos/SOURCES.md`.
- Deleted `docs/fumadocs/public/model-logos/` (194 GitHub avatars, 6.9 MB).

### Code

- `docs/fumadocs/components/model-identity-mark.tsx` and
  `benchmark-identity-mark.tsx` — resolve the org for each id; `title` shows
  the institution's formal name ("Tencent", "Peking University"); when an org
  has no logo file the mark renders the institution abbreviation (JD, SAIL,
  ZJU); only entries with no resolved institution fall back to neutral
  initials of the entry name. All call sites (catalog list pages, detail
  pages, home configurator) pick this up automatically.
- `docs/fumadocs/scripts/generate-org-logos.py` — new, human-reviewed mapping
  source of truth. Regenerates the logo map, benchmark `logoKey`s and
  `SOURCES.md`. It never derives a logo from a GitHub owner avatar.
- `docs/fumadocs/next.config.mjs`, `docs/fumadocs/scripts/dev-local-ssd.sh` —
  updated `model-logos` references to `org-logos`.

## Institution coverage

| Bucket | Count | Examples |
| --- | --- | --- |
| Official logo file | 85 orgs | Tencent, Alibaba, NVIDIA, Google DeepMind, Meituan, Physical Intelligence, Stanford, Oxford, PKU, Tsinghua, SNU, Mila |
| Abbreviation only (no redistributable official file found) | 29 orgs | JD.com (JD), Shanghai AI Laboratory (SAIL), ZJU, SJTU, HUST, HKU, CUHK, CAS, USTC, NUS, NTU, HKUST, Waterloo, BUAA, CityU |
| Unresolved (neutral initials fallback) | 20 models, 8 benchmarks | entries with no verifiable publishing institution in catalog/paper metadata |

University emblems for the abbreviation-only bucket exist on Wikipedia only as
non-free fair-use media, so they cannot legally be committed; those orgs render
their abbreviation with the full formal name in the tooltip.

## Acceptance checks

- Zero entries in `model-logo-map.json` reference personal GitHub avatars or
  `avatars.githubusercontent.com`.
- Spot checks: `videophy` → Google, `t2v-compbench` → HKU (abbr),
  `worldscore` → Stanford, `wrbench` → USTC (abbr), `likephys` → Oxford,
  `vbench` → Shanghai AI Laboratory (abbr), `physics-iq` → Google DeepMind,
  `wbench` → Meituan, `vmbench` → Amap (Alibaba).
- Hunyuan* → Tencent, Qwen/Wan* → Alibaba, Cosmos* → NVIDIA,
  LongCat → Meituan, π0 family → Physical Intelligence.
- `npm run types:check` and `npm run build` (static export) pass; exported
  HTML contains no `model-logos` references and ships all 85 org logos.
