import Link from 'next/link';
import {
  ArrowRight,
  CircleDot,
  GitCompareArrows,
  Layers3,
  MessageCircle,
  MessagesSquare,
  Puzzle,
  UserPlus,
} from 'lucide-react';
import { CatalogCoverage } from '@/components/catalog-coverage';
import { HomeConfigureSection } from '@/components/home-configure-section';
import { HomeHeroMedia } from '@/components/home-hero-media';
import { type HomeRecipeOption } from '@/components/home-run-configurator';
import { SiteFooter } from '@/components/site-footer';
import { SiteHeader } from '@/components/site-header';
import { WorldFoundryWorkflow } from '@/components/worldfoundry-system-map';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  OPENENVISION_GAIA_REPO,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
  WORLDFOUNDRY_WECHAT_QR,
} from '@/lib/site-links';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { withBasePath } from '@/lib/site-path';

const pillars = [
  {
    title: 'Shared',
    description:
      'Environment setup, checkpoints, input shaping, previews, and reporting stay shared while model-specific behavior stays explicit.',
    Icon: Puzzle,
  },
  {
    title: 'Durable',
    description:
      'Expensive outputs survive the process that created them, so they can be inspected, rescored, compared, or audited without rerunning inference.',
    Icon: Layers3,
  },
  {
    title: 'Comparable',
    description:
      'Runs, artifacts, and scorecards share stable IDs and provenance, so teams can compare models and benchmarks on the same evidence.',
    Icon: GitCompareArrows,
  },
];

const capabilities = [
  {
    index: '01',
    title: 'Know what exists',
    description:
      'Manifests expose stable IDs, sources, capabilities, assets, runtime bindings, readiness, and blockers before compute is allocated.',
    href: '/docs/overview/capabilities',
    link: 'See what is included',
  },
  {
    index: '02',
    title: 'Run through shared boundaries',
    description:
      'Pipelines preserve model-native I/O while TUI, CLI, Studio, Python, and MCP reuse the same execution contracts.',
    href: '/docs/overview/design',
    link: 'Understand the design',
  },
  {
    index: '03',
    title: 'Inspect before scoring',
    description:
      'Videos, geometry, actions, trajectories, and traces stay visible on disk and in Studio instead of disappearing inside scripts.',
    href: '/docs/guides/studio',
    link: 'Explore Studio',
  },
  {
    index: '04',
    title: 'Evidence, not headlines',
    description:
      'Benchmark runners preserve per-sample outcomes, coverage, provenance, blockers, reports, and scorecards.',
    href: '/docs/evaluation',
    link: 'Read about evaluation',
  },
];

const featuredModelIds = [
  'wan2.2',
  'ltx-video',
  'cosmos-predict-2.5',
  'hunyuanvideo-1.5',
  'longcat-video',
  'lingbot-video',
  'bernini',
  'matrix-game-2',
];

const featuredModels: HomeRecipeOption[] = [
  ...featuredModelIds
    .map((id) => modelRecipeIndex.recipes.find((recipe) => recipe.id === id))
    .filter((recipe): recipe is NonNullable<typeof recipe> => Boolean(recipe)),
  ...modelRecipeIndex.recipes.filter(
    (recipe) =>
      recipe.status.group === 'verified' && !featuredModelIds.includes(recipe.id),
  ),
]
  .slice(0, 8)
  .map((recipe) => ({
    id: recipe.id,
    name: recipe.name,
    provider: recipe.provider,
    category: recipe.category,
    tasks: recipe.tasks,
    status: recipe.status.label,
    environment: recipe.runtime.environmentName,
    python: recipe.runtime.python,
    cuda: recipe.runtime.cudaLabel,
  }));

const communityLinks = [
  {
    title: 'Join Slack',
    description: 'Real-time help & discussions',
    href: WORLDFOUNDRY_SLACK_INVITE,
    Icon: MessagesSquare,
  },
  {
    title: 'GitHub Issues',
    description: 'Bug reports & feature requests',
    href: WORLDFOUNDRY_GITHUB_ISSUES,
    Icon: CircleDot,
  },
  {
    title: 'GitHub',
    description: 'Source, stars, and contributions',
    href: WORLDFOUNDRY_GITHUB_REPO,
    Icon: MessageCircle,
  },
];

const resourceLinks = [
  {
    label: 'Blog',
    text: 'Updates, technical notes, and release highlights.',
    href: '/blog',
  },
  {
    label: 'Events',
    text: 'Meetups, demos, benchmark sprints, and milestones.',
    href: '/events',
  },
  {
    label: 'OpenEnvision',
    text: 'The lab organization behind WorldFoundry.',
    href: '/openenvision',
  },
];

const ecosystemLinks = [
  {
    label: 'Awesome World Modeling',
    text: 'Curated papers, models, and resources for world modeling.',
    href: OPENENVISION_AWESOME_WORLD_MODELING,
    external: true,
  },
  {
    label: 'BlogrXiv',
    text: 'Curated index for technical AI research blogs and writing.',
    href: OPENENVISION_BLOGXIV_SITE,
    external: true,
  },
  {
    label: 'Gaia',
    text: 'Sibling open-vision project in the same organization.',
    href: OPENENVISION_GAIA_REPO,
    external: true,
  },
  {
    label: 'WorldFoundry on GitHub',
    text: 'Source, issues, discussions, and releases.',
    href: WORLDFOUNDRY_GITHUB_REPO,
    external: true,
  },
];

const workflowRail = ['Discover', 'Prepare', 'Run', 'Inspect', 'Evaluate'] as const;

export default function HomePage() {
  return (
    <main className="pi-home-shell wf-home-shell">
      <div className="wf-home-stage">
        <SiteHeader
          variant="hero"
          active="home"
          wordmarkClassName="wf-home-wordmark"
          languageLinks={[
            { href: '/', label: 'English', current: true },
            { href: '/zh/docs', label: '中文' },
          ]}
        />

        <section className="wf-home-hero" aria-labelledby="wf-home-title">
          <div className="wf-home-hero-showcase">
            <HomeHeroMedia>
              <div className="wf-home-hero-content">
                <p className="wf-home-hero-kicker">Infrastructure for world models</p>
                <h1 id="wf-home-title">WorldFoundry</h1>
                <p className="wf-home-hero-lead">
                  Run, inspect, and evaluate world models in one reproducible workflow.
                </p>
                <div className="wf-home-hero-actions">
                  <Link href="/docs/guides/supported-models" className="wf-home-button wf-home-button-primary">
                    <span>Explore model recipes</span>
                    <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
                  </Link>
                  <Link href="/docs/quickstart" className="wf-home-button wf-home-button-secondary">
                    <span>Start the quickstart</span>
                  </Link>
                </div>
                <Link className="wf-home-hero-catalog-link" href="/docs/guides/supported-models">
                  {modelRecipeIndex.total} manifest-backed model recipes
                  <ArrowRight aria-hidden="true" size={13} strokeWidth={1.7} />
                </Link>
              </div>
            </HomeHeroMedia>
          </div>
        </section>
      </div>

      <div className="wf-home-main">
        <HomeConfigureSection models={featuredModels} />

        <section className="wf-home-pillars wf-home-reveal" aria-labelledby="wf-pillars-title">
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Why WorldFoundry
            </p>
            <h2 id="wf-pillars-title">
              Infrastructure for <span>operating</span> world models.
            </h2>
            <p>
              Shared setup, durable artifacts, and comparable evidence — without forcing every model
              into the same internals.
            </p>
          </header>
          <div className="wf-home-pillar-grid">
            {pillars.map(({ title, description, Icon }) => (
              <article className="wf-home-pillar" key={title}>
                <span className="wf-home-pillar-icon" aria-hidden="true">
                  <Icon size={22} strokeWidth={1.7} />
                </span>
                <h3>{title}</h3>
                <p>{description}</p>
              </article>
            ))}
          </div>
          <div className="wf-home-center-action">
            <Link href="/docs/overview/why-worldfoundry" className="wf-home-text-link">
              Read the benefits and tradeoffs
              <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
            </Link>
          </div>
        </section>

        <section
          className="wf-home-catalog-section wf-home-reveal"
          aria-labelledby="wf-catalog-title"
        >
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Catalog
            </p>
            <h2 id="wf-catalog-title">
              One engine, <span>many</span> models.
            </h2>
            <p>Browse integrated models and benchmarks with the same manifest contracts.</p>
          </header>
          <CatalogCoverage />
        </section>

        <section
          className="wf-home-capabilities wf-home-reveal"
          aria-labelledby="wf-capabilities-title"
        >
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              End to end
            </p>
            <h2 id="wf-capabilities-title">
              Operate with <span>shared</span> contracts.
            </h2>
            <p>
              Catalog, runtime, workspace, and evaluation share identities and durable outputs.
            </p>
          </header>
          <div className="wf-home-capability-grid">
            {capabilities.map((capability) => (
              <article className="wf-home-capability-card" key={capability.title}>
                <span className="wf-home-capability-index" aria-hidden="true">
                  {capability.index}
                </span>
                <h3>{capability.title}</h3>
                <p>{capability.description}</p>
                <Link href={capability.href}>
                  {capability.link}
                  <ArrowRight aria-hidden="true" size={15} strokeWidth={1.8} />
                </Link>
              </article>
            ))}
          </div>
        </section>

        <section
          className="wf-home-workflow-section wf-home-reveal"
          aria-labelledby="wf-workflow-title"
        >
          <header className="wf-home-workflow-intro">
            <p className="wf-home-workflow-badge">
              <span aria-hidden="true" />
              Five-stage pipeline
            </p>
            <h2 id="wf-workflow-title">
              How it <span>works</span>
            </h2>
            <p className="wf-home-workflow-lead">
              Every surface follows the same sequence. The artifact is the handoff between model
              execution and benchmark evaluation.
            </p>
            <div className="wf-home-workflow-rail" aria-hidden="true">
              <span className="wf-home-workflow-rail-progress" />
              <ol>
                {workflowRail.map((label, index) => (
                  <li key={label} style={{ '--wf-rail-index': index } as React.CSSProperties}>
                    <span />
                    <strong>{label}</strong>
                  </li>
                ))}
              </ol>
            </div>
          </header>
          <div className="wf-home-workflow-frame">
            <div className="wf-home-workflow-ambient" aria-hidden="true">
              <span className="wf-home-workflow-ambient-glow" />
              <span className="wf-home-workflow-ambient-grid" />
            </div>
            <WorldFoundryWorkflow variant="home" />
          </div>
        </section>

        <section
          className="wf-home-community wf-home-reveal"
          aria-labelledby="wf-community-title"
        >
          <div className="wf-home-community-panel">
            <div className="wf-home-community-copy">
              <p className="wf-home-community-badge">
                <span aria-hidden="true" />
                Everyone welcome
              </p>
              <h2 id="wf-community-title">
                Got questions?
                <span> We&apos;re here to help.</span>
              </h2>
              <p>
                Whether you&apos;re just getting started or debugging a complex run, the community is
                open to everyone. No question is too basic.
              </p>
              <ul className="wf-home-community-signals">
                <li>
                  <MessageCircle aria-hidden="true" size={16} strokeWidth={1.8} />
                  Fast &amp; friendly responses
                </li>
                <li>
                  <UserPlus aria-hidden="true" size={16} strokeWidth={1.8} />
                  Active maintainers
                </li>
              </ul>
            </div>

            <div className="wf-home-community-actions">
              <ul className="wf-home-community-links">
                {communityLinks.map(({ title, description, href, Icon }) => (
                  <li key={title}>
                    <a href={href} target="_blank" rel="noreferrer" className="wf-home-community-link">
                      <Icon aria-hidden="true" size={20} strokeWidth={1.7} />
                      <span>
                        <strong>{title}</strong>
                        <em>{description}</em>
                      </span>
                      <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
                    </a>
                  </li>
                ))}
              </ul>

              <figure className="wf-home-community-qr">
                <img
                  src={withBasePath(WORLDFOUNDRY_WECHAT_QR)}
                  alt="WorldFoundry WeChat community QR code"
                  width={280}
                  height={280}
                />
                <figcaption>
                  <strong>WeChat Community</strong>
                  <span>Scan to join. The QR code is updated periodically if it expires.</span>
                </figcaption>
              </figure>
            </div>
          </div>
        </section>

        <section className="wf-home-resources wf-home-reveal" aria-labelledby="wf-resources-title">
          <header className="wf-home-center-intro">
            <p className="wf-home-section-badge">
              <span aria-hidden="true" />
              Around the project
            </p>
            <h2 id="wf-resources-title">
              Resources &amp; <span>ecosystem</span>
            </h2>
            <p>Notes, events, and sibling projects in the OpenEnvision network.</p>
          </header>
          <div className="wf-home-resource-grid">
            {resourceLinks.map((item) => (
              <Link className="wf-home-resource-card" href={item.href} key={item.label}>
                <h3>{item.label}</h3>
                <p>{item.text}</p>
                <span aria-hidden="true">
                  <ArrowRight size={15} strokeWidth={1.8} />
                </span>
              </Link>
            ))}
          </div>
          <ul className="wf-home-ecosystem-chips" aria-label="Ecosystem links">
            {ecosystemLinks.map((item) => (
              <li key={item.label}>
                <a href={item.href} target="_blank" rel="noreferrer">
                  <strong>{item.label}</strong>
                  <span>{item.text}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <SiteFooter />
    </main>
  );
}
