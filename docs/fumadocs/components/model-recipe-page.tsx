import Link from 'next/link';
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  ExternalLink,
  FileCode2,
  GitBranch,
  Package,
  ShieldCheck,
} from 'lucide-react';

import { DocsMobileNavToggle } from '@/components/docs-mobile-nav';
import { DocsReadingProgress } from '@/components/docs-reading-progress';
import { DocsScrollBridge } from '@/components/docs-scroll-bridge';
import { ModelCommandBuilder } from '@/components/model-command-builder';
import { ModelIdentityMark } from '@/components/model-identity-mark';
import { SiteHeader } from '@/components/site-header';
import type { ModelRecipe, ModelRecipeSource } from '@/lib/model-recipe-types';
import { getRelatedModelRecipes, modelRecipeData } from '@/lib/model-recipes';

type Locale = 'en' | 'zh';

const copy = {
  en: {
    docs: 'Docs',
    models: 'Model recipes',
    back: 'Back to all models',
    menu: 'Menu',
    close: 'Close',
    language: 'Language',
    overview: 'Overview',
    use: 'Use this model',
    whatIs: 'What this model is',
    whatIsIntro:
      'Plain-language summary assembled from the catalog manifest — what the model does, what it takes in, and what it writes out.',
    curatedNarrative: 'Curated in the catalog manifest',
    synthesizedNarrative: 'Synthesized from recorded manifest fields',
    architecture: 'Architecture',
    usageNotes: 'Usage notes',
    publisher: 'Publisher',
    publisherKindCompany: 'Company',
    publisherKindUniversity: 'University',
    publisherKindLab: 'Research lab',
    highlights: 'At a glance',
    modalities: 'Modalities',
    modalityInputs: 'Inputs',
    modalityOutputs: 'Outputs',
    useCases: 'Typical uses',
    hardwareGuidance: 'Hardware guidance',
    minVram: 'Minimum VRAM',
    recommendedHardware: 'Recommended',
    tasksLabel: 'Tasks',
    inputsOverview: 'Inputs',
    outputsOverview: 'Outputs',
    compatibility: 'Compatibility & versions',
    sourcesSection: 'Papers, code & weights',
    sourcesIntro:
      'Where this model comes from: the recorded project page, paper, upstream source, and weight repositories, with pinned revisions when the manifest records them.',
    sourceKindProject: 'Project page',
    sourceKindPaper: 'Paper',
    sourceKindDocs: 'Documentation',
    sourceKindSource: 'Source code',
    sourceKindWeights: 'Weights',
    sourceLink: 'Link',
    sourceRevisionCol: 'Pinned revision',
    noSources: 'No official sources are recorded in the catalog manifest.',
    install: 'Environment & install',
    assets: 'Weights & local assets',
    launch: 'Run & outputs',
    launchBuilderHint: 'Use the command builder above to pick a variant and copy its run command.',
    variantsSection: 'Variants & pipeline bindings',
    variantsIntro:
      'Every recorded variant with the runtime profile, pipeline binding, and integration status that actually serves it.',
    variantCol: 'Variant',
    variantTaskCol: 'Task',
    variantProfileCol: 'Runtime profile',
    variantBindingCol: 'Pipeline binding',
    variantStatusCol: 'Status',
    noVariants: 'This model is recorded as a single route without named variants.',
    benchmarksSection: 'Evaluation benchmarks',
    benchmarksIntro:
      'Benchmarks recommended by the manifest or referenced in its evidence notes. Open one in the Benchmark Hub for metrics and runtime details.',
    benchmarkFromDocs: 'Recommended',
    benchmarkFromManifest: 'Referenced in evidence',
    benchmarksBrowse: 'Browse all benchmarks in the Evaluation section',
    evidence: 'Evidence & limitations',
    limitations: 'Known limitations',
    manifestNotes: 'Manifest notes',
    advanced: 'Advanced — manifest fields',
    advancedIntro:
      'Internal pipeline and runtime metadata recorded by the catalog manifest. Most users do not need these to run a model.',
    notRecorded: 'Not recorded',
    runtime: 'Runtime',
    python: 'Python',
    cuda: 'CUDA',
    pytorch: 'PyTorch',
    checkpoint: 'Checkpoint',
    checkpointRevision: 'Checkpoint revision',
    sourceRevision: 'Source revision',
    output: 'Output',
    integration: 'Integration',
    runnerEvidence: 'Runner evidence',
    demoEvidence: 'Native demo evidence',
    environment: 'Environment',
    environmentKind: 'Environment kind',
    profile: 'Runtime profile',
    binding: 'Pipeline binding',
    runner: 'Runner',
    pipeline: 'Pipeline target',
    backend: 'Backend stage',
    runtimeStatus: 'Runtime status',
    driverStatus: 'Driver status',
    packages: 'Package constraints',
    condaPackages: 'Conda packages',
    validationImports: 'Validation imports',
    installCommand: 'Install command',
    checkCommand: 'Check local assets before allocating compute',
    prepareCommand: 'Prepare checkpoints and source assets',
    inputs: 'Input contract',
    field: 'Field',
    recordedContract: 'Recorded contract',
    artifacts: 'Artifact contract',
    artifactKind: 'Artifact kind',
    filename: 'Filename / path',
    noContract: 'No input schema is recorded in the selected runtime profile.',
    noArtifacts: 'No artifact contract is recorded for this model.',
    noCheckpoints: 'No checkpoint repository is recorded in the catalog manifest.',
    variantContractIntro:
      'Select a variant in the run builder above to inspect its pipeline-specific input and artifact contract.',
    installIntro:
      'The environment resolver reads the recorded profile and chooses the unified or dedicated environment shown here. Copy the install command to materialize it.',
    assetsIntro:
      'Run the local check before allocating compute. Gated, private, and license fields below come directly from the checkpoint manifest.',
    launchIntro:
      'The generated command selects the recorded task profile and writes durable result manifests and artifacts.',
    evidenceIntro:
      'Catalog integration, native-demo parity, and runner parity are independent records. Anything unverified stays "Not recorded" — it is never upgraded to a support claim.',
    useIntro: 'Choose a recorded route, prepare its requirements, then run it.',
    readyNow: 'Runnable route recorded',
    readyNowDescription: 'A runnable WorldFoundry pipeline is recorded for this model — copy the command below.',
    needsValidation: 'Route recorded; parity still in progress',
    needsValidationDescription: 'A runnable route is recorded; upstream parity is still being verified.',
    referenceOnly: 'Reference or planned entry',
    referenceOnlyDescription: 'This entry records upstream provenance, not a runnable WorldFoundry route yet.',
    referenceHeadline: 'Not runnable in WorldFoundry yet',
    referenceBody:
      'This catalog entry records where the model comes from and what it is, but no runnable WorldFoundry pipeline is bound to it. Commands and environments below are intentionally omitted.',
    relatedRunnable: 'Runnable recipes in the same family',
    supportedRoutes: 'Runnable variants',
    officialSources: 'Official sources',
    provenance: 'Recipe provenance',
    catalogManifest: 'Catalog manifest',
    related: 'Related recipes',
    details: 'Open recipe',
  },
  zh: {
    docs: '文档',
    models: '模型配方',
    back: '返回全部模型',
    menu: '菜单',
    close: '关闭',
    language: '语言',
    overview: '概览',
    use: '使用此模型',
    whatIs: '这个模型是什么',
    whatIsIntro: '由 catalog manifest 生成的平实说明——模型做什么、接受什么输入、写出什么。',
    curatedNarrative: '来自 catalog manifest 的人工撰写内容',
    synthesizedNarrative: '由 manifest 已记录字段自动合成',
    architecture: '架构',
    usageNotes: '使用要点',
    publisher: '发表机构',
    publisherKindCompany: '企业',
    publisherKindUniversity: '高校',
    publisherKindLab: '研究机构',
    highlights: '速览',
    modalities: '模态',
    modalityInputs: '输入',
    modalityOutputs: '输出',
    useCases: '典型用途',
    hardwareGuidance: '硬件建议',
    minVram: '最低显存',
    recommendedHardware: '推荐配置',
    tasksLabel: '任务',
    inputsOverview: '输入',
    outputsOverview: '输出',
    compatibility: '兼容性与版本',
    sourcesSection: '论文、代码与权重出处',
    sourcesIntro: '模型的来源：已记录的项目主页、论文、上游源码与权重仓库；若 manifest 固定了 revision 也一并显示。',
    sourceKindProject: '项目主页',
    sourceKindPaper: '论文',
    sourceKindDocs: '文档',
    sourceKindSource: '源码',
    sourceKindWeights: '权重',
    sourceLink: '链接',
    sourceRevisionCol: '固定 revision',
    noSources: 'Catalog manifest 没有记录官方来源。',
    install: '环境与安装',
    assets: '权重与本地资产',
    launch: '运行与输出',
    launchBuilderHint: '用上方的命令构建器选择 variant，复制对应的运行命令。',
    variantsSection: '变体与 Pipeline Binding',
    variantsIntro: '每个已记录 variant 及其实际使用的 runtime profile、pipeline binding 与集成状态。',
    variantCol: 'Variant',
    variantTaskCol: '任务',
    variantProfileCol: 'Runtime profile',
    variantBindingCol: 'Pipeline binding',
    variantStatusCol: '状态',
    noVariants: '该模型记录为单一路径，没有命名 variant。',
    benchmarksSection: '评测 Benchmark',
    benchmarksIntro: 'Manifest 推荐或其证据记录中提到的评测。点开可在 Benchmark Hub 查看指标与运行环境细节。',
    benchmarkFromDocs: '推荐',
    benchmarkFromManifest: '证据中提及',
    benchmarksBrowse: '在 Evaluation 部分浏览全部 benchmark',
    evidence: '证据与限制',
    limitations: '已知限制',
    manifestNotes: 'Manifest 备注',
    advanced: '高级 — manifest 字段',
    advancedIntro: 'catalog manifest 记录的内部 pipeline 与 runtime 元数据。多数用户运行模型时用不到这些。',
    notRecorded: '未记录',
    runtime: '运行时',
    python: 'Python',
    cuda: 'CUDA',
    pytorch: 'PyTorch',
    checkpoint: 'Checkpoint',
    checkpointRevision: 'Checkpoint revision',
    sourceRevision: '源码 revision',
    output: '输出',
    integration: '集成状态',
    runnerEvidence: 'Runner 证据',
    demoEvidence: '原生 Demo 证据',
    environment: '环境',
    environmentKind: '环境类型',
    profile: 'Runtime profile',
    binding: 'Pipeline binding',
    runner: 'Runner',
    pipeline: 'Pipeline target',
    backend: 'Backend stage',
    runtimeStatus: 'Runtime 状态',
    driverStatus: 'Driver 状态',
    packages: '依赖版本约束',
    condaPackages: 'Conda 依赖',
    validationImports: '验证 Imports',
    installCommand: '安装命令',
    checkCommand: '分配算力前先检查本地资产',
    prepareCommand: '准备 checkpoint 与源码资产',
    inputs: '输入契约',
    field: '字段',
    recordedContract: '已记录契约',
    artifacts: 'Artifact 契约',
    artifactKind: 'Artifact 类型',
    filename: '文件名 / 路径',
    noContract: '所选 runtime profile 没有记录输入 schema。',
    noArtifacts: '该模型没有记录 artifact 契约。',
    noCheckpoints: 'Catalog manifest 没有记录 checkpoint 仓库。',
    variantContractIntro: '请在上方运行构建器中选择 variant，查看该 Pipeline 对应的输入与 Artifact 契约。',
    installIntro: '环境解析器会读取已记录 profile，并选择此处显示的统一或独立环境。复制安装命令即可创建它。',
    assetsIntro: '分配算力前先做本地检查；gated、private 与 license 字段直接来自 checkpoint manifest。',
    launchIntro: '生成的命令会选择已记录的 task profile，并持久保存结果 manifest 与 artifact。',
    evidenceIntro: 'Catalog 集成、原生 demo parity 与 runner parity 是三条独立记录。未验证的内容保持“未记录”，绝不升级为支持声明。',
    useIntro: '选择已记录路径，确认环境与资产，然后运行。',
    readyNow: '已记录可运行路径',
    readyNowDescription: 'WorldFoundry 已记录该模型的可运行 Pipeline——复制下方命令即可。',
    needsValidation: '已记录路径，仍在验证 parity',
    needsValidationDescription: '已记录可运行路径；上游 parity 仍在验证中。',
    referenceOnly: '参考或计划条目',
    referenceOnlyDescription: '此条目记录了上游来源，但尚无可运行的 WorldFoundry 路径。',
    referenceHeadline: '尚不可在 WorldFoundry 中运行',
    referenceBody:
      '该 catalog 条目记录了模型来源与基本信息，但未绑定可运行的 WorldFoundry Pipeline。下方命令与环境因此省略。',
    relatedRunnable: '同族中可运行的配方',
    supportedRoutes: '可运行变体',
    officialSources: '官方来源',
    provenance: '配方溯源',
    catalogManifest: 'Catalog manifest',
    related: '相关配方',
    details: '打开配方',
  },
} as const;

const sourceKindLabelKeys = {
  project: 'sourceKindProject',
  paper: 'sourceKindPaper',
  docs: 'sourceKindDocs',
  source: 'sourceKindSource',
  weights: 'sourceKindWeights',
} as const;

const modalityLabelsZh: Record<string, string> = {
  text: '文本',
  image: '图像',
  video: '视频',
  audio: '音频',
  action: '动作',
  depth: '深度',
  trajectory: '轨迹',
  'camera pose': '相机位姿',
  '3D scene': '3D 场景',
  '4D scene': '4D 场景',
  'world state': '世界状态',
  mesh: '网格',
  'point cloud': '点云',
};

function modelBasePath(locale: Locale) {
  return `${locale === 'zh' ? '/zh' : ''}/docs/guides/supported-models`;
}

function formatStatus(value: string, fallback: string) {
  if (!value || value === 'not_recorded') return fallback;
  return value.replaceAll('_', ' ').replaceAll('-', ' ');
}

function revision(recipe: ModelRecipe, kind: 'checkpoint' | 'source') {
  if (kind === 'checkpoint') {
    return recipe.checkpoints.find((item) => item.revision)?.revision;
  }
  return recipe.sources.find((item) => item.revision)?.revision;
}

function artifactSummary(recipe: ModelRecipe, fallback: string) {
  if (recipe.artifacts.length === 0) return fallback;
  return recipe.artifacts
    .slice(0, 2)
    .map((artifact) => artifact.filename || artifact.kind)
    .join(', ');
}

function runnerIsVerified(value: string) {
  const normalized = value.toLowerCase();
  return !normalized.includes('partial') && (
    normalized.includes('verified') || normalized.includes('validated') || normalized.includes('passed')
  );
}

function isRunnableVariant(value: { pipelineTarget: string | null; status: string }) {
  const status = value.status.toLowerCase().replaceAll('-', '_');
  const unavailable = ['not_recorded', 'planned', 'profile', 'profile_only', 'blocked', 'unavailable', 'missing'];
  return Boolean(value.pipelineTarget) && !unavailable.some((marker) => status.includes(marker));
}

function modalityLabel(value: string, locale: Locale) {
  return locale === 'zh' ? modalityLabelsZh[value] ?? value : value;
}

function sourceHost(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

function DefinitionRow({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value || '—'}</dd>
    </div>
  );
}

function CommandBlock({ label, children }: { label?: string; children: string }) {
  return (
    <div className="wf-recipe-command-block">
      {label ? <span className="wf-recipe-command-label">{label}</span> : null}
      <pre className="wf-recipe-static-command">
        <code>{children}</code>
      </pre>
    </div>
  );
}

function SourcesTable({ sources, locale }: { sources: ModelRecipeSource[]; locale: Locale }) {
  const t = copy[locale];
  return (
    <table className="wf-recipe-sources-table">
      <thead>
        <tr>
          <th>{locale === 'zh' ? '类型' : 'Kind'}</th>
          <th>{t.sourceLink}</th>
          <th>{t.sourceRevisionCol}</th>
        </tr>
      </thead>
      <tbody>
        {sources.map((source) => (
          <tr key={`${source.kind}:${source.url}`}>
            <td>
              <span className={`wf-recipe-source-kind is-${source.kind}`}>
                {t[sourceKindLabelKeys[source.kind]]}
              </span>
            </td>
            <td>
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.label}
                <small>{sourceHost(source.url)}</small>
                <ExternalLink aria-hidden="true" size={11} />
              </a>
            </td>
            <td>{source.revision ? <code>{source.revision.slice(0, 12)}</code> : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ModelRecipePage({ recipe, locale }: { recipe: ModelRecipe; locale: Locale }) {
  const t = copy[locale];
  const basePath = modelBasePath(locale);
  const related = getRelatedModelRecipes(recipe, 4);
  const taskLabel = recipe.tasks.length > 0 ? recipe.tasks.join(' · ') : t.notRecorded;
  const docs = recipe.docs;
  const overviewParagraphs = locale === 'zh' && docs.overviewZh.length > 0 ? docs.overviewZh : docs.overview;
  const architectureParagraphs =
    locale === 'zh' && docs.architectureZh.length > 0 ? docs.architectureZh : docs.architecture;
  const usageNoteParagraphs = locale === 'zh' && docs.usageNotesZh.length > 0 ? docs.usageNotesZh : docs.usageNotes;
  const highlights = locale === 'zh' && docs.highlightsZh.length > 0 ? docs.highlightsZh : docs.highlights;
  const useCases = locale === 'zh' && docs.useCasesZh.length > 0 ? docs.useCasesZh : docs.useCases;
  const limitations = locale === 'zh' && docs.limitationsZh.length > 0 ? docs.limitationsZh : docs.limitations;
  const publisher = docs.publisher;
  const publisherName = publisher ? (locale === 'zh' ? publisher.nameZh : publisher.name) : null;
  const publisherKindLabel = publisher?.kind === 'company'
    ? t.publisherKindCompany
    : publisher?.kind === 'university'
      ? t.publisherKindUniversity
      : publisher?.kind === 'lab'
        ? t.publisherKindLab
        : null;
  const hasModalities = docs.modalities.inputs.length > 0 || docs.modalities.outputs.length > 0;
  const hasHardware = docs.hardware.minVramGb !== null
    || Boolean(docs.hardware.recommended)
    || docs.hardware.notes.length > 0;
  const environmentKind =
    recipe.runtime.environmentKind === 'dedicated'
      ? locale === 'zh'
        ? '独立环境'
        : 'Dedicated environment'
      : recipe.runtime.environmentKind === 'unified'
        ? locale === 'zh'
          ? '统一环境'
          : 'Unified environment'
        : t.notRecorded;
  const routedVariants = recipe.variants.filter(isRunnableVariant);
  const hasRunnableRoute = routedVariants.length > 0 || (
    recipe.variants.length === 0
    && Boolean(recipe.runtime.pipelineTarget)
    && !['planned', 'profile', 'blocked'].includes(recipe.status.group)
  );
  const routeIsVerified = runnerIsVerified(recipe.status.runner);
  const isReferenceOnly = !hasRunnableRoute;
  const usageState = isReferenceOnly
    ? 'reference'
    : routeIsVerified
      ? 'ready'
      : 'caution';
  const usageTitle = usageState === 'ready'
    ? t.readyNow
    : usageState === 'caution'
      ? t.needsValidation
      : t.referenceOnly;
  const usageDescription = usageState === 'ready'
    ? t.readyNowDescription
    : usageState === 'caution'
      ? t.needsValidationDescription
      : t.referenceOnlyDescription;
  const runnablePeers = isReferenceOnly
    ? modelRecipeData.recipes
        .filter((candidate) => candidate.id !== recipe.id
          && candidate.category === recipe.category
          && ['verified', 'integrated'].includes(candidate.status.group))
        .slice(0, 4)
    : [];
  const benchmarkHubBase = `${locale === 'zh' ? '/zh' : ''}/docs/evaluation/benchmark-hub`;
  const sections: Array<[string, string]> = [
    ['overview', t.overview],
    ['use', t.use],
    ['what-is', t.whatIs],
    ['sources', t.sourcesSection],
    ['launch', t.launch],
    // The install section is fully omitted for reference-only entries, so its
    // anchor must not appear in the sidebar either.
    ...(isReferenceOnly ? [] : ([['install', t.install]] as Array<[string, string]>)),
    ['assets', t.assets],
    ['variants', t.variantsSection],
    ...(docs.benchmarks.length > 0 ? ([['benchmarks', t.benchmarksSection]] as Array<[string, string]>) : []),
    ['evidence', t.evidence],
    ['advanced', t.advanced],
  ];
  let sectionNumber = 0;
  const nextSectionNumber = () => String(sectionNumber++).padStart(2, '0');

  return (
    <main className="pi-doc-shell wf-recipe-shell" lang={locale}>
      <DocsScrollBridge />
      <SiteHeader
        variant="solid"
        active="models"
        docsHref={locale === 'zh' ? '/zh/docs' : '/docs'}
        docsLabel={t.docs}
        languageAriaLabel={t.language}
        beforeInner={<DocsReadingProgress />}
        brandLeading={<DocsMobileNavToggle openLabel={t.menu} closeLabel={t.close} />}
        languageLinks={[
          {
            href: `/docs/guides/supported-models/${recipe.id}`,
            label: 'English',
            current: locale === 'en',
          },
          {
            href: `/zh/docs/guides/supported-models/${recipe.id}`,
            label: '中文',
            current: locale === 'zh',
          },
        ]}
      />

      <div className="pi-doc-frame wf-recipe-frame">
        <aside className="pi-doc-sidebar wf-recipe-sidebar" id="pi-doc-sidebar" aria-label={t.models}>
          <Link className="wf-recipe-back" href={basePath}>
            <ArrowLeft aria-hidden="true" size={14} />
            {t.back}
          </Link>
          <div className="wf-recipe-sidebar-identity">
            <ModelIdentityMark
              id={recipe.id}
              name={recipe.name}
              provider={recipe.provider}
              category={recipe.category}
              size="small"
            />
            <div>
              <span>{recipe.categoryLabel}</span>
              <strong>{recipe.name}</strong>
              <code>{recipe.id}</code>
            </div>
          </div>
          <nav className="wf-recipe-section-nav" aria-label={t.models}>
            {sections.map(([id, label]) => (
              <a href={`#${id}`} key={id}>
                {label}
              </a>
            ))}
          </nav>
          <div className="wf-recipe-sidebar-source">
            <FileCode2 aria-hidden="true" size={14} />
            <span>{t.catalogManifest}</span>
            <code>{recipe.catalogPath}</code>
          </div>
        </aside>

        <div className="pi-doc-main">
          <article className="pi-doc-article wf-recipe-article">
            <div className="pi-doc-article-inner">
              <nav className="wf-recipe-breadcrumb" aria-label="Breadcrumb">
                <Link href={basePath}>{t.models}</Link>
                <span aria-hidden="true">/</span>
                <span>{recipe.categoryLabel}</span>
                <span aria-hidden="true">/</span>
                <strong>{recipe.name}</strong>
              </nav>

              <header className="wf-recipe-hero" id="overview">
                <p className="wf-recipe-eyebrow">
                  {recipe.categoryLabel} · {taskLabel}
                </p>
                <div className="wf-recipe-title-row">
                  <ModelIdentityMark
                    id={recipe.id}
                    name={recipe.name}
                    provider={recipe.provider}
                    category={recipe.category}
                    size="large"
                  />
                  <div>
                    <h1>{recipe.name}</h1>
                    <p>
                      {publisherName ?? recipe.provider}
                      {publisherKindLabel ? (
                        <span className="wf-recipe-publisher-kind"> · {publisherKindLabel}</span>
                      ) : null}
                    </p>
                  </div>
                </div>
                <p className="wf-recipe-summary">{recipe.summary}</p>
                <div className="wf-recipe-hero-meta">
                  <span className={`wf-recipe-status wf-recipe-status-${recipe.status.group}`}>
                    <ShieldCheck aria-hidden="true" size={13} />
                    {recipe.status.label}
                  </span>
                  <span>{environmentKind}</span>
                  {revision(recipe, 'source') ? <span>{locale === 'zh' ? '源码已固定' : 'Pinned source'}</span> : null}
                  <code>{recipe.id}</code>
                </div>
                {hasModalities ? (
                  <div className="wf-recipe-modality-strip" aria-label={t.modalities}>
                    {docs.modalities.inputs.map((item) => (
                      <span className="wf-recipe-modality is-input" key={`in:${item}`}>
                        {modalityLabel(item, locale)}
                      </span>
                    ))}
                    {docs.modalities.inputs.length > 0 && docs.modalities.outputs.length > 0 ? (
                      <ArrowRight aria-hidden="true" size={13} />
                    ) : null}
                    {docs.modalities.outputs.map((item) => (
                      <span className="wf-recipe-modality is-output" key={`out:${item}`}>
                        {modalityLabel(item, locale)}
                      </span>
                    ))}
                  </div>
                ) : null}
                {recipe.sources.length > 0 ? (
                  <nav className="wf-recipe-source-links" aria-label={t.officialSources}>
                    {recipe.sources.slice(0, 5).map((source) => (
                      <a href={source.url} target="_blank" rel="noreferrer" key={`${source.kind}:${source.url}`}>
                        {source.label}
                        <ExternalLink aria-hidden="true" size={11} />
                      </a>
                    ))}
                  </nav>
                ) : null}
              </header>

              <div className={`wf-recipe-runbar is-${usageState}`} id="use">
                <div className="wf-recipe-runbar-head">
                  <ShieldCheck aria-hidden="true" size={15} />
                  <div>
                    <span className="wf-recipe-runbar-label">{t.integration}</span>
                    <strong>{usageTitle}</strong>
                    <p>{usageDescription}</p>
                  </div>
                </div>
                {routedVariants.length > 0 ? (
                  <div className="wf-recipe-runbar-routes">
                    <span>{t.supportedRoutes}</span>
                    <div>
                      {routedVariants.map((variant) => (
                        <code key={variant.id}>{variant.id}</code>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              {isReferenceOnly ? (
                <section className="wf-recipe-reference-card" aria-labelledby="wf-reference-heading">
                  <h2 id="wf-reference-heading">{t.referenceHeadline}</h2>
                  <p>{t.referenceBody}</p>
                  <dl className="wf-recipe-reference-meta">
                    {publisherName || recipe.provider ? (
                      <DefinitionRow
                        label={publisherName ? t.publisher : 'Provider'}
                        value={publisherName ?? recipe.provider}
                      />
                    ) : null}
                    <DefinitionRow label={t.tasksLabel} value={recipe.tasks.length > 0 ? recipe.tasks.join(' · ') : null} />
                    <DefinitionRow label={t.integration} value={formatStatus(recipe.status.integration, t.notRecorded)} />
                    <DefinitionRow label={t.runnerEvidence} value={formatStatus(recipe.status.runner, t.notRecorded)} />
                    <DefinitionRow label={t.demoEvidence} value={formatStatus(recipe.status.demo, t.notRecorded)} />
                  </dl>
                  {runnablePeers.length > 0 ? (
                    <div className="wf-recipe-reference-peers">
                      <span>{t.relatedRunnable}</span>
                      <div>
                        {runnablePeers.map((peer) => (
                          <Link href={`${basePath}/${peer.id}`} key={peer.id}>
                            <strong>{peer.name}</strong>
                            <small>{peer.provider}</small>
                          </Link>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </section>
              ) : (
                <ModelCommandBuilder recipe={recipe} locale={locale} />
              )}

              <section className="wf-recipe-section wf-recipe-what-is" id="what-is">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.whatIs}</h2>
                    <p>{t.whatIsIntro}</p>
                  </div>
                </header>
                <div className="wf-recipe-overview-prose">
                  {overviewParagraphs.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </div>
                <p className="wf-recipe-narrative-origin">
                  {docs.curated ? t.curatedNarrative : t.synthesizedNarrative}
                </p>
                {highlights.length > 0 ? (
                  <div className="wf-recipe-highlights" aria-label={t.highlights}>
                    <span>{t.highlights}</span>
                    <ul>
                      {highlights.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {architectureParagraphs.length > 0 ? (
                  <div className="wf-recipe-usecases wf-recipe-architecture">
                    <h3>{t.architecture}</h3>
                    {architectureParagraphs.map((paragraph) => (
                      <p key={paragraph}>{paragraph}</p>
                    ))}
                  </div>
                ) : null}
                {usageNoteParagraphs.length > 0 ? (
                  <div className="wf-recipe-usecases wf-recipe-usage-notes">
                    <h3>{t.usageNotes}</h3>
                    {usageNoteParagraphs.map((paragraph) => (
                      <p key={paragraph}>{paragraph}</p>
                    ))}
                  </div>
                ) : null}
                {useCases.length > 0 ? (
                  <div className="wf-recipe-usecases">
                    <h3>{t.useCases}</h3>
                    <ul>
                      {useCases.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                <dl className="wf-recipe-what-is-meta">
                  {publisherName ? (
                    <DefinitionRow
                      label={t.publisher}
                      value={
                        publisherKindLabel
                          ? locale === 'zh'
                            ? `${publisherName}（${publisherKindLabel}）`
                            : `${publisherName} (${publisherKindLabel})`
                          : publisherName
                      }
                    />
                  ) : null}
                  <DefinitionRow label={t.tasksLabel} value={recipe.tasks.length > 0 ? recipe.tasks.join(' · ') : t.notRecorded} />
                  <DefinitionRow label={t.environment} value={recipe.runtime.environmentName} />
                  <DefinitionRow label={t.output} value={artifactSummary(recipe, t.notRecorded)} />
                </dl>
                {hasHardware ? (
                  <div className="wf-recipe-hardware">
                    <h3>{t.hardwareGuidance}</h3>
                    <dl>
                      {docs.hardware.minVramGb !== null ? (
                        <DefinitionRow label={t.minVram} value={`${docs.hardware.minVramGb} GB`} />
                      ) : null}
                      {docs.hardware.recommended ? (
                        <DefinitionRow label={t.recommendedHardware} value={docs.hardware.recommended} />
                      ) : null}
                    </dl>
                    {docs.hardware.notes.length > 0 ? (
                      <ul>
                        {docs.hardware.notes.map((note) => (
                          <li key={note}>{note}</li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ) : null}
                <div className="wf-recipe-contracts">
                  <div>
                    <h3>{t.inputsOverview}</h3>
                    {recipe.inputContract.length > 0 ? (
                      <table>
                        <thead>
                          <tr>
                            <th>{t.field}</th>
                            <th>{t.recordedContract}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recipe.inputContract.map((item) => (
                            <tr key={item.field}>
                              <td><code>{item.field}</code></td>
                              <td>{item.detail}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p>{t.noContract}</p>
                    )}
                  </div>
                  <div>
                    <h3>{t.outputsOverview}</h3>
                    {recipe.artifacts.length > 0 ? (
                      <table>
                        <thead>
                          <tr>
                            <th>{t.artifactKind}</th>
                            <th>{t.filename}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recipe.artifacts.map((artifact, index) => (
                            <tr key={`${artifact.kind}:${artifact.filename}:${index}`}>
                              <td><code>{artifact.kind}</code></td>
                              <td><code>{artifact.filename || '—'}</code></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p>{t.noArtifacts}</p>
                    )}
                  </div>
                </div>
              </section>

              <section className="wf-recipe-section" id="sources">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.sourcesSection}</h2>
                    <p>{t.sourcesIntro}</p>
                  </div>
                </header>
                {recipe.sources.length > 0 ? (
                  <SourcesTable sources={recipe.sources} locale={locale} />
                ) : (
                  <p className="wf-recipe-empty-note">{t.noSources}</p>
                )}
                <div className="wf-recipe-provenance">
                  <div>
                    <GitBranch aria-hidden="true" size={16} />
                    <strong>{t.provenance}</strong>
                  </div>
                  <code>{recipe.catalogPath}</code>
                </div>
              </section>

              <section className="wf-recipe-section" id="launch">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.launch}</h2>
                    <p>{isReferenceOnly ? t.referenceOnlyDescription : recipe.variants.length > 0 ? t.variantContractIntro : t.launchIntro}</p>
                  </div>
                </header>
                {isReferenceOnly ? null : recipe.variants.length === 0 && recipe.runtime.pipelineTarget ? (
                  <CommandBlock>{recipe.commands.run}</CommandBlock>
                ) : null}
                {isReferenceOnly || recipe.variants.length === 0 ? null : (
                  <p className="wf-recipe-builder-pointer">{t.launchBuilderHint}</p>
                )}
              </section>

              {isReferenceOnly ? null : (
              <section className="wf-recipe-section" id="install">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.install}</h2>
                    <p>{t.installIntro}</p>
                  </div>
                </header>
                <dl className="wf-recipe-environment-grid">
                  <DefinitionRow label={t.environment} value={recipe.runtime.environmentName} />
                  <DefinitionRow label={t.environmentKind} value={environmentKind} />
                  <DefinitionRow label={t.python} value={recipe.runtime.python} />
                  <DefinitionRow label={t.cuda} value={recipe.runtime.cudaLabel} />
                  <DefinitionRow label={t.pytorch} value={recipe.runtime.packageVersions.torch} />
                  <DefinitionRow label={t.driverStatus} value={recipe.runtime.driverStatus} />
                </dl>
                <CommandBlock label={t.installCommand}>{recipe.commands.install}</CommandBlock>
                {recipe.runtime.pipPackages.length === 0 && recipe.runtime.condaPackages.length === 0 ? (
                  <p className="wf-recipe-empty-note">{t.notRecorded}</p>
                ) : (
                  <div className="wf-recipe-package-columns">
                    <details>
                      <summary>
                        {t.packages} <span>{recipe.runtime.pipPackages.length}</span>
                      </summary>
                      {recipe.runtime.pipPackages.length > 0 ? (
                        <ul>
                          {recipe.runtime.pipPackages.map((item) => (
                            <li key={item}>
                              <code>{item}</code>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p>{t.notRecorded}</p>
                      )}
                    </details>
                    <details>
                      <summary>
                        {t.condaPackages} <span>{recipe.runtime.condaPackages.length}</span>
                      </summary>
                      {recipe.runtime.condaPackages.length > 0 ? (
                        <ul>
                          {recipe.runtime.condaPackages.map((item) => (
                            <li key={item}>
                              <code>{item}</code>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p>{t.notRecorded}</p>
                      )}
                    </details>
                  </div>
                )}
                {recipe.runtime.validationImports.length > 0 ? (
                  <p className="wf-recipe-validation-imports">
                    <span>{t.validationImports}</span>
                    {recipe.runtime.validationImports.map((item) => (
                      <code key={item}>{item}</code>
                    ))}
                  </p>
                ) : null}
              </section>
              )}

              <section className="wf-recipe-section" id="assets">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.assets}</h2>
                    <p>{isReferenceOnly ? t.referenceOnlyDescription : t.assetsIntro}</p>
                  </div>
                </header>
                {isReferenceOnly ? null : (
                  <>
                    <CommandBlock label={t.checkCommand}>{recipe.commands.check}</CommandBlock>
                    <CommandBlock label={t.prepareCommand}>{recipe.commands.prepare}</CommandBlock>
                  </>
                )}
                {recipe.checkpoints.length > 0 ? (
                  <div className="wf-recipe-checkpoints">
                    {recipe.checkpoints.map((checkpoint) => (
                      <article key={`${checkpoint.id}:${checkpoint.revision ?? ''}`}>
                        <div>
                          <Package aria-hidden="true" size={16} />
                          <strong>{checkpoint.id}</strong>
                        </div>
                        <dl>
                          <DefinitionRow label="Revision" value={checkpoint.revision} />
                          <DefinitionRow label="License" value={checkpoint.license} />
                          <DefinitionRow label="Gated" value={typeof checkpoint.gated === 'boolean' ? String(checkpoint.gated) : null} />
                          <DefinitionRow label="Private" value={typeof checkpoint.private === 'boolean' ? String(checkpoint.private) : null} />
                          {checkpoint.role ? <DefinitionRow label="Role" value={checkpoint.role} /> : null}
                        </dl>
                        {checkpoint.notes && checkpoint.notes.length > 0 ? (
                          <ul className="wf-recipe-checkpoint-notes">
                            {checkpoint.notes.map((note) => (
                              <li key={note}>{note}</li>
                            ))}
                          </ul>
                        ) : null}
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="wf-recipe-empty-note">{isReferenceOnly ? t.referenceOnlyDescription : t.noCheckpoints}</p>
                )}
              </section>

              <section className="wf-recipe-section" id="variants">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.variantsSection}</h2>
                    <p>{t.variantsIntro}</p>
                  </div>
                </header>
                {recipe.variants.length > 0 ? (
                  <div className="wf-recipe-variants-scroll">
                    <table className="wf-recipe-variants-table">
                      <thead>
                        <tr>
                          <th>{t.variantCol}</th>
                          <th>{t.variantTaskCol}</th>
                          <th>{t.variantProfileCol}</th>
                          <th>{t.variantBindingCol}</th>
                          <th>{t.variantStatusCol}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recipe.variants.map((variant) => (
                          <tr key={variant.id}>
                            <td><code>{variant.id}</code></td>
                            <td>{variant.task || '—'}</td>
                            <td>{variant.runtimeProfile ? <code>{variant.runtimeProfile}</code> : '—'}</td>
                            <td>{variant.pipelineBinding ? <code>{variant.pipelineBinding}</code> : '—'}</td>
                            <td>
                              <span className={`wf-recipe-variant-status ${isRunnableVariant(variant) ? 'is-runnable' : 'is-pending'}`}>
                                {formatStatus(variant.status, t.notRecorded)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="wf-recipe-empty-note">{t.noVariants}</p>
                )}
                <dl className="wf-recipe-runtime-matrix">
                  <DefinitionRow label={t.binding} value={recipe.runtime.bindingId} />
                  <DefinitionRow label={t.runner} value={recipe.runtime.runner ?? recipe.runtime.runnerTarget} />
                  <DefinitionRow label={t.pipeline} value={recipe.runtime.pipelineTarget} />
                </dl>
              </section>

              {docs.benchmarks.length > 0 ? (
                <section className="wf-recipe-section" id="benchmarks">
                  <header>
                    <span>{nextSectionNumber()}</span>
                    <div>
                      <h2>{t.benchmarksSection}</h2>
                      <p>{t.benchmarksIntro}</p>
                    </div>
                  </header>
                  <div className="wf-recipe-bench-grid">
                    {docs.benchmarks.map((bench) => (
                      <Link
                        href={`${benchmarkHubBase}/${bench.id}`}
                        className="wf-recipe-bench-card"
                        key={bench.id}
                      >
                        <div className="wf-recipe-bench-head">
                          <BarChart3 aria-hidden="true" size={15} />
                          <strong>{bench.name}</strong>
                          <span className={`wf-recipe-bench-source is-${bench.source}`}>
                            {bench.source === 'docs' ? t.benchmarkFromDocs : t.benchmarkFromManifest}
                          </span>
                        </div>
                        <p>{locale === 'zh' ? bench.summaryZh : bench.summary}</p>
                        <small>{locale === 'zh' ? bench.reasonZh : bench.reason}</small>
                      </Link>
                    ))}
                  </div>
                  <p className="wf-recipe-bench-browse">
                    <Link href={`${locale === 'zh' ? '/zh' : ''}/docs/evaluation`}>
                      {t.benchmarksBrowse}
                      <ArrowRight aria-hidden="true" size={12} />
                    </Link>
                  </p>
                </section>
              ) : null}

              <section className="wf-recipe-section" id="evidence">
                <header>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.evidence}</h2>
                    <p>{t.evidenceIntro}</p>
                  </div>
                </header>
                <dl className="wf-recipe-evidence-grid">
                  <DefinitionRow label={t.integration} value={formatStatus(recipe.status.integration, t.notRecorded)} />
                  <DefinitionRow label={t.runnerEvidence} value={formatStatus(recipe.status.runner, t.notRecorded)} />
                  <DefinitionRow label={t.demoEvidence} value={formatStatus(recipe.status.demo, t.notRecorded)} />
                  <DefinitionRow label={t.validationImports} value={recipe.runtime.validationImports.join(', ') || null} />
                </dl>
                {limitations.length > 0 ? (
                  <div className="wf-recipe-limitations">
                    <h3>{t.limitations}</h3>
                    <ul>
                      {limitations.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {recipe.notes.length > 0 ? (
                  <details className="wf-recipe-manifest-notes">
                    <summary>
                      {t.manifestNotes} <span>{recipe.notes.length}</span>
                    </summary>
                    <ul>
                      {recipe.notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </details>
                ) : null}
                <div className="wf-recipe-provenance">
                  <div>
                    <GitBranch aria-hidden="true" size={16} />
                    <strong>{t.provenance}</strong>
                  </div>
                  <code>{recipe.catalogPath}</code>
                  {recipe.sources.map((source) => (
                    <a href={source.url} target="_blank" rel="noreferrer" key={`${source.kind}:${source.url}`}>
                      <span>{source.label}</span>
                      <code>{source.revision ?? source.url}</code>
                      <ExternalLink aria-hidden="true" size={11} />
                    </a>
                  ))}
                </div>
              </section>

              <details className="wf-recipe-advanced" id="advanced">
                <summary>
                  <span>{nextSectionNumber()}</span>
                  <div>
                    <h2>{t.advanced}</h2>
                    <p>{t.advancedIntro}</p>
                  </div>
                </summary>
                <div className="wf-recipe-advanced-body">
                  <dl className="wf-recipe-version-matrix">
                    <DefinitionRow label={t.integration} value={formatStatus(recipe.status.integration, t.notRecorded)} />
                    <DefinitionRow label={t.runnerEvidence} value={formatStatus(recipe.status.runner, t.notRecorded)} />
                    <DefinitionRow label={t.environment} value={recipe.runtime.environmentName} />
                    <DefinitionRow label={t.python} value={recipe.runtime.python} />
                    <DefinitionRow label={t.cuda} value={recipe.runtime.cudaLabel} />
                    <DefinitionRow label={t.pytorch} value={recipe.runtime.packageVersions.torch} />
                    <DefinitionRow label={t.sourceRevision} value={revision(recipe, 'source')} />
                    <DefinitionRow label={t.checkpointRevision} value={revision(recipe, 'checkpoint')} />
                  </dl>

                  <dl className="wf-recipe-runtime-matrix">
                    <DefinitionRow label={t.profile} value={recipe.runtime.profileId} />
                    <DefinitionRow label={t.binding} value={recipe.runtime.bindingId} />
                    <DefinitionRow label={t.runner} value={recipe.runtime.runner ?? recipe.runtime.runnerTarget} />
                    <DefinitionRow label={t.pipeline} value={recipe.runtime.pipelineTarget} />
                    <DefinitionRow label={t.backend} value={recipe.runtime.backendStage} />
                    <DefinitionRow label={t.runtimeStatus} value={recipe.runtime.runtimeStatus} />
                    <DefinitionRow label={t.driverStatus} value={recipe.runtime.driverStatus} />
                    <DefinitionRow label={t.environmentKind} value={environmentKind} />
                  </dl>
                </div>
              </details>

              {related.length > 0 ? (
                <section className="wf-recipe-related" aria-labelledby="wf-related-recipes">
                  <h2 id="wf-related-recipes">{t.related}</h2>
                  <div>
                    {related.map((item) => (
                      <Link href={`${basePath}/${item.id}`} key={item.id}>
                        <div className="wf-recipe-related-identity">
                          <ModelIdentityMark
                            id={item.id}
                            name={item.name}
                            provider={item.provider}
                            category={item.category}
                            size="small"
                          />
                          <span>
                            <span>{item.provider}</span>
                            <strong>{item.name}</strong>
                          </span>
                        </div>
                        <small>{t.details} <ArrowRight aria-hidden="true" size={12} /></small>
                      </Link>
                    ))}
                  </div>
                </section>
              ) : null}
            </div>
          </article>
        </div>

        <aside className="pi-doc-right-rail wf-recipe-rail" aria-label={t.compatibility}>
          <h2>{t.compatibility}</h2>
          <dl>
            <DefinitionRow label={t.runtime} value={environmentKind} />
            <DefinitionRow label={t.python} value={recipe.runtime.python} />
            <DefinitionRow label={t.cuda} value={recipe.runtime.cudaLabel} />
            <DefinitionRow label={t.pytorch} value={recipe.runtime.packageVersions.torch} />
            <DefinitionRow label={t.checkpoint} value={recipe.checkpoints[0]?.id} />
            <DefinitionRow label={t.checkpointRevision} value={revision(recipe, 'checkpoint')} />
            <DefinitionRow label={t.output} value={artifactSummary(recipe, t.notRecorded)} />
          </dl>
          <div className="wf-recipe-rail-status">
            <span>{t.integration}</span>
            <strong>{formatStatus(recipe.status.integration, t.notRecorded)}</strong>
            <span>{t.runnerEvidence}</span>
            <strong>{formatStatus(recipe.status.runner, t.notRecorded)}</strong>
          </div>
        </aside>
      </div>
    </main>
  );
}
