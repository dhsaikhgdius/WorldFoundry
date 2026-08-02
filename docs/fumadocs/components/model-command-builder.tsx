'use client';

import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { ModelRecipe, ModelRecipeTask } from '@/lib/model-recipe-types';

type Locale = 'en' | 'zh';
type CommandTab = 'prepare' | 'install' | 'inspect' | 'check' | 'run';

const labels = {
  en: {
    title: 'Build your run',
    description: 'Choose a recorded variant and task profile, then copy the exact command for that route.',
    variant: 'Variant',
    task: 'Task',
    taskProfile: 'Task profile',
    pipeline: 'Inference pipeline',
    loading: 'Loading',
    invocation: 'Invocation',
    runner: 'Runner',
    environment: 'Environment',
    device: 'Device',
    taskContract: 'Task-specific contract',
    inputs: 'Inputs',
    artifacts: 'Artifacts',
    recordedContract: 'Recorded contract',
    artifactKind: 'Artifact kind',
    filename: 'Filename / path',
    noPipeline: 'No runnable inference pipeline is recorded for this model.',
    noContract: 'No contract is recorded for this variant.',
    noArtifacts: 'No artifact contract is recorded for this task.',
    prepare: 'Prepare',
    install: 'Install',
    inspect: 'Inspect',
    check: 'Check assets',
    run: 'Run',
    copy: 'Copy',
    copied: 'Copied',
    defaultVariant: 'Default model ID',
    notRecorded: 'Not recorded',
  },
  zh: {
    title: '构建运行命令',
    description: '选择仓库中已记录的 variant 和 task profile，然后复制对应 route 的准备、检查或运行命令。',
    variant: 'Variant',
    task: '任务',
    taskProfile: '任务 Profile',
    pipeline: '推理 Pipeline',
    loading: '加载方式',
    invocation: '调用方式',
    runner: 'Runner',
    environment: '环境',
    device: '设备',
    taskContract: 'Task 专属契约',
    inputs: '输入',
    artifacts: '输出 Artifact',
    recordedContract: '已记录契约',
    artifactKind: 'Artifact 类型',
    filename: '文件名 / 路径',
    noPipeline: '该模型没有记录可运行的推理 Pipeline。',
    noContract: '该 variant 没有记录契约。',
    noArtifacts: '该 task 没有记录 Artifact 契约。',
    prepare: '准备',
    install: '安装',
    inspect: '查看 Manifest',
    check: '检查资产',
    run: '运行',
    copy: '复制',
    copied: '已复制',
    defaultVariant: '默认 Model ID',
    notRecorded: '未记录',
  },
} as const;

function commandPlaceholder(field: { field: string; required?: boolean; default?: unknown; kind?: string }) {
  if (!field.required || (field.default !== undefined && field.default !== null && field.default !== '')) {
    return null;
  }
  if (['prompt', 'instruction', 'text', 'caption'].includes(field.field)) {
    return '"Describe the desired output."';
  }
  if (['input_path', 'image', 'video', 'audio', 'images', 'input'].includes(field.field)) {
    return '/path/to/input';
  }
  if (field.kind === 'json' || field.kind === 'interaction_tokens') {
    return "'{}'";
  }
  return 'VALUE';
}

function shellWord(value: string) {
  return /^[A-Za-z0-9._/-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;
}

function runCommand(modelId: string, task: ModelRecipeTask) {
  const lines = [
    'worldfoundry-eval run \\',
    `  ${shellWord(modelId)} \\`,
    `  --pipeline.task-profile ${shellWord(task.id)} \\`,
  ];
  for (const field of task.inputs) {
    const placeholder = commandPlaceholder(field);
    if (!placeholder) continue;
    lines.push(`  --pipeline.${field.field.replaceAll('_', '-')} ${placeholder} \\`);
  }
  lines.push('  --json');
  return lines.join('\n');
}

function commandFor(tab: CommandTab, recipe: ModelRecipe, runtimeModelId: string, task: ModelRecipeTask) {
  switch (tab) {
    case 'prepare':
      return `bash scripts/inference/prepare_model_infer.sh ${runtimeModelId}`;
    case 'install':
      return `bash scripts/setup/model_env_install.sh --model ${runtimeModelId}`;
    case 'inspect':
      return recipe.commands.inspect;
    case 'check':
      return recipe.commands.check;
    case 'run':
      return runCommand(runtimeModelId, task);
  }
}

function isRunnableVariant(value: { pipelineTarget: string | null; status: string }) {
  const status = value.status.toLowerCase().replaceAll('-', '_');
  const unavailable = ['not_recorded', 'planned', 'profile', 'profile_only', 'blocked', 'unavailable', 'missing'];
  return Boolean(value.pipelineTarget) && !unavailable.some((marker) => status.includes(marker));
}

function variantLabel(value: { id: string; pipelineTarget: string | null; status: string }, fallback: string) {
  if (isRunnableVariant(value)) return value.id;
  const status = value.status && value.status !== 'not_recorded'
    ? value.status.replaceAll('_', ' ').replaceAll('-', ' ')
    : fallback;
  return `${value.id} — ${status}`;
}

function ContractTable({
  title,
  items,
  empty,
  fieldLabel,
  valueLabel,
  artifactTable = false,
}: {
  title: string;
  items: Array<{ field?: string; detail?: string; kind?: string; filename?: string }>;
  empty: string;
  fieldLabel: string;
  valueLabel: string;
  artifactTable?: boolean;
}) {
  return (
    <div className="wf-command-builder-contract">
      <h3>{title}</h3>
      {items.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>{fieldLabel}</th>
              <th>{valueLabel}</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, index) => (
              <tr key={`${item.field ?? item.kind}:${item.detail ?? item.filename}:${index}`}>
                <td><code>{item.field ?? item.kind}</code></td>
                <td>{item.detail ?? item.filename ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>{empty}</p>
      )}
    </div>
  );
}

export function ModelCommandBuilder({ recipe, locale = 'en' }: { recipe: ModelRecipe; locale?: Locale }) {
  const t = labels[locale];
  const choices = useMemo(
    () =>
      recipe.variants.length > 0
        ? recipe.variants
        : [
            {
              id: recipe.id,
              label: recipe.name,
              // A model without explicit variants can still expose several
              // task modes. Keep the task field empty here so the renderer
              // below can show the complete model-level task list instead of
              // silently reducing it to tasks[0].
              task: '',
              runtimeProfile: recipe.runtime.profileId ?? '',
              pipelineBinding: recipe.runtime.bindingId ?? '',
              status: recipe.status.integration,
              pipelineTarget: recipe.runtime.pipelineTarget,
              runner: recipe.runtime.runner ?? recipe.runtime.runnerTarget,
              loadingMethod: recipe.runtime.loadingMethod,
              invocationMode: recipe.runtime.invocationMode,
              environmentName: recipe.runtime.environmentName,
              environmentKind: recipe.runtime.environmentKind,
              python: recipe.runtime.python,
              cudaLabel: recipe.runtime.cudaLabel,
              backendStage: recipe.runtime.backendStage,
              runtimeStatus: recipe.runtime.runtimeStatus,
              inputContract: recipe.inputContract,
              artifacts: recipe.artifacts,
            },
          ],
    [recipe],
  );
  const taskChoices: ModelRecipeTask[] =
    recipe.inferenceTasks.length > 0
      ? recipe.inferenceTasks
      : [
          {
            id: recipe.tasks[0] ?? 'default',
            label: recipe.tasks[0] ?? t.notRecorded,
            description: '',
            source: 'catalog',
            variantIds: [],
            inputs: [],
            artifacts: [],
          },
        ];
  const defaultChoice = choices.find(isRunnableVariant) ?? choices[0];
  const defaultTask = taskChoices.find(
    (task) => task.variantIds.length === 0 || task.variantIds.includes(defaultChoice?.id ?? ''),
  ) ?? taskChoices[0];
  const [selectedId, setSelectedId] = useState(defaultChoice?.id ?? recipe.id);
  const [selectedTaskId, setSelectedTaskId] = useState(defaultTask?.id ?? 'default');
  const [tab, setTab] = useState<CommandTab>(defaultChoice && isRunnableVariant(defaultChoice) ? 'run' : 'inspect');
  const [copied, setCopied] = useState(false);
  const selected = choices.find((choice) => choice.id === selectedId) ?? choices[0];
  const selectedTask = taskChoices.find((task) => task.id === selectedTaskId) ?? taskChoices[0];
  const selectedPipeline = recipe.variants.length > 0
    ? selected?.pipelineTarget
    : selected?.pipelineTarget ?? recipe.runtime.pipelineTarget;
  const selectedRunner = selected?.runner ?? recipe.runtime.runner ?? recipe.runtime.runnerTarget;
  const selectedEnvironment = selected?.environmentName ?? recipe.runtime.environmentName;
  const selectedDevice = selected?.cudaLabel ?? recipe.runtime.cudaLabel;
  const selectedProfile = selected?.runtimeProfile || recipe.runtime.profileId;
  const selectedInputContract = selectedTask.inputs.length
    ? selectedTask.inputs
    : selected?.inputContract?.length
      ? selected.inputContract
      : recipe.inputContract;
  const selectedArtifacts = selectedTask.artifacts.length
    ? selectedTask.artifacts
    : selected?.artifacts?.length
      ? selected.artifacts
      : recipe.artifacts;
  const hasPipeline = recipe.variants.length > 0
    ? Boolean(selected && isRunnableVariant(selected))
    : Boolean(selectedPipeline);
  const hasProfile = Boolean(selectedProfile);
  const command = commandFor(tab, recipe, selected?.id ?? recipe.id, selectedTask);

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  const tabs: CommandTab[] = [
    ...(hasProfile ? (['prepare', 'install'] as CommandTab[]) : []),
    'inspect',
    'check',
    ...(hasPipeline ? (['run'] as CommandTab[]) : []),
  ];
  const runUnavailable = tab === 'run' && !hasPipeline;

  return (
    <section className="wf-command-builder" aria-labelledby="wf-command-builder-title">
      <div className="wf-command-builder-heading">
        <div>
          <h2 id="wf-command-builder-title">{t.title}</h2>
          <p>{t.description}</p>
        </div>
        <div className="wf-command-builder-heading-route">
          <span>{t.pipeline}</span>
          <code>{selectedPipeline ?? t.noPipeline}</code>
        </div>
      </div>

      <div className="wf-command-builder-controls">
        <label>
          <span>{t.variant}</span>
          <select
            value={selectedId}
            onChange={(event) => {
              const nextId = event.target.value;
              const nextChoice = choices.find((choice) => choice.id === nextId);
              setSelectedId(nextId);
              if (selectedTask.variantIds.length > 0 && !selectedTask.variantIds.includes(nextId)) {
                const compatible = taskChoices.find((task) => task.variantIds.includes(nextId));
                if (compatible) setSelectedTaskId(compatible.id);
              }
              if (tab === 'run' && (!nextChoice || !isRunnableVariant(nextChoice))) setTab('inspect');
              setCopied(false);
            }}
          >
            {choices.map((choice) => (
              <option value={choice.id} key={choice.id}>
                {variantLabel(choice, t.notRecorded)}
              </option>
            ))}
          </select>
        </label>
        <div>
          <span>{t.task}</span>
          {taskChoices.length > 1 ? (
            <select
              value={selectedTask.id}
              onChange={(event) => {
                const nextTask = taskChoices.find((task) => task.id === event.target.value) ?? taskChoices[0];
                setSelectedTaskId(nextTask.id);
                if (nextTask.variantIds.length > 0 && !nextTask.variantIds.includes(selected?.id ?? '')) {
                  const nextVariantId = nextTask.variantIds[0];
                  const nextChoice = choices.find((choice) => choice.id === nextVariantId);
                  setSelectedId(nextVariantId);
                  if (tab === 'run' && (!nextChoice || !isRunnableVariant(nextChoice))) setTab('inspect');
                }
                setCopied(false);
              }}
            >
              {taskChoices.map((task) => (
                <option value={task.id} key={task.id}>
                  {task.label} ({task.id})
                </option>
              ))}
            </select>
          ) : (
            <strong>
              {selectedTask.label} <code>{selectedTask.id}</code>
            </strong>
          )}
        </div>
        <div>
          <span>{t.environment}</span>
          <strong>{selectedEnvironment ?? t.notRecorded}</strong>
        </div>
        <div>
          <span>{t.device}</span>
          <strong>{selectedDevice ?? t.notRecorded}</strong>
        </div>
      </div>

      <div className="wf-command-builder-runtime">
        <div>
          <span>{t.runner}</span>
          <code>{selectedRunner ?? t.notRecorded}</code>
        </div>
        <div>
          <span>{t.loading}</span>
          <strong>{selected?.loadingMethod ?? recipe.runtime.loadingMethod ?? t.notRecorded}</strong>
        </div>
        <div>
          <span>{t.invocation}</span>
          <strong>{selected?.invocationMode ?? recipe.runtime.invocationMode ?? t.notRecorded}</strong>
        </div>
      </div>

      {taskChoices.length > 0 ? (
        <details className="wf-command-builder-contracts" open>
          <summary>
            {t.taskContract} <code>{selectedTask.id}</code>
          </summary>
          <div>
            <ContractTable
              title={t.inputs}
              items={selectedInputContract}
              empty={t.noContract}
              fieldLabel={t.taskProfile}
              valueLabel={t.recordedContract}
            />
            <ContractTable
              title={t.artifacts}
              items={selectedArtifacts}
              empty={t.noArtifacts}
              fieldLabel={t.artifactKind}
              valueLabel={t.filename}
              artifactTable
            />
          </div>
        </details>
      ) : null}

      <div className="wf-command-builder-tabs" role="tablist" aria-label={t.title}>
        {tabs.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={tab === item}
            className={tab === item ? 'is-active' : undefined}
            key={item}
            onClick={() => {
              setTab(item);
              setCopied(false);
            }}
          >
            {t[item]}
          </button>
        ))}
      </div>

      <div className="wf-command-builder-code">
        {runUnavailable ? (
          <p className="wf-command-builder-unavailable">{t.noPipeline}</p>
        ) : (
          <>
            <pre key={`${selectedId}:${tab}`}>
              <code>{command}</code>
            </pre>
            <button type="button" onClick={copyCommand} aria-live="polite">
              {copied ? <Check aria-hidden="true" size={14} /> : <Copy aria-hidden="true" size={14} />}
              {copied ? t.copied : t.copy}
            </button>
          </>
        )}
      </div>
    </section>
  );
}
