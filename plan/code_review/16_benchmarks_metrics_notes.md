# 16. Benchmarks / Metrics 观察笔记（infra 任务 6，只读调研）

> 本文件由「任务 6：其余 infra 代码审查」产出。以下树对本任务**禁改**，问题只记录在此，留给
> metrics in-tree agent（`bc-2804e344-0764-5dc8-9d99-b8934ed8cda2`）与各 bench in-tree agent 处置：
> - `worldfoundry/evaluation/tasks/metrics/**`
> - `worldfoundry/evaluation/tasks/execution/runners/<bench>/**`
> - hub mdx / fumadocs 文档（任务 4/5 在写）
>
> 调研方式：`rg` / 目录清点 / 读源码，均为静态事实；未跑任何 GPU 评测。

## 1. 重复的 FID / CLIP / FVD 实现位置

通用 metric 在 metrics 树、runners 共享层（`_scorers`、`_benchmark_metrics`）和各 bench vendored
runtime 里至少各有一份。建议长期收敛到 `tasks/metrics/registry.py` 的共享 registry；本任务**未**在
framework/api 层复制任何一份。

### 1.1 FID（Frechet Inception Distance）及其变体

| 位置 | 说明 |
| --- | --- |
| `tasks/metrics/fid/compute.py` + `fid/vendor/swav/fid_score.py` | 规范 in-tree 实现（含 SwAV vendored 打分） |
| `tasks/metrics/clean_fid/cleanfid/fid.py`（+ `inception_pytorch.py`、`wrappers.py`） | clean-fid vendored 全套，含独立 Inception 权重加载 |
| `tasks/metrics/_shared/vendor/torch_fidelity/metric_fid.py` | torch-fidelity vendored 第三份 FID |
| `tasks/metrics/fld/fld/metrics/FID.py` | FLD vendored 树里又一份 frechet 计算 |
| `tasks/metrics/cfid/` | 条件 FID 变体（独立实现） |
| `tasks/metrics/fsim/vendor/piq/feature_extractors/fid_inception.py` | piq vendored 的 FID Inception 特征器（仅特征器，仍是重复权重路径） |
| `runners/fetv/runtime/fetv_eval/compute_fid.py` | FETV vendored runtime 私有 FID |
| `runners/devil_dynamics/runtime/official/tools/evaluate_dynamic_ddp_metrics.py` | DEVIL vendored 工具内嵌 frechet 计算 |

frechet 距离公式本体（`sqrtm(sigma1 @ sigma2)` 一族）在 `fvmd/vendor/fvmd/frechet_distance.py`、
`fwd/vendor/pytorchfwd/fwd.py`、`fjd/fjd_core.py`、`fld/.../FID.py`、`clean_fid/cleanfid/fid.py`、
`fid/vendor/swav/fid_score.py` 至少 6 处重复。

### 1.2 FVD（Frechet Video Distance）

| 位置 | 说明 |
| --- | --- |
| `tasks/metrics/fvd/fvd_core.py` | 规范 in-tree 实现 |
| `runners/mirabench/runtime/mirabench/evaluation/fvd.py` + `inception.py` | MiraBench vendored 私有 FVD（自带 I3D/Inception 加载） |
| `runners/world_in_world/runtime/official/evaluation/FVD/calculate_fvd.py`（+ `cal_4metrics.py`） | World-in-World vendored 私有 FVD |
| `runners/{mirabench,genai_bench,ipv_bench,fetv,aigcbench}/*_video_quality_contract.py` | wrapper 层各自声明 FVD 结果字段（契约面，不算实现重复，但字段命名未对齐 registry id） |

### 1.3 CLIP score

| 位置 | 说明 |
| --- | --- |
| `runners/_scorers/clip_score/**` | runners 共享层的完整 CLIP/LanguageBind/UMT/HPSv2 打分树（`clipscore.py` + `models/clipscore_models/*`）——事实上的「第二 metrics 树」，与 `tasks/metrics/` 平行 |
| `tasks/metrics/quality_loss/wrapper.py` | metrics 树内的 CLIP-score 包装 |
| `runners/worldscore/runtime/.../metrics/{torchmetrics,iqa_pytorch}/clip_score_metrics.py` + `metric_impls/clip_mlp_aesthetic_metrics.py` | WorldScore vendored **一家就有 3 份** CLIP score 实现 |
| `runners/four_d_worldbench/runtime/.../metric/torchmetrics/clip_score_metrics.py` | 与 worldscore 的 torchmetrics 版同源复制 |
| `runners/fetv/runtime/fetv_eval/metrics/clips.py` | FETV 私有 CLIP 打分 |
| `runners/ewmbench/runtime/ewmbench/EWMBench/semantics.py` | EWMBench 私有 CLIP 语义打分 |
| `runners/worldarena/runtime/video_quality/WorldArena/semantic_alignment.py` | WorldArena 私有 |
| `runners/worldolympiad/runtime/worldolympiad/interaction/clip_score.py` | WorldOlympiad 私有 |
| `runners/videoscore/runtime/videoscore/benchmark/{eval_feature_metric,feature_metric_tools/t2v_align_eval}.py` | VideoScore 私有 |
| `runners/vbench_plus_plus/runtime/vbench2_beta_long/utils.py`、`runners/aigcbench/runtime/aigcbench/eval.py` | 另两份 vendored 私有 |

**建议（给 metrics agent）**：vendored runtime 内的副本可保留（上游快照，改了就不是 official），
但 wrapper 层（`runners/*/*.py` 非 runtime）与 `_scorers/` 应统一走 `tasks/metrics/registry.py`
的 id/alias；至少要求 wrapper 产出的 metric 键名对齐 registry 词表，避免同名不同义。

## 2. Runner 与 hub 文档不一致（只记代码侧事实）

任务 4/5 正在写 hub mdx；以下是**代码侧**可核实的事实，供其对照：

1. `docs/fumadocs/lib/benchmark-catalog-status.json` 缺 `apple-pi` / `larybench` /
   `physical-ai-bench` 三个 id（catalog 侧数据已齐，见 `fixes/04_evaluation_tasks_fixes.md`
   ET-21——docs 部分越界 deferred，需重跑 docs 生成器）。
2. runners 目录共 **49 个 bench 子目录**（不含 `_scorers`、`_benchmark_metrics`、
   `workspace_registry.py`）；catalog（`tasks/catalog/`）与 zoo manifest 三方对齐为
   72=72=72（ET-01 修复后）——即约 23 个 catalog id 没有独立 runner 子目录（走共享框架或
   hosted 路径），文档侧「有 runner」列表须以 `workspace_registry.py` 为准而非目录名。
3. `phygenbench_runtime.py` 等 official runner 的默认 backend 是 **`mock`**（见 §4）；hub 文档如把
   这些 bench 标为「已跑通官方评测」需以 scorecard 中 `backend` 字段为准，不能以「runner 存在」为准。

## 3. Catalog 与 runner 的缺口

1. **状态词表靠 warning 降级**：`tasks/catalog/schema.py` 的
   `_normalize_source_status` / `_normalize_integration_status` 对未识别状态打一次性 warning 后降级
   到 fallback（ET-17 已加观测），但 `BenchmarkZooEntry` 仍有 7 个互为回退的状态字段（ET-20，P3
   deferred）——词表未收敛前，catalog 生成的任何「状态徽章」都可能来自 fallback 而非作者本意。
2. **evidence 位全库 false**：catalog 顶层三证据位（`official_gpu_validation` 等）全库为 false，
   `official_gpu_validation.scorecard` 引用运行产物路径而非入库文件（ET-22，P3 deferred）。
   official-run 事实上**未接线**到 catalog——文档不应从 catalog 读「已验证」。
3. **CLI 契约不统一**：6 个 runner 不接受 `--benchmark-id`（ET-02，P2 deferred），
   `workspace_registry.py` 里用 `pass_benchmark_id` 字段绕过；接口收敛前 catalog 无法假设统一调用面。
4. **子进程工具二分**：13 个 runner 文件仍直接用 `subprocess.run/Popen` 而非 `core.process` 工具族
   （ET-07 部分收敛后余量），超时/进程组清理行为与共享框架不一致。

## 4. Hosted judge / 仿真器 / mock backend 诚实边界

1. **默认 mock 的 official runner**：`phygenbench_runtime.py`（默认 backend=`mock`，用
   `phygenbench-mock:{prompt_id}` 种子生成确定性假分数）；`mirabench`、`ewmbench`、`iworldbench`、
   `genai_bench`、`apple_pi`、`phyfps_bench_gen` 的 runtime wrapper 同样带 mock 路径。scorecard 里
   记录了 `backend` 字段——**任何汇报必须透传该字段**，mock 产出的数字不是评测结果。
2. **hosted VLM judge**：`videoverse_judge.py`（Gemini，`GEMINI_API_KEY`/`GOOGLE_API_KEY`，默认
   `gemini-2.5-pro`）、`physvidbench_judge.py` + `run_multi_ask_with_api_key.py`（多次询问 API）、
   `worldolympiad`（VLM interaction）、`rbench`/`memobench`/`videobench`/`devil_dynamics` wrapper
   均含 hosted key 路径。judge 模型版本进 scorecard 与否未统一——hosted judge 打分不可复现时，
   文档不应写成确定性 benchmark 分数。
3. **`missing_judge_requirements()` 模式值得推广**：videoverse 的「先检查 key/依赖、缺了列出来」
   写法比静默 fallback 到 mock 更诚实，建议 metrics/bench agent 统一采用。

## 5. 禁改树内仍开放的 P0/P1（留给 in-tree agents）

第一轮 in-tree P1（ET-01/04/10/11/14）已由 evaluation tasks agent 修复
（见 `fixes/04_evaluation_tasks_fixes.md`）。当前仍开放：

| 条目 | 级别 | 位置 | 摘要 |
| --- | --- | --- | --- |
| ET-02 | P2 | `runners/*` 6 个 runner | 不接受 `--benchmark-id`，CLI 契约不统一（接口级重构） |
| ET-07 余量 | P2 | 13 个 runner 文件 | 子进程调用未收敛到 `core.process` 工具族 |
| ET-20 | P3 | `catalog/schema.py` + 72 manifest | `BenchmarkZooEntry` 7 状态字段互为回退，schema 需重设计 |
| ET-21 docs | — | `docs/fumadocs/lib/benchmark-catalog-status.json` | 缺 3 个 id，重跑生成器即可 |
| ET-22 | P3 | catalog manifests | evidence 位全 false、scorecard 引用未入库路径 |
| ET-23 | P3 | framework `GenerationRequest` | 无 seed 管理约定（本任务已核实 framework 侧未加；建议在 runtime profile 约定可选 `seed`） |
| 本文 §1 | P2（建议） | `_scorers/` + wrapper 层 | CLIP/FVD 键名未对齐 metrics registry 词表 |
| 本文 §4 | P1（建议） | mock-backend runners | scorecard `backend` 字段需强制透传到任何聚合报表，防止 mock 分数被当真 |

## 6. 本任务在可写树内已做的关联修复（交叉引用）

- **SA-10 (P1)**：`runtime/{assets,conda,benchmark_repos}.py` 不再 import
  `worldfoundry.evaluation.utils`；manifest 加载器下沉到 `worldfoundry/core/io/manifests.py`，
  `evaluation.utils` 改为 re-export（公共契约不变）。评测侧 manifest IO 从此有单一权威实现。
- **TR-12/13**、**TE-13**、**PL-04**：详见 `fixes/second_round_fixes.md` 与各模块 fixes 文档。
