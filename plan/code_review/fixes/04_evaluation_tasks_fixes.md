# 04 evaluation tasks 层修复日志

- 评审报告：`plan/code_review/04_evaluation_tasks.md`（ET-01～ET-24）
- 修复日期：2026-08-14
- 约束：只改 `worldfoundry/evaluation/tasks/` 与 `worldfoundry/data/`；不改 evaluation 顶层 framework.py/runner.py/public.py/utils.py/api/models/reporting；不改 vendored `runtime/` 第三方代码（wrapper 层可改）；不装新依赖；不改既有测试；不 git commit/add/stash；不重命名/删除公共符号
- 新增测试（纯 CPU，全部位于 `test/eval_core/`，共 36 条，全部通过）：
  - `test_eval_tasks_fix_official_runner.py`（8 条，ET-04/10/12/15）
  - `test_eval_tasks_fix_metric_failfast.py`（7 条，ET-11）
  - `test_eval_tasks_fix_workspace_timeout.py`（4 条，ET-05）
  - `test_eval_tasks_fix_videoscore_syspath.py`（5 条，ET-08）
  - `test_eval_tasks_fix_devil_aliases.py`（3 条，ET-13）
  - `test_eval_tasks_fix_catalog_hygiene.py`（5 条，ET-17/18/19）
  - `test_eval_tasks_fix_runner_timeouts.py`（4 条，ET-03/06）

## eval_core collect 前后对比（兼容性红线验证）

命令：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q -p no:cacheprovider 2>&1 | tail -3`

- 修复前（本次会话开始时）：1436 tests collected，**18 errors**（均为既有收集错误，另一 agent 负责修复）
- 修复后：**1516 tests collected，0 errors**（收集错误已由另一 agent 并行修完；本次新增 7 个测试文件贡献 36 条；**本次改动未引入任何新收集错误**）

---

## P1 已修

### ET-04 / ET-14 官方 runner 失败路径不写 scorecard、异常白名单缺 TimeoutExpired、子进程不清理 —— 已修

改动：

- `execution/framework/official_runner.py`
  - `run_official_pipeline`：`subprocess.run` 改为 `worldfoundry.core.process.run_logged_subprocess`（进程组隔离 + 超时/异常时 `terminate_process_group` 清理整棵子进程树 + 日志落盘）；`TimeoutExpired` 不再被吞掉，向上传播到失败路径。
  - 新增 `write_failed_scorecard(...)` 公共 helper：任何失败路径都产出最小 `run.status="failed"` scorecard（含 error 类型/消息、benchmark_id、时间戳），恢复"失败必有 scorecard"契约。
  - `run_main`：异常白名单（原来仅少数几类）改为 `except Exception`，统一走 `write_failed_scorecard` 后返回非零退出码；`KeyboardInterrupt`/`SystemExit` 不拦截。
  - `build_scorecard` 增加 `run_status_override` 可选参数（默认 None，不影响既有调用方）。
- `execution/runners/worldscore/run_worldscore_official_runner.py`
  - `run_command_with_timeout`：内部改为 `subprocess.Popen(start_new_session=True)` + 超时后 `terminate_process_group`（复用 `worldfoundry.core.process`），杀整个进程组而非只杀直接子进程；外部签名不变。
  - `main`：独立的异常白名单同样改为 `except Exception` → `write_failed_scorecard`，退出码非零。

验证：`test_eval_tasks_fix_official_runner.py` —— 模拟命令超时（fake 慢子进程）后 scorecard.json 存在且 `run.status == "failed"`；模拟意外异常（如 OSError）同样产出 failed scorecard；worldscore 超时路径同样断言。8/8 通过。

### ET-15 非零退出码仍解析旧结果冒充新分数 —— 已修

改动（`execution/framework/official_runner.py`）：

- `run_official_pipeline` 在启动官方命令前记录 `run_started_wall`；
- 命令退出码非零或超时 → 直接判失败，不再进入结果 glob/解析分支；
- 命令成功时，glob 发现的结果文件经 `_is_fresh_result(path, run_started_wall)` 校验：`mtime` 早于本次 run 开始时间（留 2s 时钟余量）的旧文件被拒绝并记 warning，不会被当作本次产物解析。

验证：同上测试文件 —— 预先放置旧结果文件 + 命令以非零退出 → 不解析、scorecard 为 failed；旧结果 + 命令成功但结果 mtime 早于 run 开始 → 拒绝解析；新写入的结果正常通过。通过。

### ET-11 约 60 个 metric id 通过校验但静默不产出 —— 已修（fail-fast）

改动：

- `metrics/registry.py`
  - 新增 `OFFLINE_COMPUTABLE_METRIC_IDS` / `OFFLINE_COMPUTABLE_METRIC_PREFIXES` 与 `is_offline_computable_metric_id()`：显式声明 `BuiltinExistingResultsMetric` 实际有实现分支的指标集合。
  - `BuiltinExistingResultsMetric.__init__` fail-fast：请求的 canonical metric id 若无实现分支，立即抛 `MetricRegistryError`，错误消息列出不可计算的 id 与该场景下实际支持的指标。
  - 对无实现分支的注册表条目，`spec.implementation` 改为 `None`（声明与事实一致）。
  - `__all__` 导出新增符号（只增不删）。
- `execution/orchestration/evaluate.py`
  - `_metric_callable` 捕获 `MetricRegistryError` 后补充上下文重抛：附上该 benchmark 在 catalog 中声明的受支持指标列表，错误可执行。

验证：`test_eval_tasks_fix_metric_failfast.py` —— 请求已注册但无实现的 id 时构造即抛错且消息含支持列表；请求有实现的 id 正常构造；前缀族（如 `vbench2_*`）判定正确。7/7 通过。
说明：未实现这 60 个指标本身（按要求）；修复只把"静默不产出"变为"校验期报错"。

### ET-10 normalize_unit_score 对 (1,100] 一律 /100 歪曲 1-5 分制 —— 已修（**分数语义变化，见标注**）

改动（`execution/framework/official_runner.py`）：

- 新增 `declared_metric_normalizers(benchmark_id)`（`lru_cache` + `MappingProxyType` 只读视图）：从 benchmark catalog 的指标元数据读取显式 per-metric `normalizer` 声明。
- 新增 `normalized_metric_score(...)`：有显式声明 → 按声明换算（权威路径）；无声明 → 回落到原 `normalize_unit_score` 启发式**并打 warning**（提示该指标未声明量纲）。
- `generic_extract_metrics` / `catalog_fallback` / `metric_row` 统一改走 `normalized_metric_score`（`metric_row` 新增可选 `config` 参数，默认 None 保持旧签名兼容）。

**分数语义变化标注**：

- 有 catalog `normalizer` 声明的指标：换算依据从"盲目启发式"变为"显式声明"，两者不一致时**分数会变**——这是本条修复的目的（声明为权威）。
- 无声明的指标：数值行为与旧版**完全一致**（仍走 `normalize_unit_score` 启发式），仅新增 warning。选择保留回落而非直接去掉启发式，因为 `rg` 调用面排查显示既有 scorecard 消费方依赖 `normalized_score` 非空；直接置 None 会破坏消费方，属报告允许的"收窄"路线。
- `normalize_unit_score` 本身未改（公共符号行为不变）。

验证：`test_eval_tasks_fix_official_runner.py` —— 声明 `scale: [1,5]` 类 normalizer 的指标按声明换算（不再 /100）；未声明的 3.5 分值维持旧启发式产出并可捕获 warning 日志。通过。

### ET-01 BENCHMARK_INTEGRATION_REGISTRY 缺 5 个 benchmark + vbench-2.0 id 漂移 —— 已修

改动：

- `execution/framework/integration.py`：补齐 5 个缺失条目——`4dworldbench`、`larybench`、`sana-wm-bench`、`worldolympiad`、`worldreasonbench`（tier/runner_script/hf_dataset_id/judge_model_id 按 catalog 与 runner 实况填写）。
- `execution/runners/workspace_registry.py`：`validate_workspace_registry` 扩展为同时交叉校验 `VIDEO_RUNNER_REGISTRY` ↔ `BENCHMARK_INTEGRATION_REGISTRY`（防再漂移）。
- vbench-2.0：统一为带点的 `"vbench-2.0"` 规范形，三张表与 catalog/task/profile 全部对齐。

验证（一致性脚本 `/tmp/wf_registry_consistency_check.py`，修复后输出摘录）：

```text
A video_runner_registry     : 50
B integration_registry      : 50
C workspace (CLI + vbench)  : 50
D catalog ids               : 72
E task yaml ids             : 72
F runtime profile ids       : 72
OK   A-B / B-A / A-C / C-A / A-D / A-E / A-F / D-E / E-D / D-F（全部无缺漂）
vbench-family ids observed  : ['vbench', 'vbench-2.0', 'vbench-plus-plus']
OK   vbench ids use the canonical dot/hyphen forms everywhere
OK   validate_workspace_registry: no issues
ALL CONSISTENCY CHECKS PASSED (registries aligned)
```

---

## P2 已修

### ET-05 workspace `_run_cli_command` 无超时 —— 已修

改动（`execution/runners/workspace_registry.py`）：`subprocess.run` 改为 `run_logged_subprocess`（stdout/stderr 落盘 + 进程组清理）；新增 `_workspace_subprocess_timeout()`（config 显式值优先，否则 `WORLDFOUNDRY_BENCHMARK_TIMEOUT` 环境变量，均未设保持无超时的历史行为）；超时抛 `RuntimeError` 并附日志尾部（`read_text_tail`）。

验证：`test_eval_tasks_fix_workspace_timeout.py` —— fake 慢命令 + 1s 超时 → 抛错且错误消息含日志尾；未设超时时行为不变；config 值优先于 env。4/4 通过。

### ET-06 7 处 runner 运行时 subprocess 无超时 —— 已修

改动：`official_runner.py` 新增 `default_benchmark_timeout()`（读 `WORLDFOUNDRY_BENCHMARK_TIMEOUT`，未设/非法返回 None = 保持历史无界行为），并接线到全部 7 处：

- `runners/phygenbench/phygenbench_runtime.py`（upstream overall.py 聚合）
- `runners/videophy/videophy_runtime.py`、`runners/videophy2/videophy2_runtime.py`（judge 子进程）
- `runners/wbench/run_wbench_official_runner.py`、`runners/worldarena/run_worldarena_official_runner.py`（官方管线子进程）
- `runners/wrbench/wrbench_runtime.py`（D1-D6 评测子进程）
- `runners/_scorers/vqa_score/_backend.py`：`ffmpeg -version` 探测加固定 60s 超时（探测应瞬时完成），异常白名单补 `TimeoutExpired`。

验证：`test_eval_tasks_fix_runner_timeouts.py` —— env 解析矩阵（未设/合法/非法/空串）；phygenbench 真子进程挂死 + 1s 超时 → 抛 `TimeoutExpired`；未设 env 时管线照常跑通（历史行为保持）。4/4 通过。

### ET-08 videoscore sys.path 永久污染 + transformers 全局 monkeypatch 不可逆 —— 已修

改动（`execution/runners/videoscore/run_videoscore_official_runner.py`）：

- `load_official_videoscore_module`：`sys.path` 插入仅限 import 期间，`try/finally` 恢复；只移除自己插入的项（已在 path 中的不动）。
- `patch_transformers_dynamic_cache_api`：改为返回 `undo_patch` 可调用（未打补丁时返回 None）；`run_bounded_videoscore` 以 `try/finally` 确保 bounded 推理结束后撤销补丁。既有调用方以"调用后忽略返回值"方式使用，兼容（已核对 `test_benchmark_zoo_scripts.py:3833` 的调用形态）。

验证：`test_eval_tasks_fix_videoscore_syspath.py` —— import 成功/失败后 `sys.path` 均复原；预先存在的路径不被误删；undo 回调恢复原方法。5/5 通过。

### ET-12 别名子串模糊匹配可能错配 —— 已修

改动（`official_runner.py` `metric_id_from_key`）：子串匹配仅在**唯一**命中一个 metric id 时接受；多义命中直接拒绝（返回 None）；唯一但非精确的命中打 warning（提示建议补精确别名）。精确别名/canonical 命中路径不变。

验证：`test_eval_tasks_fix_official_runner.py` 内 alias 用例 —— 多义子串返回 None、唯一模糊命中带 warning、精确命中无 warning。通过。

### ET-16 evalcrafter dover 脚本 bare except + resume 路径 `pkl.dump` 应为 `pkl.load` —— 已修

改动（`execution/runners/evalcrafter/metric_scripts/dover_evaluate_a_set_of_videos.py`，wrapper 层脚本非 vendored）：resume 读取改为 `pkl.load`（原 `pkl.dump` 在读取分支，恢复功能实际从未生效且可能截断已有结果文件）；bare `except:` 拆为 `except FileNotFoundError:`（正常冷启动）与 `except Exception as exc:`（打日志不再静默）。

验证：`py_compile` 通过；逻辑为 wrapper 内自包含改动，行为由代码审读 + 既有 evalcrafter 测试收集无回归确认。

### ET-17 / ET-18 / ET-19 catalog 卫生三连 —— 已修

改动：

- ET-17（`catalog/schema.py`）：`_normalize_source_status` / `_normalize_integration_status` 遇到不认识的非空状态串降级前调用 `_warn_unrecognized_status`（模块级去重集合，每个 `(kind, value)` 只警告一次）。降级取值本身不变（`unknown`/`planned`），纯可观测性增强。运行既有测试时已观察到真实 manifest 中 `official_runtime_start_ready` 等未收录状态触发该 warning（证明其价值）。
- ET-18（`catalog/benchmark_catalog.py`）：新增 `_entry_benchmark_id()` 同时认 `benchmark_id:` 与 `id:` 两种键，`_catalog_benchmark_path_index` 与 `benchmark_catalog_ids` 统一走它——修复"只写 `benchmark_id:` 的 shard 在 ids 列表里消失"的不一致。
- ET-19（`catalog/benchmark_catalog.py`）：shard 解析的 `except Exception` 收窄为 `except (OSError, TypeError, ValueError, yaml.YAMLError)` 并对每个被跳过的坏 shard 记 `warning`（含路径与异常），坏 YAML 不再无声消失。
- ET-19 附注（lru_cache 缓存陈旧/共享可变实例）：报告建议"缓存键纳入 mtime **或**提供显式失效接口"——两处缓存族均已有显式失效接口（`benchmark_catalog.clear_benchmark_catalog_cache()`、`zoo_registry.clear_benchmark_zoo_registry_cache()`），满足其一，**无需改动**。

验证：`test_eval_tasks_fix_catalog_hygiene.py` —— 未识别状态触发一次性 warning 且值仍按旧规则降级；`benchmark_id:`-only shard 在 path index 与 ids 两处均可见；坏 YAML shard 触发 warning 且其余 shard 正常索引。5/5 通过。

### ET-21 catalog/runtime profile/docs 缺漂（ai2thor、larybench 无 profile）—— 已修（docs 部分越界，deferred）

改动（均在 `worldfoundry/data/`）：

- 新增 `benchmarks/runtime_profiles/official/ai2thor.yaml`、`benchmarks/runtime_profiles/official/larybench.yaml`（environment_id、required_assets/env/packages/paths、status、validation_command 等按 runner 实况填写）。
- `benchmarks/catalog/embodied/larybench.yaml`：补 `checkpoint_refs: []`、`validation_command`，`runner_availability` 增加 `verification_scope` 与 task manifest 对齐。
- `benchmarks/tasks/external/larybench.yaml`：metadata 补齐 `contract_validation_command`、`requires`、`blockers`、`checkpoint_refs`、`validation_command`、`artifact_layout`。
- 所有 YAML 改动经 `python -c "yaml.safe_load(...)"` 语法验证 + 一致性脚本 id 交叉验证（见 ET-01 输出：catalog/task/profile 三方 72=72=72 全对齐）。

验证：既有测试 `test_zoo_readiness_contracts.py::test_formal_benchmark_inventory_has_catalog_task_and_runtime_profile_for_every_id` 由修复前失败转为通过。
docs 状态表（`docs/fumadocs/lib/benchmark-catalog-status.json` 缺 3 个 id）在 `docs/` 下，超出本次允许改动范围 —— **deferred**（重新生成 docs 即可，见 deferred 清单）。

---

## P3 已修

### ET-03 `VideoRunnerSpec.results_flag` 50 项全部重复同一值 —— 已修

改动（`execution/framework/runner_registry.py`）：`results_flag` 字段加默认值 `"--official-results-path"`；既有 50 处显式传参保持原样（值相同，无行为变化），新增条目不再重复。

验证：`test_eval_tasks_fix_runner_timeouts.py` —— 缺省构造取默认值；注册表全部条目仍解析为同一 flag。通过。

### ET-13 devil_dynamics 通用名别名错配风险 —— 已修

改动（`execution/runners/devil_dynamics/run_devil_dynamics_official_runner.py`）：从 `CONFIG.metric_aliases` 移除 4 个通用名别名（`subject_consistency`/`background_consistency`/`motion_smoothness`/`naturalness`→DEVIL 专有指标的映射）。XLSX 解析走 `DEVIL_KEY_TO_METRIC` 专用映射、官方 JSON 走 canonical id，均不受影响；移除后通用名不再被劫持为 DEVIL 指标。

验证：`test_eval_tasks_fix_devil_aliases.py` —— 通用键不再映射到 DEVIL 指标；DEVIL 专有键/canonical id 解析不变。3/3 通过。

### ET-09 wrapper 顶部 sys.path shim 不一致 —— 核实后无需改动

核查结论：报告指认的无条件 `sys.path.insert`（devil_dynamics 等）在当前代码中已全部为条件插入形式（`if str(REPO_ROOT) not in sys.path:`）；`rg` 全量扫描确认 wrapper 层（`runners/*.py`、`run_*_official_runner.py`）无一处无条件插入，剩余无条件插入均在 vendored `runtime/` 下（本次范围禁改）。评审快照落后于当前代码，**无需改动**；"抽公共 helper"属锦上添花的重构，不做。

---

## Deferred 清单（含原因与建议方案）

| 条目 | 内容 | 原因 | 建议 |
| --- | --- | --- | --- |
| ET-02 (P2) | 6 个 runner 不接受 `--benchmark-id`，CLI 契约不统一 | 接口级重构：需改 6 个 runner 的参数面并同步 `pass_benchmark_id` 消费方，公共接口变化与"另一 agent 正在修测试"冲突，回归面大 | 统一到 `run_main` 标准参数集，`pass_benchmark_id` 字段随之退役；配合 ET-01 的注册表交叉校验做 CI 断言 |
| ET-07 (P2) | 子进程工具二分（run_logged_subprocess vs subprocess.run/Popen 共 16 文件） | 本次已把最关键的三处（official_runner 主管线、workspace_registry、worldscore）收敛到 core.process 工具族；其余 13 处逐文件迁移属机械大改，收益边际递减 | 提炼 `run_vendored_runtime()` 公共入口后分批迁移 |
| ET-20 (P3) | `BenchmarkZooEntry` 7 个状态字段互为回退 | schema 语义重设计，牵动全部 72 个 manifest 与消费方，非最小修复 | 收敛为源码可得性/集成程度/验证程度三正交维度；ET-17 的 warning 已先行提供可观测性 |
| ET-21（docs 部分） | docs 状态 JSON 缺 `apple-pi`/`larybench`/`physical-ai-bench` | `docs/fumadocs/` 不在允许改动范围 | 重跑 docs 生成器即可（catalog 侧数据已齐） |
| ET-22 (P3) | catalog 顶层三证据位全库 false、`official_gpu_validation.scorecard` 引用运行产物路径 | 正面确认为主；改进需重设计 evidence 字段并入库 evidence 文件，涉及 reporting 消费方（禁改范围） | evidence 指向入库文件或哈希；内外两份同名布尔保留一处 |
| ET-23 (P3) | framework 层无 seed 管理约定 | `GenerationRequest` 在 evaluation framework/models 范围（本次禁改） | 在 GenerationRequest/runtime profile 约定可选 `seed` 键并写入 scorecard reproduction 块 |
| ET-24 (P3) | 导入卫生正面确认 | 无需修复 | — |

---

## 验证汇总

1. **collect 红线**：修复前 1436 collected / 18 errors → 修复后 1516 collected / **0 errors**（无新增收集错误；18 个既有错误由另一 agent 并行修复，非本次改动）。
2. **新增测试**：7 文件 36 条全部通过（`36 passed in 32.71s`）。
3. **静态验证**：全部改动文件 `py_compile` 通过；关键模块 `PYTHONPATH=. python -c "import ..."` 导入通过（official_runner、runner_registry、workspace_registry、metrics.registry、benchmark_catalog、schema、各 runtime wrapper）。
4. **一致性脚本**：`/tmp/wf_registry_consistency_check.py` 全绿（三注册表 50=50=50，catalog/task/profile 72=72=72，vbench id 规范形统一，`validate_workspace_registry` 无 issue）。
5. **既有测试回归**（触及模块的定向回归）：`test_benchmark_registry.py` / `test_metric_registry.py` / `test_catalog_core.py` / `test_benchmark_zoo_schema.py` / `test_phygenbench_official_runner.py` 合跑 35 passed / 10 failed——10 个失败**全部为既有漂移**，与本次改动无关，证据如下：
   - `test_benchmark_zoo_schema.py` 8 个失败：断言 `from_dict` 合成 `contract_validation_command`、顶层证据位为 true、robotwin run_command 指向旧脚本等——涉及字段均非本次触碰（本次 schema.py 改动仅 +22 行 warning）；修复前同一会话已记录同批失败。
   - `test_benchmark_registry.py::test_benchmark_zoo_exports_catalog_v2_specs`：期望 `schema_version == "worldfoundry-catalog-benchmark"`，当前值 `"worldfoundry-benchmark"`——该常量不在本次任何改动文件中（并行改动所致）。
   - `test_phygenbench_official_runner.py::test_phygenbench_official_run_with_mock_backend_writes_scorecard`：期望 `evaluation.kind == "phygenbench_official_in_tree"`，该字符串在 `worldfoundry/` 全库 rg 零命中（只存在于测试文件），runner 从未产出过此值；本次对 phygenbench 仅加 `timeout=default_benchmark_timeout()`（env 未设时 = None，与旧行为逐字节一致）。
   - `test_benchmark_zoo_scripts.py::test_videoscore_runner_patches_transformers_dynamic_cache_api`：import 不存在的 `framework.script_paths` 模块失败（测试侧引用未落地的模块，属测试修复 agent 范围）；本次对 `patch_transformers_dynamic_cache_api` 的返回值变更与该测试的"调用后忽略返回值"形态兼容。
   - `test_phyeduvideo_official_runner.py` / `test_physvidbench_official_runner.py` 各 1 个失败：会话开始时（任何改动前）即失败，已记录为既有问题。
