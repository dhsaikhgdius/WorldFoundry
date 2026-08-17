# Evaluation 框架层修复日志（EF-01 ~ EF-38）

对应评审报告：`plan/code_review/03_evaluation_framework.md`

修复范围（硬边界）：仅 `worldfoundry/evaluation/` 下的 `framework.py`、`runner.py`、`public.py`、`utils.py`、`__init__.py`、`__main__.py`、`api/`、`models/`、`reporting/`；允许在 `test/` 下新建 `test_eval_framework_fix_*.py` 纯 CPU 测试。

## 基线（修复前实测）

- `PYTHONPATH=. python -m pytest test/eval_core --collect-only -q`：1419 tests collected；收集错误数在 9–16 间波动（多次运行取并集共 15 个文件），全部为环境缺第三方依赖（`ftfy`/`fastapi`/`imageio` 等），与本层代码无关。
- `python -X importtime -c "import worldfoundry.evaluation.models.runtime.profiles"`：总耗时 ~1.50s，输出含 669 行 torch 相关模块（顶层 `BaseSynthesis` 拉起完整 torch 栈）。
- `python -X importtime -c "import worldfoundry.evaluation.models.runners.builtins"`：顶层导入 `worldfoundry.evaluation.tasks.embodied.rollout_runner`，连带 tasks 层与 numpy。
- `python -X importtime -c "import worldfoundry.evaluation.runner"`：~445ms（无 numpy/torch，但 eager 拉起 9 个 orchestration 模块）。
- 并行现状：本仓库同时有其他 agent 在改 `worldfoundry/evaluation/tasks/`、`worldfoundry/data/`、`core/` 等区域；运行既有测试时观察到的失败若病因位于上述区域（文件布局断言、docker 镜像串、`environments.py` 签名），一律记为并行基线，不在本层修复范围。

## 最终验证汇总（修复后实测）

- **collect-only**：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q` → **1516 tests collected, 0 errors**（基线 1419 collected + 9–16 errors；错误消失系并行 agent 修复了缺依赖测试文件，本层无新增收集错误——红线满足）。
- **importtime 终态**（本 NFS 环境墙钟波动大，以 torch/numpy 行数为准）：
  - `models.runtime.profiles`：torch 行 **669 → 0**，numpy 行 0；墙钟 1.50s → 0.27–0.99s。
  - `models.runners.builtins`：`evaluation.tasks.*`/numpy 行 **→ 0**（惰性化后仅剩 `api.tasks` 字符串误匹配）。
  - `evaluation.runner`：445ms → **~208ms**，torch/numpy 行 0（懒 facade）。
- **py_compile**：31 个改动/新建 .py 全部通过。
- **新建测试**：`test/test_eval_framework_fix_redact_and_status.py` + `test/test_eval_framework_fix_registry_and_artifacts.py` 共 **22 passed**（纯 CPU）。
- **定向回归**：`test_api_contracts / test_base_contracts / test_metric_registry / test_model_runner_registry / test_model_manifests / test_pipeline_contract / test_pipeline_plugin_discovery` 等本层契约测试通过（52 + 40 passed 两轮）；全部残余失败逐一溯源均为并行基线（证据见文末"并行基线失败清单"）。EF-02 第一版全量自去重曾使既有 `test_duplicate_keys_and_aliases_are_left_to_registry` 失败，已按"评审建议 ∩ 既有测试契约"收敛语义后恢复全绿（详见 EF-02 条目）。

## 已修复

### EF-30 (P1) profiles.py 顶层 import BaseSynthesis 拉起 torch

- 改动：
  - 新建 `worldfoundry/evaluation/models/runtime/profile_synthesis.py`，将 `RuntimeProfileSynthesis` 及其专属 helper（`_coerce_path_input`、`_json_safe`）整体迁出 `profiles.py`；该模块才 import `BaseSynthesis`（继而 torch）。
  - `profiles.py` 回归 stdlib+yaml：删除顶层 `BaseSynthesis`/`resolve_conda_env_context`/`json`/`sys`/`tempfile` 导入；新增 PEP 562 `__getattr__`/`__dir__`，按名惰性暴露 `RuntimeProfileSynthesis` 与 `BaseSynthesis`（首访问时才 import `profile_synthesis`），既有 `from ...runtime.profiles import RuntimeProfileSynthesis` 的调用方（synthesis/ 侧 100+ 模块）不受影响。
  - `models/runtime/__init__.py`：移除对 `RuntimeProfileSynthesis` 的 eager re-export，改为模块级 `__getattr__` 惰性暴露；`__all__` 不变。
- 验证：
  - `python -X importtime -c "import worldfoundry.evaluation.models.runtime.profiles" | rg -c torch` → **0**（基线 669 行）；总导入耗时 1.50s → **0.27s**。
  - 惰性路径身份校验：`profiles.BaseSynthesis is worldfoundry.synthesis.base_synthesis.BaseSynthesis`、`RuntimeProfileSynthesis.__mro__[1] is BaseSynthesis`、`runtime.RuntimeProfileSynthesis is profiles.RuntimeProfileSynthesis` 全部通过；访问前 `torch not in sys.modules`。
  - `test/eval_core/test_runtime_profile_base_synthesis.py` 全绿。

### EF-31 (P2) RuntimeProfileSynthesis.predict 死代码与误导性语义

- 改动（随 EF-30 迁移一并完成，位于 `profile_synthesis.py`）：
  - 删除从未被引用的 `predict(timeout_seconds=21600)` 参数（rg 确认无调用方传参；为兼容仍以 `kwargs.pop("timeout_seconds", None)` 吞掉传入值）。
  - 删除从未被调用的 `_runtime_env()` 方法（rg 确认零引用）。
  - 类与 `predict` docstring 明示"仅 plan（prepared）或 blocked，从不执行命令"。
  - 删除 `profiles.py` 中无调用点的第三份 `project_root()` 实现（rg 确认零引用）。
- 验证：py_compile 通过；`predict` 行为不变（仅签名收窄 + docstring 修正）。

### EF-23 (P2) bindings.runtime_profile_execution_metadata 吞掉一切异常

- 改动（`models/pipelines/bindings.py`）：
  - 区分两类失败：`KeyError`（profile id 不存在，`load_runtime_profile` 的约定异常）→ warning 后返回 `{}`（路由可继续降级）；其余异常（YAML 损坏、加载环境坏掉）→ warning 后 **re-raise**，不再静默降级路由。
  - 模块新增 `LOGGER = logging.getLogger(__name__)`（与同族 discovery.py 一致）。
  - 与 EF-30 的组合修复后，此路径已不再 import torch（元数据加载轻量化），"坏 torch 环境被吞"的组合拳不复存在。
- 验证：未知 profile id 实测输出 `WARNING ... Runtime profile 'xxx' not found; continuing without profile metadata.` 且返回 `{}`；元数据路径全程 `torch not in sys.modules`。

### EF-24 (P2) 插件绑定与内置冲突被静默丢弃

- 改动（`models/pipelines/bindings.py` `merge_pipeline_binding_plugins`）：`except ValueError: continue` → `LOGGER.warning("Skipping plugin pipeline binding %r from slug %r: %s", ...)` 后 continue，与 discovery.py:133 的重复告警风格一致。
- 验证：py_compile + import 通过（该函数行为除日志外不变）。

### EF-17 (P1) builtins.py 顶层 import tasks.embodied 反向依赖

- 改动：
  - `models/runners/builtins.py`：删除顶层 `from ...tasks.embodied.rollout_runner import EmbodiedClosedLoopRunner`。`BuiltinRuntimeRunnerEntry` 新增 `runner_target: str` 字段、`runner_class` 放宽为 `type | None`，新增 `resolve_runner_class()`（惰性 `import_dotted_attr`）。embodied 条目只声明字符串 target；`worldfoundry.pipeline` 条目为同层模块（stdlib 级导入成本）保持 eager class（既有测试断言注册表条目 `.runner_class.__name__`）。
  - `to_dict()`：`runner_class` 键输出 target 字符串（与原 `f"{module}:{qualname}"` 逐字节一致），新增 `runner_target` 键（additive）。
  - `__all__` 不变；`EmbodiedClosedLoopRunner` 经模块 `__getattr__` 惰性暴露，兼容既有 `from ...builtins import EmbodiedClosedLoopRunner`。
  - `models/runners/registry.py`：注册 builtin 时 `runner_target=entry.runner_target or entry.name`，embodied 条目 `runner_class=None` → `resolve_runner_class` 走惰性 import；`worldfoundry.pipeline` 行为完全不变。
- 验证：
  - `import worldfoundry.evaluation.models.runners.registry` 后断言 `worldfoundry.evaluation.tasks.*`、`numpy` 均不在 `sys.modules`；构建 `ModelRunnerRegistry()` 后依然不在；`resolve_runner_class('embodied.rollout')` 才触发 tasks 导入并正确返回 `EmbodiedClosedLoopRunner`。
  - `python -X importtime -c "import ...runners.builtins" | rg "tasks|numpy"` 仅剩 `worldfoundry.evaluation.api.tasks`（api 层任务描述模块，非 `evaluation.tasks`，属正则误匹配）。
  - `test/eval_core/test_model_runner_registry.py` 全绿（含 `.runner_class.__name__` 与插件碰撞两条既有断言）。

### EF-08 (P1) utils.py import 时向 sys.path[0] 插入仓库根

- 改动（`evaluation/utils.py`）：
  - 删除模块顶层 `if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))` 副作用。
  - 新增显式 `ensure_repo_root_on_sys_path() -> Path`，需要 repo-root 动态导入的调用方自行调用（docstring 说明安装态风险）。
  - rg 全仓确认没有模块依赖"import utils 即改 sys.path"的隐式行为（源码态跑通全部定向测试佐证）。
  - 附带清理：`models/catalog/registry.py`、`models/catalog/zoo_registry.py` 中"为触发该副作用而保留"的 `REPO_ROOT` 死导入一并删除。
- 验证：py_compile；`PYTHONPATH=. python -c "import worldfoundry.evaluation.utils; import sys; assert sys.path[0] != '<repo>'"`；定向契约测试 52 passed。

### EF-04 (P2) 缺失/空 status 默认判"成功"（fail-open）

- 改动：
  - `api/generation.py::normalize_generation_status`：`text or "succeeded"` → `text or GENERATION_UNKNOWN_STATUS`（缺失/空白 status 归一为 `"unknown"`，不再默认成功）；`is_generation_status_successful("unknown")` 为 False（fail-closed）。
  - `GenerationResult.from_dict`：反序列化缺 `status` 键时默认 `GENERATION_UNKNOWN_STATUS`（外部 runner 写出的残缺结果行不再被当成功样本进指标）。
  - `api/__init__.py`：导出 `GENERATION_UNKNOWN_STATUS`（additive）。
  - `models/pipelines/results.py::pipeline_result_status`：in-process pipeline 返回 mapping 但未带 `status` 键 → 仍判 `succeeded`（进程内失败必然抛异常，这是原契约且有既有测试断言）；显式给了空白 status 或跨进程恢复的行 → 走新的 unknown 归一。语义边界写入 docstring。
- 验证：新增单测覆盖 `None/""/"  "/unknown/failed/succeeded` 全矩阵 + `from_dict` 缺键路径 + in-process mapping 无 status 仍成功；`test/eval_core/test_api_contracts.py`、`test_pipeline_contract.py` 全绿（往返序列化兼容）。

### EF-35 (P2) redact_secrets 子串匹配误伤 max_new_tokens/tokenizer

- 改动（`reporting/run_manifest.py::_is_sensitive_key`）：子串 `in` 匹配 → 按 `[^a-z0-9]+` 切词后的**全词/词对匹配**：单词命中 `{token, secret, password, passwd, credential, credentials, auth, bearer, apikey}`；相邻词对命中 `{api key, access token, auth token, secret key, ...}`；`key` 单独出现仅在与敏感限定词成对时命中。`max_new_tokens`、`tokenizer`、`tokens_per_second`、`monkey`、`keyframes` 等不再误伤。
- 验证：新增单测 12 组正反例（`api_key/API-KEY/auth_token/hf_token/password` 仍脱敏；`max_new_tokens/tokenizer/num_tokens/keyframe_interval` 不脱敏；嵌套 dict/list 递归行为不变）。

### EF-01 (P1) registry 重复注册报错路径 AttributeError

- 改动（`api/registry.py::_Registry._existing_name_for_key`）：`self._items[key].name` → `self._key_fn(self._items[key])`。`MetricSpec` 无 `.name` 字段（key_fn 是 `lambda s: s.metric_id`），原实现在"重复注册 MetricSpec"的报错路径上先炸 `AttributeError`，掩盖真正的冲突信息。
- 验证：新增单测：重复注册 `MetricSpec` 现在抛出带 metric id 的 `ValueError`（而非 AttributeError）；`test_metric_registry.py` 全绿。

### EF-02 (P2) 注册自碰撞：name 大小写变体 / alias 列表内部重复即报错

- 语义收敛（评审建议与既有测试契约的调和）：既有测试 `test_model_manifests.py::test_duplicate_keys_and_aliases_are_left_to_registry` 明确要求"manifest 构建器把重复源键留给 registry 报错"（显式 alias 与自身 canonical key 碰撞必须抛 `DuplicateRegistryKeyError`，作为源表重复键的第二道防线），不能全量自去重。最终边界：
  - **`name` 与 `model_id` 仅大小写不同 → 可注册**（评审动机用例）。根因修在 `_model_aliases`：原先用大小写敏感的 `name != model_id` 决定是否把 name 追加为 alias、注册时却按 casefold 归一——改为 `lookup_key(name) != lookup_key(model_id)` 同一归一化比较，大小写变体不再被追加成自碰撞 alias。
  - **alias 列表内部 casefold 重复 → 静默去重**（`dict.fromkeys`，与 `AliasRegistryStore.register_with_conflict` 语义一致）。
  - **显式 alias 与自身 canonical key 碰撞 → 保持报错**（既有测试契约），但消息从误导性的 `'alpha' conflicts with 'alpha'` 改为明示自碰撞：`an alias duplicates the item's own canonical key (redundant alias or duplicate source keys)`。
  - 跨条目重名语义不变（仍报错）。相对 HEAD 严格更宽松，真实 catalog（273 manifests）注册无新失败。
- 验证：`ModelManifestRegistry().register(WorldModelManifest(model_id='foo', name='Foo'))` 实测注册成功且 `get('Foo')` 解析正确（评审动机用例，HEAD 上抛错）；新增单测覆盖三条边界；既有 `test_model_manifests.py::test_duplicate_keys_and_aliases_are_left_to_registry`、`test_build_model_manifest_registry_returns_core_model_registry` 全绿。

### EF-05 (P3) registry.py docstring 位置错误 + 从包 __init__ 反向 import

- 改动（`api/registry.py`）：模块 docstring 移到文件首行（原先位于 import 之后成为无效字符串语句）；`from . import MetricSpec, WorldModelManifest`（对包 `__init__` 的反向导入）→ `from .metrics import MetricSpec`、`from .world_model_manifest import WorldModelManifest` 直接从定义模块导入。
- 验证：`python -c "import worldfoundry.evaluation.api.registry; assert registry.__doc__"`；api 包导入顺序不再依赖 `__init__` 执行完成。

### EF-11 (P2) framework.py canonical-id 解析吞掉加载失败

- 改动（`framework.py`）：`_canonical_model_id_or_self` 的 `except (KeyError, TypeError, ValueError)` → 仅捕 `UnknownModelZooKeyError`；`_canonical_benchmark_id_or_self` 同理仅捕 `UnknownBenchmarkZooKeyError`。未知 id 保留"原样返回"的宽松语义，但 YAML 损坏/schema 错误等真实加载失败现在向上冒泡，不再被静默当成"未知 id"。
- 验证：py_compile；定向跑 framework 相关契约测试通过。

### EF-13 (P3) framework._model_manifest_dir_for except Exception

- 改动（`framework.py`）：`except Exception` → `except ImportError`（该 try 块内唯一预期失败是可选依赖导入；其余异常冒泡）。
- 验证：py_compile + import。

### EF-18 (P2) YAML 解析错误不带文件路径

- 改动（统一模式：`raise yaml.YAMLError(f"failed to parse ... {path}: {exc}") from exc`；schema 错误同理带路径）：
  - `utils.py::load_manifest`（所有 manifest 加载的公共入口）。
  - `models/catalog/manifest.py::_iter_catalog_mappings` + `load_model_catalog_manifests`（schema `TypeError/ValueError` 亦带路径）。
  - `models/catalog/schema.py::load_entries`、`models/catalog/zoo_registry.py::_load_entries_with_priority`（`ModelZooEntry.from_dict` 失败带 manifest 路径）。
  - `models/pipelines/bindings.py::load_pipeline_binding`、`models/pipelines/aliases.py::load_pipeline_alias_groups`。
  - `models/runtime/_shared.py::iter_manifest_mappings`（EF-32 合并后 runtime 三个 loader 共享同一带路径报错）。
- 验证：实测坏 YAML 报错串包含完整文件路径；新增单测覆盖 `load_manifest` 坏 YAML 路径注入。

### EF-19 (P2) zoo_registry 重复 model_id 静默去重

- 改动（`models/catalog/zoo_registry.py::_dedupe_entries_by_target_priority`）：同优先级下重复 `model_id` 被丢弃时 `LOGGER.warning`，消息含**两个文件路径**（保留者 + 被丢弃者）；跨优先级覆盖（target override legacy 的设计语义）保持静默。
- 验证：构造双文件重复 id 实测 warning 输出两个路径；真实 catalog 加载无 warning（无同级重复）。

### EF-21 (P2) 未知 integration/demo 状态静默归为 planned/pending

- 改动（`models/catalog/schema.py`）：新增 `_warn_unknown_status_once`：进程内首个未知状态值 `WARNING`（附"启用 DEBUG 看全量、有意的新状态请注册"提示），后续每个新未知值 `DEBUG`（真实 catalog 有 64 种未注册 demo_status，逐条 WARNING 会刷屏）。`_normalize_integration_status`/`_normalize_demo_status` 接入。归一化结果不变（仍降级 planned/pending——语义变更需 catalog 数据侧配合，见 deferred）。
- 验证：真实 catalog 加载输出单条 WARNING；DEBUG 级别可见全部未知值明细。

### EF-25 (P2) resolver 未知 model id 报错无建议、异常类型泄漏

- 改动（`models/runners/resolver.py`）：
  - `resolve_model_zoo_config`：`UnknownModelZooKeyError` → 转 `ModelResolutionError`，消息含 `difflib.get_close_matches` 从全部已注册 id+alias 算出的"did you mean"建议。
  - `resolve_world_model_runner`：runner 解析失败的报错串补充原异常类型名（`f"{type(exc).__name__}: {exc}"`），不再丢失根因类别。
- 验证：新增单测：拼错 id 报 `ModelResolutionError` 且建议列表含正确 id；未知 id 无相近项时消息含"available models"计数。

### EF-06 (P3) ArtifactRef 无条件全文件 SHA-256

- 改动（`api/artifacts.py`）：`ArtifactRef.from_path`/`from_uri`/`enrich_artifact_ref` 新增 `compute_hash: bool = True` 关键字参数（默认行为完全不变）；`compute_hash=False` 时跳过哈希但仍填 `size_bytes`，且**保留**已有 `sha256`（不会把已算好的哈希抹成 None）。
- 验证：新增单测：默认路径哈希照算；`compute_hash=False` 跳算、已有哈希保留、size 仍回填。

### EF-07 (P3) coerce_artifact_refs 非 mapping 值报错不可行动

- 改动（`api/artifacts.py::coerce_artifact_refs`）：裸 `dataclasses` 构造错误 → 显式 `TypeError`，消息含 artifact 名、实际类型与期望形态（`ArtifactRef | mapping with 'uri'`）。
- 验证：新增单测断言报错消息含键名与类型名。

### EF-14 (P3) runner.py facade eager 拉起 9 个 orchestration 模块

- 改动（`evaluation/runner.py`）：全部 re-export 改为 `_LAZY_EXPORTS: dict[symbol -> module]` + PEP 562 `__getattr__`/`__dir__` 惰性解析；`__all__` 收窄为运行入口与请求/结果类型，移除混入公共面的通用 cache/hash 工具（`file_sha256` 等仍可从 `utils` 导入，additive 兼容别名保留一版）。
- 验证：`python -X importtime -c "import worldfoundry.evaluation.runner"` 445ms → **208ms**；`from worldfoundry.evaluation.runner import <每个旧符号>` 逐一断言可导入且与源模块同一对象。

### EF-16 (P3) engine_version 恒定字符串 + git 元数据 lru_cache 跨 run 过期

- 改动（`evaluation/utils.py`）：`EVALUATION_ENGINE_VERSION` 加数字修订位（`"worldfoundry-eval-engine/1"`），后续行为变更有可比对的版本锚点；`git_metadata()` 去掉 `lru_cache`（长驻进程内跨 run 的 commit/dirty 状态不再陈旧），每次 run 采集一次的调用频率下开销可忽略（子进程 ~20ms）。
- 验证：同进程内两次调用之间制造 dirty 状态，第二次结果反映变化；version-context 相关测试通过。

### EF-20 (P2，部分) lru_cache 注册表无清理入口

- 改动：补齐缺失的清理入口——`models/pipelines/bindings.py::clear_pipeline_binding_registry_cache()`、`models/pipelines/aliases.py::clear_pipeline_alias_registry_cache()`（与 zoo 既有 `clear_model_zoo_registry_cache` 对齐；测试/长驻进程可显式失效）。
- 未做（见 deferred）：缓存返回冻结视图/内容指纹 key。

### EF-22 (P3) schema.py official_sources/sources 两段逐行复制的解析逻辑

- 改动（`models/catalog/schema.py`）：
  - `_hf_repo_ids_from_entry`：两段逐字相同的 HF 解析块 → `for source_key in ("official_sources", "sources")` 单一代码路径（优先级顺序不变）。
  - `_checkpoint_refs_from_entry`：两段近似块（official 接受 4 种 HF 键拼写、sources 仅 1 种）→ 参数化 `(source_key, hf_keys)` 循环。
- 验证：**行为快照对比**——真实 catalog 全部 283 个 entry 的派生 `hf_repo_ids` + `checkpoint_refs.to_dict()` 修复前后 JSON 逐字节一致（`diff` 为空）；py_compile 通过。

### EF-28 (P3) 每次 default_model_runner_registry() 重扫 entry-points

- 改动（`models/runners/plugins.py`）：`importlib.metadata.entry_points(group=...)` 扫描提为 `@lru_cache(maxsize=1)` 的 `_entry_point_discovery`（进程内一次）；环境变量开关仍每次动态读取（行为可控性不变）。
- 验证：两次 `discover_model_runner_plugins()` 第二次不再触发 metadata 扫描（计时对比）；`test_pipeline_plugin_discovery.py` 全绿。

### EF-29 (P3) GenerationResult.timings 恒为空

- 改动：`models/pipelines/lifecycle.py::generate_in_context` 用 `time.perf_counter()` 测量 pipeline 调用耗时，经 `normalize(..., timings={"generate_seconds": round(elapsed, 6)})` 传入 `models/pipelines/results.py::generation_result_from_pipeline`（新增可选 `timings` 参数，additive），最终落入 `GenerationResult.timings`。评审误判澄清：`runtime` 字段存放的是运行时标识串（如 `worldfoundry.runtime_profile.plan`）而非耗时，保持不变。
- 验证：pipeline 契约测试全绿；in-context 生成路径实测 `timings["generate_seconds"] > 0`。

### EF-32 (P3) runtime 三个 loader 各持一份相同 helper

- 改动：新建 `models/runtime/_shared.py` 收敛 `tuple_of_str`/`schema_version_or_none`/`yaml_manifest_paths`/`iter_manifest_mappings`（带 kind+路径的统一报错）；`profiles.py`/`environments.py`/`assets.py` 删除各自的 `_tuple_of_str`/`_schema_version`/`_manifest_paths`/`_iter_*_mappings` 副本改用共享实现（collection_keys/id_keys 参数化保留各自 payload 形态差异）。
- 验证：三个模块加载真实 data/ 清单结果与修复前一致（profile/environment/asset 计数与 id 集合相同）；py_compile + 定向测试通过。

### EF-33 (P3) validate_runtime_registry O(N²) 文件 I/O

- 改动（`models/runtime/validate.py`）：`validate_runtime_profile_references` 新增可选 `environments`/`assets` 预加载映射参数（缺省时行为不变，逐个重扫）；`validate_runtime_registry` 先一次性加载三类清单再逐 profile 校验（O(N²) → O(N)）。
- 验证：真实 data/ 全量校验循环 **29s → 1s**（NFS 上效果显著）；校验结论（通过/失败集合）前后一致。

### EF-37 (P3) run_index/run_comparison 跨模块 import 6 个下划线私有函数

- 改动（`reporting/run_report.py`）：`_run_summary_path`/`_run_summary_candidate`/`_normalise_roots`/`_number_or_none`/`_dedupe_labels`/`_row_from_summary` 提升为公共 API（`resolve_run_summary_path`/`find_run_summary_candidate`/`normalise_report_roots`/`number_or_none`/`dedupe_labels`/`row_from_summary`），加入 `__all__`；旧下划线名保留为兼容别名（一版后可删）。`run_index.py`/`run_comparison.py` 改导入公共名。
- 验证：`compare-runs`/`index-runs` 相关测试全绿；旧名 `is` 新名（别名同一对象）。

### EF-38 (P3) reporting/_demo.py import 时 mkdir

- 改动：`DEMO_DIR.mkdir(exist_ok=True)` 从模块顶层移入 `main()`；import 该模块不再在磁盘上创建目录。
- 验证：`python -c "import worldfoundry.evaluation.reporting._demo"` 后目录不存在；`main()` 运行时正常创建。

### EF-09/EF-10（部分）utils.py 拼接清理 + hash/JSON 实现收敛

- 改动（`evaluation/utils.py`）：
  - 重复 4 次的 `from pathlib import Path` 等 import 全部归拢到文件头，删除 `# io.py`/`# versioning.py` 等分节注释残留；`benchmark_task_sample_path` 与常量块的粘连排版修正。
  - `file_sha256` 改为委托 `api/json_contract.py::sha256_file`（全仓单一分块哈希实现）。
  - `stable_json_dumps`/`stable_hash` 序列化前先过 `api.json_contract.to_plain`（set/frozenset 先排序再 JSON 化，"stable_hash 对 set 不稳定"的问题消除）。
  - `project_root` 全 evaluation 层收敛到 utils 一处（profiles.py 的第三份副本随 EF-31 删除；`worldfoundry_repository_root = project_root` 别名保留兼容）。
- 验证：`stable_hash({"a", "b"})` 多进程/多次运行结果一致；哈希值与 json_contract 直接调用一致。
- 未做（见 deferred）：utils.py 拆回子模块的完整重构；core/io/serialization.py 侧的第三份实现（越界，见移交）。

## 新建测试

- `test/test_eval_framework_fix_redact_and_status.py`：EF-35 redact 正反例矩阵、EF-04 status fail-closed 全矩阵（normalize/is_successful/from_dict/pipeline_result_status 边界）。
- `test/test_eval_framework_fix_registry_and_artifacts.py`：EF-01 MetricSpec 重复注册报错类型、EF-02 三条注册语义边界（name 大小写变体可注册 / alias 撞自身 key 报自碰撞错 / 跨条目冲突仍报错）、EF-18/19 错误路径与重复告警、EF-25 建议消息、EF-29 timings 透传、EF-37 公共 helper、EF-06/07 artifacts compute_hash 与报错可行动性。
- 合计 **22 passed**（纯 CPU，无 torch/numpy 依赖）。

## 移交项

- **EF-34 (P2) `write_json`/`_atomic_write_text` 固定临时文件名并发竞态**：位于 `worldfoundry/core/io/serialization.py`，超出本层修复边界。移交 core 基础层 agent（该文件在其范围内，CF 报告亦有相同发现）。建议：与同模块 `write_jsonl` 一致改用 uuid/pid 唯一临时名 + `os.replace`。
- **EF-10（core 侧）`core/io/serialization.py::stable_hash` 第三份实现**：evaluation 侧已收敛到 `api/json_contract.py`；core 侧副本的去留移交 core agent。
- **EF-15 (P3) SKILL.md 命令/路径过时**：`.cursor/skills/` 不在允许修改路径内，移交文档维护者。
- **EF-27 (P3) RUNTIME_VIDEO_ALIASES 硬编码与 data/ 别名 YAML 双轨**：修复方向是把硬编码表迁入 `worldfoundry/data/models/bindings/aliases/*.yaml`——写 data/ 越界（data agent 正在改）。移交时建议：迁移后 `aliases.py` 硬编码表改为空元组 + deprecation 注释，一版后删除。
- **EF-30 附带项（synthesis 侧）**：`worldfoundry/synthesis/base_synthesis.py` 顶层 import torch 本体、以及 synthesis/ 侧 100+ 模块反向顶层 import `evaluation.models.runtime.profiles` 的双向耦合，属 synthesis 层，本次仅在 evaluation 侧切断（profiles 不再 eager 触碰 synthesis）。
- **EF-21 数据侧**：catalog 中 64 种未注册 demo_status/integration_status 值的清洗或注册（`worldfoundry/data/models/catalog/**`），移交 data agent；evaluation 侧已提供 WARNING 可见性。

## Deferred（写方案不动）

- **EF-03 (P2) api/registry.py 双注册表实现（`AliasRegistryStore` vs `_Registry`）**：两者语义已随 EF-01/02 对齐（自 alias 去重、跨条目报错），但合并涉及 `AliasRegistryStore` 的 models/tasks 层调用方（tasks/ 正被并行修改）。方案：下一版将 `_Registry` 改为 `AliasRegistryStore` 的薄包装（`register` 委托 + 保留 `_key_fn` 定制），行为测试先行（本次新增的注册语义单测即为该重构的安全网）。
- **EF-12 (P2) WorldFoundryRunRequest God-request（50 字段）+ 启发式 dispatch**：公共契约层 dataclass 字段不能删改，拆分必然破坏兼容。方案：新增 `ExistingResultsRequest`/`SingleBenchmarkRequest`/`SuiteRequest` 三个子请求类型 + `WorldFoundryRunRequest.to_typed()` 迁移期适配器；dispatch 由"字段启发式"改为显式 `request.mode` 字段（additive，默认 auto 保持现行为）；两版后废弃旧启发式。
- **EF-20 (P2) 缓存返回可变 registry 实例**：本次已补清理入口。冻结方案：`load_*_registry` 返回 `MappingProxyType` 视图或在实例上置 `_frozen` 标志使 `register` 抛错（需先审计全部调用方是否有"拿缓存实例再 register"的用法，`merge_pipeline_binding_plugins` 的复制模式表明存在此类需求）；缓存 key 纳入目录 `(path, max_mtime)` 指纹可解决进程内数据过期。
- **EF-26 (P3) torch 后端全局 patch（TF32/SDPA）未入 version context**：方案：`build_version_context` 增加可选 `inference_infra` 段，运行路径（已加载 torch 之后）从 `worldfoundry.core` 读取 `inference_infra_state()` 快照注入；必须保持 utils.py 本身零 torch 导入（惰性、仅在 runner 执行路径调用），否则回退 EF-30 的成果。等 core agent 提供稳定的状态快照 API 后接入。
- **EF-36 (P3) scorecard/summary/index 载荷冗余 + artifacts 绝对路径**：载荷键收敛（`metrics.summary` vs `evaluation.summary` 二选一、`runs`/`rows` 合并）会破坏下游消费者，需要一版"双写 + deprecation 警告"过渡；artifacts 内部引用改相对路径需与 `validate-artifact`/`index-runs` 的解析逻辑同步改。方案已明确，待与报告消费方（leaderboard 工具）确认后一并做。
- **zoo_registry 与 catalog/registry 双轨**（评审"结构性问题"节）：`models/catalog/zoo_registry.py`（YAML 优先级去重 + 惰性 manifest 导出）与 `models/catalog/registry.py`（`WorldModelManifest` 注册表）是两套 id→模型解析链路，`resolver.py` 两边都查。方案：registry.py 退化为 zoo_registry 之上的**视图**（`discover_model_registry()` 内部改为 `load_model_zoo_registry().to_world_model_registry()`，manifest 转换已有 `model_zoo_entries_to_world_model_manifests` 可复用），公共函数签名不变；差异行为（registry 的 alias 大小写策略）以 zoo 为准归一。牵涉 tasks/ 层调用方，等并行修改收敛后执行。
- **EF-09（完整版）utils.py 拆分为子模块**：本次已做 import 归拢与死代码清理；完整拆分 `evaluation/utils/{io,manifest,paths,versioning}.py` 涉及全仓 100+ 处 `from ...utils import X` 的导入路径（含 tasks/ 层），方案是保留 `utils.py` 为纯 re-export facade 一版后再收紧——等 tasks/ 并行修改收敛后执行。

## 并行基线失败清单（与本层改动无关的失败，均已溯源）

- `test_model_zoo_registry.py` 5 项 + `test_model_manifests.py::test_model_package_facade_imports_are_stdlib_or_local`：测试硬编码 `<repo>/src/worldfoundry/...` 路径，仓库从无 `src/` 布局（git 历史确认），HEAD 上同样失败；抛错点（`utils.py::manifest_paths` 的 `FileNotFoundError` / `Path.read_text`）在 HEAD 与修复后行为相同。
- `test_model_runtime_layering.py` 2 项：断言 `load_runtime_environment_profiles(legacy_root=...)`，该参数在 HEAD 上就不存在（`git show HEAD` 确认 0 处 `legacy_root`）。
- `test_contract_stability.py` 1 项：CLI 缺 `contract` 子命令，子命令注册位于 `tasks/execution/orchestration/`（并行 tasks agent 区域）；本层 `framework.py` 的 40 行 diff 无任何 contract 子命令相关改动。
- `test_zoo_readiness_contracts.py` 等 316 项（全量跑）：断言 `worldfoundry/data/**` YAML 内容（如 `4dworldbench` 缺 `dataset.not_applicable`）与 `tasks/catalog/schema.py` 状态归一，均在并行 agent 修改区域（git status 显示 data/ + tasks/ 20+ 文件在改）。
- `test_embodied_simulator_registry.py::test_harness_docker_images_match_official_runtime_profiles`：断言 data/ 中 docker 镜像串，同上。
- `test_eval_runner_writes_version_context_and_fingerprint`：`test-contract-model` 无 zoo 声明（fixture 与 data/ 不同步），隔离验证与本层改动无关。
