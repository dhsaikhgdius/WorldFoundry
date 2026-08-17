# evaluation 框架层评审（framework/runner/api/models/reporting）

> 状态：已完成。最后更新：2026-08-14

## 评审范围与方法

**范围**（`worldfoundry/evaluation/`，tasks/ 由另一 agent 负责，仅报跨界依赖问题）：

| 模块 | 文件 | 行数 |
| --- | --- | --- |
| 顶层 facade | `framework.py`(485) `runner.py`(208) `public.py`(134) `utils.py`(452) `__init__.py`(91) `__main__.py`(7) | 1377 |
| api/ | `__init__.py`(82) `artifacts.py`(213) `generation.py`(154) `json_contract.py`(162) `metrics.py`(279) `models.py`(78) `registry.py`(341) `tasks.py`(245) `world_model_manifest.py`(58) | 1612 |
| models/catalog | `__init__.py`(92) `manifest.py`(706) `policy.py`(131) `registry.py`(157) `schema.py`(1144) `zoo_registry.py`(440) | 2670 |
| models/pipelines | `__init__.py`(93) `aliases.py`(317) `bindings.py`(407) `discovery.py`(147) `invocation.py`(248) `lifecycle.py`(154) `loading.py`(494) `results.py`(175) | 2035 |
| models/runners | `builtins.py`(107) `pipeline.py`(286) `plugins.py`(164) `registry.py`(334) `resolver.py`(418) | 1309 |
| models/runtime | `__init__.py`(65) `assets.py`(433) `environments.py`(380) `profiles.py`(1237) `validate.py`(311) | 2426 |
| models 顶层 | `__init__.py`(60) `import_target.py`(83) | 143 |
| reporting/ | `__init__.py`(169) `_demo.py`(216) `comparison_identity.py`(246) `run_browser.py`(321) `run_comparison.py`(338) `run_index.py`(493) `run_manifest.py`(330) `run_report.py`(399) `scorecard.py`(263) `validation.py`(619) | 3394 |

合计约 14966 行（含 tasks 外全部范围文件）。

**方法**：
1. 先读 `.cursor/skills/worldfoundry-evaluation-guide/SKILL.md` 及 references，确认设计原则（stdlib-first 契约、GPU/API import 下沉运行路径、api→models→tasks 依赖方向）。
2. 逐文件 Read 精读；`rg` 扫描反模式（bare except、顶层重依赖、全局可变状态、非原子写、subprocess 无超时等）。
3. 每条发现附 `path:line` 与代码摘录；严重度 P0（损坏/危险）/ P1（严重设计缺陷）/ P2（应修复）/ P3（改进建议）。

## 发现（按主题分组）

### 主题 A：api/ 契约层（artifacts / generation / metrics / registry / tasks）

总体评价：契约层质量较高——stdlib-only 属实（仅 `dataclasses/json/hashlib/mimetypes/urllib` 等），每个 dataclass 带 `schema_version` 常量并在构造时校验，`from_dict` 对缺省字段有默认值，`JsonContract.to_dict/from_dict` 大体对称（`task_id`/`metric_id` 兼容属性不落盘、读取时兼容）。以下为具体问题。

### [EF-01] P1 重复注册 MetricSpec 时抛 AttributeError 而非 DuplicateRegistryKeyError（错误路径本身是坏的）
- 位置：`worldfoundry/evaluation/api/registry.py:291-296`
- 证据：

```291:296:worldfoundry/evaluation/api/registry.py
    def _existing_name_for_key(self, key: str) -> str | None:
        if key in self._items:
            return self._items[key].name
        if key in self._aliases:
            return self._items[self._aliases[key]].name
        return None
```

- 问题：`_Registry` 同时服务 `WorldModelManifest` 和 `MetricSpec`（`SpecT = TypeVar("SpecT", WorldModelManifest, MetricSpec)`），但 `MetricSpec` 没有 `name` 字段（只有 `id`/`display_name`，见 `api/metrics.py:23-42`）。冲突检测路径访问 `.name` 直接崩。实测：

```text
>>> r = MetricSpecRegistry(); r.register(MetricSpec(id='fvd')); r.register(MetricSpec(id='FVD'))
AttributeError: 'MetricSpec' object has no attribute 'name'
```

- 影响：所有依赖 `MetricSpecRegistry` 报重复指标的调用方拿到的是 AttributeError，掩盖真实原因（重复 id/alias），且说明该错误路径从未被测试覆盖。
- 建议：改用 `self._key_fn(self._items[key])`，别硬编码 `.name`；补一条重复注册的单测。

### [EF-02] P2 manifest.name 与 model_id 仅大小写不同（或 name 同时出现在 aliases 里）时注册直接失败
- 位置：`worldfoundry/evaluation/api/registry.py:275-281`（配合 `_model_aliases`，`registry.py:176-182`）
- 证据：

```275:281:worldfoundry/evaluation/api/registry.py
        seen = {canonical_key}
        for alias in aliases:
            if alias in seen:
                raise DuplicateRegistryKeyError(
                    f"duplicate {self._kind} key {alias!r} in {item_name!r}"
                )
            seen.add(alias)
```

- 问题：`_model_aliases` 用大小写敏感比较 `manifest.name != manifest.model_id` 决定是否把 name 追加为 alias，而注册时按 `casefold` 归一。`WorldModelManifest(model_id='foo', name='Foo')` 实测抛 `DuplicateRegistryKeyError: duplicate model manifest key 'foo' in 'foo'`。同理 aliases 列表中与 name 重复的项也会触发。自碰撞应当去重而不是报错——同文件里另一套实现 `AliasRegistryStore.register_with_conflict`（`registry.py:71-73`）恰恰是 `if alias_key in seen: continue` 静默去重，两套语义不一致。
- 影响：合法 manifest 被拒注册；YAML 作者难以从报错理解"自己和自己冲突"。
- 建议：`_validate_new_keys` 内对 item 自身 alias 去重（continue），只对跨 item 冲突报错。

### [EF-03] P2 同一文件内两套几乎等价的 alias-registry 实现（AliasRegistryStore 与 _Registry）
- 位置：`worldfoundry/evaluation/api/registry.py:29-106` 与 `registry.py:197-296`
- 证据：

```29:31:worldfoundry/evaluation/api/registry.py
class AliasRegistryStore(Generic[ItemT]):
    """Ordered item store with normalized canonical keys and aliases."""
```

```197:198:worldfoundry/evaluation/api/registry.py
class _Registry(Generic[SpecT]):
    """Base registry class managing canonical keys, aliases, and insertion order."""
```

- 问题：两者都维护 `_entries/_items + _aliases + _order`、都做 casefold 归一、都做冲突检测，但行为细节分叉（自 alias 去重 vs 报错；冲突时返回三元组 vs 抛错）。`AliasRegistryStore` 被 models/tasks 层复用，`_Registry` 只服务本文件两个子类。
- 影响：注册语义随调用路径漂移（见 EF-02），维护双份逻辑。
- 建议：`_Registry` 基于 `AliasRegistryStore` 实现或直接合并。

### [EF-04] P2 状态缺失/为空一律视为"成功"：normalize_generation_status 的默认值方向危险
- 位置：`worldfoundry/evaluation/api/generation.py:23-48`
- 证据：

```29:30:worldfoundry/evaluation/api/generation.py
    text = str(status or "").strip().lower()
    return text or "succeeded"
```

- 问题：`status=None/""/空白` 全部归一为 `succeeded`，`is_generation_result_successful` 只要 `error` 为空即判成功。外部 runner 写出的残缺结果行（如崩溃时只写了 sample_id）会被当作可打分的成功样本进入指标。
- 影响：静默把坏数据算进分数，破坏 leaderboard 可信度；宁可 fail-closed。
- 建议：缺失 status 归一为 `"unknown"` 并判为不成功，或至少在 restore 处 WARN。

### [EF-05] P3 registry.py 模块 docstring 位于 import 之后成为无效语句；且从包 `__init__` 反向 import
- 位置：`worldfoundry/evaluation/api/registry.py:1-8`
- 证据：

```3:8:worldfoundry/evaluation/api/registry.py
from collections.abc import Callable as CollectionsCallable
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, TypeVar

from . import MetricSpec, WorldModelManifest

"""Registries and store implementations for WorldFoundry models and metrics."""
```

- 问题：docstring 在 import 后，不再是 `__doc__`；`from . import ...` 使子模块依赖包 `__init__`（父包 barrel），形成"子模块→父包→其它子模块"的隐式初始化顺序耦合（api/__init__.py:3-45 先 import artifacts/generation/metrics 等）。当前恰好不循环，但任何人把 registry 提前加入 barrel 就会炸。
- 建议：docstring 上移；改 `from .metrics import MetricSpec` / `from .world_model_manifest import WorldModelManifest`。

### [EF-06] P3 ArtifactRef.from_uri / enrich_artifact_ref 对本地文件无条件全量 SHA-256
- 位置：`worldfoundry/evaluation/api/artifacts.py:139-147`、`198-213`
- 证据：

```139:147:worldfoundry/evaluation/api/artifacts.py
        if local_path is not None and local_path.is_file():
            return cls.from_path(
                local_path,
                kind=kind,
                uri=text_uri,
                mime_type=mime_type or _mime_type_for_path(text_uri),
```

- 问题：`from_path` 必算 `sha256_file`（1MiB 分块读全文件）。视频类 artifact 单文件可达 GB 级，逐样本 hash 会显著拖慢 result 收集/enrich 流程，且无开关。
- 影响：大规模 run 的 IO 放大；hash 对"引用完整性"有价值但应可选/可延迟。
- 建议：`from_uri`/`enrich_artifact_ref` 增加 `compute_hash: bool = True` 或大小阈值；或懒计算。

### [EF-07] P3 coerce_artifact_refs 对非 mapping 值报错不可操作
- 位置：`worldfoundry/evaluation/api/artifacts.py:189-195`
- 证据：

```192:195:worldfoundry/evaluation/api/artifacts.py
    artifacts: dict[str, ArtifactRef] = {}
    for name, artifact in (value or {}).items():
        artifacts[str(name)] = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    return artifacts
```

- 问题：runner 返回 `{"video": "/path/to.mp4"}`（字符串而非 dict）时，`ArtifactRef.from_dict(str)` 抛 `AttributeError: 'str' object has no attribute 'get'`，不含 artifact 名与期望格式。
- 建议：类型检查并抛 `TypeError(f"artifact {name!r} must be ArtifactRef or mapping, got str")`；也可考虑善意接受 str 为 `uri`（但需明确 kind）。

---

### 主题 B：顶层 facade（framework.py / runner.py / public.py / utils.py / __init__.py）

### [EF-08] P1 utils.py 在 import 时向 sys.path 插入仓库根目录（全局副作用）
- 位置：`worldfoundry/evaluation/utils.py:208-210`
- 证据：

```208:210:worldfoundry/evaluation/utils.py
# Side effect: benchmarks and model-registry discovery rely on repo-root imports.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- 问题：`import worldfoundry.evaluation.utils`（几乎所有 evaluation 模块都会传递触发，framework.py:9 顶层 import）就会把 `REPO_ROOT` 插到 `sys.path[0]`。`REPO_ROOT` 来自 `project_root()`——向上找 `pyproject.toml`，装到 site-packages 时回退 `package_root().parents[1]`（`worldfoundry/core/io/paths.py:72-78`）。这意味着：a) 安装态下可能把 site-packages 的上级目录插为最高优先级 import 路径；b) 用户工程若有与仓库根下同名的顶层包（如 `test/`、`scripts/`、`configs/`）会被本仓库同名目录遮蔽；c) 副作用发生与否取决于"谁先 import 了 utils"，不可控。
- 影响：经典的 import 污染，评测框架作为库被嵌入宿主进程时尤其危险。
- 建议：删除；需要 repo-root 发现的调用方（benchmark/model discovery）应显式用 `importlib` 按路径加载，或仅在 CLI 入口做一次。

### [EF-09] P2 utils.py 是 5 个旧模块的机械拼接：重复 import、常量与函数交错、命名混杂
- 位置：`worldfoundry/evaluation/utils.py:7-9,67-71,136-149,163-174,213-222`
- 证据：

```67:71:worldfoundry/evaluation/utils.py
# manifest.py
from pathlib import Path
from typing import Any

import yaml
```

```171:174:worldfoundry/evaluation/utils.py
        path = BENCHMARK_TASK_ROOT / f"{benchmark_id}.sample_results{suffix}"
        if path.is_file():
            return path
    return None
MODEL_ZOO_DIR = DATA_ROOT / "models" / "catalog"
```

- 问题：文件内以 `# io.py`、`# manifest.py`、`# resources.py`、`# paths.py`、`# versioning.py` 注释分节，`from pathlib import Path` 重复 import 4 次，`benchmark_task_sample_path` 函数插在两段常量定义中间（`return None` 与 `MODEL_ZOO_DIR` 之间连空行都没有）。此外 `worldfoundry_repository_root = project_root` 等别名赋值（141-144 行）让同一能力有 3 个名字。
- 影响：可读性/可维护性差；分节注释说明本该是 5 个模块，合并后既没有收敛导出面也没有清理。
- 建议：要么拆回子模块（`evaluation/utils/{io,manifest,paths,versioning}.py`），要么彻底去重、把 import 归拢到文件头。

### [EF-10] P2 三套并行的 hash/JSON 规范化实现，"stable_hash" 对 set 实际不稳定
- 位置：`worldfoundry/evaluation/utils.py:307-328` vs `worldfoundry/evaluation/api/json_contract.py:77-94`；`worldfoundry/core/io/serialization.py:80-81`
- 证据：

```322:328:worldfoundry/evaluation/utils.py
def file_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file on disk by reading in chunks."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
```

`api/json_contract.py:88-94` 的 `sha256_file` 逐字节相同实现。更要紧的是规范化差异：`utils.stable_hash` 走 `jsonable`（`core/io/serialization.py:80-81`：`set/frozenset → [jsonable(item) for item in value]`，**不排序**，受 PYTHONHASHSEED 影响），而 `api.json_contract.to_plain`（`json_contract.py:109-110`）对 set 按 `repr` 排序。`build_run_fingerprint`（utils.py:432-452）用前者，声称产出"stable hashes"。
- 问题：同一逻辑对象经两条路径得到不同 hash；且 payload 里只要出现 set，run fingerprint 跨进程就不可复现。
- 影响：破坏 run fingerprint 的可复现承诺（可复现性正是它存在的意义）；三处 file-sha256 实现（还有 `runner.py` 经 orchestration.cache 再导出的 `file_sha256`）漂移风险。
- 建议：全部收敛到 `api/json_contract.py` 一处；`jsonable` 对 set 排序或禁止 set。

### [EF-11] P2 model/benchmark ID 规范化时静默吞掉 catalog 加载错误
- 位置：`worldfoundry/evaluation/framework.py:246-251`（benchmark 侧 261-272 同构）
- 证据：

```246:251:worldfoundry/evaluation/framework.py
    try:
        from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

        return load_model_zoo_registry(root).get(value).model_id
    except (KeyError, TypeError, ValueError):
        return value
```

- 问题：本意是"未知 ID 原样透传，让下游报错"，但 `except (KeyError, TypeError, ValueError)` 同时吞掉了 catalog YAML 损坏、schema 校验失败等真实错误（zoo 加载器大量抛 ValueError）。坏 catalog 的表现变成"别名解析悄悄失效"，下游用原始字符串继续跑，报错点远离病灶。
- 影响：错误诊断成本高；别名行为随 catalog 健康状态静默漂移。
- 建议：只捕获"未知 key"类异常（如 `UnknownRegistryKeyError`）；catalog 加载失败应显式 raise 或至少 warning。

### [EF-12] P2 WorldFoundryRunRequest 76 行 50 个字段的 God-request + 启发式 dispatch
- 位置：`worldfoundry/evaluation/framework.py:23-76`、`423-437`
- 证据：

```23:29:worldfoundry/evaluation/framework.py
@dataclass(frozen=True)
class WorldFoundryRunRequest:
    """One public run request for existing results, one benchmark, or a model x benchmark suite."""

    output_dir: str | Path
    model_ids: Sequence[str] = ()
    benchmark_ids: Sequence[str] = ()
```

```429:437:worldfoundry/evaluation/framework.py
    if (
        _suite_ids(request)
        or len(model_ids) > 1
        or len(benchmark_ids) > 1
        or request.model_workers != 1
        or bool(request.worker_cuda_devices)
    ):
        return True
    return bool(benchmark_ids) and not request.execute
```

- 问题：单个 dataclass 承载三种互斥运行形态（existing-results / 单 benchmark / suite）的全部参数，字段间约束（如 `results_path` 与 `benchmark_ids` 互斥、`task_*` 只在 generation 路径有效）完全靠隐式 dispatch 决定，用户传了无效组合不会报错，只是被静默忽略。`model_workers=2` 会悄悄把单 benchmark 变成 suite 运行（输出目录布局随之改变）。
- 影响：公共 API 语义脆弱；参数被静默丢弃是框架大忌。
- 建议：dispatch 后对"该形态下未消费的字段"做冲突校验并报错/警告；文档化三形态的字段矩阵。

### [EF-13] P3 framework.py 广义 except Exception 掩盖 orchestration 导入失败
- 位置：`worldfoundry/evaluation/framework.py:334-340`
- 证据：

```334:340:worldfoundry/evaluation/framework.py
    try:
        from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import CONTRACT_VALIDATION_ID

        if model_id == CONTRACT_VALIDATION_ID:
            return None
    except Exception:  # noqa: BLE001 - fallback to normal model-zoo resolution.
        pass
```

- 问题：orchestration 模块真实损坏（语法错误、依赖缺失）时该分支静默走"正常 zoo 解析"，之后同一模块的 import 又会在 `_run_single_benchmark` 内部炸——同一原因两种表现。捕 `ImportError` 足矣。

### [EF-14] P3 runner.py 自称"lazy loading"，实际 eager 拉起 9 个 orchestration 模块，且把 cache/hash 工具再导出为公共 runner API
- 位置：`worldfoundry/evaluation/runner.py:1-31,108-122`
- 证据：

```1:8:worldfoundry/evaluation/runner.py
"""In-process evaluation runners.

Provides lazy loading for embodied run symbols to optimize lean CLI entry paths.
"""

# ruff: noqa: F822 - embodied symbols listed in __all__ are provided lazily by __getattr__.

from worldfoundry.evaluation.api import GenerationRequest
```

- 问题：只有 5 个 VLA 符号懒加载（108-122 行），其余 `cache/contract/evaluate/existing_results/fidelity/materialize/model_benchmark/suite/plan/service` 十个模块全部 eager import；`sha256_hex`、`normalize_json`、`canonical_json_dumps` 等通用工具被列进 `__all__`（168-201 行），成为 `evaluation.runner` 的公共表面——与 `api.json_contract` 的同名函数形成第二个官方出口。runner.py 本身 208 行没有任何逻辑，纯 barrel。
- 影响：facade 职责名不符实；公共表面过宽，未来收缩即 breaking change。
- 建议：cache/hash 工具从 `__all__` 移除；docstring 与现实对齐或把 orchestration 也改为懒加载。

### [EF-15] P3 skill 文档与代码漂移：`preparation.py` 已不存在
- 位置：`.cursor/skills/worldfoundry-evaluation-guide/SKILL.md:31`；`worldfoundry/evaluation/`（实际无该文件）
- 证据：`ls worldfoundry/evaluation/preparation.py` → No such file or directory；SKILL.md 架构图仍列 `├── preparation.py   # readiness/prepare reports`，且未列出实际存在的 `public.py`、`reporting/`。
- 影响：按 skill 导航的 agent/新人会找错文件。
- 建议：更新 SKILL.md 架构节。

### [EF-16] P3 版本上下文的 engine_version 是常量字符串而非版本号；git 状态经 lru_cache 可能过期
- 位置：`worldfoundry/evaluation/utils.py:240`、`274-282`
- 证据：

```238:240:worldfoundry/evaluation/utils.py
VERSION_CONTEXT_SCHEMA_VERSION = "worldfoundry-version-context"
RUN_FINGERPRINT_SCHEMA_VERSION = "worldfoundry-run-fingerprint"
EVALUATION_ENGINE_VERSION = "worldfoundry-eval-engine"
```

- 问题：`EVALUATION_ENGINE_VERSION` 无任何数字/日期成分，engine 变更无法从 run manifest 分辨（复现性检查点 10 的弱项）。`_git_metadata_cached` 以 lru_cache 缓存 commit/dirty，长驻进程（服务化、notebook）中跨 run 记录到过期 git 状态。
- 建议：engine version 挂接包版本或显式递增；git 元数据按 run 采集（去缓存或 TTL）。

### 主题 C：models/catalog（schema / manifest / registry / zoo_registry / policy）

总体评价：schema 归一化能力强（大小写、别名、多形态 YAML 兼容），zoo registry 对"多模型共享 HF repo"的 alias 歧义处理（`zoo_registry.py:283-300`）相当细致；policy 层（in-tree target 校验）职责清晰。问题集中在错误上下文、静默去重、缓存失效。

### [EF-17] P1 models→tasks 反向依赖：runners/builtins.py 顶层 import 具身 rollout runner，模型解析路径被拖入 numpy/simulator 子树
- 位置：`worldfoundry/evaluation/models/runners/builtins.py:13-14`；`models/runners/registry.py:21,273`
- 证据：

```13:14:worldfoundry/evaluation/models/runners/builtins.py
from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.evaluation.tasks.embodied.rollout_runner import EmbodiedClosedLoopRunner
```

```273:273:worldfoundry/evaluation/models/runners/registry.py
_BUILTIN_MODEL_RUNNER_REGISTRY = ModelRunnerRegistry()
```

- 问题：架构文档（skill references/architecture.md）规定 models/ 管"模型身份/解析"，tasks/ 管"benchmark 侧"；tasks 侧已有 `tasks/embodied/adapters/runtime_policy_adapters.py:11` 反向 import `models.runtime.profiles`，加上本处 models→tasks，形成双向耦合（当前未成环仅因 rollout_runner 的传递 import 恰好不回到 models）。且 registry.py 在模块级实例化 `_BUILTIN_MODEL_RUNNER_REGISTRY`（273 行），使 import `models.runners.registry` 即触发 `tasks.embodied` 全链：实测 `python -X importtime` 总耗时 ~431ms，其中 `tasks.embodied` 占 ~299ms（含 numpy ~159ms）。所有模型解析入口（resolver.py:18 顶层 import registry）都背上这笔成本，违反 skill 首页"Preserve lightweight imports"守则。
- 影响：分层被破坏；`zoo models` 类轻量 CLI 命令 import 变重；未来 tasks.embodied 增加 GPU/模拟器顶层 import 时会直接炸掉全部模型解析路径（或成环）。
- 建议：builtins 里的 embodied 条目改为惰性（`runner_target="worldfoundry.evaluation.tasks.embodied.rollout_runner:EmbodiedClosedLoopRunner"` 字符串目标，registry 已支持 `create` 时延迟 import），删除顶层 import。

### [EF-18] P2 YAML 解析错误不含文件路径：一个损坏文件让整个 catalog 加载失败且无法定位
- 位置：`worldfoundry/evaluation/utils.py:77-84`；同模式：`models/catalog/manifest.py:355`、`models/pipelines/bindings.py:217`、`models/pipelines/aliases.py:278`
- 证据：

```77:84:worldfoundry/evaluation/utils.py
def load_manifest(path: str | Path) -> Any:
    """Load a checked-in YAML WorldFoundry manifest."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix not in MANIFEST_SUFFIXES:
        raise ValueError(f"unsupported manifest suffix for {resolved}: expected .yaml or .yml")
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))
```

- 问题：`yaml.safe_load(<str>)` 抛出的 ParserError 位置标记是 `in "<unicode string>", line 2 ...`（实测），不含文件名。`load_model_zoo_registry`/`load_model_catalog_manifests`/`load_pipeline_bindings` 都是遍历目录逐文件加载，任何一个文件坏掉，整个 catalog 加载抛错且用户不知道是哪个文件。schema 层的 `ModelZooEntry.from_dict` 校验错误（如 `model_id is required`）同样不带文件路径。
- 影响：坏数据报错质量差（评审标准 4）；目录里上百个 YAML 时排障成本高。
- 建议：`load_manifest` 用 `try/except yaml.YAMLError as exc: raise ValueError(f"{path}: {exc}")`；zoo/binding 加载循环里同样包一层带 path 的上下文。

### [EF-19] P2 跨文件重复 model_id 静默"高优先级/先到先得"，与 alias 冲突硬报错的语义不一致
- 位置：`worldfoundry/evaluation/models/catalog/zoo_registry.py:204-221`；`models/catalog/manifest.py:396-405`
- 证据：

```215:221:worldfoundry/evaluation/models/catalog/zoo_registry.py
            existing = selected.get(entry.model_id)
            if existing is None:
                selected[entry.model_id] = (priority, sequence, entry)
                sequence += 1
            elif priority > existing[0]:
                selected[entry.model_id] = (priority, existing[1], entry)
```

- 问题：同一 `model_id` 出现在多个 YAML 中时按 schema_version 优先级去重、同优先级第一个赢，全程无日志/警告；而两个不同模型 alias 撞车时 `ModelZooRegistry.register` 抛 `DuplicateModelZooKeyError`（zoo_registry.py:309-313）。维护者复制一个 entry 文件改错 id 时，一半情况静默吞并、一半情况报错，行为不可预期。
- 影响：catalog 数据问题被掩盖；条目"谁生效"取决于文件排序这种隐晦规则。
- 建议：同优先级重复至少 `logging.warning`（含两个文件路径）；或提供 strict 模式。

### [EF-20] P2 四处 lru_cache 缓存可变 registry 对象：无文件失效机制、可被调用方污染
- 位置：`worldfoundry/evaluation/models/catalog/zoo_registry.py:426-435`、`models/pipelines/bindings.py:230-240`、`models/pipelines/aliases.py:287-305`、`models/catalog/registry.py:142-150`
- 证据：

```426:435:worldfoundry/evaluation/models/catalog/zoo_registry.py
@lru_cache(maxsize=32)
def _load_model_zoo_registry_cached(resolved_root: str) -> ModelZooRegistry:
    """Load the model zoo registry into memory and cache the registry instance."""
    return ModelZooRegistry.from_directory(Path(resolved_root))


def load_model_zoo_registry(path: str | Path | None = None) -> ModelZooRegistry:
    """Load and cache the ModelZooRegistry from a directory path."""
    root = Path(path) if path is not None else default_model_zoo_dir()
    return _load_model_zoo_registry_cached(str(root.resolve()))
```

- 问题：a) 缓存 key 只有路径，无 mtime——同进程内编辑 YAML 后 `load_model_zoo_registry` 返回旧数据（只有 zoo 提供了手动 `clear_model_zoo_registry_cache`，bindings/aliases/discover_model_registry 连清理入口都没有）；b) 返回的是**可变**实例（`ModelZooRegistry.register`、`PipelineBindingRegistry.register` 都是公开方法），任何调用方对返回值 register 会永久污染全进程共享缓存。`merge_pipeline_binding_plugins`（bindings.py:252）小心地复制了一份，说明作者意识到了，但契约本身没有防护。
- 影响：长驻进程（TUI/服务/测试套件）中数据过期与串扰；测试之间需要手工清缓存。
- 建议：缓存返回冻结视图或在 register 上加"已缓存不可变"保护；缓存 key 纳入目录内容指纹，或统一提供 `clear_*_cache()`。

### [EF-21] P2 状态归一化 fail-open：未知 integration/demo 状态静默归为 planned/pending
- 位置：`worldfoundry/evaluation/models/catalog/schema.py:165-198`
- 证据：

```174:182:worldfoundry/evaluation/models/catalog/schema.py
    if normalized in INTEGRATION_STATUSES:
        return normalized
    if normalized in _INTEGRATED_INTEGRATION_ALIASES:
        return "integrated"
    if normalized.startswith("blocked"):
        return "blocked"
    if normalized in _PENDING_INTEGRATION_ALIASES or normalized.startswith("pending"):
        return "planned"
    return "planned"
```

- 问题：`integration_status: integarted`（拼写错误）会被静默归为 `planned`，模型无声地从可运行列表消失；`_normalize_demo_status` 同样把任何未知值归 `pending`。与之相对，`_validate_status`（schema.py:126-131）对直接构造的 dataclass 是硬校验——同一字段"YAML 进来宽松、代码构造严格"。
- 影响：catalog 数据的拼写/新增状态错误无提示；readiness 报表失真。
- 建议：未知值 warning 一次（含 model_id 与原始值），或收紧为仅接受已登记 alias。

### [EF-22] P3 schema.py 内 official_sources/sources 两段完全复制的解析逻辑
- 位置：`worldfoundry/evaluation/models/catalog/schema.py:337-351` vs `353-367`；`590-600` vs `602-611`
- 证据：

```353:356:worldfoundry/evaluation/models/catalog/schema.py
    sources = entry.get("sources")
    if isinstance(sources, Mapping):
        huggingface = sources.get("huggingface")
        if isinstance(huggingface, (list, tuple)):
```

- 问题：`official_sources` 与 `sources` 的 HF 解析、checkpoint 提取是两段逐行复制的代码（各出现两次，共 4 段），1144 行的 schema.py 中此类"多键位兼容"复制是主要膨胀源。
- 建议：提取 `for key in ("official_sources", "sources")` 循环或公共 helper。

---

### 主题 D：models/pipelines 与 models/runners（绑定解析、加载、执行生命周期）

总体评价：这一层是设计亮点——`resolve_pipeline_route` 的三级解析优先级有清晰 docstring；plugin 发现（discovery.py / plugins.py）对坏插件"警告并跳过"的容错正确；`WorldFoundryPipelineRunner` 有逐样本失败隔离、`cleanup()` 用 `sys.modules.get("torch")` 避免引入 torch、reset 契约用签名检查防误调用。GPU/重依赖 import 确实都在运行路径内（`load_pipeline_from_spec`、`create`）。以下为问题。

### [EF-23] P2 runtime profile 元数据加载 except Exception 返回空 dict：坏 profile 静默降级路由
- 位置：`worldfoundry/evaluation/models/pipelines/bindings.py:164-182`
- 证据：

```169:174:worldfoundry/evaluation/models/pipelines/bindings.py
    try:
        from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile

        profile = load_runtime_profile(profile_id, check_conda_env_exists=False)
    except Exception:
        return {}
```

- 问题：`resolve_pipeline_route` 的第 2 优先级依赖此函数。runtime profile YAML 损坏、profile id 拼错、甚至 profiles 模块本身的代码 bug，全部被吞成"无 profile 元数据"，路由静默落到下一优先级（或返回 None 报"no binding found"——与真实病因无关的错误）。resolver.py:284 也用它组装 config metadata，坏 profile 时诊断字段悄悄缺失。
- 影响：错误报告远离病灶（评审标准 6）；catalog 与 profile 的引用完整性问题不可见。
- 建议：区分"profile 不存在"（可容忍，返回 {}）与"存在但加载失败"（raise 或 warning）；至少 log 一次。

### [EF-24] P2 插件绑定与内置绑定冲突时被静默丢弃（无任何日志）
- 位置：`worldfoundry/evaluation/models/pipelines/bindings.py:246-258`
- 证据：

```252:258:worldfoundry/evaluation/models/pipelines/bindings.py
    merged = PipelineBindingRegistry(registry.list())
    for _, binding in sorted(plugin_bindings.items(), key=lambda item: item[0]):
        try:
            merged.register(binding)
        except ValueError:
            continue
```

- 问题：同文件族的 discovery.py（`LOGGER.warning("Skipping duplicate ...")`，discovery.py:133）和 runners/registry.py（冲突记为 `ModelRunnerRegistryIssue`，registry.py:319-328）都会给出信号，唯独这里静默 `continue`。插件作者的绑定被内置绑定遮蔽时无从得知。
- 建议：与 runner 插件一致，warning 或返回 issues。

### [EF-25] P2 resolver 对"未知模型 ID"抛的异常类型与 docstring 承诺不符，且无候选提示
- 位置：`worldfoundry/evaluation/models/runners/resolver.py:228-237`（docstring 229-230 行 vs 实际 237 行）
- 证据：

```228:237:worldfoundry/evaluation/models/runners/resolver.py
    Raises:
        ModelResolutionError: If the model ID is not found, the variant does
            not exist, or no ``runner_target`` can be resolved.
    ...
    """
    ...
    # ── Load zoo entry and resolve variant ───────────────────────────────
    entry = load_model_zoo_registry(manifest_dir).get(model_id)
```

- 问题：未知 ID 时 `.get()` 抛 `UnknownModelZooKeyError(KeyError)`，而非 docstring 承诺的 `ModelResolutionError(ValueError)`——两者无继承关系，按文档写 `except ModelResolutionError` 的调用方拦不住。错误消息 `unknown model-zoo entry: 'xyz'` 也没有近似候选（did-you-mean），对一个含数百条目的 catalog 不友好。另外 `resolve_world_model_runner`（190-191 行）把底层异常压成 `ModelResolutionError(str(exc))`，`KeyError` 的 str 只剩带引号的 key，上下文丢失。
- 建议：`.get` 包一层转成 `ModelResolutionError` 并附 `difflib.get_close_matches` 候选；`str(exc)` 改为带异常类型的格式化。

### [EF-26] P3 pipeline 加载路径全局修改 torch 后端（TF32/SDPA patch），版本上下文未记录
- 位置：`worldfoundry/evaluation/models/pipelines/loading.py:452-454`；`worldfoundry/core/inference.py:4645-4682`
- 证据：

```452:455:worldfoundry/evaluation/models/pipelines/loading.py
    from worldfoundry.core import install_worldfoundry_inference_infra

    install_worldfoundry_inference_infra()
    pipeline_cls = import_pipeline_target(spec.pipeline_target)
```

- 问题：`install_worldfoundry_inference_infra` 默认开 TF32、设置 matmul 精度并 monkeypatch torch SDPA（进程级全局状态）。这直接影响生成数值，从而影响评测分数的跨环境可比性；但 `build_version_context`（utils.py:397-429）没有记录 attention_backend/matmul_precision/TF32 状态。作为库嵌入宿主进程时还会改写宿主的 torch 全局配置。
- 建议：把 `inference_infra_state()` 快照并入 version context / run manifest；文档标注该副作用。

### [EF-27] P3 运行时视频别名表硬编码在代码里，与 data/ 别名 YAML 双轨
- 位置：`worldfoundry/evaluation/models/pipelines/aliases.py:25-45`
- 证据：

```25:29:worldfoundry/evaluation/models/pipelines/aliases.py
RUNTIME_VIDEO_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("allegro_ti2v", ("allegro",)),
    ("cogvideox_2b_t2v", ("cogvideox-2b-t2v",)),
    ("cogvideox_5b_i2v", ("cogvideox-5b-i2v",)),
    ("cogvideox_5b_t2v", ("cogvideox-5b-t2v", "cogvideox", "cogvideox-5b", "cogvideox-t2v")),
```

- 问题：同文件下半部分是"数据驱动"的 alias YAML 加载体系（`DEFAULT_PIPELINE_ALIASES_ROOT`），上半部分却是硬编码表，两套并存；新增模型需要记得改两处。违背项目"data-backed"原则。
- 建议：把 RUNTIME_VIDEO_ALIASES 迁入 `data/models/bindings/aliases/video.yaml`。

### [EF-28] P3 每次 default_model_runner_registry() 都重扫 entry-points
- 位置：`worldfoundry/evaluation/models/runners/registry.py:276-295`；`plugins.py:63`
- 证据：

```290:295:worldfoundry/evaluation/models/runners/registry.py
def model_runner_registry_snapshot(*, include_plugins: bool = True) -> ModelRunnerRegistry:
    """Return fresh registry snapshot with optional plugin overlay."""
    registry = ModelRunnerRegistry(entries=_BUILTIN_MODEL_RUNNER_REGISTRY.list(), include_builtins=False)
    if not include_plugins:
        return registry
    return _overlay_plugin_runners(registry)[0]
```

- 问题：`resolve_world_model_runner` 每次调用 `default_model_runner_registry()` → 重建 registry + `entry_points(group=...)` 全量扫描 + 可能重复 import 插件模块。"snapshot 不可变、每次新建"的思路是对的（避免了 EF-20 的污染问题），但 discovery 结果本身可以按进程缓存。
- 建议：`discover_model_runner_plugins` 结果加 lru_cache（entry-points 在进程内不变）。

### [EF-29] P3 GenerationResult.timings 契约字段在 pipeline 结果路径从未填充
- 位置：`worldfoundry/evaluation/models/pipelines/results.py:136-144`（对照 `api/generation.py:129`）
- 证据：

```136:144:worldfoundry/evaluation/models/pipelines/results.py
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id=context.model_id,
        artifacts=artifacts,
        status=status,
        error=pipeline_result_error(result, status),
        metadata=pipeline_metadata(result=result, context=context),
    )
```

- 问题：契约专门设计了 `timings` 字段，但内置 runner 把耗时（`result.get("runtime")`）塞进 `metadata["runtime"]`，`timings` 恒为空 dict。契约字段与实际使用脱节，下游按 `timings` 聚合性能的消费者拿不到数据。
- 建议：`timings={"runtime": result.get("runtime")}`（非空时）或删字段。

---

### 主题 E：models/runtime（profiles / environments / assets / validate）

总体评价：environments/assets 两个模块干净对称（frozen dataclass + from_mapping + validate + to_dict）；validate.py 把 `except Exception` 转成结构化 `RuntimeValidationIssue` 是正确的容错模式；`_runtime_profile_env_cache_key` 把 10 个环境变量与 conda env 存在性纳入缓存 key（profiles.py:732-747），是全 models/ 里唯一认真做了缓存失效的地方。核心问题是 torch 顶层泄漏和大块死代码。

### [EF-30] P1 profiles.py（纯 YAML 元数据加载器）顶层 import BaseSynthesis → torch：读个配置要付 1.5s + GPU 栈
- 位置：`worldfoundry/evaluation/models/runtime/profiles.py:34`；`worldfoundry/synthesis/base_synthesis.py:12-13`
- 证据：

```34:34:worldfoundry/evaluation/models/runtime/profiles.py
from worldfoundry.synthesis.base_synthesis import BaseSynthesis
```

```12:13:worldfoundry/synthesis/base_synthesis.py
try:
    import torch
```

- 问题：`base_synthesis` 的 try/except 只保护"torch 未安装"，装了 torch 就全量加载。实测 `import worldfoundry.evaluation.models.runtime.profiles` 后 `torch in sys.modules == True`，耗时 1.52s。而 profiles.py 的主要职责是加载 YAML 元数据——`runtime_profile_execution_metadata`（bindings.py:170，路由第 2 优先级）、`validate.py`（readiness 校验）、`runtime/__init__.py`（eager barrel）都会触发。这直接违反 skill 首条设计原则"GPU/API imports live inside runtime execution paths"。更糟的组合拳：torch import 若在坏 CUDA 环境下抛错，bindings.py:173 的 `except Exception` 把它吞成"无 profile 元数据"（见 EF-23），路由静默降级。附带的分层问题：唯一消费 `BaseSynthesis` 的是 `RuntimeProfileSynthesis`（942 行起），而 synthesis/ 侧 100+ 个模型模块又反向顶层 import `evaluation.models.runtime.profiles`（如 `synthesis/action_generation/base_action_synthesis.py:11`），evaluation↔synthesis 双向耦合。
- 影响：所有"读 profile 元数据"的轻量路径（含 CLI readiness、pipeline 路由）都背上 torch 导入成本与失败面；违反项目核心 import 卫生守则。
- 建议：把 `RuntimeProfileSynthesis` 拆到独立模块（或 `BaseSynthesis` import 移入类方法内），让 profiles.py 回归 stdlib+yaml。

### [EF-31] P2 RuntimeProfileSynthesis.predict 从不执行命令：timeout_seconds/_runtime_env/project_root 全是死代码，返回语义误导
- 位置：`worldfoundry/evaluation/models/runtime/profiles.py:39-46,1136-1147,1161,1211-1223`
- 证据：

```1161:1161:worldfoundry/evaluation/models/runtime/profiles.py
        timeout_seconds: int = 21600,
```

```1211:1221:worldfoundry/evaluation/models/runtime/profiles.py
        return {
            "status": "blocked",
            ...
            "runtime": "worldfoundry.runtime_profile.vendor_blocked",
```

- 问题：`predict()` 组装完 command 后只写 plan JSON，永远返回 "prepared"（plan_only）或 "blocked"——整个文件没有任何 subprocess 调用。于是：`timeout_seconds=21600` 参数从未被引用（rg 证实）；`_runtime_env`（为子进程准备 PATH/LD_LIBRARY_PATH 的方法）从未被调用；模块头的 `project_root()`（profiles.py:39-46）也无调用点，且与 `utils.project_root` 是第三份不同 fallback 的实现（`current.parents[5]`）。类 docstring 说 "planning or executing model inference"，实际只有 planning。调用方看到签名里的 timeout 会以为存在执行路径。
- 影响：死代码误导维护者与调用方；`_context` 里 40+ 个模板变量默认值（`unnorm_key="bridge_orig"`、`class_id=207`、`master_port=25000`，1102-1117 行）是具体模型的参数泄漏进通用层，无执行路径消费它们。
- 建议：删除 `timeout_seconds`/`_runtime_env`/`project_root`；docstring 明示"仅 plan/blocked"；模型特定默认值移入各 profile YAML。

### [EF-32] P3 runtime 三文件各复制一份 _tuple_of_str/_schema_version/_manifest_paths/_iter_*_mappings
- 位置：`worldfoundry/evaluation/models/runtime/profiles.py:127-135,407-411,440-460`；`assets.py:41-49,288-318`；`environments.py:20-28,269-299`
- 证据：

```269:273:worldfoundry/evaluation/models/runtime/environments.py
def _schema_version(value: Any) -> int | None:
    """Coerce any schema version value to int or return ``None``."""
    if value in (None, ""):
        return None
    return int(value)
```

- 问题：同一签名的 4 个 helper 在 3 个文件里逐字重复（`_schema_version` 三份完全一致；`_manifest_paths`/`_iter_*_mappings` 仅默认 root 和 key 名不同）。schema 版本校验规则（"只支持 2"）也在 3 个 validate 里各写一遍，将来升 schema v3 要改 6 处。
- 建议：抽 `runtime/_common.py`；`_iter_*_mappings` 参数化 collection key。

### [EF-33] P3 validate_runtime_registry 对每个 profile 全量重载 environments/assets：O(N²) 文件 IO
- 位置：`worldfoundry/evaluation/models/runtime/validate.py:100-148,244-245`；`environments.py:320-351`
- 证据：

```244:245:worldfoundry/evaluation/models/runtime/validate.py
    for profile in load_runtime_profile_manifests(profile_root or DEFAULT_RUNTIME_PROFILES_ROOT):
        issues.extend(validate_runtime_profile_references(profile))
```

- 问题：`validate_runtime_profile_references` 内部经 `load_runtime_environment_profile_by_id` → `load_runtime_environment_profiles`（无缓存，rglob 全目录 + 逐文件 yaml.safe_load）。N 个 profile 各触发一次全量加载，catalog 变大后 readiness 校验时间平方增长。另外 `RuntimeEnvironmentProfile.to_dict(check_exists=False)` 把 `exists` 硬编码为 `False`（environments.py:245），下游无法区分"未检查"与"确认不存在"，语义上应为 `None`。
- 建议：在 `validate_runtime_registry` 里加载一次 environments/assets 传入；`exists` 用三态。

---

### 主题 F：reporting/（scorecard / run_manifest / run_index / run_comparison / validation / browser）

总体评价：这是全包质量最高的子层。`validation.py` 的"累积 errors/warnings 而非抛错 + NaN/Infinity 检测 + artifact 路径存在性检查 + strict 模式"是教科书式契约校验；`run_index` 有 invalid-row 容错、重复 run_id 检测、label 去重；`comparison_identity` 的 strict-fields + comparison_key + missing_required/recommended 设计能支撑严肃的跨 run 可比性判定；`run_browser` 的 HTML/JSON 注入转义（`_json_for_script`）正确。以下问题多为局部缺陷。

### [EF-34] P2 write_json 的"原子写"用固定临时文件名：并发写同一 output dir 会互相踩踏
- 位置：`worldfoundry/core/io/serialization.py:149-157`（对照同文件 `write_jsonl` 264 行）
- 证据：

```149:155:worldfoundry/core/io/serialization.py
def _atomic_write_text(path: Path, text: str, *, atomic: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
```

- 问题：所有 scorecard/summary/run_manifest/index JSON 都经 `write_json` → `_atomic_write_text`，临时名固定为 `.{name}.tmp`。两个进程同时刷新同一 `index.json`（或用户并发对同一 output_dir 跑两次）时：A 写 tmp → B 覆写 tmp → A rename，落盘的是 B 的半成品或混合内容；B 的 rename 还可能因文件已被移走而 `FileNotFoundError`。同文件的 `write_jsonl`（264 行）用 `uuid4().hex` 临时名 + `open("x")` + 异常清理，是正确写法——同一模块内两套原子性标准。框架层面也没有 run 目录锁：两个 run 写同一 output_dir 时 scorecard/summary 互相覆盖，无任何警告（评审标准 5）。
- 影响：并发 index 刷新/重复提交场景下报告文件损坏；损坏发生在"最终产物"层，事后难以归因。
- 建议：`_atomic_write_text` 改用 uuid 临时名（与 write_jsonl 对齐）；可选：run 目录内放 lockfile 或对已存在 run_manifest 的目录发警告。

### [EF-35] P2 redact_secrets 子串匹配 "token"：max_new_tokens/tokenizer 等生成参数被误脱敏，破坏 manifest 的复现价值
- 位置：`worldfoundry/evaluation/reporting/run_manifest.py:28-48`
- 证据：

```45:48:worldfoundry/evaluation/reporting/run_manifest.py
def _is_sensitive_key(key: str) -> bool:
    """Return whether *key* looks like a secret field that should be redacted."""
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)
```

- 问题：`_SENSITIVE_KEY_PARTS` 含 `"token"`/`"secret"`/`"credential"` 且用 **子串** 匹配。实测 `redact_secrets({'max_new_tokens': 512, 'tokenizer': 'gpt2', 'num_tokens': 8})` → 三个键全部 `<redacted>`。run manifest 的 `config` 恰恰要记录生成参数以支撑复现（模块 docstring："captures the full reproducibility context"），LLM/VLM 类 runner 的 `max_tokens`、`tokenizer` 配置会从 manifest 里消失。
- 影响：复现所需的关键配置被静默抹掉；且用户无从发现（值只剩 `<redacted>`）。
- 建议：改精确词段匹配（split by `_` 后整词比较，或仅匹配 `api_key`/`*_token` 结尾等模式），并为 `tokenizer`/`*_tokens` 加白名单；补单测。

### [EF-36] P3 scorecard/summary/index 载荷冗余：同一数据落盘 2-3 份、artifacts 记录绝对路径
- 位置：`worldfoundry/evaluation/reporting/scorecard.py:223-239,261`；`run_index.py:382-383`；`run_comparison.py:290-291`
- 证据：

```226:236:worldfoundry/evaluation/reporting/scorecard.py
            "per_metric": per_metric,
            "summary": dict(metrics_summary_payload),
        },
        "evaluation": {
            ...
            "summary": dict(metrics_summary_payload),
```

- 问题：`metrics_summary` 在 scorecard 里完整出现两次（`metrics.summary` 与 `evaluation.summary`），leaderboard 出现两次；index/comparison 的 `runs` 与 `rows` 是同一列表落两个键；eligibility 的 `reasons`/`blocking_reasons`、`leaderboard_valid`/`leaderboard_eligible` 互为别名。另外 `write_scorecard` 把 `artifacts.scorecard` 记为 `resolve()` 后的**绝对路径**（261 行，run_manifest/环境文件同样），run 目录整体挪动/归档后所有内部引用失效，而 `samples_ref` 用的却是相对路径——同一文档两种约定。
- 影响：大规模 run 下载荷体积翻倍；消费者不知该读哪个键；目录不可搬迁。
- 建议：收敛为单一键（保留别名一个版本后删除）；artifacts 内部引用改相对路径。

### [EF-37] P3 run_index/run_comparison 跨模块 import run_report 的 6 个下划线私有函数
- 位置：`worldfoundry/evaluation/reporting/run_index.py:27-34`；`run_comparison.py:21`
- 证据：

```27:34:worldfoundry/evaluation/reporting/run_index.py
from .run_report import (
    _dedupe_labels,
    _mapping,
    _normalise_roots,
    _row_from_summary,
    _run_summary_candidate,
    load_run_summary,
)
```

- 问题：`_row_from_summary`/`_dedupe_labels` 实际是三个模块共享的核心行构造逻辑，却以私有命名挂在 run_report 下；`_mapping = mapping_or_empty` 之类别名再包一层。重构 run_report 时无法从命名判断哪些"私有"函数有外部消费者。
- 建议：升级为公共命名放入 `reporting/_common.py` 或 utils。

### [EF-38] P3 _demo.py 在 import 时于包目录内创建 _demo_out 目录
- 位置：`worldfoundry/evaluation/reporting/_demo.py:30-31`
- 证据：

```30:31:worldfoundry/evaluation/reporting/_demo.py
DEMO_DIR = Path(__file__).parent / "_demo_out"
DEMO_DIR.mkdir(exist_ok=True)
```

- 问题：`mkdir` 在模块顶层执行——任何 import（文档生成器、`pkgutil.walk_packages`、pytest 收集）都会在 site-packages 里写目录；只读安装环境下直接 `PermissionError`。demo 脚本本身也随包发布。
- 建议：`mkdir` 移入 `main()`；demo 移到 `examples/` 或 docs。

---

## 汇总

### 按严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0（损坏/危险） | 0 | — |
| P1（严重设计缺陷） | 4 | EF-01, EF-08, EF-17, EF-30 |
| P2（应修复） | 17 | EF-02, EF-03, EF-04, EF-09, EF-10, EF-11, EF-12, EF-18, EF-19, EF-20, EF-21, EF-23, EF-24, EF-25, EF-31, EF-34, EF-35 |
| P3（改进建议） | 17 | EF-05, EF-06, EF-07, EF-13, EF-14, EF-15, EF-16, EF-22, EF-26, EF-27, EF-28, EF-29, EF-32, EF-33, EF-36, EF-37, EF-38 |
| 合计 | 38 | |

### 按主题分布

| 主题 | 发现数 | 重点 |
| --- | --- | --- |
| A. api/ 契约层 | 7（1×P1, 3×P2, 3×P3） | 注册错误路径本身有 bug；双套 registry 实现 |
| B. 顶层 facade | 9（1×P1, 4×P2, 4×P3） | sys.path 污染；utils 大杂烩；hash 三套并存 |
| C. models/catalog | 6（1×P1, 4×P2, 1×P3） | models→tasks 反向依赖；错误缺文件上下文；缓存无失效 |
| D. models/pipelines+runners | 7（3×P2, 4×P3） | 吞错降级路由；异常类型与文档不符 |
| E. models/runtime | 4（1×P1, 1×P2, 2×P3） | profiles 顶层拉 torch；大块死代码 |
| F. reporting/ | 5（2×P2, 3×P3） | 原子写竞态；redact 误伤复现参数 |

### 值得肯定的设计

- `api/` 契约层确实 stdlib-only（rg 验证零第三方 import），全部 dataclass 带 `schema_version` 并在构造期校验。
- `reporting/validation.py` 的累积式校验（errors/warnings + NaN 检测 + artifact 存在性 + strict 模式）与 `comparison_identity` 的可比性判定是高水准设计。
- `_runtime_profile_env_cache_key`（profiles.py:732）把环境变量与 conda env 存在性纳入缓存 key，是全包唯一认真处理缓存失效的地方。
- 插件体系（runner plugins / pipeline discovery）对坏插件"警告并跳过 + 结构化 issue"的容错方向正确。
- `WorldFoundryPipelineRunner` 的逐样本失败隔离与 `sys.modules.get("torch")` 式轻量清理。

### Top 5 最重要问题

1. **[EF-30] profiles.py 顶层 import torch**：纯 YAML 元数据加载路径（pipeline 路由、readiness 校验）被迫加载完整 GPU 栈（实测 1.52s），直接违反项目第一设计原则；与 EF-23 的 `except Exception` 组合后，坏 torch 环境会静默降级路由而非报错。
2. **[EF-17] models→tasks 反向依赖 + 模块级 registry 实例化**：`runners/builtins.py` 顶层 import 具身 rollout runner，令一切模型解析入口背上 ~300ms 的 tasks.embodied/numpy 导入，分层契约（models 不依赖 tasks）已破。
3. **[EF-08] utils.py import 时改写 sys.path[0]**：几乎所有 evaluation import 都会触发，作为库嵌入宿主进程时可遮蔽宿主同名包，安装态下行为不可预期。
4. **[EF-01] MetricSpec 重复注册抛 AttributeError**：冲突检测路径访问不存在的 `.name` 字段，错误处理路径本身是坏的且从未被测试覆盖（实测复现）。
5. **[EF-04] 缺失 status 默认按"成功"处理**：外部 runner 写出的残缺结果会被当作可打分样本计入指标，fail-open 方向威胁 leaderboard 可信度（配合 EF-35 的 redact 误伤、EF-34 的并发写竞态，报告层的"可信/可复现"承诺存在多处缺口）。

### 修复优先级建议

- **立即**（≤1 天）：EF-01（一行改 `_key_fn`）、EF-13（收窄为 ImportError）、EF-23/EF-24 的 warning 补齐、EF-35（redact 匹配收紧）。
- **短期**（1 周内）：EF-30（拆 RuntimeProfileSynthesis）、EF-17（builtins 改字符串 runner_target）、EF-08（删 sys.path 副作用并修显式加载点）、EF-34（uuid 临时名）、EF-18（YAML 报错带路径）。
- **规划**（一个迭代）：EF-12（RunRequest 拆形态/加冲突校验）、EF-09/EF-10（utils 拆分与 hash 收敛）、EF-03（registry 合并）、EF-20（缓存失效策略统一）。
