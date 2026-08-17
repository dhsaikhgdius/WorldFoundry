# core 基础层评审（configuration/registry/model_loading/checkpoint/io/utils）

> 状态：已完成（48 条发现：P0×1 / P1×9 / P2×26 / P3×12）
> 评审人：infra 代码评审 agent
> 日期：2026-08-14

## 评审范围与方法

### 范围

`worldfoundry/core/` 的"基础设施"半区，共约 88 个文件、2.2 万行：

| 子目录/文件 | 行数 | 角色 |
|---|---|---|
| `configuration/`（cosmos_config, flags, hydra, lazy_config/, model_config） | ~1100 | 配置体系（多套并存） |
| `registry.py` | 170 | 全局注册表（全框架依赖） |
| `model_loading/`（13 文件） | ~1900 | 模型工厂/加载/LoRA |
| `checkpoint/`（8 文件） | ~1750 | 权重加载/DCP/safetensors |
| `io/`（30 文件） | ~7600 | 文件/视频/S3/HF/序列化/路径 |
| `structures/` | ~730 | tree/shape/validator |
| `utils/`（17 文件） | ~5300 | 杂项工具 |
| `logging_setup.py`, `log_filters.py` | ~850 | 日志 |
| `prompting.py`, `time.py`, `process.py`, `runtime_cache.py` | ~430 | 杂项 |
| `safety/` | ~160 | 安全护栏 |
| `visualization/` | ~360 | 动作叠加可视化 |
| `__init__.py` | 296 | 包入口（lazy attr） |

### 方法

1. `rg` 全范围扫描反模式：`torch.load`、`pickle.load`、`shell=True`、bare `except:`、`os.environ` 写、模块级副作用、上层反向依赖（evaluation/pipelines/training/studio）、硬编码路径。
2. `Read` 精读优先级文件：registry → configuration → model_loading → checkpoint → io，其余按依赖关系精读。
3. 每条发现附 `path:line` + 代码摘录。

### 严重度定义

- **P0**：损坏/危险（bug、数据损坏、安全漏洞、崩溃路径）
- **P1**：严重设计缺陷（可维护性/扩展性/正确性）
- **P2**：应修复（一致性/健壮性/性能）
- **P3**：改进建议

## 发现（按主题分组）

### 主题 A：分层与依赖方向

### [CF-1] P1 core→runtime 反向依赖形成包级循环（video_tiling 顶层 import runtime）

- 位置：`worldfoundry/core/io/video_tiling.py:22`
- 证据：

```python
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_callable_cached
```

  而 runtime 层反过来大量依赖 core（`worldfoundry/runtime/jobs.py:18` `from worldfoundry.core.time import utc_now_iso`；`worldfoundry/runtime/env.py:22` `from worldfoundry.core.io.paths import ...`；`runtime/compile_cache.py` 自身 import `worldfoundry.runtime.env`→`core.io.paths`）。
- 问题：core 作为最底层不应 import 上层 runtime。`video_tiling` 是模块顶层 import，形成 `core.io.video_tiling → runtime.compile_cache → runtime.env → core.io.paths` 的跨层环。当前因 core.io 的 lazy `__getattr__` 和 import 顺序侥幸可用，但任何一侧改成急切导入就会触发 ImportError（partially initialized module）。
- 影响：分层被破坏后无法单独发布/测试 core；import 顺序敏感，重构极易引入难排查的循环导入崩溃。
- 建议：把 `CompilePolicy`/`compile_callable_cached` 下沉到 core（如 `core/acceleration/`），或让 video_tiling 通过依赖注入接收 compile 函数；同类的 `utils/inference_runtime.py:181`（函数内 lazy import runtime）也应一并处理。

### [CF-2] P2 utils/inference_runtime 在函数内 import 上层 runtime（分层反转的第二处）

- 位置：`worldfoundry/core/utils/inference_runtime.py:181`
- 证据：

```python
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_callable_cached
```

- 问题：与 CF-1 同源，虽为惰性导入不会在 import 期爆炸，但 core 的执行路径仍依赖上层包的存在。
- 影响：core 单元测试必须携带 runtime 包；依赖方向文档化的"core 最底层"承诺失效。
- 建议：与 CF-1 一起下沉 compile_cache 或引入回调注入。

### 主题 B：configuration/ 配置体系

### [CF-3] P1 lazy_config 导入时全局 monkey-patch `OmegaConf.to_object`

- 位置：`worldfoundry/core/configuration/lazy_config/__init__.py:12`
- 证据：

```python
OmegaConf.to_object = to_object
```

- 问题：import `worldfoundry.core.configuration`（或其任何 re-export）即替换 omegaconf 库的全局方法。被替换后的 `to_object` 对含 `_target_` 的 DictConfig 直接原样返回（`omegaconf_patch.py:56-57`），改变了第三方库对所有调用方的语义——包括 vendored 模型代码、hydra 内部和用户自己的 omegaconf 用法。
- 影响：全进程行为漂移：任何依赖 `OmegaConf.to_object` 把含 `_target_` 的配置转成 plain dict 的代码会静默拿到 DictConfig；难以定位（补丁发生在离现场很远的 import 处）。
- 建议：不打全局补丁。在自己的 instantiate/序列化路径里调用本地 `to_object`；如必须兼容旧行为，至少提供开关并在文档中显著标注。

### [CF-4] P1 `LazyConfig.load` 用 `yaml.unsafe_load` 且 Python 配置直接 exec，配置文件即代码执行

- 位置：`worldfoundry/core/configuration/lazy_config/config.py:128,131`
- 证据：

```python
exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
...
result = OmegaConf.create(yaml.unsafe_load(path.read_text(encoding="utf-8")), flags={"allow_objects": True})
```

- 问题：`.py` 配置 exec 是 detectron2 LazyConfig 的既有设计（可接受，需文档警示）；但 YAML 分支用 `yaml.unsafe_load`，可实例化任意 Python 对象——YAML 通常被当作"数据"分发（从 checkpoint 目录、HF 仓库、评测资产目录加载），信任边界与 .py 不同。
- 影响：加载不可信 YAML（如下载的模型附带 config）即任意代码执行；安全审计通常将此列为高危。
- 建议：默认 `yaml.safe_load`；确需还原 python object 的场景走显式 `allow_objects=True` 的专用入口并要求调用方声明信任。保存侧 `save_yaml` 已经做了 `_plain_config` 降级，说明常规链路并不需要 unsafe load。

### [CF-5] P2 `_patch_relative_imports` 运行期替换 `builtins.__import__`，非线程安全

- 位置：`worldfoundry/core/configuration/lazy_config/config.py:48-79`
- 证据：

```python
builtins.__import__ = patched_import
try:
    yield
finally:
    builtins.__import__ = original_import
```

- 问题：加载 .py 配置期间全进程的 `__import__` 被替换。若其它线程（数据加载 worker、后台日志、并发评测任务）同时 import，将经过补丁函数；两个线程嵌套使用时恢复顺序也可能交错，留下污染的 `__import__`。
- 影响：偶发、难复现的并发 import 异常。
- 建议：优先用 `importlib` 自定义 Loader/Finder（sys.meta_path 局部注册 + 唯一前缀过滤）实现相对导入，避免动 builtins；至少加锁并文档化"配置加载须在主线程"。

### [CF-6] P2 hydra.py 依赖全局 ConfigStore/GlobalHydra 状态且用 assert 做输入校验

- 位置：`worldfoundry/core/configuration/hydra.py:93-101,123-132`
- 证据：

```python
cs = ConfigStore.instance()
cs.store(name="config", node=config_omegaconf)
if not GlobalHydra().is_initialized():
    with initialize(version_base=None):
        config_omegaconf = compose(config_name="config", overrides=overrides)
...
assert ref_fields == keys or keys.issubset(ref_fields), (...)
```

- 问题：(1) 每次 `override()` 向进程级 ConfigStore 写入同名 "config" 节点，并依赖 GlobalHydra 是否初始化走不同路径——并发或多次调用互相覆盖；(2) 字段校验用 `assert`，`python -O` 下直接失效；(3) `get_config_module`（150-158 行）对非 .py 输入只 `log.error` 不 raise，继续往下算出错误模块名。
- 影响：多任务/嵌入式使用（studio、评测并发）时配置互相污染；-O 运行时校验静默消失。
- 建议：assert 改 `raise ValueError`；ConfigStore 节点名带 uuid 或用局部 Hydra 实例；`get_config_module` 校验失败应立即 raise。

### [CF-7] P2 flags.py import 时读环境变量、env 前缀不统一、TRAINING/SMOKE 死常量

- 位置：`worldfoundry/core/configuration/flags.py:27-32`
- 证据：

```python
INTERNAL = _env_bool("COSMOS_INTERNAL")
VALIDATION = _env_bool("COSMOS_VALIDATION")
VERBOSE = _env_bool("COSMOS_VERBOSE")
EXPERIMENTAL_CHECKPOINTS = _env_bool("COSMOS_EXPERIMENTAL_CHECKPOINTS")
SMOKE = _env_bool("COSMOS_SMOKE")
TRAINING = False
```

- 问题：(1) import 时固化 env——之后 `os.environ` 的修改（测试 monkeypatch、CLI 内部 set）不生效，且 import 顺序决定行为；(2) 框架其余部分用 `WORLDFOUNDRY_*` 前缀（`logging_setup.py:62-64`），这里沿用 `COSMOS_*`，用户面难以发现；(3) 全仓库检索 `TRAINING`/`SMOKE` 无消费方，属死代码。
- 影响：可测试性差（需 reload 模块才能改 flag）；配置面碎片化。
- 建议：改为函数或缓存属性（首次访问时读 env），统一 `WORLDFOUNDRY_` 前缀并保留旧名 fallback；删除死常量。

### [CF-8] P2 cosmos_config.py 顶层 try-import megatron.core，配置模块可能拖入整个训练栈

- 位置：`worldfoundry/core/configuration/cosmos_config.py:11-19`
- 证据：

```python
try:
    from megatron.core import ModelParallelConfig as _ModelParallelConfig
except ImportError:
    @attrs.define(slots=False)
    class _ModelParallelConfig:
        context_parallel_size: int = 1
```

- 问题：`configuration/__init__.py:3` 急切导入 cosmos_config，因此 import `worldfoundry.core.configuration` 时，若环境装有 megatron，会连带 import megatron.core（内部 import torch 等重依赖）。与"配置/目录查询 stdlib-first、重依赖延迟到执行路径"的项目自我要求冲突。fallback 只有 1 个字段，与真实 `ModelParallelConfig` 的 API 面差异巨大，两种环境下行为漂移。
- 影响：CLI 冷启动被拖慢；同一配置对象在不同环境下字段集不同，序列化/校验结果不可复现。
- 建议：配置层定义自有的 `ModelParallelConfig`（字段显式），在真正构建 megatron 运行时的地方做转换；或延迟到 `Config.model_parallel` 首次实例化时再解析。

### [CF-9] P3 make_freezable 的 freeze 语义不完整、EMAConfig 漏装饰

- 位置：`worldfoundry/core/configuration/cosmos_config.py:29-49,95-101`
- 证据：

```python
def freeze(self) -> None:
    for value in attrs.asdict(self, recurse=False).values():
        if _is_attrs_instance(value) and hasattr(value, "freeze"):
            value.freeze()
    self._is_frozen = True
```

- 问题：(1) 冻结只拦截 `__setattr__`，容器字段（list/dict）内容仍可变；(2) `EMAConfig` 未加 `@make_freezable`，嵌套冻结时被跳过，与同文件其它 config 不一致；(3) `make_freezable` 重复装饰会叠加多层 `setattr_override`。
- 影响：freeze 提供的"不可变"保证是弱承诺，易被误信。
- 建议：文档标注浅冻结语义；补齐 EMAConfig；装饰器加幂等检查。

### [CF-10] P3 lazy_call.py `_CONVERT_TARGET_TO_STRING` 模块级 ClassVar 误用 + get_default_params 分支死代码

- 位置：`worldfoundry/core/configuration/lazy_config/lazy_call.py:29-43`
- 证据：

```python
def get_default_params(cls_or_func):
    if callable(cls_or_func):
        signature = inspect.signature(cls_or_func)
    else:
        signature = inspect.signature(cls_or_func.__init__)
...
_CONVERT_TARGET_TO_STRING: ClassVar[bool] = False
```

- 问题：(1) `ClassVar` 注解只在类体内有意义，模块级使用是类型注解误用；(2) 类对象本身 `callable()` 为 True，`else` 分支只对字符串 target 生效——此时取的是 `str.__init__` 签名，返回空 dict，即字符串 target 的默认参数从不被复制，行为与 docstring 暗示不符；(3) 该全局量"Used by tests"，生产代码路径受测试开关影响。
- 影响：轻微行为不一致与可读性问题。
- 建议：字符串 target 先 `locate()` 再取签名，或明确不支持；删除 ClassVar 注解。

### [CF-11] P2 configuration 多套体系并存、职责边界未文档化

- 位置：`worldfoundry/core/configuration/`（整体）
- 证据：同目录并存 5 套机制——`cosmos_config.py`（attrs+freeze，Cosmos 推理）、`flags.py`（env 开关）、`hydra.py`（Hydra compose/override）、`lazy_config/`（detectron2 LazyCall/LazyConfig + OmegaConf）、`model_config.py`（dataclass + `__getattr__` 委托）。`configuration/__init__.py` 把 4 套一起 re-export，但没有任何模块级文档说明"什么场景用哪套"。
- 问题：`ModelConfig`（model_config.py:36）与 model_loading/config.py 的 `ModelConfig`（见 CF-17）重名不同物；`Config.validate` 只检查 model 非 None；hydra.py 与 lazy_config 各有一套 instantiate 语义（hydra compose vs `_target_` locate）。新增配置时无从判断落点。
- 影响：认知负担与误用风险高；两个 `ModelConfig` 在 IDE 补全/grep 时极易混淆。
- 建议：在 `configuration/__init__.py` 写明每套的适用场景与来源（vendored detectron2/cosmos）；重命名其一（如 `DiffusionModelConfig` vs `LoaderModelConfig`）；长期收敛到 lazy_config + attrs 两套。

### 主题 C：registry 与全局状态

### [CF-12] P2 TypedRegistry 无并发保护，check-then-act 竞态

- 位置：`worldfoundry/core/registry.py:68-101`
- 证据：

```python
normalized = normalize_registry_key(key)
if normalized in self._items:
    raise DuplicateRegistryKeyError(f"duplicate registry key: {key!r}")
...
self._items[normalized] = item
self._aliases.update(alias_lookup)
```

- 问题：`register` 的查重与写入之间无锁。注册通常发生在 import 期（受 import lock 保护），但该类型也被运行期动态注册使用（如 evaluation 运行器按需注册），多线程下两个线程可同时通过查重并各自写入。
- 影响：并发场景重复键静默互相覆盖（后写胜），破坏"Duplicate 必须报错"的契约。
- 建议：加 `threading.Lock`（注册频率低，成本可忽略）；顺带在错误消息里列出已有 keys 的近似匹配（现在只有 `unknown registry key: 'x'`，可附 `did you mean ...`，提升可操作性）。
- 备注（正面）：注册失败时不留半写状态（alias_lookup 全量构建后才提交）、确定性排序枚举、大小写规范化——这些设计是对的。

### 主题 D：日志基础设施

### [CF-13] P3 log_filters 在 import 时安装全局 logging filter（已文档化的例外）

- 位置：`worldfoundry/core/log_filters.py:82`；`worldfoundry/core/__init__.py:12`
- 证据：

```python
install_inductor_autotune_demote()
```

```python
from worldfoundry.core import log_filters as _log_filters  # noqa: F401
```

- 问题：import `worldfoundry.core` 即修改 `torch._inductor.select_algorithm` logger 的 filter 链。副作用是幂等且良性的（把误报 ERROR 降级 WARNING），且 docstring 明示；但它建立了"import 有副作用"的先例，且改变了第三方 logger 的可观测行为（监控按 ERROR 告警的系统将看不到这类记录）。
- 影响：低；主要是行为透明度。
- 建议：保留亦可；更干净的做法是并入 `configure_logging()` 的 opt-in 路径。

### [CF-14] P3 logging_setup 全局 `_CONFIGURED` 无锁；`_parse_bytes` 边角解析错误

- 位置：`worldfoundry/core/logging_setup.py:721-724,401-419`
- 证据：

```python
if _CONFIGURED and not force:
    return
...
base = float(num) if num else float(default)
factor = _BYTES_UNITS.get(unit, 1)
return int(base * factor)
```

- 问题：(1) `configure_logging` 的幂等检查非原子，两线程同时首调会各配一遍（loguru `remove()`+`add()` 交错可产生重复 sink）；(2) `_parse_bytes("mb")`（无数字）会把 default 字节数再乘 1024²。均为边角。
- 影响：低。
- 建议：加模块锁；`num` 为空时直接返回 default。
- 备注（正面）：redaction（`_SENSITIVE_*` 正则 + `_json_safe`）、rank 后缀文件、loguru/stdlib 双路径统一 schema 的设计质量高于同类项目平均水平。

### 主题 E：model_loading/

### [CF-15] P1 `ModelConfig.download_if_necessary` 的 `dist.barrier(device_ids=[dist.get_rank()])` 在多机下传错设备号

- 位置：`worldfoundry/core/model_loading/config.py:150-153`
- 证据：

```python
if use_usp:
    import torch.distributed as dist

    dist.barrier(device_ids=[dist.get_rank()])
```

- 问题：`device_ids` 期望**本地 GPU 序号**（local rank / device index），这里传了**全局 rank**。单机 8 卡时两者恰好相等掩盖了问题；两机 16 卡时 rank 8-15 会请求不存在的 CUDA device 8-15。
- 影响：多机 USP 推理路径崩溃（invalid device ordinal）或 barrier 绑错设备导致 NCCL hang。
- 建议：用 `torch.cuda.current_device()` 或 `LOCAL_RANK` 环境变量；或直接省略 `device_ids`（NCCL 会用当前设备）。

### [CF-16] P2 `WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD` 只认 "true"/"false"，其它取值静默落入 None

- 位置：`worldfoundry/core/model_loading/config.py:83-94`
- 证据：

```python
if os.environ["WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD"].lower() == "true":
    return True
elif os.environ["WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD"].lower() == "false":
    return False
```

- 问题：设 `=1`/`yes`（flags.py 的 `_env_bool` 均接受）时两个分支都不命中，函数隐式返回 `None`，`require_downloading` 里 `not None` → 继续联网下载。同一框架内两套 env 布尔解析规则不一致，且无告警。
- 影响：离线集群上用户设 `=1` 期望跳过下载，实际仍发起网络请求（可能 hang 在无外网环境）。
- 建议：复用统一的 env 布尔解析（如 `worldfoundry.core.utils.env_is_true`），无法解析时 raise 或 warn。

### [CF-17] P2 `reset_local_model_path` 让环境变量静默覆盖调用方显式传入的路径

- 位置：`worldfoundry/core/model_loading/config.py:135-138`
- 证据：

```python
def reset_local_model_path(self):
    if os.environ.get("WORLDFOUNDRY_MODEL_DIR") is not None or self.local_model_path is None:
        self.local_model_path = str(local_model_root_path())
```

- 问题：优先级倒置——只要设置了 `WORLDFOUNDRY_MODEL_DIR`，用户在代码里显式传的 `local_model_path` 就被覆盖。常规约定是"显式实参 > 环境变量 > 默认值"。
- 影响：多模型目录共存场景下模型被下载/解析到意料之外的目录，难排查。
- 建议：改为 `if self.local_model_path is None: ...`（`local_model_root_path()` 内部已消费 env）。

### [CF-18] P2 model.py 顶层 import transformers；`load_model_with_disk_offload` 硬编码 `vram_limit=80`

- 位置：`worldfoundry/core/model_loading/model.py:4-5,218`
- 证据：

```python
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers
...
enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
```

- 问题：(1) 加载纯 torch 模型也要 import transformers（数百 ms 到数秒），违反轻量导入约定，`is_deepspeed_zero3_enabled` 完全可以惰性导入；(2) `vram_limit=80`（GiB）是按 A100/H100-80G 写死的魔数，40G/24G 卡上预算失真。
- 影响：import 变慢；消费级/中端 GPU 上磁盘 offload 策略失效导致 OOM。
- 建议：函数内惰性 import transformers；`vram_limit` 默认 None 并在运行时按 `torch.cuda.get_device_properties(...).total_memory` 推导。

### [CF-19] P3 dataclass 字段类型注解与默认值不符（`str = None`）；两个同名 `ModelConfig` 并存

- 位置：`worldfoundry/core/model_loading/config.py:40-45`；`worldfoundry/core/configuration/model_config.py:36`
- 证据：

```python
path: Union[str, list[str]] = None
model_id: str = None
...
download_source: str = None
```

- 问题：(1) 非 Optional 注解配 None 默认，类型检查工具全部报错，等于放弃静态检查；(2) `worldfoundry.core` 命名空间下有两个 `ModelConfig`（model_loading/config.py 的下载/放置配置 vs configuration/model_config.py 的 DiT 架构配置），`core/__init__.py:37` 把前者作为 `worldfoundry.core.ModelConfig` 导出，极易 import 错。
- 影响：可读性/工具链体验、误用风险。
- 建议：补 `Optional[...]`；重命名其一。

### [CF-20] P3 LightX2VLoRALoader 覆写 `get_name_dict` 改变返回类型；load 全量复制 state_dict

- 位置：`worldfoundry/core/model_loading/lora.py:101-120,122-140`
- 证据：

```python
@staticmethod
def get_name_dict(lora_state_dict):
    ...
    return pairs, diffs   # 父类返回 dict，子类返回 tuple
...
state_dict = model.state_dict()
...
model.load_state_dict(state_dict)
```

- 问题：(1) 子类覆写静态方法但返回类型从 `dict` 变为 `(dict, dict)`，破坏里氏替换，任何按父类契约调用 `get_name_dict` 的代码在子类实例上会拿到 tuple；(2) `load()` 先取全量 `state_dict()` 再整体 `load_state_dict`，14B 模型瞬时多占一份权重内存——同文件 `merge_rank_scaled_lora_` 的 docstring 明确说要避免这种做法。
- 影响：API 契约不稳；大模型加载 LoRA 时内存峰值翻倍。
- 建议：改成 in-place `parameter.add_()` 路径；`get_name_dict` 拆成两个方法。

### 主题 F：checkpoint/

### [CF-21] P0 本地 checkpoint 缓存写入非原子且多 rank 并发写同一路径，损坏后被永久优先加载

- 位置：`worldfoundry/core/checkpoint/load.py:763-789,590-597,630-638,706-710`
- 证据：

```python
def _save_to_local_cache(state_dict, path, ext, *, label="checkpoint cache"):
    ...
    if ext == ".safetensors":
        save_safetensors(dict(state_dict), path)
    else:
        torch.save(state_dict, path)
```

```python
if local_cache_checkpoint_path is not None and os.path.exists(local_cache_checkpoint_path):
    state_dict = _validated_tensor_state_dict(
        torch.load(local_cache_checkpoint_path, map_location="cpu", weights_only=True), ...)
```

- 问题：三条缓存链路（S3 单文件缓存、DCP 合并缓存、sharded-safetensors 合并缓存）都直接写最终路径：(1) 进程在写入中途被 kill/OOM 会留下截断文件，而读取侧只用 `os.path.exists` 判断命中，损坏文件从此**永远优先于源数据**被加载，报错形如反序列化失败甚至静默错权重；(2) `load_distributed_checkpoint` 在每个 rank 上都会执行（DCP 本来就是所有 rank 调用），无 rank0 门控、无文件锁——单机 8 卡即 8 个进程同时 `torch.save` 同一路径，交错写必然损坏；(3) 缓存无任何内容校验（size/etag/hash），S3 源更新后本地陈旧缓存永久生效，可复现性受损。
- 影响：数据损坏 + 分布式竞态 + 陈旧缓存三重问题叠加在所有 checkpoint 加载路径上；损坏后无自愈（需要人工找到 `~/.cache/worldfoundry` 删文件）。
- 建议：写临时文件（同目录 `tempfile.NamedTemporaryFile`）+ `os.replace` 原子改名；rank0 写 + barrier，或 `fcntl.flock`/`O_EXCL` 锁；缓存旁存 etag/size 元数据做命中校验；加载失败时删除缓存并回源。

### [CF-22] P2 `_load_checkpoint_from_local` 把整个 safetensors 读进内存，放弃零拷贝 mmap

- 位置：`worldfoundry/core/checkpoint/load.py:736-741`
- 证据：

```python
if ext == ".safetensors":
    with open(path, "rb") as f:
        payload = load_safetensors(f.read())
```

- 问题：safetensors 的核心优势是 mmap 零拷贝加载，`f.read()` 把几十 GB 的权重先完整读入 Python bytes，再解析成 tensor，内存峰值约 2 倍且首字节延迟高。同文件已 import `load_safetensors_file`（mmap 版）却未使用。
- 影响：大模型加载显著变慢、宿主内存压力翻倍，容器内存受限时直接 OOM。
- 建议：改用 `safetensors.torch.load_file(path, device=...)`。

### [CF-23] P2 `_parallel_hf_hub_download_shards` 用 fork 的 ProcessPoolExecutor，CUDA/线程环境下不安全

- 位置：`worldfoundry/core/checkpoint/load.py:296-299`
- 证据：

```python
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    for shard_file, path in pool.map(_hf_hub_download_shard_task, work):
        shard_to_path[shard_file] = path
```

- 问题：Linux 默认 fork start method。该函数在模型加载路径中被调用，此时进程往往已初始化 CUDA、已有 HF/requests 后台线程；fork 这类进程是 CUDA 官方明确不支持的行为，轻则 warning 重则子进程死锁。下载本身是 IO-bound，用进程池并无必要。
- 影响：偶发 hang/崩溃，且与环境强相关难复现。
- 建议：改 `ThreadPoolExecutor`（hf_hub_download 释放 GIL 的网络 IO 足够并行），或显式 `mp.get_context("spawn")`。

### [CF-24] P2 S3 credential 默认相对路径 `credentials/s3_checkpoint.secret`，行为取决于 CWD

- 位置：`worldfoundry/core/checkpoint/load.py:45,515-530`
- 证据：

```python
_OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH = "credentials/s3_checkpoint.secret"
...
def get_storage_reader(checkpoint_path: str, credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH) -> ...:
```

- 问题：默认凭证路径是相对路径，随进程工作目录漂移；从不同目录启动同一命令结果不同。命名前缀 `_OMNIDREAMS_` 是遗留产品名，与 WorldFoundry 命名体系不一致。
- 影响：S3 checkpoint 在部分启动方式下必现 credential 找不到；排查成本高。
- 建议：默认从 env（如 `WORLDFOUNDRY_S3_CREDENTIALS`）或用户级配置目录解析为绝对路径；清理 OMNIDREAMS 前缀。

### [CF-25] P2 `_validated_tensor_state_dict` 与 `safe_loading.tensor_state_dict` 同包内重复实现

- 位置：`worldfoundry/core/checkpoint/load.py:533-554`；`worldfoundry/core/checkpoint/safe_loading.py:47-67`
- 证据：

```python
# load.py
candidates: list[object] = [payload]
if isinstance(payload, Mapping):
    candidates.extend(payload.get(key) for key in ("state_dict", "model_state_dict", "model", "module") if key in payload)
```

```python
# safe_loading.py
DEFAULT_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "module")
...
candidates.extend(payload[key] for key in wrapper_keys if key in payload)
```

- 问题：同一"解包 wrapper key + 校验 str→Tensor"逻辑在同一子包写了两遍，wrapper key 列表重复维护；错误消息措辞不同。
- 影响：将来加 wrapper key（如 `"ema"`）只改一处的概率很高，行为分叉。
- 建议：load.py 改调 `safe_loading.tensor_state_dict`。

### [CF-26] P2 dcp.py 依赖多枚 torch 私有符号并临时改写 `_version._derived_version` 全局

- 位置：`worldfoundry/core/checkpoint/dcp.py:22-31,76-84`
- 证据：

```python
from torch.distributed.checkpoint.default_planner import (
    DTensor, LoadPlan, _create_read_items, _version, flatten_state_dict,
)
...
_version._derived_version = "2_3"
try:
    old_state_dict, old_mappings = flatten_state_dict(self.original_state_dict)
    ...
finally:
    _version._derived_version = None
```

- 问题：(1) `_create_read_items`/`_version` 是 torch 私有 API，torch 升级即碎；虽有 `except ImportError` fallback，但 fallback 直接退化为原生 planner，`dcp_allow_mismatched_size`、旧 key 布局兼容等特性**静默消失**，调用方无从感知；(2) `_derived_version` 是进程级全局，两个线程并发 load 时互相踩。
- 影响：torch 版本升级后行为静默降级；并发加载偶发错乱。
- 建议：fallback 分支至少 `log.warning` 一次；`_derived_version` 改写段加锁；给私有 API 依赖建 CI 哨兵测试。

### [CF-27] P3 sharded_safetensors 的 zstd 子进程：`-T` 参数分离传递疑似无效/报错，stderr 管道有死锁窗口

- 位置：`worldfoundry/core/checkpoint/sharded_safetensors.py:28-38`
- 证据：

```python
cmd = ["zstd", "-d"]
if num_threads:
    cmd.extend(["-T", str(num_threads)])
process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
decompressed_data = process.stdout.read()
```

- 问题：(1) zstd 的线程参数要求连写 `-T4` 或 `--threads=4`，分离的 `"-T", "4"` 会被解析为文件参数（当前唯一调用方未传 `num_threads`，属埋雷）；(2) stderr=PIPE 但在 stdout 读完之前从不消费，zstd 若输出较多告警会填满 64KB 管道缓冲互相阻塞；(3) 未处理 zstd 二进制不存在的 FileNotFoundError。
- 影响：启用压缩 checkpoint + 线程参数时直接失败；极端情况下 hang。
- 建议：用 `subprocess.run(capture_output=True)` 或 `--threads=N` 连写形式；捕获 FileNotFoundError 给出"请安装 zstd"提示。

### [CF-28] P2 checkpoint/load.py 顶层硬依赖 loguru 与 huggingface_hub，与"loguru 可选"的日志设计矛盾

- 位置：`worldfoundry/core/checkpoint/load.py:27-28`
- 证据：

```python
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from loguru import logger
```

- 问题：`logging_setup.py:173-178` 专门为 loguru 缺失做了 stdlib fallback（"minimal runtime environment"），但 checkpoint 加载路径顶层硬 import loguru，最小环境下加载本地 checkpoint 也会 ImportError；加载纯本地文件同样被迫 import huggingface_hub。
- 影响："最小运行环境"承诺被单点破坏。
- 建议：改用 `worldfoundry.core.logging_setup.get_logger(__name__)`；HF 相关 import 下沉到 `_download_checkpoint_from_huggingface_url` 等函数内。

### 主题 G：io/

### [CF-29] P1 `easy_io.load` 把 `map_location`/`weights_only` 无条件透传给所有格式加载器，非 torch 格式必然 TypeError

- 位置：`worldfoundry/core/io/easy_io.py:43-52`；`worldfoundry/core/io/serialization.py:326-333`
- 证据：

```python
def load(self, path, *, map_location="cpu", weights_only=True, **kwargs):
    options = {key: value for key, value in kwargs.items() if key in {"encoding", "file_format", "loader"}}
    return load_serialized(
        resolve_checkpoint_path(path),
        map_location=map_location,
        weights_only=weights_only,
        **options,
    )
```

  `load_serialized` 中 JSON 分支为 `json.loads(_read_text(...), **kwargs)`。实测：`json.loads('{}', map_location='cpu', weights_only=True)` → `TypeError: JSONDecoder.__init__() got an unexpected keyword argument 'map_location'`。
- 问题：`easy_io` 是给 vendored cosmos 代码用的兼容 facade，声称支持任意格式，但除 torch/byte 格式外（json/yaml/pkl/jsonl/csv...）全部因多余 kwargs 崩溃。
- 影响：任何经 `easy_io.load` 读 JSON/YAML 的调用（`base_models/.../structured_conditioning.py:195` 等 vendored 调用点一旦传入非 torch 文件）直接 TypeError，且报错信息完全不指向真实原因。
- 建议：仅当推断格式属于 `_TORCH_FORMATS` 时才附加 `map_location/weights_only`。

### [CF-30] P1 `io/cache.py` 缓存写入非原子 + rank 探测失败时所有进程并发写同一路径

- 位置：`worldfoundry/core/io/cache.py:32-49,63-69`
- 证据：

```python
def _populate(source_path, cache_path: Path, backend_args=None) -> None:
    ...
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    ...
    copy_uri(source_path, cache_path, **_storage_options(backend_args))

def _distributed_rank_and_barrier():
    try:
        from worldfoundry.core.distributed import torch_process_group as distributed
        return distributed.get_rank(), distributed.barrier
    except Exception:
        return 0, lambda: None
```

- 问题：(1) `copy_uri` 直接写最终缓存路径，中途崩溃留下非空截断文件，`st_size > 0` 的命中检查会把它当有效缓存**永久使用**；(2) 进程组未初始化（普通多进程并行评测是常态）时 `except Exception` 让**每个进程都自认 rank 0**，并发 `copy_uri` 同一路径互相踩踏；(3) 与 CF-21 同根：仓库里已有正确实现（`io/download.py` 的 temp+validator+`os.replace`、`io/integrity.py` 的原子原语、`io/s3_sync.py` 的 size/sha256 校验），但这条最常用的缓存链路没有用它们。
- 影响：共享缓存目录（NFS/CPFS 上的 `~/.cache/worldfoundry`）被截断文件污染后所有后续任务加载失败或错数据。
- 建议：`_populate` 改走 `download_to_cache` 式 temp+rename；rank 探测区分"未初始化"（用文件锁）与"已初始化"（rank0+barrier）。

### [CF-31] P1 `extract_tar` 无 filter 的 `tar.extractall`，存在路径穿越（CVE-2007-4559 类）

- 位置：`worldfoundry/core/io/file_utils.py:610-621`
- 证据：

```python
def extract_tar(source_tarball, output_dir=".", members=None):
    import tarfile
    source_tarball, output_dir = f_expand(source_tarball), f_expand(output_dir)
    with tarfile.open(source_tarball, "r:*") as tar:
        tar.extractall(output_dir, members=members)
```

- 问题：不传 `filter=`，恶意 tar 中 `../../` 成员或绝对路径成员可写出 `output_dir` 之外（Python<3.14 默认不过滤，3.12+ 仅告警）。该框架的评测/资产链路会解包下载的 benchmark 数据包，输入并非总是可信。
- 影响：解包不可信归档时任意文件写 → 可升级为代码执行（写 `~/.bashrc`、site-packages 等）。
- 建议：`tar.extractall(output_dir, members=members, filter="data")`；低版本 Python 手动校验成员路径（可复用 `integrity.safe_relative_path`）。

### [CF-32] P1 python_config 的配置值 eval 与两处 eval 型 f-string：数据即代码 + 依赖 CPython locals() 实现细节

- 位置：`worldfoundry/core/io/python_config.py:97-114`；`worldfoundry/core/io/print_utils.py:60-65`；`worldfoundry/core/io/file_utils.py:666-673`
- 证据：

```python
if value.startswith("eval(") and value.endswith(")"):
    return eval(value[5:-1], {}, {"d": root})
...
resolved = re.sub(r"\${(.*)}", r"d.\1", original)
...
    return eval(resolved, {}, {"d": root})
```

```python
def fstring(fmt_str, **kwargs):
    locals().update(kwargs)
    return eval("f" + shlex.quote(fmt_str))
```

- 问题：(1) 配置**字符串值**以 `eval(` 开头即执行（globals 为 `{}` 时 `__builtins__` 仍自动注入，`__import__('os')` 可用），把"数据"升级为代码执行面；(2) `${...}` 插值正则是贪婪 `(.*)`，同一字符串出现两个 `${a}...${b}` 时会被替换成 `d.a}...${b` 然后 eval 报 SyntaxError（逻辑 bug）；(3) `fstring` 依赖 `locals().update()` 能被后续 eval 看到——这是 CPython≤3.12 的实现细节，PEP 667（Python 3.13）后失效；用 `shlex.quote`（shell 引号规则）拼 Python 字符串字面量，遇单引号即语法错乱。同一函数在 print_utils 和 file_utils 内重复实现两份。
- 影响：安全面扩大；多插值配置直接崩；Python 3.13 升级后 `next_available_file_name` 的默认模板 `"_v{i+1}"` 静默失效。
- 建议：插值改非贪婪并用受限求值（`ast.literal_eval` + 显式属性解析）；`fstring` 改 `str.format_map` 或 `string.Template`（`{i+1}` 这类表达式模板改为回调）；删除重复副本。

### [CF-33] P2 三条"下载/合并→缓存"链路各自实现且质量不一：正确的校验与原子性没有沉淀为唯一路径

- 位置：`worldfoundry/core/io/download.py:99-135`（temp+validator+replace，正确）；`worldfoundry/core/io/s3_sync.py:129-181`（size+sha256 校验+重试，正确）；`worldfoundry/core/io/cache.py:32-40` 与 `worldfoundry/core/checkpoint/load.py:763-789`（直接写最终路径，无校验）；另 `io/artifacts.py:243-257` `download_file_from_url`（requests 无 timeout、失败留半截文件、print+返回 None）
- 证据：

```python
# artifacts.py
response = requests.get(url, stream=True, allow_redirects=True)   # 无 timeout
...
except requests.exceptions.RequestException as exc:
    print(f"Error downloading file: {exc}")
    return None    # 中途断连时 filename 处已留下部分内容
```

- 问题：同一"取远端字节落地本地"的任务在 core 内至少 5 处实现，其中 2 处正确、3 处有截断/竞态/挂死风险；新代码无从知道该抄哪份。
- 影响：行为不可预测；修一处漏其余。
- 建议：以 `download_to_cache`（加可选 checksum 参数）为唯一原语，其余全部改调；`download_file_from_url` 加 timeout 并失败即删除部分文件。

### [CF-34] P2 `merge_video_audio` 捕获自己抛出的异常并静默返回；wan_video_geometry 的 `save_video` 失败仅记 INFO

- 位置：`worldfoundry/core/io/video_data.py:208-219`；`worldfoundry/core/io/wan_video_geometry.py:120-138`
- 证据：

```python
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)
        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")
    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"merge_video_audio failed with error: {e}")
```

- 问题：`raise RuntimeError` 被同函数末尾的 `except Exception` 吃掉，函数正常返回；调用方（评测产物落盘）以为合成成功。`wan_video_geometry.save_video` 同样吞掉所有异常且用 `logging.info` 记录失败。这两个文件还各自复制了一份几乎相同的 `merge_video_audio`。
- 影响：评测视频静默缺音轨/缺文件，跑完整个 benchmark 才发现产物为空，浪费大量 GPU 时。
- 建议：删掉外层 except（clean-up 用 finally），让失败冒泡；重复实现合并为一处。

### [CF-35] P2 `_atomic_write_text` 固定临时文件名，两进程并发写同一目标互相截断；与 json_utils 的非原子写并存

- 位置：`worldfoundry/core/io/serialization.py:149-157`；`worldfoundry/core/io/json_utils.py:64-69`
- 证据：

```python
if atomic:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
```

- 问题：(1) 同目标的两个并发写者共用同一个 `.name.tmp`，`write_text`（truncate+write）交错后 replace 发布的可能是混合内容——同文件 `write_jsonl`（264 行）已用 `uuid4` 临时名+`"x"` 模式，风格未统一；(2) 同包 `json_utils.json_dump` 是第三套 JSON 写实现，完全非原子。
- 影响：并行评测写 scorecard/manifest 时低概率产出损坏 JSON。
- 建议：`_atomic_write_text` 加 uuid 后缀；`json_utils` 标注 legacy 并逐步收敛到 `serialization.write_json`。

### [CF-36] P2 config_utils 是 core 内第 4 套 instantiate 体系，且 `_CLASS_REGISTRY` 静默覆盖、异常链丢失、`print_config` 已随 omegaconf 2.x 失效

- 位置：`worldfoundry/core/io/config_utils.py:15,24-26,112-129,216-219`
- 证据：

```python
_CLASS_REGISTRY = {}  # for instantiation
...
def print_config(cfg: DictConfig) -> None:
    print(cfg.pretty(resolve=True))
...
        except Exception as e:
            raise RuntimeError(f"Error instantiating {cls}: {e}")
```

- 问题：(1) core 内已有 `lazy_config.instantiate`（`_target_`）、`model_loading.factory.instantiate_from_config`（`target`/`class_path`）、`configuration/hydra.override`，这里再来一套 `cls`/`class` + 模块级 dict 注册表——4 套目标定位词汇并存；(2) `_CLASS_REGISTRY[name] = class_type` 无重复检查（与 `core/registry.py` 的 Duplicate 策略相反）；(3) `raise RuntimeError(...)` 未 `from e`，原始 traceback 丢失；(4) `DictConfig.pretty()` 在 omegaconf≥2.1 已删除，`print_config` 必然 AttributeError（死代码）；(5) `resource_file_path` 在 `importlib.resources.path` 上下文退出后返回路径，zip 安装场景下该临时路径已被删除。
- 影响：配置生态碎片化到难以教学；实例化错误难定位；隐性坏函数。
- 建议：明确该模块为 legacy（vendored jimfan-utils 系）并冻结；`raise ... from e`；删除 `print_config` 或改 `OmegaConf.to_yaml`。

### [CF-37] P2 io 的 pickle 面：`load_serialized` 的 pkl/gz 分支、`file_utils.load_pickle`、`utils.misc_utils` base64-pickle

- 位置：`worldfoundry/core/io/serialization.py:332-335`；`worldfoundry/core/io/file_utils.py:711-713`；`worldfoundry/core/utils/misc_utils.py:262`
- 证据：

```python
if fmt in _PICKLE_FORMATS:
    return pickle.loads(_read_bytes(file), **kwargs)
if fmt in _GZIP_FORMATS:
    return pickle.loads(gzip.decompress(_read_bytes(file)), **kwargs)
```

- 问题：`load_serialized` 是通用入口（`load_model_loader_registry`、`load_from_cache_or_uri` 等都走它），后缀为 `.pkl/.gz` 的远端 URI 会直接反序列化任意对象；`.gz` 被假定"必为 gzip+pickle"，语义上也不对（`.json.gz` 等）。torch 分支默认 `weights_only=True` 做对了，pickle 分支没有等价防护也无文档警示。
- 影响：从不可信源（HF 仓库、S3、URL）加载 `.pkl` 即任意代码执行；与 CF-4（yaml.unsafe_load）构成 core 的两大反序列化风险面。
- 建议：pickle 分支加显式 opt-in 参数（如 `allow_pickle=True` 默认 False），docstring 标注信任边界；`.gz` 改为嗅探内层格式或要求双后缀。

### [CF-38] P2 `easy_io.exists` 吞掉一切异常返回 False，把网络/权限错误伪装成"文件不存在"

- 位置：`worldfoundry/core/io/easy_io.py:36-41`
- 证据：

```python
def exists(self, path, **kwargs) -> bool:
    try:
        resolved = resolve_checkpoint_path(path)
        return exists_uri(resolved, **self._options(**kwargs))
    except Exception:
        return False
```

- 问题：`resolve_checkpoint_path` 可能触发 HF 下载/解析，认证失败、网络超时、参数错误全部折叠为 `False`。
- 影响：上游据此走"文件缺失"分支（重下载/报缺资产），偶发环境问题被永久误诊。
- 建议：只捕获 `FileNotFoundError`/`OSError` 的明确子集，其余冒泡。

### [CF-39] P3 `copy_uri` 远端路径整读内存；`_load_video`/`_dump_video` 魔数默认 fps=17；`f_remove` 静默吞删除错误

- 位置：`worldfoundry/core/io/storage.py:227-236`；`worldfoundry/core/io/serialization.py:559-567`；`worldfoundry/core/io/file_utils.py:315-323`
- 证据：

```python
    write_binary_uri(dst, read_binary_uri(src, **storage_options), **storage_options)
```

```python
                try:
                    os.remove(f)
                except Exception as e:  # final resort safeguard
                    pass
```

- 问题：(1) 远端 copy 把整个对象读进内存再写出，几十 GB checkpoint 直接打爆宿主内存（而它正是 `cache._populate` 的底层）；(2) `_dump_video` 默认 `fps=17` 是某模型特有值，作为通用序列化默认值易踩坑；(3) `f_remove` 对删除失败完全无声（连日志都没有）。
- 影响：大文件缓存路径 OOM；误用默认 fps；删除失败无从排查。
- 建议：copy 用分块流式（fsspec `open` 两端 + `shutil.copyfileobj`）；fps 改必填或 24；`f_remove` 至少 warning。

### 主题 H：utils/

### [CF-40] P1 `to_state_dict` 主路径返回 None：默认参数下任何对象的 state_dict 都被丢弃

- 位置：`worldfoundry/core/utils/torch_utils.py:403-413`
- 证据：

```python
    def _to_state_dict(m):
        if implements_state_dict(m):
            if isinstance(m, nn.Module) and unwrap_ddp:
                m = unwrap_ddp_model(m)
                tree = _require_tree()
                return tree.map_structure(_transfer, m.state_dict())
        else:
            return _transfer(m)
```

- 问题：当对象实现了 `state_dict()` 但不满足 `isinstance(m, nn.Module) and unwrap_ddp`（默认 `unwrap_ddp=False`，或对象是 Optimizer/LRScheduler）时，内层 if 不命中且没有对应分支，函数隐式返回 `None`。即 `to_state_dict(model)` 默认返回 None——主路径完全失效，明显是重构时丢失了分支。
- 影响：调用方拿到 None 状态字典；若用于保存 checkpoint 则静默写出空内容（数据丢失级别）。当前仓库内可能无人调用才未暴露——那它就是危险的死代码，留着必伤人。
- 建议：补上缺失的通用分支（`return tree.map_structure(_transfer, m.state_dict())`），或删除该函数；补一个最小单测。

### [CF-41] P2 torch_utils 多枚失修 API：`save_torch` 反转参数时保存成 1 元组、`np.long` 已被 numpy 移除、运行期设 `PYTHONHASHSEED` 无效

- 位置：`worldfoundry/core/utils/torch_utils.py:239-246,599,136`
- 证据：

```python
def save_torch(D, *fpath):
    if isinstance(D, str):
        assert not isinstance(fpath, str), ...
        fpath, D = D, fpath          # D 变成 (obj,) 元组
    torch.save(D, str(f_join(fpath)))
```

```python
        return np.array(D, dtype=np.long)   # numpy>=1.24 AttributeError
```

```python
    os.environ["PYTHONHASHSEED"] = str(seed)   # 解释器启动后设置无效
```

- 问题：(1) 文档声称支持 `save_torch(fpath, D)` 两种参数顺序，但交换后 `D` 是 `(obj,)` 元组，`torch.save` 存的是元组——加载方拿到的类型悄然改变；(2) `random_derangement(format="numpy")` 在现代 numpy 下直接 AttributeError；(3) `PYTHONHASHSEED` 必须在解释器启动前设置，运行期赋值不改变 str hash 行为，给出虚假的可复现性承诺。
- 影响：契约破坏、死路径、误导性的 reproducibility；这类 vendored 工具层长期无测试覆盖。
- 建议：`save_torch` 交换时取 `fpath[0]`（或删除该"便利"）；`np.long`→`np.int64`；删掉 PYTHONHASHSEED 行或在 docstring 说明其局限。

### [CF-42] P2 core 内并存 4 套注册表抽象，且除 TypedRegistry 外全部静默覆盖重名

- 位置：`worldfoundry/core/registry.py`（TypedRegistry，重复注册报错）；`worldfoundry/core/utils/functional_utils.py:278,332`（`make_registry_metaclass`/`ClassRegistry`）；`worldfoundry/core/io/config_utils.py:15,24-26`（`_CLASS_REGISTRY`）
- 证据：

```python
# functional_utils.py — 无重复检查
cls.registry[name] = new_cls
...
def add(self, cls):
    self.registry[cls.__name__] = cls
```

- 问题：`registry.py` 精心实现了确定性、大小写不敏感、重复注册即 `DuplicateRegistryKeyError` 的注册表，但同一 core 里还有 3 套注册机制，全部"后注册者静默胜出"，错误消息与行为互相矛盾。
- 影响：新增代码无从选择正确抽象；重名 bug 只在其中一套里能被发现。
- 建议：utils 两套标注 legacy（vendored）并禁止新用；`config_utils._CLASS_REGISTRY` 改用 TypedRegistry 或至少加重复告警。

### [CF-43] P3 并行执行工具三套并存、命名误导（`ProcessThreadPool` 实为线程池）、`async_return=True` 泄漏 pool

- 位置：`worldfoundry/core/utils/parallel_execution.py:7-8,83-92`
- 证据：

```python
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool as ProcessThreadPool
...
    pool = ProcessThreadPool(processes=num_processes)
    ...
    if async_return:
        return pool
```

- 问题：(1) `parallel_execution(num_processes=32)` 实际开的是**线程**，参数名与别名 `ProcessThreadPool` 双重误导；(2) `async_return=True` 返回未 close 的 pool，异常路径也不 close（无 try/finally）；(3) 与 `parallel_threads`/`parallel_processes`、`s3_sync` 的 ThreadPoolExecutor、`sharded_safetensors` 的线程池并存，仓库内至少 4 种并行 map 写法。
- 影响：读者误判并发模型（GIL 下 CPU 密集任务无加速）；资源泄漏。
- 建议：统一到 `concurrent.futures`；别名与参数改名；补 try/finally。

### [CF-44] P3 随机种子 API 三套并存且保证不一致；cuda_graph 依赖 torch 私有符号

- 位置：`worldfoundry/core/utils/torch_utils.py:107,152,783`；`worldfoundry/core/utils/cuda_graph.py:22-24`
- 证据：

```python
def set_seed_everywhere(seed, deterministic=False, set_tensorflow=False, ...)
def set_random_seed(seed: int, by_rank: bool = False) -> int:
def fix_random_seeds(seed=31):
```

```python
from torch._C import _graph_pool_handle
from torch.utils._pytree import tree_flatten as _tree_flatten
```

- 问题：三个种子函数覆盖范围各不相同（`fix_random_seeds` 不管 determinism/TF/PYTHONHASHSEED；`set_random_seed` 包装 `set_seed_everywhere` 但吞掉 rank 探测异常），调用方难以知道用哪个才"够"；cuda_graph 与 dcp.py（CF-26）同样耦合 torch 私有 API，升级 torch 时是隐性断点。
- 影响：可复现性保证碎片化；torch 升级脆弱。
- 建议：保留 `set_seed_everywhere` 一个入口，其余转发并标 deprecated；cuda_graph 加 torch 版本兼容说明/防护。

### 主题 I：structures/ 与 safety/ 与 visualization/

### [CF-45] P2 `core/visualization` 缺 `__init__.py`：setuptools `find`（非 namespace）打包会丢弃该子包

- 位置：`worldfoundry/core/visualization/`（目录内仅 `action_overlay.py`）；`pyproject.toml:509`
- 证据：

```toml
packages = { find = { where = ["."], include = ["worldfoundry*"], exclude = [...] } }
```

  目录清单：`visualization/` 下无 `__init__.py`（仅 `action_overlay.py` 与 `__pycache__`）。
- 问题：`tool.setuptools.packages.find` 默认只发现带 `__init__.py` 的常规包，`worldfoundry.core.visualization` 会被排除在 wheel 之外；源码 checkout 下靠隐式 namespace package 侥幸可导入（`tests/studio_visualization/...` 正是这样用），pip 安装后 ImportError。全 core 其余子包均有 `__init__.py`，唯独这里缺失，显然是遗漏而非设计。
- 影响：安装发行版后 `core.visualization` 缺失；测试在源码环境通过、安装环境失败的隐性差异。
- 建议：补 `__init__.py`（可带 lazy 导出）；CI 加"wheel 内容与源码包清单一致"检查。

### [CF-46] P2 safety 包 import 即拉起 torch（经 `core.distributed.logging`），Protocol 定义模块被重依赖劫持

- 位置：`worldfoundry/core/safety/__init__.py:3`；`worldfoundry/core/safety/guardrails.py:10`；`worldfoundry/core/distributed/logging.py:7`
- 证据：

```python
# safety/__init__.py — 急切导入
from .guardrails import ContentSafetyGuardrail, GuardrailRunner, PostprocessingGuardrail
# guardrails.py
from worldfoundry.core.distributed.logging import log
# distributed/logging.py
import torch
```

- 问题：safety 包的主体是两个 `Protocol` 接口和一个纯编排类，本应是最轻的模块；却因 `distributed.logging` 顶层 `import torch` 连带把 torch/numpy/imageio 全部拉进来。轻量导入原则（项目自我要求）在此失守。另注意 `run_safety_check` 在未配置任何 safety model 时返回"safe"（fail-open），仅记 warning——对安全组件这是值得显式文档化的默认。
- 影响：CLI/目录查询若触达 safety 类型注解即背上 torch 导入成本；最小环境不可用。
- 建议：`log` 改为函数内导入或用 stdlib `logging.getLogger`；`distributed.logging` 的 torch 依赖本身值得懒化（它只在取 rank 时需要）。

### [CF-47] P3 structures 导出表漂移：`Float`/`OneOf` 已实现但未导出；`tree_utils` 吞类型化异常

- 位置：`worldfoundry/core/structures/__init__.py:20-24`；`worldfoundry/core/structures/validator.py:81,108`；`worldfoundry/core/structures/tree_utils.py:29-35`
- 证据：

```python
    "Bool": "worldfoundry.core.structures.validator",
    "Int": "worldfoundry.core.structures.validator",
    "JsonDict": "worldfoundry.core.structures.validator",
    "String": "worldfoundry.core.structures.validator",
    "Validator": "worldfoundry.core.structures.validator",
    # validator.__all__ 还有 Float、OneOf —— 未列入
```

- 问题：手工维护的 `_EXPORT_MODULES` 已与 `validator.__all__` 脱节（`Float`/`OneOf` 缺席），这正是 CF-3（巨型手写导出表）预言的漂移在最小包上的实例；`tree_value_at_path` 把任意异常折叠成 `ValueError` 且未 `from e`。
- 影响：`from worldfoundry.core.structures import Float` 报 AttributeError，用户被迫直接 import 深层模块；调试信息降级。
- 建议：加"导出表与子模块 `__all__` 一致"的单测（一个 for 循环即可）；异常用 `raise ... from e`。

### 主题 J：根级杂项（prompting/time/process/runtime_cache）

### [CF-48] P3 `core/prompting.py` 顶层 `import torch`，仅为两个 `@torch.no_grad()` 装饰器

- 位置：`worldfoundry/core/prompting.py:7,25,33`
- 证据：

```python
import torch
...
    @torch.no_grad()
    def process_prompt(self, prompt, positive: bool = True):
```

- 问题：模块 docstring 自称 "framework-neutral"，但顶层 import torch；装饰器完全可以在方法体内用 `with torch.no_grad():`（延迟导入）实现。当前消费方（wan prompter）本身是 torch 重代码，实际影响小，但它经 `core/__init__` 的 `_EXPORT_MODULES` 暴露，`from worldfoundry.core import PromptProcessor` 即拉起 torch。
- 影响：轻量导入原则的一处小破口。
- 建议：改函数内 `torch.no_grad()` 上下文；或接受现状并从 core 顶层导出表移除。
- 备注（正面）：同目录 `process.py`（子进程生命周期 JSONL 化、进程组终止）、`runtime_cache.py`、`time.py`（`CudaSyncTimer` 在 `__enter__` 才读 env、懒 import torch）质量良好，是 core 内导入卫生的正面样板；唯 `time.py:37` 的 `SYNC_TIMER` 环境变量无 `WORLDFOUNDRY_` 前缀（并入 CF-8 命名问题）。

## 汇总

### 按严重度统计

| 严重度 | 数量 | 条目 |
| --- | --- | --- |
| P0 | 1 | CF-21 |
| P1 | 9 | CF-1, CF-3, CF-4, CF-15, CF-29, CF-30, CF-31, CF-32, CF-40 |
| P2 | 26 | CF-2, CF-5, CF-6, CF-7, CF-8, CF-11, CF-12, CF-16, CF-17, CF-18, CF-22, CF-23, CF-24, CF-25, CF-26, CF-28, CF-33, CF-34, CF-35, CF-36, CF-37, CF-38, CF-41, CF-42, CF-45, CF-46 |
| P3 | 12 | CF-9, CF-10, CF-13, CF-14, CF-19, CF-20, CF-27, CF-39, CF-43, CF-44, CF-47, CF-48 |
| 合计 | 48 | |

### 按主题分布（粗略）

- 分层/循环依赖：CF-1, CF-2
- configuration 体系：CF-3～CF-11
- registry 与注册表碎片化：CF-12, CF-42
- model_loading：CF-15～CF-20
- checkpoint：CF-21～CF-28
- io：CF-29～CF-39
- utils：CF-40～CF-44
- structures/safety/visualization/打包：CF-45～CF-47
- 根级杂项与日志：CF-13, CF-14, CF-48

### 本范围 Top 5 最重要问题

1. **[CF-21] P0 checkpoint 本地缓存非原子写 + 多 rank 并发写同一路径**：`_cache_checkpoint_locally` 直接写最终路径且无校验，写一半崩溃/并发踩踏产生的损坏文件此后被**永久优先加载**；同型缺陷再现于 `io/cache.py`（CF-30）与 `serialization._atomic_write_text` 的固定临时名（CF-35）。仓库内已有正确原语（`io/download.py`、`io/integrity.py`、`io/s3_sync.py` 的校验+重试），只需收敛到唯一路径。
2. **[CF-4]+[CF-32] P1 配置面 = 代码执行面**：`LazyConfig` 用 `yaml.unsafe_load`+`exec`，`python_config` 对以 `eval(` 开头的**配置值**直接 `eval`，`print_utils/file_utils` 的 eval 型 fstring 还依赖 CPython≤3.12 的 `locals()` 行为。加载一份不可信 YAML/JSON5 配置等价于运行任意代码，且 Python 3.13 下部分路径必然失效。
3. **[CF-15] P1 `dist.barrier(device_ids=[dist.get_rank()])` 多机传错设备号**：device_ids 需要**本地** device index，传全局 rank 在 node1+ 上越界/错绑 GPU，多机下载同步 barrier 直接崩溃或死锁——多机推理/评测的实际阻断点。
4. **[CF-1]+[CF-2] P1 core→runtime 反向依赖成环**：`io/video_tiling.py` 顶层、`utils/inference_runtime.py` 函数内 import `worldfoundry.runtime.compile_cache`，而 runtime 大量 import core。"core 是最底层"的架构承诺已破，当前靠 lazy `__getattr__` 和导入顺序侥幸存活。
5. **[CF-40]+[CF-29] P1 基础 API 主路径失效**：`to_state_dict` 默认参数下对任何对象返回 None（保存空 checkpoint 的数据丢失路径）；`easy_io.load` 把 `map_location/weights_only` 透传给 JSON/YAML 加载器必然 TypeError——两者都说明 utils/io 兼容层缺少最小单测防护。

### 总体评价

core 呈明显的"双层结构"：新写的基础设施（`registry.py`、`logging_setup.py`、`io/integrity.py`、`io/download.py`、`io/disk.py`、`process.py`、`checkpoint/safe_loading.py`）设计质量高——原子写、审计钩子、redaction、进程组终止都做对了；但大量 vendored 风格的工具层（jimfan-utils 系的 `file_utils/print_utils/misc_utils/torch_utils`、wan 系的 `wan_video_geometry/video_data`、cosmos 系的 `easy_io/cosmos_config`）带入了 eval/exec、静默吞错、非原子写、私有 API 依赖等历史债，且与新层并存造成 4 套注册表、4 套 instantiate、3 套下载缓存、3 套种子函数的碎片化。建议的整改主线：(1) 用 `integrity/download` 原语统一所有"远端→本地缓存"写路径；(2) 给配置加载划定信任边界，废除 eval/exec/unsafe_load；(3) 冻结 legacy 工具层（标注 deprecated、禁止新引用），把导出表与 `__all__` 一致性、easy_io 兼容层行为纳入最小单测。
