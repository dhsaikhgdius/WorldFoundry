# Core Foundation 修复日志（对应评审 `plan/code_review/01_core_foundation.md`，CF-1～CF-48）

范围：仅 `worldfoundry/core/` 基础半区（configuration/、registry.py、model_loading/、checkpoint/、io/、structures/、utils/、logging_setup.py、log_filters.py、prompting.py、time.py、process.py、runtime_cache.py、safety/、visualization/、core/`__init__.py`）。
计算半区（distributed/、attention/、kernels/、nn/、acceleration/、memory/、vram/、device.py、inference*.py、realtime*.py）与其它子包未动。

新增测试（纯 CPU，未改任何既有测试）：

- `test/test_core_foundation_fix_cache_atomicity.py`（11 项）
- `test/test_core_foundation_fix_point_defects.py`（21 项）

验证环境说明：本机为最小环境（torch 2.7+CUDA、pytest、attrs、yaml、requests、safetensors 可用；omegaconf/hydra/loguru/huggingface_hub/boto3/imageio/ffmpeg **不可用**）。凡依赖 omegaconf/hydra 的模块（configuration 包）无法整包导入验证，采用 `py_compile` + 按文件路径 importlib 加载（必要处对 `lazy_config` 打桩）做行为验证，并在下文逐条注明。

---

## P0

### CF-21 checkpoint/load.py 缓存非原子写 + 无并发防护 + 损坏缓存永久优先 —— 已修

改动（`worldfoundry/core/checkpoint/load.py`）：

- `_save_to_local_cache`：写同目录唯一临时文件（`.<name>.<uuid>.tmp`）→ `flush+os.fsync` → `os.replace` 原子发布；失败路径清理临时文件，最终缓存槽位保持为空（`os.path.exists` 命中检查不会看到半截文件）。
- 三条缓存链路（S3 单文件、DCP 合并、sharded-safetensors 合并）读取侧统一加损坏自愈：缓存读取失败 → `_discard_corrupt_cache` 删除坏文件并记 warning → 回源重建。修复"损坏后永久优先于源数据"的核心问题。
- `_should_write_shared_cache`：进程组已初始化时仅 rank0 写共享缓存（其余 rank 只读），未初始化时保持单进程语义。多 rank 并发交错写被消除。
- 陈旧缓存（源更新后本地缓存不失效）：未加 etag/size 元数据校验 —— **deferred**（见末节，需要跨三条链路设计缓存元数据 sidecar 格式，属结构性收敛的一部分）。

验证：`test_core_foundation_fix_cache_atomicity.py` —— 中断写不发布任何文件；残留 `.tmp` 不影响命中与读取；人为写坏缓存后二次加载自愈并重建出可读缓存；发布后无 tmp 残留。11/11 通过。

### CF-30 io/cache.py `_populate` 非原子 + rank 探测失败全员自认 rank0 —— 已修

改动（`worldfoundry/core/io/cache.py`）：

- `_populate`：先 `copy_uri` 到同目录唯一临时文件，成功后 `os.replace` 原子改名；失败清理临时文件（`unlink(missing_ok=True)`）。
- 新增 `_populate_lock`：基于 `fcntl.flock` 的旁路锁文件（`<cache>.lock`）串行化本机并发填充；POSIX 锁不可用时降级为无锁（记 debug）。进程组未初始化的普通多进程并行评测由文件锁保护；已初始化时保持原 rank0+barrier 门控。
- 锁内二次命中检查：拿到锁后先复查缓存是否已被并发者填充，避免重复下载。

验证：同上测试文件 —— 中断复制后缓存槽为空、重试成功；4 线程并发填充只发生 1 次下载（锁文件放 `/tmp` 保证 flock 语义）。

### CF-35 serialization._atomic_write_text 固定临时名并发互相截断 —— 已修

改动（`worldfoundry/core/io/serialization.py`）：临时名加 `uuid4` 后缀（对齐同文件 `write_jsonl` 的既有风格），`os.replace` 发布，异常清理。

验证：4 写者 × 25 轮并发写同一 JSON，最终文件始终是某一个写者的完整 payload，无混合内容、无 tmp 残留。

---

## P1

### CF-1 io/video_tiling.py 顶层 import worldfoundry.runtime —— 已修

`CompilePolicy`/`compile_callable_cached` 改为 `VideoTiler.__init__` 内惰性导入；docstring 标注 core→runtime 层级债务（正确归宿是 compile_cache 下沉或 tiler 上移，等 runtime/ 侧收敛后处理）。未移动任何文件。

### CF-2 utils/inference_runtime.py 反向依赖 —— 已修（文档化）

该文件的 `worldfoundry.runtime.compile_cache` 导入原本已在函数内，补模块级 docstring 标注层级债务，与 CF-1 同一处置口径。

### CF-3 lazy_config 导入时全局 monkey-patch `OmegaConf.to_object` —— 部分修复 + deferred

已做：`lazy_config/instantiate.py` 改为直接调用本包 `omegaconf_patch.to_object`，不再依赖全局补丁存在；`lazy_config/__init__.py` 在补丁处加显著 WARNING 注释（语义差异、影响面、新代码应直接调本地函数）。
Deferred：移除全局补丁本身。理由：全仓多处 vendored 代码调用 `OmegaConf.to_object`（dc_ae、perceptionlm、depth_anything、wan linear 等），无法在不逐一审计的前提下确认没有调用方依赖"含 `_target_` 的 DictConfig 原样返回"的补丁语义。方案：为补丁加环境开关并在 CI 中以关闭态跑全量测试，绿后默认关闭一个版本周期再删除。

### CF-4 `LazyConfig.load` 的 yaml.unsafe_load —— 已修

改为 `yaml.safe_load`。依据：rg 全仓 YAML 配置样本无 `!!python` 等对象标签；保存侧 `save_yaml` 一直经 `_plain_config` 降级为纯数据，框架自身 round-trip 不需要 unsafe load。类 docstring 写明信任边界（.py 配置=代码，YAML=数据）。`.py` 配置的 exec 是 detectron2 LazyConfig 既有设计，保留并文档警示。
验证：py_compile；本机无 omegaconf，行为验证受限（读取逻辑未变，仅 loader 收紧）；若外部环境存在依赖 python-object YAML 的配置将在加载时显式报错而非静默执行。

### CF-15 `download_if_necessary` 的 `dist.barrier(device_ids=[dist.get_rank()])` —— 已修

`worldfoundry/core/model_loading/config.py`：CUDA 可用时传 `device_ids=[torch.cuda.current_device()]`（本地设备号），否则不传 `device_ids`。与 torch.distributed 语义一致（device_ids 期望 local device index，多机下全局 rank 会越界/绑错卡）。

### CF-29 easy_io.load 无条件透传 map_location/weights_only —— 已修

`worldfoundry/core/io/easy_io.py`：先 `infer_serialization_format`，仅当格式属 `_TORCH_FORMATS` 时附加这两个 kwargs。JSON/YAML/pickle 等不再 TypeError。
验证：单测加载 JSON 正常返回。

### CF-31 extract_tar 无 filter 的路径穿越（CVE-2007-4559 类） —— 已修

`worldfoundry/core/io/file_utils.py`：`tar.extractall(..., filter="data")`；对不支持 `filter=` 的旧 Python 以 `TypeError` 回退到手工校验（复用 `integrity.safe_relative_path` 校验成员路径 + 拒绝 link 成员）后再解包。

### CF-32 eval 型 f-string（py3.13 失效）与 python_config 的 eval —— 已修

- `file_utils.next_available_file_name` 与 `print_utils.fstring`：弃用 `locals().update(kwargs)`（PEP 709 下静默失效）+ `shlex.quote` 方案，改为 `re.sub` 定位 `${...}` 占位符、在 `{"__builtins__": {}}` 受限命名空间内求值替换。两处实现统一。
- `python_config._eval_string`：`eval(...)` 前缀走 `_restricted_eval`（空 `__builtins__`）；单一整体插值与混合文本插值分流处理（`_is_single_whole_interpolation` / `_interpolate_mixed`），修复原贪婪正则在 `"${a} and ${b}"` 上的 SyntaxError，同时消除任意代码执行面。

### CF-40 to_state_dict 主路径返回 None —— 已修

`worldfoundry/core/utils/torch_utils.py`：`_to_state_dict` 重写，实现 `state_dict()` 的对象恒返回其 state dict（原代码 unwrap 分支后落空返回 None）。

---

## P2

### CF-5 `_patch_relative_imports` 替换 builtins.__import__ 非线程安全 —— 已修（最小）

`lazy_config/config.py`：模块级 `threading.RLock` 串行化补丁的安装/恢复对，消除嵌套/并发交错导致 `__import__` 被永久污染的窗口；注释说明加载期间其它线程 import 会路过补丁函数但非 config 包前缀一律委托原始 `__import__`。
Deferred：sys.meta_path 自定义 Finder/Loader 方案（不动 builtins），改动面大、需要对 relative-import 配置样本回归。

### CF-6 hydra.py assert 校验 + get_config_module 只 log 不 raise —— 已修（部分）

- `config_from_dict`：两处 `assert` 改 `TypeError`/`ValueError`（`python -O` 下不失效）。
- `get_config_module`：非 `.py` 输入直接 `raise ValueError`（原来 log.error 后继续算出错误模块名）；`.replace(".py", "")` 改为仅剥尾缀（修中段 ".py" 被误删）。调用面 rg 确认仅 lyra_2 一处，传的是 .py 路径。
- Deferred：ConfigStore 固定节点名 "config" 的并发污染（uuid 节点名/局部 Hydra 实例）。理由：GlobalHydra 已初始化分支的行为依赖进程内 Hydra 状态，本机无 hydra 无法回归；方案：节点名带 uuid + compose 后清理，配合并发用例验证。
- 验证：py_compile + AST 断言全文件无 assert；本机无 hydra，无法运行时验证（逻辑等价替换）。

### CF-7 flags.py env 前缀不统一 + TRAINING/SMOKE 死常量 —— 已修

- `_env_bool` 先读 `WORLDFOUNDRY_<NAME>` 再回退 `COSMOS_<NAME>`；模块 docstring 写明"import 时快照"语义。
- 删除 `TRAINING`/`SMOKE`（rg 全仓确认零消费方，`configuration/__init__` 也未 re-export）。
- 验证：importlib 按路径加载模块，断言两种前缀、优先级（WORLDFOUNDRY_ 覆盖 COSMOS_）、死常量移除。

### CF-8 cosmos_config.py 顶层 try-import megatron.core —— 已修

megatron 导入下沉到 `_default_model_parallel()` 工厂（首次构造 `Config` 时解析）；`model_parallel` 字段注解改 `Any` + 注释。fallback attrs 类保留（更名 `_FallbackModelParallelConfig`）。import `worldfoundry.core.configuration` 不再连带 megatron/torch 栈。有 megatron 环境下首次 `Config()` 得到与原来相同的真实 `ModelParallelConfig` 实例。
验证：stub 加载后 `Config(model=None).model_parallel` 为 fallback 且未触发 megatron 导入。

### CF-11 configuration 多套体系并存无文档 —— 已修（文档）+ deferred

`configuration/__init__.py` docstring 写明 4+1 套体系各自适用场景与来源，并显著标注两个 `ModelConfig` 重名不同物（architecture vs 下载/放置）。
Deferred：重命名其一（`worldfoundry.core.ModelConfig` 是公共导出，改名属 API 断裂）；长期收敛到 lazy_config+attrs 两套。

### CF-12 TypedRegistry 无并发保护 —— 已修

`worldfoundry/core/registry.py`：`register` 的查重+写入置于 `threading.Lock` 内（`__init__` 中先建锁再注册构造参数）；`UnknownRegistryKeyError` 消息附 `difflib.get_close_matches` 的 "did you mean" 提示；`get` 收敛为 `get_item().value` 消除重复解析逻辑。
验证：8 线程并发注册同 key 恰好 1 家胜出、其余 `DuplicateRegistryKeyError`；近似键报错含建议。

### CF-16 `WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD` 只认 true/false —— 已修

`parse_skip_download` 接受 `1/true/yes/on / 0/false/no/off`（大小写不敏感），无法解析时 `raise ValueError`（不再静默落 None 继续联网）。

### CF-17 `reset_local_model_path` 环境变量覆盖显式实参 —— 已修

改为仅 `self.local_model_path is None` 时才用 `local_model_root_path()` 填充（该函数内部本就消费 `WORLDFOUNDRY_MODEL_DIR`），恢复"显式实参 > env > 默认"约定。

### CF-18 model.py 顶层 import transformers + vram_limit=80 魔数 —— 已修

- transformers 惰性化：`is_deepspeed_zero3_enabled` 改为惰性代理（transformers 缺失时返回 False——ZeRO-3 路径本就只能经 transformers 集成到达）；`ContextManagers` 用 `contextlib.ExitStack` 等价替换，彻底移除该依赖。无 transformers 环境下模块可导入、`load_model` 纯 torch 路径可用（原来直接 ImportError，行为严格变好）。
- `load_model_with_disk_offload` 的 `vram_limit=80` 改 `_disk_offload_vram_limit(device)`：CUDA 设备取 `get_device_properties().total_memory` 的 90%，探测失败回退历史值 80。修复 40G/24G 卡上 "used<80 恒真→常驻直至 OOM" 的预算失真。80G 卡行为变化仅在 72–80GiB 已用区间（更保守，方向安全）。
- 验证：子进程断言导入后 `transformers not in sys.modules`；`load_model` 以 `state_dict` 路径加载 `nn.Linear` 权重正确、eval 模式；`_disk_offload_vram_limit("cpu") == 80.0`。

### CF-22 `_load_checkpoint_from_local` 放弃 safetensors mmap —— 已修

`.safetensors` 分支改用 `load_safetensors_file`（mmap 零拷贝，支持 device 参数），不再 `f.read()` 全量进内存。

### CF-23 `_parallel_hf_hub_download_shards` fork 进程池 —— 已修

`ProcessPoolExecutor` → `ThreadPoolExecutor`（下载为 IO-bound，hf_hub 网络 IO 释放 GIL；消除 CUDA 初始化后 fork 的未定义行为）。

### CF-24 S3 credential 相对路径随 CWD 漂移 —— 已修

默认凭证路径尊重 `WORLDFOUNDRY_S3_CREDENTIALS` 环境变量，未设置时保留原相对路径默认值以兼容既有部署（文档注明）。OMNIDREAMS 前缀清理限于内部常量命名，未改公共 API。

### CF-25 `_validated_tensor_state_dict` 与 safe_loading 重复实现 —— 已修

load.py 侧委托 `safe_loading.tensor_state_dict`，wrapper-key 列表单点维护。

### CF-26 dcp.py 依赖 torch 私有符号 + `_derived_version` 全局改写 —— 已修（点状）+ deferred

- ImportError fallback 分支加模块级 `log.warning`（明示 `dcp_allow_mismatched_size` 与旧 key 布局兼容已静默失效）。
- `_version._derived_version` 改写段置于模块级 `threading.Lock` 内，并发加载不再互踩。
- 顺带：顶层 `from worldfoundry.core.io.s3_filesystem import S3StorageReader`（拉 boto3）下沉到 `get_storage_reader` 分支内 + `TYPE_CHECKING` 注解，本地 checkpoint 部署不再要求 boto3（与 CF-28 同型）。
- Deferred：私有 API 的 CI 哨兵测试（需要多 torch 版本矩阵，超出本次范围）。
- 验证：最小环境（无 boto3）导入成功；py_compile。

### CF-28 checkpoint/load.py 顶层硬依赖 loguru/huggingface_hub —— 已修

`logger` 改 `worldfoundry.core.logging_setup.get_logger(__name__)`（loguru 缺失自动走 stdlib fallback）；`hf_hub_download`/`try_to_load_from_cache` 及 S3 读写器导入全部下沉到使用点函数内。最小环境加载本地 checkpoint 不再 ImportError（本机即为此环境，全部单测在无 loguru/hf_hub/boto3 下通过）。

### CF-33 五处下载落地实现质量不一 —— 点修 + deferred

- 点修 `io/artifacts.download_file_from_url`：加 `timeout=60`（原来无超时可永久挂死）；失败/中断删除半截文件（调用方以 `os.path.exists` 判断跳过下载，半截文件会永久毒化——与 CF-21 同根）；`print` 改 `logging.warning`；保留"失败返回 None"契约（调用方 rg 确认忽略返回值、依赖文件存在性）；docstring 指向 `download_to_cache` 为新代码首选。
- Deferred：五处实现收敛到 `download_to_cache(checksum=...)` 唯一原语。方案：`cache._populate`、`load.py` 缓存写已在本次修复中对齐原子语义，剩余收敛为纯重构，建议单独 PR 逐链路迁移 + 各自回归。

### CF-34 merge_video_audio 吞自己抛的异常 —— 已修

- `io/video_data.py`：删除外层 `except Exception` 吞错；临时 mux 输出改 `finally` 清理（成功路径已 move、清理为 no-op；失败路径原视频保持原样）；失败向上冒泡。`save_video_with_audio` docstring 同步。
- `io/wan_video_geometry.py`（vendored Wan）：`merge_video_audio` 同样改冒泡 + finally 清理；`save_video`/`save_image` 维持 vendored 吞错契约但日志从 `logging.info` 升到 `logging.error`（可见性）。
- 行为影响标注：调用方现在会在 ffmpeg 失败时收到异常而非"假成功"。这是评审判定的正确行为（评测产物静默缺音轨浪费整轮 GPU 时）。
- 验证：缺输入抛 `FileNotFoundError`（单测覆盖）；ffmpeg 失败冒泡 + temp 清理的用例写入单测但本机无 ffmpeg 自动 skip（逻辑为直线控制流）。

### CF-36 config_utils 第 4 套 instantiate + 多处点缺陷 —— 已修（点状）+ deferred

- 模块 docstring 标注 LEGACY/冻结，指向 lazy_config.instantiate 与 TypedRegistry。
- `print_config`：`cfg.pretty()`（omegaconf≥2.1 已删除，必然 AttributeError 的死函数）改 `OmegaConf.to_yaml(cfg, resolve=True)`。
- `_CLASS_REGISTRY` 静默覆盖：注册收口到 `_registry_put`，重复注册记 warning（不 raise——legacy 代码可能依赖重注册）。
- `raise RuntimeError(...) from e` 补异常链。
- `resource_file_path` 补 zip 安装场景 caveat 文档。
- Deferred：该体系向 lazy_config 收敛（结构性重构）。
- 验证：py_compile（模块依赖 hydra/tree/omegaconf，本机不可导入；改动均为等价替换/文档/日志）。

### CF-37 io 的 pickle 反序列化面 —— 文档化 + deferred

`load_serialized` docstring 显式写明 pickle/gz 分支的信任边界（任意代码执行；torch 分支 weights_only=True 无此暴露；`.gz` 假定 gzip+pickle 的语义局限）。
Deferred：`allow_pickle=False` 默认翻转。理由：`load_serialized` 是通用入口，仓库内存在合法 `.pkl` 加载路径（`load_model_loader_registry` 等），默认翻转会当场破坏它们；方案：加 `allow_pickle` 参数默认 True → 调用面逐个显式声明 → 翻转默认，分两个版本完成。

### CF-38 easy_io.exists 吞掉一切异常返回 False —— 已修（保守）

明确缺失类异常（FileNotFoundError/IsADirectoryError/NotADirectoryError）静默返回 False；其余异常（认证/网络/参数错误）记 warning 后仍返回 False（保留 vendored facade "never raise" 契约，但环境问题不再不可见）。
Deferred：让意外异常冒泡。理由：facade 服务 vendored cosmos 代码，"exists 恒不抛"可能被依赖；方案：观察 warning 日志一个周期，确认无正常路径命中后再收紧。

### CF-41 torch_utils 点缺陷（np.long / save_torch 参数反转 / PYTHONHASHSEED） —— 已修

- `random_derangement`：`np.long`（NumPy≥1.24 已删除）→ `np.int64`。
- `save_torch`：`(fpath, D)` 顺序时正确解包 payload（原实现把 1 元 tuple 存了进去）；参数个数异常显式 `ValueError`。
- `set_seed_everywhere` 的 `os.environ["PYTHONHASHSEED"]`：注释说明只影响其后启动的子进程、不影响当前进程 hash。

### CF-42 4 套注册表并存、3 套静默覆盖 —— 文档化 + deferred

`functional_utils.make_registry_metaclass`/`ClassRegistry` docstring 标注 LEGACY（vendored）+ 静默覆盖语义 + 指向 TypedRegistry；`config_utils._CLASS_REGISTRY` 已加重复告警（见 CF-36）。
Deferred：注册表收敛（涉及 vendored 调用方，属结构性重构）。

### CF-45 core/visualization 缺 `__init__.py` —— 已修

新建 `worldfoundry/core/visualization/__init__.py`（带 docstring），setuptools `find` 打包不再丢弃该子包。

### CF-46 safety 包 import 即拉起 torch —— 已修

- `guardrails.py`：`distributed.logging`（顶层 import torch）→ stdlib `logging.getLogger(__name__)`；brace-style 日志改 %-style；numpy 改 `TYPE_CHECKING` 导入；`run_safety_check` 的 fail-open 默认（无 safety model 时返回 safe）显式写入 docstring。
- `safety/__init__.py`：急切导入（`video_io` 连带 imageio）改为与 checkpoint/structures 相同的惰性导出表，`from worldfoundry.core.safety import GuardrailRunner` 不再拉 torch/imageio/numpy。
- 验证：子进程断言导入 GuardrailRunner 后 `torch`/`imageio` 均不在 `sys.modules`；空 runner 行为不变。

---

## P3（仅零风险项）

### CF-9 make_freezable 语义 + EMAConfig 漏装饰 —— 已修

- `EMAConfig` 补 `@make_freezable`（与同文件所有 config 一致；直接 `freeze()` 原来是 AttributeError，现为正常冻结）。
- `make_freezable` 幂等（`cls.__dict__` 标记，重复装饰 no-op，不再叠加 setattr 覆盖层）。
- docstring 写明浅冻结语义（容器字段内容仍可变）。
- 验证：stub 加载 cosmos_config，冻结递归/EMAConfig 冻结/幂等三项断言通过。

### CF-10 lazy_call ClassVar 误用 + get_default_params 死分支 —— 已修

模块级 `ClassVar` 注解移除；`get_default_params` 对非 callable（字符串 target）直接返回 `{}` 并以 docstring 写明（原 else 分支取 `str.__init__` 签名，行为等价但有误导性）。

### CF-13 log_filters import 时装全局 filter —— 无改动（记录）

评审认定"已文档化的例外、副作用幂等且良性"，建议档位为"保留亦可"。为不引入行为变化，保持现状。若后续要求并入 `configure_logging()` opt-in，需通知依赖 import 即生效的现有部署。

### CF-14 logging_setup `_CONFIGURED` 无锁 + `_parse_bytes` 边角 —— 已修

- `configure_logging` 全程置于模块级 `threading.Lock` 内（幂等检查与 sink 手术原子化，并发首调不再重复配置）。
- `_parse_bytes("mb")`（无数字裸单位）直接返回 default，不再乘单位因子。
- 验证：单测 `_parse_bytes` 四种输入 + 6 线程并发 configure 无异常。

### CF-19 dataclass 注解与 None 默认不符 —— 已修

`model_loading/config.py` 的 `path/model_id/origin_file_pattern/download_source/local_model_path/skip_download/state_dict` 补 `Optional[...]`。两个 `ModelConfig` 重名的处置见 CF-11（文档化 + rename deferred）。

### CF-20 LightX2VLoRALoader 覆写返回类型 + 全量复制 state_dict —— deferred

理由：`get_name_dict` 返回类型拆分是 API 变更，需审计按父类契约调用的代码；`load()` 改 in-place `parameter.add_()` 涉及 14B 级模型的 GPU 显存行为，本机无法做等价性验证（评审自身也指出正确路径是 `merge_rank_scaled_lora_` 风格）。方案：`get_name_dict` 拆 `get_pairs`/`get_diffs` 两方法并保留旧名过渡；`load()` 改走 `merge_rank_scaled_lora_` 的逐参数原位路径，配 GPU 冒烟验证后合入。

### CF-27 zstd 子进程 `-T` 分离传参 + stderr 死锁窗口 —— 已修

`checkpoint/sharded_safetensors._load_shard`：`["-T", "4"]`（被 zstd 当文件参数）→ `-T4` 连写；`stdout.read()+wait()` → `communicate()`（并发排空双管道，消除 stderr 缓冲填满互锁）；zstd 二进制缺失捕获 `FileNotFoundError` 转带安装提示的 RuntimeError；顺带消除 BytesIO 二次拷贝（多 GB 解压 payload 少一份内存峰值）。
验证：本机有 zstd —— `num_threads=2` 的压缩分片 roundtrip 通过（旧代码此路径必失败）；缺 zstd 二进制的报错路径以 monkeypatch 覆盖。

### CF-39 copy_uri 整读内存 / f_remove 无声吞错 / fps=17 魔数 —— 已修（前两项）+ deferred（fps）

- `io/storage.copy_uri`：远端传输改 `open_uri` 双端句柄 + `shutil.copyfileobj`（16MiB 分块流式），数十 GB checkpoint 不再整体进内存（此函数正是 `cache._populate` 底层）；`local_path_for_uri` 同型修复。local→local 的 `shutil.copy` 路径不变。
- `file_utils.f_remove`：删除失败从完全无声改为 `logging.warning`（保留"不抛"语义）。
- Deferred：`_dump_video` 默认 `fps=17` 改必填/24 —— 是行为变更，依赖该默认值的调用方（某模型链路特有值）需先 rg 审计；方案：先加 DeprecationWarning 再改默认。
- 验证：copy_uri/local_path_for_uri 本地 roundtrip 单测；缓存原子性测试全量复跑通过（copy_uri 是其底层）。

### CF-43 parallel_execution 泄漏 pool + 命名误导 —— 已修（点状）

- 异常路径 `terminate()+join()`（原来 action 抛错即泄漏整个 pool）；成功路径行为不变。
- docstring 显著标注"线程池而非进程池"（GIL 提示、指向 `parallel_processes`）及 `async_return=True` 的所有权约定。
- Deferred：`ProcessThreadPool` 别名与 `num_processes` 参数改名（公共 API 断裂）；统一到 concurrent.futures。
- 验证：正常 map、异常冒泡、async_return 手动 close 三用例通过。

### CF-44 随机种子 API 三套并存 / cuda_graph 私有符号 —— 文档化 + deferred

`set_random_seed`/`fix_random_seeds` docstring 标注为 `set_seed_everywhere`（canonical）的包装/子集，注明 `by_rank` 在进程组不可用时静默回退。
Deferred:收敛与 deprecate（公共 API）；cuda_graph 的 `torch._C._graph_pool_handle` 防护属计算半区语境（该文件在 utils/ 但服务 CUDA graph 执行路径），留待计算半区评审的修复轮次统筹。

### CF-47 structures 导出表漂移 + tree_utils 吞异常 —— 已修

- `structures/__init__._EXPORT_MODULES` 补 `Float`/`OneOf`。
- `tree_value_at_path`/`tree_assign_at_path` 的 `raise ValueError` 补 `from e`。
- 防回归：单测断言导出表每项可解析、公开名与 `validator.__all__` 一致（私有哨兵 `_UNSET` 除外）。

### CF-48 prompting.py 顶层 import torch —— 已修

`@torch.no_grad()` 装饰器改方法体内 `with torch.no_grad():`（函数内惰性 import torch），模块与 docstring 声称的 framework-neutral 一致；经 `core/__init__` 导出表访问 `PromptProcessor` 不再拉起 torch。

---

## Deferred 汇总（原因 + 方案）

| 条目 | 内容 | 原因 | 方案 |
| --- | --- | --- | --- |
| CF-3 | 移除 OmegaConf.to_object 全局补丁 | vendored 调用方语义依赖未审计 | env 开关 + CI 关闭态回归 + 分版本移除 |
| CF-5 | meta_path Finder 替代 __import__ 补丁 | 改动面大，需配置样本回归 | 独立 PR，锁方案已消除急性风险 |
| CF-6 | ConfigStore 固定节点名并发污染 | 依赖 GlobalHydra 进程状态，本机无 hydra 无法回归 | uuid 节点名 + 并发用例 |
| CF-11/19 | 两个 ModelConfig 重名收敛 | 公共导出改名属 API 断裂 | 文档已加；DiffusionModelConfig/LoaderModelConfig 改名走 deprecation |
| CF-20 | LoRA loader 返回类型 + 全量复制 | GPU 行为无法本机验证，API 变更需调用方审计 | in-place 路径 + GPU 冒烟 |
| CF-21(子项) | 缓存 etag/size 元数据校验陈旧缓存 | 需跨 3 链路设计 sidecar 格式 | 并入 CF-33 收敛 |
| CF-26(子项) | torch 私有 API 的 CI 哨兵 | 需多 torch 版本矩阵 | 单独测试基建任务 |
| CF-33 | 5 处下载实现收敛到 download_to_cache | 结构性重构 | 逐链路迁移 + 各自回归（急性缺陷已点修） |
| CF-36/42 | instantiate/注册表体系收敛 | 结构性重构，涉及 vendored 调用方 | legacy 已标注冻结；新代码有明确指向 |
| CF-37 | pickle 分支 allow_pickle 默认翻转 | 会破坏仓库内合法 .pkl 加载 | 参数先行（默认 True）→ 调用面显式化 → 翻转 |
| CF-38 | exists 意外异常冒泡 | vendored facade "never raise" 契约可能被依赖 | 已加 warning 观察，后续收紧 |
| CF-39(子项) | _dump_video fps=17 默认改动 | 行为变更需调用方审计 | DeprecationWarning 过渡 |
| CF-43(子项) | 并行工具改名/统一 concurrent.futures | 公共 API 断裂 | 文档已加；统一走 deprecation |
| CF-44 | 种子 API 收敛；cuda_graph 私有符号防护 | 公共 API；cuda_graph 属计算半区语境 | deprecate 转发；计算半区轮次统筹 |
| CF-13 | log_filters 并入 configure_logging opt-in | 评审认可现状（documented exception），改动反而影响现有部署 | 维持现状 |

## 验证汇总

- 逐文件 `python -m py_compile`：全部通过（每次修改后即时执行）。
- 模块导入冒烟（最小环境，无 loguru/hf_hub/boto3/omegaconf/imageio）：`checkpoint.load`、`checkpoint.dcp`、`checkpoint.sharded_safetensors`、`io.*`（cache/serialization/easy_io/storage/video_data/artifacts）、`model_loading.model`、`registry`、`safety`、`logging_setup`、`utils.*` 均可导入；configuration 包依赖 omegaconf 无法整包导入（本机环境限制，非本次改动引入——改动前同样不可导入），flags/cosmos_config/hydra 以文件路径加载/AST 方式验证。
- 新增单测：`PYTHONPATH=. python -m pytest test/test_core_foundation_fix_cache_atomicity.py test/test_core_foundation_fix_point_defects.py -q -p no:cacheprovider` → **31 passed, 1 skipped**（skip 为 ffmpeg 缺失的环境性跳过）。
- 评测框架导入链门禁：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q -p no:cacheprovider` → **1507 tests collected**，exit code 0，无收集错误（未新增破坏）。
- 删除符号复查：`flags.TRAINING`/`flags.SMOKE` rg 全仓零引用后删除；其余修复未删除任何公共符号。
