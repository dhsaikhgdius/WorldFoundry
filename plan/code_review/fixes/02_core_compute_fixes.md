# 02 core 计算层修复日志

- 评审报告：`plan/code_review/02_core_compute.md`（CC-01～CC-38）
- 修复日期：2026-08-14
- 约束：只改授权目录（attention/、distributed/、vram/、device.py、inference.py、spatial_warp.py 等）；不装新依赖；数值行为变化显式标注；没把握一律 deferred
- 新增测试：`test/test_core_compute_fix_attention_env.py`、`test_core_compute_fix_dist_init.py`、`test_core_compute_fix_misc.py`、`test_core_compute_fix_xformers_mask.py`（共 47 条，纯 CPU，全部通过）

---

## 一、已修复清单

### CC-12 (P0) dist_init 失败静默伪装单进程
- 文件：`worldfoundry/core/distributed/generic_collectives.py`
- 改动：`dist_init()` 失败时先检查分布式启动指示环境变量（`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`TORCHELASTIC_RUN_ID`，刻意不含 `MASTER_PORT`——多个工具会预设默认端口）。有指示 → `logger.error` 后 re-raise，禁止把失败 rank 伪装成 RANK=0/WORLD_SIZE=1（那会跳过所有 collective、写 rank0 输出路径、让其余 rank 挂到 NCCL 超时——数据损坏路径）；无指示（裸 `python script.py`）→ 保留单进程 fallback，`print` 改为 `logger.warning`。另补充 `init_process_group` 返回后二次确认 `is_initialized()`。
- 数值行为影响：无（控制流语义修复）。
- 验证：`test_core_compute_fix_dist_init.py` 8 条测试覆盖两种路径（mock `init_process_group` 抛异常）：有指示 raise 且不改写 env、无指示 fallback 且写入 RANK=0/WORLD_SIZE=1/LOCAL_RANK=0、已初始化时直接返回、`MASTER_PORT` 单独存在不算指示。全部通过。

### CC-31 (P1) WORLDFOUNDRY_ATTENTION_BACKEND 两套词汇表崩溃
- 文件：`worldfoundry/core/inference.py`（`_normalize_attention_backend`）
- 改动：SDPA 策略解析器现在同时接受两套词汇。SDPA 策略词（auto/flash/cudnn/efficient/math 及别名 sdpa/default/mem_efficient/flash_attention 等）直接归一；dispatch 层词汇（flash_attention_2/3、sage_attention、xformers 等，经 `worldfoundry.core.attention.backends.normalize_attention_backend` 校验）不再 `ValueError` 崩溃，而是解析为 SDPA 策略 `auto` 并打一次 `DeprecationWarning`，明示该值由 attention dispatch 层负责兑现；真正未知的取值仍 raise（报错消息列出两套合法词汇）。torch 不可用时不因 import 失败误报。
- 数值行为影响：无（此前是崩溃，现在合法运行；SDPA 策略取 auto 与不设该变量时一致）。
- 验证：`test_core_compute_fix_attention_env.py` 覆盖：SDPA 策略词直通、dispatch 词不崩＋DeprecationWarning＋解析为 auto、未知词 raise、别名归一、大小写/连字符归一。GPU 冒烟：`WORLDFOUNDRY_ATTENTION_BACKEND=flash_attention_2` 下 import dispatch + `install_worldfoundry_inference_infra` 全链路不崩。

### CC-06 (P1) 显式请求后端不可用时静默降级
- 文件：`worldfoundry/core/attention/backends.py`（`resolve_attention_backend`）
- 改动：显式请求的后端不可用 → 降级到 torch 前打 `logger.warning`，格式"请求 X 实际用 Y 及原因（capability.reason）"，按 (requested, resolved, reason) 去重只打一次；`flash_attention_auto` 路径把 FA3/FA2 两个原因拼接后同样告警。
- 数值行为影响：无（仅新增日志，降级行为本身不变）。
- 验证：CPU 单测断言 caplog 内容；GPU 冒烟实测 `resolve_attention_backend('sage_attention')` → `torch` 并输出 `Reason: sageattention is not installed`。

### CC-07 (P1) dispatch.py import 时初始化 CUDA
- 文件：`worldfoundry/core/attention/dispatch.py`
- 改动：删除 import 时求值的 `_CAPABILITIES` 与 `FLASH_ATTN_3/2_AVAILABLE`、`SAGE_ATTN_AVAILABLE`、`XFORMERS_AVAILABLE` 模块常量，改为模块级 `__getattr__` 惰性求值（每次访问走 `probe_attention_backends()`，其内部已按 runtime 签名缓存），import 阶段不再触碰 `torch.cuda.get_device_capability`。
- 数值行为影响：无。副作用变化：import 不再在 GPU0 上创建 CUDA context（fork worker、`set_device` 前 import 均安全）。
- 验证：GPU 机器实测 `import worldfoundry.core.attention.dispatch` 后 `torch.cuda.is_initialized() == False`；访问 `dispatch.FLASH_ATTN_2_AVAILABLE` 时才惰性探测。

### CC-05 (P1) auto 链只选 torch
- 文件：`worldfoundry/core/attention/backends.py`
- 结论：查证 docstring/模块文档后确认"auto 只解析到 in-tree PyTorch SDPA"是**声明的设计意图**（no-external-repo contract，外部 FA/sage/xformers 一律显式 opt-in），`_DEFAULT_PRIORITY` 只含 torch 并非笔误；报告中矛盾来自 `attention_backend_report()` 的展示排序被误读为调度链。
- 改动：按用户指令走"意图已声明"分支的保守侧——不改优先级链；在 auto 解析到 torch 且存在可用外部后端时打一次 `logger.info`"检测到更快后端可用但未启用，设 WORLDFOUNDRY_ATTENTION_BACKEND=<name> 显式启用"；docstring 明确 auto 语义、`attention_backend_report()` 标注"展示排序≠自动调度链"。
- 数值行为影响：无（默认后端不变）。改变默认链需 owner 决策 → 见 deferred。
- 验证：GPU 冒烟实测 auto → torch 且 info 日志列出 `flash_attention_2`；CPU 单测覆盖日志去重。

### CC-32 (P1) 全局 monkey-patch F.scaled_dot_product_attention 不可逆
- 文件：`worldfoundry/core/inference.py`
- 改动：不改默认行为（默认仍安装 patch，`WORLDFOUNDRY_PATCH_SDPA=0` 可关）。新增：patch 安装时保存原函数（`_ORIGINAL_SDPA`）并打 `logger.info`（明示进程级影响与关闭/恢复方式）；`_unpatch_torch_sdpa()`；`uninstall_worldfoundry_inference_infra()`（恢复原 SDPA + 安装前捕获的 matmul precision/TF32 快照）；`worldfoundry_inference_infra_disabled()` context manager（作用域内临时恢复原函数，退出后重新安装）。patch 函数带 `_worldfoundry_core_sdpa` 标记保证幂等与可识别。
- 数值行为影响：默认路径无变化。`uninstall`/context manager 是新增 opt-in 入口，调用后 SDPA 语义回到 torch 原生（fully-masked-row 归一化包装移除）——在 API docstring 与日志中已标注。
- 验证：CPU 单测覆盖 install→patched→uninstall→原函数恢复、TF32 快照回滚；GPU 冒烟：patch 后 forward 正常、context manager 内恢复原函数且输出 diff=0、uninstall 后 `F.scaled_dot_product_attention is orig` 且 TF32 标志回滚。

### CC-13 (P1) torch.cuda.set_stream monkey-patch 全局副作用 + 线程不安全
- 文件：`worldfoundry/core/distributed/sequence_parallel/cuda_utils.py`
- 改动：跟踪缓存从模块级单值改为 `threading.local`（PyTorch current stream 本就是 per-thread 状态，进程级单值会跨线程串流；新线程回退 `torch.cuda.current_stream()`，与未打 patch 行为一致）；patch 逻辑收敛为 `install_torch_set_stream_patch()`（幂等、带 `_worldfoundry_sp_set_stream_patch` 标记、安装时打 info 日志）＋`restore_torch_set_stream()`＋`torch_set_stream_patch_removed()` context manager。模块 import 时仍安装（历史默认，sequence-parallel pynccl 依赖）。
- 数值行为影响：无。行为修正：多线程下 `current_stream_fast()` 不再串别的线程设置的流（此前是错误行为）。
- 验证：CPU 单测覆盖幂等安装、restore、context manager 恢复；import 校验日志输出正确指向本模块的 restore 函数。

### CC-19 (P2) builtins.print 全局替换不可逆
- 文件：`worldfoundry/core/distributed/metric_sync.py`
- 改动：`setup_for_distributed` 保存原 `builtins.print`（`_ORIGINAL_BUILTINS_PRINT`），重复 setup 时基于原函数重包（不再链式嵌套导致时间戳翻倍/master 标志陈旧）；patch 带 `_worldfoundry_rank_filtered_print` 标记并打 info 日志；新增 `restore_builtins_print()` 与 `builtins_print_unpatched()` context manager。world_size>8 全员打印的历史怪癖**保留原样**并在 docstring 标注（MAE 系历史行为，贸然改会改变大规模作业日志量，owner 决策）。
- 数值行为影响：无。
- 验证：CPU 单测覆盖 patch/restore/重复 setup 不嵌套。

### CC-33 (P2) TF32 配置写错属性路径
- 文件：`worldfoundry/core/inference.py`（`_configure_torch_backends`）
- 改动：改为 torch 2.7 验证过的正确路径 `torch.backends.cuda.matmul.allow_tf32` 与 `torch.backends.cudnn.allow_tf32`（原 `setattr(torch.backends.cuda, "allow_tf32", ...)` 只创建死属性）。特例：`matmul_precision="highest"` 时即使 `enable_tf32=True` 也不强开 matmul TF32（显式全精度请求优先，避免静默降级），打 info 说明。
- 数值行为影响：**有，已在代码注释与运行日志显式标注**——(a) `enable_tf32=False` 现在真正关闭 matmul TF32（此前默认 `matmul_precision="high"` 下静默保持开启）；(b) 翻转发生时打 `logger.info("Numerical-behavior impact: torch.backends.cuda.matmul.allow_tf32 changed X -> Y")`。默认参数（enable_tf32=True + high）下与旧行为等效（TF32 开）。
- 验证：CPU 单测断言属性翻转与恢复；GPU 冒烟观察到 `changed True -> False` 日志与 uninstall 后回滚。

### CC-25 (P1) vram/memory.py import 时初始化 CUDA 并固化 gpu 常量
- 文件：`worldfoundry/core/vram/memory.py`
- 改动：`gpu = torch.device(f"cuda:{torch.cuda.current_device()}")` 模块常量改为 `_default_gpu_device()` + 模块级 `__getattr__("gpu")` 惰性解析且不固化——每次访问反映当前 `torch.cuda.current_device()`（`set_device` 之后 import 的旧行为会把所有 rank 固化到 cuda:0）。CUDA 不可用时返回 CPU 设备。
- 数值行为影响：无。副作用变化：import 不再创建 CUDA context；多卡进程 `gpu` 现在指向正确的当前设备（此前指向 import 时刻设备——通常是错的 cuda:0）。
- 验证：CPU 单测（无 CUDA context 断言 + `gpu` 属性可访问）；`from worldfoundry.core.vram.memory import gpu` 兼容性确认。

### CC-26 (P2) DynamicSwapInstaller 重复安装破坏还原（部分修复）
- 文件：`worldfoundry/core/vram/memory.py`
- 改动：`_install_module` 幂等——`forge_backup_original_class` 已存在时直接返回（重复安装会把 hacked class 备份为"原 class"，导致 uninstall 永远还原不回真身）。模块内 `print` 全部改 `logger.info`。
- 未改部分：`gpu_complete_modules` 保持强引用列表——这是 `unload_complete_models()` 的工作集（需要强引用把模型搬回 CPU），改弱引用会改变卸载语义；已在注释中标注调用方职责。见 deferred。
- 数值行为影响：无。
- 验证：CPU 单测覆盖幂等安装（二次安装后 backup class 仍为真实原 class）。

### CC-27 (P2) layerwise offload hook 无法移除
- 文件：`worldfoundry/core/vram/layerwise_offload.py`
- 改动：`LayerwiseOffloadHandle` 从 frozen dataclass 改为可变，持有 `(layer, state, hook_handles)`；新增 `disable()`——remove 全部 forward hook、把参数 data 还原为 CPU 主副本、清空 GPU/CPU 参数表与 next_state、摘除层与模型上的 offload 标记属性。`enable_layerwise_cpu_offload()` 现在把 hook 句柄存进 handle（此前 `RemovableHandle` 直接丢弃）。"already enabled" 的 handle 不持有 hook，`disable()` 返回 False。
- 数值行为影响：无（enable 路径逐字节不变；disable 是新增 API）。
- 验证：CPU 单测 + CUDA 条件测试（`disable()` 后 hook 移除、参数回 CPU、可再次 enable）。

### CC-36 (P2) Sparse3DCache 无容量上限（部分修复）
- 文件：`worldfoundry/core/spatial_warp.py`
- 改动：`Sparse3DCache.__init__` 新增 `max_entries: int | None = None`——默认 None 保持历史无上限行为（不改变现有调用方），设定后 FIFO 淘汰最旧帧；新增 `__len__`/`clear()`/`_evict_to_capacity()`；非法容量 raise ValueError。
- 未改部分：`retrieve` 每次全量 stack 的性能问题 → deferred（改增量结构影响检索数值路径，需 owner 验证）。
- 数值行为影响：默认参数下无；显式设 `max_entries` 后检索候选集变小（调用方 opt-in 的预期语义）。
- 验证：CPU 单测覆盖默认无上限、有界 FIFO 淘汰顺序、clear、非法参数。

### CC-09 (P2) xformers bool mask 被静默转成 0/1 加性 bias
- 文件：`worldfoundry/core/attention/model_backends.py`（`XFormersAttention.__call__`）
- 改动：bool mask 显式转 SDPA 语义的加性 bias——True→0.0、False→-inf（此前 bool 直接赋值进 float tensor 被 cast 成 1.0/0.0，掩码完全失效，被"掩"位置只是 logit-1）；顺带修 padding 公式 `pad = (-mask.shape[-1]) % 8`（原公式在已对齐时多 pad 8 列）；保留"分配 8 对齐 buffer 再切回逻辑宽度"的 xformers 对齐要求，pad 列永不被读。
- 数值行为影响：**有，这是语义 bug 修复**——xformers 路径下带 bool mask 的注意力输出从"几乎无掩码"变为正确掩码结果（与 SDPA bool mask 语义一致）。已在代码注释标注。float mask 路径不变。
- 验证：`test_core_compute_fix_xformers_mask.py`（fake xformers 模块拦截 attn_bias）：bool mask → bias 形状/-inf 位置/0 位置精确断言、数值结果与 SDPA 参考一致、float mask 直通、对齐 pad 行为。本机未装 xformers，真实 kernel 路径无法冒烟（已在测试 docstring 标注）。

### CC-10 (P3) FlashAttention4 探测与实际 import 不一致
- 文件：`worldfoundry/core/attention/model_backends.py`
- 改动：`FLASH_ATTENTION_4` 分支的可用性探测从只查 `flash_attn` 改为同时探测 `flash_attn.cute`（实际 import 的子模块），避免装了 flash_attn 2.x 但无 cute 时报"可用"然后 ImportError。
- 数值行为影响：无。
- 验证：py_compile + import 校验（本机无 FA4，逻辑分支由单测的 fake module 覆盖）。

### CC-08 (P2) varlen FA2 请求失败静默回退（部分修复）
- 文件：`worldfoundry/core/attention/varlen.py`
- 改动：`flash_attention()` 与 `varlen_scaled_dot_product_attention()` 中 version=2 请求但 FA2 不可用时 `warnings.warn`（与 v3 路径对齐，此前只有 v3 有警告）；import 失败原因 `logger.debug` 留痕。
- 未改部分：热路径 `q_lens.max().item()` 的 D2H 同步 → deferred（FA 接口要求 python int，去同步需要缓存/预计算 max_seqlen 的接口改动）。
- 数值行为影响：无（仅日志）。
- 验证：GPU 冒烟 version=2 实际走 FA2 输出形状/dtype 正确；CPU 路径回退带警告由单测覆盖。

### CC-14 (P1) rank0_first：rank0 异常时其余 rank 永久挂死
- 文件：`worldfoundry/core/distributed/torch_process_group.py`
- 改动：rank0 的 `func` 异常被捕获后**仍然到达 barrier**（原实现 rank0 抛异常直接跳过 barrier，其余 rank 挂到 NCCL 超时）；barrier 后 `broadcast_object_list` 广播成功标志——rank0 重抛原异常，非 0 rank 收到失败标志后 raise RuntimeError（不再对半初始化状态继续跑 `func`）。非分布式路径行为不变。
- 数值行为影响：无（挂死/静默错误 → 显式失败）。
- 验证：CPU 单测覆盖非分布式路径异常直抛、成功路径返回值；分布式路径逻辑经代码走查（多 rank 行为无法单机验证，已标注）。

### CC-15 (P2) device_with_rank 用全局 rank 构造 cuda 设备号
- 文件：`worldfoundry/core/distributed/torch_process_group.py`
- 改动：改用 `LOCAL_RANK`（env 缺失时回退 `torch.cuda.current_device()`）构造 `cuda:<local_rank>`，多机下不再产出 `cuda:8+` 这类非法设备号。
- 数值行为影响：无（多机下原行为直接 crash 或错卡）。
- 验证：CPU 单测（monkeypatch env）覆盖 LOCAL_RANK 优先、缺失回退。

### CC-16 (P2) is_local_rank0 在 CUDA_VISIBLE_DEVICES 隔离下全员判真
- 文件：`worldfoundry/core/distributed/torch_process_group.py`
- 改动：优先读 `LOCAL_RANK` env（隔离启动器下每进程 `current_device()==0` 恒真），env 缺失再回退设备号判断。
- 数值行为影响：无（修复"每进程都自认 local rank0 抢写文件/重复下载"的竞态）。
- 验证：CPU 单测覆盖 LOCAL_RANK=0/非 0/缺失三态。

### CC-18 (P2) metric_sync.init_distributed 未绑定变量 + 无条件 set_device
- 文件：`worldfoundry/core/distributed/metric_sync.py`
- 改动：env 解析失败从 `NameError`（未绑定 `gpu`）改为带上下文的 `RuntimeError`（列出 RANK/WORLD_SIZE/LOCAL_RANK 或 SLURM 变量实际值）；`torch.cuda.set_device(gpu)` 加 `torch.cuda.is_available()` 守卫；rank/world_size/gpu 三者判空兜底 raise。
- 数值行为影响：无。
- 验证：CPU 单测覆盖 RANK 存在但 LOCAL_RANK 缺失 → RuntimeError（信息含变量值）。

### CC-20 (P2) multiprocess_launch：NCCL barrier 在 set_device 之前
- 文件：`worldfoundry/core/distributed/multiprocess_launch.py`
- 改动：`torch.cuda.set_device(local_rank)` 提前到 `init_process_group` 与 `object_collectives.synchronize()` 之前（原顺序所有本地进程都指着 cuda:0 做 barrier，NCCL 在单卡上建多通信器 → "Duplicate GPU detected" 或挂死）。设备数校验一并前移。
- 数值行为影响：无（原路径本身已损坏/挂死）。
- 验证：py_compile + import；多进程路径无法单机 CI 验证，代码走查确认顺序与 torchrun 惯例一致。

### CC-21 (P2) RankGenerator 错误路径引用未初始化的 self.order
- 文件：`worldfoundry/core/distributed/model_parallel_groups.py`
- 改动：校验循环内的错误消息引用局部 `order`（`self.order` 在循环后才赋值，原实现抛 AttributeError 掩盖真实报错）。
- 数值行为影响：无。
- 验证：CPU 单测触发该分支断言 RuntimeError 消息完整。

### CC-23 (P3) 无条件覆盖 NCCL 环境变量
- 文件：`worldfoundry/core/distributed/torch_process_group.py`
- 改动：`NCCL_*` 环境变量写入全部改 `os.environ.setdefault`，尊重用户/调度器已设值。
- 数值行为影响：无（通信参数默认值不变，仅不再覆盖显式配置）。
- 验证：CPU 单测覆盖已设值不被覆盖、未设值取默认。

### CC-28 (P2) fp8_linear 每次 forward 在 CPU 上创建 scale 再搬 GPU
- 文件：`worldfoundry/core/vram/layers.py`
- 改动：`scale_b = torch.ones((weight.shape[0], 1), device=device)` 直接在目标设备分配（原实现 CPU 建 tensor + `.to(device)`，每层每 forward 一次 H2D 拷贝＋潜在同步）。
- 数值行为影响：无（同值同 dtype，仅分配位置）。
- 验证：py_compile + import；数值等价性由构造直接保证（全 1 tensor）。

### CC-04 (P3) get_torch_device 失败路径用 print
- 文件：`worldfoundry/core/device.py`
- 改动：fallback 分支 `print` → `logger.warning`。
- 数值行为影响：无。
- 验证：py_compile + import。

---

## 二、Deferred 清单（含原因与方案）

### D-1. CC-01/CC-02 core 顶层 import 上层 pipelines / worldfoundry.runtime（结构性）
- 原因：跨半区改动（涉及 core/__init__.py 与上层包，均在本次授权边界外），且需要全仓 import 图重排。
- 方案：(1) 用 `worldfoundry.core.registry` 反转依赖——上层在自身 import 时向 core 注册实现，core 内部只留 protocol/接口；(2) 顶层 import 改函数内延迟 import 作为过渡；(3) 引入 import-linter（见 D-2）锁住方向。预计 2～3 个 PR，先 runtime 后 pipelines。

### D-2. import-linter 引入（结构性）
- 原因：需要新增 dev 依赖（本机 pypi 不可用）与 pyproject.toml 改动（授权边界外）。
- 方案：`pyproject.toml` 增加 `[tool.importlinter]`，contract 三条：core 不得 import worldfoundry.{pipelines,runtime,studio}；core.attention/distributed/vram 不得互相环依赖；device.py 不得被 configuration 层 import。CI 挂 `lint-imports` 步骤。落地前先跑一次基线豁免清单。

### D-3. CC-17 四套 dist_init / 三套 get_rank 收敛（结构性）
- 原因：调用方遍布全仓（training/pipelines/studio），收敛属跨半区行为变更，单机无法回归多机路径。
- 方案：以 `torch_process_group.init_torch_distributed` 为唯一入口（它已有 timeout/device_id/env 处理最完整）；`generic_collectives.dist_init`、`metric_sync.init_distributed`、`multiprocess_launch` 内部初始化改为薄包装并打 DeprecationWarning；rank 查询统一走 `torch_process_group.get_rank/get_local_rank`。分三步：先加包装与警告（无行为变化）→ 迁移仓内调用方 → 删旧入口。
- 本次已做的减害：三套入口各自的 P0/P1/P2 缺陷（CC-12/14/18/20）已就地修复，收敛前不再有数据损坏/挂死路径。

### D-4. SDPA/set_stream/builtins.print patch 默认改 opt-in（CC-32/13/19 的后续）
- 原因：默认关闭会改变依赖 patch 的现有 pipeline 行为（SDPA fully-masked-row 归一化、sequence-parallel 流跟踪、rank 过滤日志），需 owner 决策与全量回归。
- 方案：三个 patch 统一增加环境开关（`WORLDFOUNDRY_PATCH_SDPA` 已存在，补 `WORLDFOUNDRY_PATCH_SET_STREAM`、`WORLDFOUNDRY_PATCH_PRINT`），一个 release 周期内默认 on + 安装时 info 日志（本次已加日志与 restore API），下一周期翻默认为 off，迁移指南指向显式 `install_*()` 调用点（pipeline 入口统一调用）。
- 本次已落地的前置：原函数保存、restore 函数、context manager、幂等标记——翻开关时无需再动核心代码。

### D-5. CC-03 IS_CUDA_AVAILABLE import 时缓存 + NPU 全局配置副作用
- 原因：`IS_CUDA_AVAILABLE` 是被全仓 `from ... import` 的模块常量，改惰性（模块 `__getattr__`）后凡是 `from device import IS_CUDA_AVAILABLE` 的调用方拿到的仍是 import 时刻快照，半改反而制造两套语义；NPU `allow_internal_format=False` 移除影响 NPU 数值路径，无 NPU 硬件可验。
- 方案：新增 `is_cuda_available()` 函数入口 → 全仓调用方批量迁移 → 常量退役打 DeprecationWarning。NPU 配置副作用移到 `get_torch_device()` 首次 NPU 解析时执行。

### D-6. CC-22 destroy_model_parallel 只置 None 不销毁进程组
- 原因：正确做法要区分"销毁子组"与"销毁默认组"，并处理与 `dist.destroy_process_group()` 全局析构的交互；多组场景单机无法回归，改错会把作业收尾从"泄漏"变成"崩溃"。
- 方案：`destroy_model_parallel(destroy_groups: bool = False)` 增量参数，True 时对每个非默认组调 `dist.destroy_process_group(group)` 后置 None；配套多机冒烟脚本（2 节点建 TP/DP 组→destroy→重建）验证后再翻默认。

### D-7. CC-24 NativeAttention.set_context_parallel_group 修改 torch 全局 CP 选项不恢复
- 原因：P3；恢复语义需要明确"谁拥有 CP 生命周期"（模块级 vs 进程级），与 sequence_parallel 的收敛（D-3）耦合。
- 方案：保存 set 前的 `torch.distributed.tensor.experimental._attention` 相关全局值，提供 `clear_context_parallel_group()` 恢复；随 D-3 一起做。

### D-8. CC-08（后半）varlen 热路径 .item() 同步；CC-37 LitEma 每步 .item() + per-param 循环
- 原因：性能问题非正确性；FA varlen 接口要求 python int 的 max_seqlen，去同步需要调用方传入预计算值（接口变更）；LitEma 改 `torch._foreach_*` 批量化会改变浮点求和顺序（数值行为变化），需 owner 用训练曲线验证。
- 方案：varlen——`flash_attention(..., max_seqlen_q=None, max_seqlen_k=None)` 新增可选参数，调用方（packing 层已知长度）直传；LitEma——`_foreach_mul_ + _foreach_add_` 重写并在日志标注数值行为影响，跑 1k step EMA 权重 bitwise/容差对比后合入。

### D-9. CC-11 rope/kvcache 默认 device="cuda" 硬编码
- 原因：P3；改默认值影响所有未显式传 device 的调用方（当前在 CUDA 机器上行为正确），收益仅是 CPU-only 环境可用性。
- 方案：默认改 `None` → 内部 `get_current_torch_device()` 解析；随调用方审计一起做。

### D-10. CC-29/CC-30 vram_limit 每层 mem_get_info、AutoWrappedModule deepcopy、offload_to_disk for-else
- 原因：P3 性能/可读性；`mem_get_info` 节流需要选定失效策略（步进计数 vs 时间窗），deepcopy 移除需确认无调用方依赖副本语义；for-else 重写纯重构无行为收益，风险大于收益。
- 方案：mem_get_info 加 `WORLDFOUNDRY_VRAM_CHECK_INTERVAL`（默认 1=现行为）；deepcopy → 浅拷贝＋文档标注；for-else 改显式 matched 标志＋对未 offload 参数打 warning。

### D-11. CC-34 core 内嵌模型专属常量/绝对路径
- 原因：迁移目的地（configuration 层）在授权边界外。
- 方案：常量迁到 `worldfoundry/core/configuration/model_defaults.py`（另一 agent 半区）或各模型包，core 只留读取接口。

### D-12. CC-26（后半）gpu_complete_modules 强引用
- 原因：改弱引用会改变 `unload_complete_models()` 语义（弱引用死亡后无法搬回 CPU，静默泄漏 GPU 显存反而更糟）；现有强引用+显式 unload 是自洽的。
- 方案：保持强引用；补充 `unload_complete_models(clear=True)` 已有清空路径的调用方审计，必要时在 `load_model_as_complete` 打 debug 日志提示当前驻留数量。

### D-13. CC-36（后半）Sparse3DCache.retrieve 全量 stack
- 原因：改增量/预分配结构触碰检索数值路径（coverage 排序），需 owner 用真实轨迹回归。
- 方案：容量上限已落地（本次），retrieve 改为维护 padded 常驻 tensor + 有效计数，脏标记增量更新；对齐现实现输出做 bitwise 对比后合入。

### 无需处理
- CC-35 / CC-38：报告的正面确认项（realtime、kernels/nn/acceleration/video_postprocess 质量良好），无动作。

---

## 三、验证汇总

### 单元测试（纯 CPU，pytest -p no:cacheprovider）
```
test/test_core_compute_fix_attention_env.py   \
test/test_core_compute_fix_dist_init.py       \
test/test_core_compute_fix_misc.py            \
test/test_core_compute_fix_xformers_mask.py
→ 47 passed in 27.18s
```

### 编译与 import 校验
- 所有改动文件 `py_compile` 通过。
- 全部 16 个触碰模块 `PYTHONPATH=. python -c "import ..."` 一次性通过（device、inference、spatial_warp、attention.{backends,dispatch,model_backends,varlen}、distributed.{generic_collectives,metric_sync,torch_process_group,model_parallel_groups,multiprocess_launch,sequence_parallel.cuda_utils}、vram.{memory,layers,layerwise_offload}）。

### GPU 冒烟（单卡 A800，小 tensor，秒级）
- `WORLDFOUNDRY_ATTENTION_BACKEND=flash_attention_2`：dispatch 实际选 FA2，输出 vs SDPA 参考 max diff = 9.77e-4（bf16 容差内）；varlen version=2 走 FA2 输出形状/dtype 正确。
- `resolve_attention_backend('sage_attention')` → `torch`，warning 含"Reason: sageattention is not installed"（CC-06 实证）。
- `import worldfoundry.core.attention.dispatch` 后 `torch.cuda.is_initialized() == False`（CC-07 实证）。
- auto → torch，info 日志列出可用未启用后端 `flash_attention_2`（CC-05 实证）。
- SDPA patch：install → patched forward 正常；`worldfoundry_inference_infra_disabled()` 作用域内恢复原函数（输出 diff=0.0）；uninstall 后原函数与 TF32 标志完整回滚；TF32 翻转日志 `allow_tf32 changed True -> False` 出现（CC-32/33 实证）。

### 既有测试回归确认（非本次改动引入）
- `test/eval_core/test_attention_dispatch_safety.py` 5 个失败、`test/eval_core/test_core_primitives.py` 6 个失败：将改动文件逐一换回 `git show HEAD:` 版本复测，失败集合完全一致——均为工作区内其它进行中改动/测试漂移的既有问题，与本次修复无关。其余 27+ 条通过项在改动前后一致通过。
- xformers 本机未安装：CC-09 真实 kernel 路径无法冒烟，语义由 fake-module 单测锁定（bias 值精确断言 + SDPA 参考对比）。
