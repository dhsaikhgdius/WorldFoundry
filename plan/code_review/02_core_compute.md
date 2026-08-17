# core 计算层评审（attention/kernels/nn/distributed/memory/device/realtime）

> 评审人：infra 深度代码评审（自动化辅助）
> 日期：2026-08-14
> 状态：已完成（38 条发现：P0×1、P1×10、P2×17、P3×10，其中 2 条为正面确认）

## 评审范围与方法

**范围**（worldfoundry/core 的"计算/GPU"半区，共约 150 个文件、3.9 万行）：

- `worldfoundry/core/attention/`（43 文件，~12k 行）
- `worldfoundry/core/kernels/`（12 文件，~4.8k 行）
- `worldfoundry/core/nn/`（24 文件，~3.6k 行）
- `worldfoundry/core/acceleration/`（12 文件，~2.2k 行）
- `worldfoundry/core/memory/`（7 文件，~1.7k 行）
- `worldfoundry/core/vram/`（6 文件，~1.5k 行）
- `worldfoundry/core/distributed/`（43 文件，~7.5k 行）
- 顶层文件：`device.py`、`gradient.py`、`inference.py`（209KB）、`inference_state.py`、`realtime.py`、`realtime_timing.py`、`torchprofile.py`、`video_postprocess.py`、`video_postprocess_rtx.py`、`geometry.py`、`camera_conditioning.py`、`camera_pose.py`、`camera_trajectory.py`、`action_normalization.py`、`spatial_warp.py`、`world_explorer.py`

**方法**：

1. `rg` 全范围扫描反模式：硬编码 `"cuda"`/`.cuda()`、import 时 CUDA 初始化、bare except、热路径 `.item()/.cpu()` 同步、`torch.distributed` 使用、上层 import（分层违规）、全局可变状态。
2. `Read` 精读优先级文件：attention 后端选择链（backends/dispatch/model_backends）、distributed（进程组、collectives、SP/CP runtime）、device.py、vram/memory offload、acceleration（quantization/cuda graph）、realtime 路径。
3. 每条发现给出 `path:line` + 代码摘录证据。

**严重度定义**：P0=损坏/危险（bug、数据损坏、崩溃路径、死锁）；P1=严重设计缺陷；P2=应修复；P3=改进建议。

## 发现（按主题分组）

### 主题 A：分层与依赖方向

### [CC-01] P1 core 顶层 import 上层 pipelines（硬性分层违规）
- 位置：`worldfoundry/core/inference.py:34`
- 证据：

```python
from worldfoundry.pipelines.gen3c.constants import (
```

- 问题：`core` 是最底层，被 evaluation/pipelines/training 依赖；`inference.py` 却在模块顶层 import `worldfoundry.pipelines.gen3c.constants`。这是方向反转，且是顶层 import（非延迟），任何 `import worldfoundry.core.inference` 都会把 pipelines 包拉进来。
- 影响：形成 core→pipelines→core 的循环依赖隐患（pipelines 必然 import core）；破坏 core 的可独立测试性与最小依赖面；pipelines 的任何 import 副作用都会传染给所有 core 用户。
- 建议：把 gen3c 常量下沉到 core（或独立的 constants 模块），或在 pipelines 侧注入。core 内禁止出现 `worldfoundry.pipelines` 字样，加 import-linter 契约。

### [CC-02] P1 core 多处顶层 import worldfoundry.runtime（跨层依赖 + import 副作用）
- 位置：`worldfoundry/core/kernels/triton_group_norm_silu.py:18`、`worldfoundry/core/attention/triton_piecewise_attention.py:15`、`worldfoundry/core/attention/rope_kernel.py:39`、`worldfoundry/core/kernels/triton_diffusion.py:11`、`worldfoundry/core/acceleration/triton_nvfp4.py:7`
- 证据（triton_group_norm_silu.py:18）：

```python
from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache
```

- 问题：`runtime` 属于自研上层（与 core 平级或更高）。core 的 5 个 triton kernel 模块在**顶层** import 它，另有 7 处函数内延迟 import（`inference.py:4755`、`kernels/diffusion.py:215`、`kernels/autotune_cache.py:174`、`attention/piecewise.py:36` 等）。同一依赖两种引入方式并存，说明分层边界未定义清楚。
- 影响：core 无法脱离 runtime 单独发布/测试；如果 runtime 反向 import core（大概率），则形成真实的循环 import，靠 import 顺序侥幸不崩。
- 建议：`compile_cache` 是纯基础设施，应下沉进 core（如 `core/compile_cache.py`），runtime 层 re-export 兼容旧路径。

### 主题 B：设备管理（device.py）

### [CC-03] P2 import 时缓存 CUDA 可用性 + import 副作用修改 NPU 全局配置
- 位置：`worldfoundry/core/device.py:15-19`
- 证据：

```python
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available() and hasattr(torch, "npu") and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False
```

- 问题：三个副作用发生在 import 时：(1) `torch.cuda.is_available()` 在 fork-based multiprocessing 场景可能污染子进程（CUDA driver 初始化后 fork 不安全，除非设置 `PYTORCH_NVML_BASED_CUDA_CHECK=1`）；(2) 可用性被缓存为模块常量，之后修改 `CUDA_VISIBLE_DEVICES` 或延迟初始化的场景读到陈旧值；(3) 静默修改 `torch.npu.config.allow_internal_format` 全局配置，import 一个工具模块不应改变全局数值行为。
- 影响：`device.py` 被 core 各处 import，实际上让"import worldfoundry.core"就固化设备判定。同文件内 `get_current_torch_device()`（line 62）用的是实时 `torch.cuda.is_available()`，与缓存常量并存，行为不一致。
- 建议：把 `IS_CUDA_AVAILABLE` 改为惰性函数/`functools.lru_cache`，NPU 配置修改移到显式的 `setup_npu()` 入口。

### [CC-04] P3 get_torch_device 失败路径用 print 而非 logger
- 位置：`worldfoundry/core/device.py:38-42`
- 证据：

```python
    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda
```

- 问题：底层库直接 `print` 到 stdout；且 fallback 到 `torch.cuda` 的语义可疑——`get_device_type()` 只会返回 cuda/npu/cpu，`torch.cpu` 在新版本存在，但没有 `torch.cpu.current_device()` 之外的完整 API（如 `empty_cache` 是 no-op 但存在）。真正会触发这个分支的是 `cpu` 早期版本，fallback 到 `torch.cuda` 在无 GPU 机器上后续调用直接抛错。
- 影响：无 GPU 环境下 `synchronize()/empty_cache()` 可能沿此路径崩溃；日志不可控。
- 建议：换 `logging`，cpu 分支返回显式 no-op shim 或让调用方分支。

### 主题 C：attention 后端选择链

### [CC-05] P1 "auto" 后端实际永远只选 torch SDPA，与文档/报告 API 自相矛盾
- 位置：`worldfoundry/core/attention/backends.py:96-104,269-273`；`worldfoundry/core/attention/dispatch.py:345-354`
- 证据（backends.py）：

```python
_DEFAULT_PRIORITY = (_TORCH,)
_REPORT_PRIORITY = (
    "flash_attention_3",
    "flash_attention_2",
    "sage_attention",
    "xformers",
    _TORCH,
)
```

（dispatch.py）：

```python
def _auto_attention_backends(device: torch.device | str | None) -> tuple[str, ...]:
    """Use the in-tree exact PyTorch provider for automatic dispatch. ..."""
    del device
    return ()
```

- 问题：`resolve_attention_backend("auto")` 的 docstring 写"traverses priority list to select the first runnable package"，模块头注释写"Fallback Priority Chain ... flash-attn → ... → SDPA"，`attention_backend_report()` 也按 FA3>FA2>sage>xformers>torch 的"dispatch priority"输出报告；但真实的 auto 链是空元组/只有 torch——装了 flash-attn 也永远不会被 auto 使用。这是（注释里承认的）刻意决策（"no-external-repo contract"），但公开 API 的名字、docstring、报告输出与真实行为矛盾。
- 影响：用户装好 flash-attn 后以为默认加速已生效（报告还显示 FA3 usable），实际全程跑 SDPA，性能差数倍且极难发现；`_REPORT_PRIORITY` 的"dispatch priority"是假的。
- 建议：二选一：要么 auto 链真的按 report 优先级探测（推荐，探测已有 usable 判定），要么把 docstring/报告改为明确"auto=torch SDPA only，外部后端需显式指定"，并在启动日志打印 resolved 后端。

### [CC-06] P1 显式请求的后端不可用时静默降级到 torch，无任何日志
- 位置：`worldfoundry/core/attention/backends.py:279-282`；`worldfoundry/core/attention/dispatch.py:431-433`
- 证据（backends.py）：

```python
    capability = capabilities.get(requested)
    if capability is not None and capability.usable:
        return requested
    return _TORCH
```

- 问题：用户显式设置 `WORLDFOUNDRY_ATTENTION_BACKEND=flash_attention_2`，若探测判定不可用（包没装/算力不符），`resolve_attention_backend` 静默返回 "torch"。dispatch 侧运行期失败有 `_warn_backend_fallback` 警告，但**选择期**降级完全无声。
- 影响：拼错包名之外的所有"配置了但没生效"场景不可观测；生产训练/推理跑在慢路径上无人知晓。
- 建议：显式请求降级时 `warnings.warn` 或 logger.warning 一次（带 reason 字段，capability.reason 已经有现成文案）。

### [CC-07] P1 dispatch.py import 时初始化 CUDA（违反自身注释的设计意图）
- 位置：`worldfoundry/core/attention/dispatch.py:27-39`
- 证据：

```python
def initialize_attention_priority():
    # Keep the user's preference (usually ``auto``) unresolved until a tensor
    # device is known. Resolving at import time can permanently select CPU SDPA
    # before a worker calls torch.cuda.set_device().
    return attention_backend_from_env()

ATTENTION_IMPLEMENTATION = initialize_attention_priority()
_CAPABILITIES = probe_attention_backends()
```

- 问题：line 27-31 的注释刻意避免 import 时解析设备，但紧接着 line 35 `probe_attention_backends()` 内部调用 `torch.cuda.get_device_capability(device)`（backends.py:364），该调用触发 CUDA lazy init——在 **import 时**创建 CUDA context。派生的 `FLASH_ATTN_*_AVAILABLE` 模块常量也被永久固化。
- 影响：(1) fork 型 DataLoader/multiprocessing 在 import 后 fork 会撞 "CUDA initialization error"；(2) 在尚未 `torch.cuda.set_device()` 的 worker 里，context 建在 GPU0 上，造成多进程全部在 GPU0 留下 ~500MB context；(3) 与注释声明的意图直接矛盾。
- 建议：`_CAPABILITIES` 与 `FLASH_ATTN_*_AVAILABLE` 改为惰性（首次调用时探测）；probe 的 capability 查询在 `torch.cuda.is_initialized()` 为 False 时可用 NVML 或延后。

### [CC-08] P2 varlen.py：FA2 请求失败静默回退（v3 有警告、v2 没有）+ 热路径 .item() 同步
- 位置：`worldfoundry/core/attention/varlen.py:128-148,159-176`
- 证据：

```python
    elif version == 2:
        try:
            import flash_attn as flash_attn_module
        except Exception:
            pass
    ...
    if version == 3 and not use_fa3:
        warnings.warn(
            "FlashAttention 3 is unavailable on this GPU/runtime; falling back to PyTorch SDPA."
        )
```

- 问题：(1) `version==2` 且 flash_attn import 失败/不可用时没有对应警告，直接走 SDPA——与 v3 分支不对称的静默降级；(2) `except Exception: pass` 吞掉 flash_attn import 的所有错误（包括 ABI mismatch 的 ImportError 细节），无日志（同文件 368-373 行第二处相同模式）；(3) line 159-176 `int(q_lens.max().item())` 若 q_lens 在 GPU 上则每次调用强制 device sync，这是 attention 热路径。
- 影响：静默慢路径；热路径隐藏同步拖慢 pipeline（尤其 CUDA graph/异步流场景）。
- 建议：v2 分支补警告；import 失败记录 debug 日志；`max_seqlen` 让调用方传入或在 CPU 上计算 lengths。

### [CC-09] P2 xformers 掩码路径：bool mask 被静默转成 0/1 加性 bias，掩码失效
- 位置：`worldfoundry/core/attention/model_backends.py:112-126`
- 证据：

```python
            pad = 8 - mask.shape[-1] % 8
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )
            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
```

- 问题：(1) 若调用方传 bool 掩码（SDPA 路径合法），`mask_out[...] = mask` 把 True/False 静默 cast 成 1.0/0.0 的加性 bias，掩码几乎无效（被掩位置只是 -0 而非 -inf），且不报错——同一个 `MaskedAttentionFunction` 协议下 SDPA 与 xformers 语义不一致；(2) `pad = 8 - x % 8` 在整除时多 pad 8（浪费无害）；(3) 代码里留着 "doesn't this remove the padding again??" 的疑问注释——实际上 slice 回原宽度是 xformers 对齐 stride 的正规技巧，但注释表明作者自己不确定，需要写清楚。
- 影响：bool mask 调用者得到错误的 attention 结果（数值污染，无异常）；维护者无法信任该路径。
- 建议：入口检查 `mask.dtype == torch.bool` 则显式转 `masked_fill` 语义（0/-inf），并把疑问注释改为解释对齐技巧的正式注释。

### [CC-10] P3 FlashAttention4 的可用性探测与实际 import 不一致
- 位置：`worldfoundry/core/attention/model_backends.py:404-409,166-169`
- 证据：

```python
            case AttentionFunction.FLASH_ATTENTION_4:
                if not _module_available("flash_attn"):
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but `flash-attn-4` is not installed."
                    )
                return FlashAttention4()
```

- 问题：探测的是 `flash_attn`（FA2 的包名），但 `FlashAttention4.__call__` 实际 import `flash_attn.cute`（只有 FA4 构建才有）。装了 FA2 的环境探测通过、首次 forward 才报错，违背 to_callable "resolution 期 fail loudly" 的自述契约。
- 影响：错误延迟到 forward 期，训练启动数分钟后才崩。
- 建议：探测 `flash_attn.cute`。

### [CC-11] P3 rope.py / kvcache.py 函数签名默认 device 硬编码 cuda
- 位置：`worldfoundry/core/attention/rope.py:159,194,364,496`；`worldfoundry/core/attention/kvcache.py:99`
- 证据（rope.py:159）：

```python
    device: torch.device = torch.device("cuda"),
```

- 问题：库层 API 的默认参数硬编码 `cuda`（不带 index），绕过了 `device.py` 的统一选择逻辑；无 GPU/NPU 环境下默认值直接不可用。
- 建议：默认 `None`，函数内 `get_current_torch_device()` 解析。

### 主题 D：分布式正确性

### [CC-12] P0 dist_init 失败时静默伪装成单进程运行（数据损坏风险）
- 位置：`worldfoundry/core/distributed/generic_collectives.py:118-128`
- 证据：

```python
def dist_init() -> None:
    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(backend="nccl")
        assert torch.distributed.is_initialized()
    except Exception:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init")
```

- 问题：进程组初始化失败（NCCL 网络抖动、MASTER_ADDR 配错、端口冲突）被 `except Exception` 吞掉，然后**改写环境变量把本进程伪装成 RANK=0/WORLD_SIZE=1**，仅 `print` 一行警告继续运行。
- 影响：一个 8 卡任务里若某 rank init 失败，它会以"独立单进程"身份继续跑：写相同的输出路径、跳过所有 collective——其余 7 个 rank 在第一个 collective 处永久挂死（直到 NCCL timeout），而失败 rank 可能产出损坏/覆盖的结果文件。分布式 init 失败必须 fail fast。
- 建议：直接 re-raise；确需单机降级的场景应由调用方显式选择，且用 logger.error。

### [CC-13] P1 sequence_parallel/cuda_utils.py import 时 monkey-patch torch.cuda.set_stream（全局副作用 + 线程不安全）
- 位置：`worldfoundry/core/distributed/sequence_parallel/cuda_utils.py:34-46`
- 证据：

```python
prev_set_stream = torch.cuda.set_stream

_current_stream = None

def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    ...
torch.cuda.set_stream = _patched_set_stream
```

- 问题：（vLLM 移植代码）import 该模块（`pynccl.py:9` 引用，随 SP 通信链路加载）就替换全进程的 `torch.cuda.set_stream`。缓存的 `_current_stream` 是**进程级单值**：(1) PyTorch 的 current stream 本质是 per-thread、per-device 状态，realtime 路径多线程 + 多卡场景下缓存值会串台（线程 A 的 stream 被线程 B 读到）；(2) C++ 侧/`torch.cuda.graph` 内部切换 stream 不经过 Python `set_stream`，缓存失效后 `current_stream()` 返回陈旧 stream，pynccl collective 提交到错误 stream 上是静默的正确性/竞态问题。
- 影响：多线程推理或与 CUDA graph 混用时 NCCL 通信与计算可能不同步，产生偶发的错误数值或 hang，极难排查。
- 建议：去掉全局 patch，改为在 pynccl 调用点显式传 stream；至少将缓存改为 `threading.local` + per-device dict，并在 README 标注该副作用。

### [CC-14] P1 rank0_first：rank0 异常时其余 rank 永久挂死
- 位置：`worldfoundry/core/distributed/torch_process_group.py:179-192`
- 证据：

```python
    def wrapper(*args, **kwargs):  # noqa: ANN202
        result = None
        if is_rank0():
            result = func(*args, **kwargs)
        barrier()
        if not is_rank0():
            result = func(*args, **kwargs)
        return result
```

- 问题：rank0 执行 `func`（典型场景：下载数据/建缓存）抛异常时，rank0 不会到达 `barrier()` 就退出/进入异常处理，其余 rank 卡在 barrier 直到 NCCL heartbeat 超时（默认 1800s）。此外 rank0 的 `func` 耗时超过 process-group timeout 时，其他 rank 的 barrier 也会先超时。
- 影响：多机任务里 rank0 单点失败会表现为"全体卡死 30 分钟后 NCCL abort"，掩盖真实错误。
- 建议：`try/finally` 保证 barrier 必达 + 通过 broadcast 一个 success 标志让非 rank0 决定 raise；文档标注 func 耗时上限受 pg timeout 约束。

### [CC-15] P2 device_with_rank 用全局 rank 构造 cuda 设备号（多机必错）
- 位置：`worldfoundry/core/distributed/torch_process_group.py:156-159`
- 证据：

```python
def device_with_rank(device: str) -> str:
    if device == "cuda":
        return f"cuda:{get_rank()}"
    return device
```

- 问题：`get_rank()` 是全局 rank。两台 8 卡机器上 rank 8 会得到 `cuda:8`——不存在的设备。正确做法是用 LOCAL_RANK。当前仓库内无调用方（latent API），但它在 `__all__` 里作为公共 API 导出。
- 影响：任何多机使用者踩到即崩；单机超卖（rank 数 > GPU 数）同样崩。
- 建议：改用 `int(os.getenv("LOCAL_RANK", 0))` 或 `torch.cuda.current_device()`；或直接删除死 API。

### [CC-16] P2 is_local_rank0 在 CUDA_VISIBLE_DEVICES 隔离启动器下全员判真
- 位置：`worldfoundry/core/distributed/torch_process_group.py:150-153`
- 证据：

```python
def is_local_rank0() -> bool:
    if torch.cuda.is_available():
        return torch.cuda.current_device() == 0
    return int(os.getenv("LOCAL_RANK", 0)) == 0
```

- 问题：常见调度器（每进程 `CUDA_VISIBLE_DEVICES=<单卡>`）下每个进程的 `current_device()` 都是 0，所有 rank 都自认 local rank 0。LOCAL_RANK env 明明可用却只作为无 CUDA 时的 fallback。
- 影响：以 `is_local_rank0()` 保护的"每节点一次"操作（下载、解压、写缓存）会被并发执行，产生文件竞争。
- 建议：优先 `LOCAL_RANK` env，CUDA current_device 作为最后手段。

### [CC-17] P2 多套并存的进程组初始化/rank 辅助函数（4 个 dist_init、3 套 get_rank）
- 位置：`worldfoundry/core/distributed/torch_process_group.py:90`（`init`）、`generic_collectives.py:118`（`dist_init`）、`evaluation_collectives.py:21`（`dist_init`）、`metric_sync.py:47`（`init_distributed`）；get_rank/get_world_size/barrier 分别定义于 `torch_process_group.py:130,138,174`、`generic_collectives.py:51,59,103`、`evaluation_collectives.py:89`、`sequence_parallel/parallel_state.py:792`
- 证据（三种行为不一致的初始化）：`torch_process_group.init` 设置 NCCL env + CPU affinity；`generic_collectives.dist_init` 失败伪装单进程（CC-12）；`evaluation_collectives.dist_init` setdefault MASTER_ADDR/PORT 后初始化。
- 问题：同一包内 4 个初始化入口行为各异（timeout、env 副作用、失败语义都不同），rank 辅助函数三处重复。调用方混用会导致"初始化路径取决于谁先被 import"。
- 影响：行为不可预测、修 bug 要改多处；`metric_sync.init_distributed` 还有变量未绑定 bug（见 CC-18）。
- 建议：收敛到单一 `bootstrap_distributed()`，其余仅保留薄别名并标注 deprecated。

### [CC-18] P2 metric_sync.init_distributed：env 解析失败后使用未绑定变量 + 无条件 set_device
- 位置：`worldfoundry/core/distributed/metric_sync.py:47-81`
- 证据：

```python
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            logger.info("torchrun env vars not set")
    ...
    torch.cuda.set_device(gpu)
```

- 问题：(1) 若 `RANK` 存在但 `LOCAL_RANK` 缺失，except 只打 info 日志继续走，`gpu`（以及 rank/world_size 为 None）未绑定，line 74 直接 `NameError`/`init_process_group(world_size=None)` 崩溃且报错信息误导；(2) `torch.cuda.set_device(gpu)` 无 CUDA 可用性检查，CPU 机器直接崩；(3) 硬编码 `backend="nccl"`。
- 影响：错误路径产生与根因无关的异常；无 GPU 环境无法降级。
- 建议：解析失败即 raise 带说明的 RuntimeError；set_device 前判断 `torch.cuda.is_available()`；backend 走 `device.py:parse_nccl_backend`。

### [CC-19] P2 setup_for_distributed 全局替换 builtins.print，且 world_size>8 时逻辑反转
- 位置：`worldfoundry/core/distributed/metric_sync.py:31-44`
- 证据：

```python
    def print(*args, **kwargs):  # noqa: A001
        force = kwargs.pop("force", False)
        force = force or get_world_size() > 8
        if is_master or force:
            ...
    builtins.print = print
```

- 问题：(1) 库代码 monkey-patch `builtins.print`，影响整个进程（含第三方库输出），且不可逆（无 teardown）；(2) `force = force or get_world_size() > 8`——超过 8 卡时**所有 rank 都强制打印**，与"只有 master 打印"的本意相反（该行为是从 MAE 代码库照搬的历史怪癖）；(3) 每次 print 都调用 `get_world_size()` 走 dist 查询。
- 影响：大规模任务日志爆炸（N 份重复输出），且抢占了用户/其他库的 print 语义。
- 建议：删除 builtins patch，改用 logging + rank filter（仓库已有 `distributed/logging.py:print_rank_0`）。

### [CC-20] P2 multiprocess_launch：NCCL barrier 在 set_device 之前（launch 路径自身已损坏，属死代码）
- 位置：`worldfoundry/core/distributed/multiprocess_launch.py:54-70`
- 证据：

```python
    dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank)
    ...
    object_collectives.synchronize()   # dist.barrier() on NCCL
    ...
    torch.cuda.set_device(local_rank)
```

- 问题：NCCL barrier 发生在 `torch.cuda.set_device(local_rank)` 之前，此时每个本地进程的 current device 都是 cuda:0，NCCL 会对同一 GPU 建多个 rank 的 communicator → "Duplicate GPU detected" 报错或挂死。即该 `launch()` 在 n_gpu_per_machine>1 时根本跑不通。全仓库对此模块的引用只有 `find_free_port`（`synthesis/visual_generation/minwm/worldfoundry_runtime.py:11`），launch/distributed_worker 是死代码。
- 影响：留着一个必然死锁的公共 API 误导使用者。
- 建议：将 barrier 移到 set_device 之后（或删除 launch/distributed_worker，仅保留 find_free_port）。

### [CC-21] P2 RankGenerator 错误路径引用未初始化的 self.order（异常被 AttributeError 掩盖）
- 位置：`worldfoundry/core/distributed/model_parallel_groups.py:203-212`
- 证据：

```python
        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            ...
        self.order = order
```

- 问题：f-string 里的 `self.order` 在赋值（line 212）之前被引用；触发该校验时抛出的是 `AttributeError: 'RankGenerator' object has no attribute 'order'` 而非预期的 RuntimeError 信息。
- 影响：并行配置错误时用户看到的是无关的 AttributeError。
- 建议：改用局部变量 `order`。

### [CC-22] P2 destroy_model_parallel 只置 None，不销毁进程组（NCCL 资源泄漏）
- 位置：`worldfoundry/core/distributed/model_parallel_groups.py:647-682`
- 证据：

```python
def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
```

- 问题：`initialize_model_parallel` 每次创建 ~10 类 `new_group`（含 gloo 副本）；destroy 仅把模块级引用置 None，不调用 `dist.destroy_process_group(group)`。长驻进程（studio/评测服务）里反复 init/destroy 会累积 NCCL communicator 与其显存。另外 `get_dp_world_size`（line 615-620）在未初始化时返回 0 而非 1，除法调用方会得到 ZeroDivisionError。
- 影响：长驻服务显存缓慢上涨；返回 0 的语义陷阱。
- 建议：destroy 时逐组 `destroy_process_group`；未初始化时 world_size 返回 1（与 torch 惯例一致）。

### [CC-23] P3 torch_process_group.init 无条件覆盖 NCCL 环境变量、写死 nccl backend
- 位置：`worldfoundry/core/distributed/torch_process_group.py:107-117`
- 证据：

```python
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        ...
        dist.init_process_group(backend="nccl", ...)
```

- 问题：用户显式设置的 `TORCH_NCCL_BLOCKING_WAIT=1`（调试用）会被静默覆盖；backend 写死 nccl，与 `device.py:get_nccl_backend()`（支持 hccl/NPU）不一致——core 声称支持 NPU，但主初始化路径只支持 CUDA。
- 建议：`os.environ.setdefault`；backend 由 `get_nccl_backend()` 决定。

### [CC-24] P3 NativeAttention.set_context_parallel_group 修改 torch 全局 CP 选项且不恢复
- 位置：`worldfoundry/core/attention/native.py:406-415`
- 证据：

```python
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")
            from torch.distributed.tensor.experimental._attention import (
                _cp_options, set_rotate_method,
            )
            _cp_options.enable_load_balance = False
            set_rotate_method("allgather")
```

- 问题：启用某一个模块的 CP 时修改 torch 私有全局 `_cp_options` 与 rotate method，进程内其他 `context_parallel` 用户被连带改变；禁用 CP（传 None）时也不会恢复。依赖 torch 私有 API `_attention`。
- 建议：文档化该全局副作用；disable 时恢复原值；跟踪 torch 版本变化。

### 主题 E：显存管理（vram/）与 offload

### [CC-25] P1 vram/memory.py import 时初始化 CUDA 并固化 `gpu` 常量
- 位置：`worldfoundry/core/vram/memory.py:7-9`
- 证据：

```python
cpu = torch.device("cpu")
gpu = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else cpu
gpu_complete_modules: list[torch.nn.Module] = []
```

- 问题：`torch.cuda.current_device()` 触发 CUDA lazy init——**import 该模块就在 GPU0 上建 CUDA context**；且 `gpu` 是模块常量，worker 之后 `set_device(3)` 后模块内所有默认 device 引用仍指向 import 时的设备。`vram/__init__.py` re-export 该模块，传染面大。
- 影响：fork DataLoader 崩溃风险；多进程各 rank 在 GPU0 留下 context（每个 ~0.5GB）；错误设备上的 free-memory 估算导致 offload 决策失真。
- 建议：`gpu` 改为函数（每次调用解析 current_device）；避免模块级求值。

### [CC-26] P2 全局 gpu_complete_modules 强引用模型永不释放 + DynamicSwapInstaller 重复安装破坏还原
- 位置：`worldfoundry/core/vram/memory.py:9,211-228`；`worldfoundry/core/vram/memory.py:13-43`
- 证据：

```python
def load_model_as_complete(model: torch.nn.Module, target_device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()
    model.to(device=target_device)
    ...
    gpu_complete_modules.append(model)
```

- 问题：(1) 模块级 list 持有模型强引用，调用方 del 模型后显存/内存仍不释放，直到某处恰好调用 `unload_complete_models()`；多模型服务下这是泄漏源。(2) `DynamicSwapInstaller._install_module` 把 `forge_backup_original_class` 存进 `module.__dict__`，二次 install 时备份的已是 hacked class，此后 uninstall 永远还原不到真实类；无 install 幂等保护。(3) 全文件用 `print` 输出。
- 影响：长驻进程显存泄漏；双重 install 后模块类被永久污染。
- 建议：改用 `weakref.WeakSet`；install 前检查 `forge_backup_original_class` 已存在则跳过；print→logger。

### [CC-27] P2 layerwise offload hook 无法移除（无 disable API，句柄未保存）
- 位置：`worldfoundry/core/vram/layerwise_offload.py:54-65`
- 证据：

```python
        layer.register_forward_pre_hook(_make_pre_hook(state), with_kwargs=True)
        layer.register_forward_hook(_make_post_hook(state), with_kwargs=True)
        setattr(layer, "_worldfoundry_layerwise_cpu_offload", True)
        setattr(layer, "_worldfoundry_layerwise_cpu_offload_state", state)
```

- 问题：`register_forward_*` 返回的 RemovableHandle 被丢弃，`LayerwiseOffloadHandle` 只是元数据，没有任何 disable/remove 入口；`_LayerwiseOffloadState` 与 module 互相强引用。一旦启用，模型永远处于 offload 模式（想切回全量 GPU 驻留只能重建模型）。此外 forward 中途抛异常时 post_hook 不执行，GPU 副本滞留。
- 影响：评审标准 5（hook 未移除）的直接命中；A/B 性能对比、动态迁移场景不可用。
- 建议：Handle 持有 hook 句柄并提供 `disable()`（移除 hook + 还原参数 + 释放 pinned 缓存）。
- 备注：该文件的 CUDA stream 语义（prefetch 流 + `record_stream` + `wait_stream`）经核对是正确的，值得肯定。

### [CC-28] P2 AutoWrappedLinear.fp8_linear 每次 forward 在 CPU 上创建 scale 再搬运 GPU
- 位置：`worldfoundry/core/vram/layers.py:532-537`
- 证据：

```python
        scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)
        scale_b = torch.ones((weight.shape[0], 1)).to(device=device)
        input = input / (scale_a + 1e-8)
        input = input.to(self.computation_dtype)
        weight = weight.to(self.computation_dtype)
```

- 问题：(1) `torch.ones((N,1))` 先在 CPU 分配再 `.to(device)`——每层每次 forward 一次 H2D 拷贝与潜在同步；应 `torch.ones(..., device=device)` 或缓存为 buffer。(2) `input / (scale_a + 1e-8)` 后又把 `scale_a`（不含 1e-8）交给 `_scaled_mm` 反量化，存在 1e-8 相对偏差（可忽略但不严谨）。(3) weight 直接 `.to(fp8)` 且 `scale_b=1`，权重完全未做 per-tensor/per-channel scaling，幅值 >448 的权重饱和截断，静默精度损失。
- 影响：offload+FP8 的热路径性能损耗；权重饱和时输出质量下降且无告警。
- 建议：scale_b 缓存到 device；权重量化时计算真实 scale；对权重 absmax>fp8_max 记录一次 warning。

### [CC-29] P3 vram_limit 检查在每层 forward 调用 mem_get_info；AutoWrappedModule 每次 forward deepcopy
- 位置：`worldfoundry/core/vram/layers.py:100-107,248-249`
- 证据：

```python
    def check_free_vram(self):
        ...
        gpu_mem_state = getattr(torch, self.computation_device_type).mem_get_info(device)
    ...
    def cast_to(self, module, dtype, device):
        return copy.deepcopy(module).to(dtype=dtype, device=device)
```

- 问题：设置 `vram_limit` 后每个 wrapped 层每次 forward 都调用一次 `mem_get_info`（CUDA driver 调用）；offload 态的 `AutoWrappedModule.computation()` 每次 forward `copy.deepcopy` 整个子模块再 `.to()`（Python 级深拷贝 + 全量 H2D + 用后即弃），产生显存碎片和显著延迟。这是 DiffSynth 式设计的固有成本，但缺乏文档与频率控制。
- 建议：mem_get_info 结果按 N 次 forward/时间窗缓存；deepcopy 路径对大模块给出一次性性能警告。

### [CC-30] P3 offload_to_disk 的 for-else 控制流难以理解且部分参数静默不 offload
- 位置：`worldfoundry/core/vram/layers.py:211-219`
- 证据：

```python
    def offload_to_disk(self, model: torch.nn.Module):
        for buf in model.buffers():
            # If there are some parameters are registed in buffers (not in state dict),
            # We cannot offload the model.
            for children in model.children():
                self.offload_to_disk(children)
            break
        else:
            model.to("meta")
```

- 问题：借助 for-else + break 表达"有 buffer 则只递归子模块，否则整体置 meta"；带 buffer 的模块其**直接参数**永远不会被 offload（静默滞留内存），行为没有任何日志或文档。
- 建议：重写为显式 `if any(model.buffers())`，并对滞留参数记 debug 日志。

### 主题 F：inference.py（运行时引导 + 模型目录）

### [CC-31] P1 同一环境变量 WORLDFOUNDRY_ATTENTION_BACKEND 存在两套不兼容词汇表，合法配置会直接崩溃
- 位置：`worldfoundry/core/inference.py:4935-4951`；对照 `worldfoundry/core/attention/backends.py:53-95,115-123`
- 证据（inference.py）：

```python
    if normalized not in {"auto", "flash", "cudnn", "efficient", "math"}:
        raise ValueError(
            f"WORLDFOUNDRY_ATTENTION_BACKEND must be one of auto, flash, cudnn, efficient, or math (got {value!r})."
        )
```

（backends.py 接受同一环境变量的另一套值：`flash_attention_2`、`sage_attention`、`xformers` 等 30+ 别名）

- 问题：`attention/backends.py` 与 `inference.py` 都读取 `WORLDFOUNDRY_ATTENTION_BACKEND`，但词汇表互不兼容。用户按 backends.py 文档设置 `=flash_attention_2` 或 `=sage_attention` 后，`install_worldfoundry_inference_infra()`（被 `wrap_runner_for_worldfoundry_core`/`worldfoundry_inference_context` 在标准推理路径自动调用）在 `_normalize_attention_backend` 抛 ValueError，**整个推理进程启动即崩**；反之按 inference.py 设置 `=cudnn`，backends.py 又把它 normalize 成 `torch`，两套语义漂移。
- 影响：合法配置组合导致崩溃；三套 attention 策略系统（backends/dispatch、model_backends 的 AttentionFunction、inference 的 SDPA policy）互相独立，用户无从知道哪套生效。
- 建议：统一为一个 env var + 一套词汇表（inference.py 的 policy 应复用 `normalize_attention_backend` 并把外部后端名映射到 SDPA 策略或忽略并告警），补集成测试覆盖两模块同时 import 的场景。

### [CC-32] P1 默认全局 monkey-patch torch F.scaled_dot_product_attention（语义 + 性能影响全进程且不可逆）
- 位置：`worldfoundry/core/inference.py:4839-4861,4897-4899`
- 证据：

```python
    def worldfoundry_core_sdpa(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        return _call_sdpa_with_backend(query, key, value, *args, **kwargs)

    setattr(worldfoundry_core_sdpa, "_worldfoundry_core_sdpa", True)
    F.scaled_dot_product_attention = worldfoundry_core_sdpa
```

- 问题：`WORLDFOUNDRY_PATCH_SDPA` 默认为真，标准推理路径会把**全进程**的 `F.scaled_dot_product_attention` 替换掉：(1) 无 unpatch API，`_ORIGINAL_SDPA` 保存了原函数但没有任何恢复入口；(2) 每次 SDPA 调用都追加 `normalize_fully_masked_rows`（bool mask 时 O(S²) 的 broadcast+any 额外读写）——vendored 第三方模型的数值行为被静默改变；(3) 对 torch.compile 的模型，patch 后的 Python wrapper 会引入 graph break/重新 guard；(4) 与 dispatch.py 的后端选择器叠加时，同一次 attention 可能经过两层策略包装。
- 影响：核心张量算子被隐式替换，跨越 core 边界影响所有 vendored 模型；调试"为什么我的 attention 输出/速度变了"极其困难。
- 建议：默认改为 opt-in；提供 `uninstall_worldfoundry_inference_infra()`；文档醒目标注；对 compile 场景直接跳过 patch。

### [CC-33] P2 TF32 配置写错属性路径，matmul TF32 从未真正开启
- 位置：`worldfoundry/core/inference.py:4830-4836`
- 证据：

```python
    for backend in (getattr(torch.backends, "cuda", None), getattr(torch.backends, "cudnn", None)):
        if backend is None:
            continue
        try:
            setattr(backend, "allow_tf32", bool(enable_tf32))
        except Exception:
            pass
```

- 问题：matmul 的 TF32 开关是 `torch.backends.cuda.matmul.allow_tf32`；`setattr(torch.backends.cuda, "allow_tf32", ...)` 只是在模块对象上新建一个无效属性（已在 torch 2.7 实机验证 `hasattr(torch.backends.cuda, "allow_tf32") == False`）。cudnn 那半是有效的。结果：`WORLDFOUNDRY_ENABLE_TF32=1`（默认）只开了 cudnn TF32，matmul TF32 从未开启，但 `_STATE.tf32_enabled=True` 汇报"已开启"。
- 影响：性能策略静默失效一半；诊断状态与事实不符；`except Exception: pass` 把任何真实错误也吞了。
- 建议：改为 `torch.backends.cuda.matmul.allow_tf32 = enable_tf32`；同时同步 `torch.set_float32_matmul_precision` 的语义（两者有重叠）。

### [CC-34] P2 核心 core 文件内嵌大量模型专属常量、绝对路径与 import 时文件系统探测
- 位置：`worldfoundry/core/inference.py:43-120`（及后续 ~3000 行模型 spec）
- 证据：

```python
_WOW_OFFICIAL_LOCAL_CHECKPOINT = _WORKSPACE_ROOT / "ckpt" / "WoW-1-Wan-14B-600k"
WOW_LOCAL_CHECKPOINT = (
    str(_WOW_OFFICIAL_LOCAL_CHECKPOINT)
    if _WOW_OFFICIAL_LOCAL_CHECKPOINT.exists()
    else "X-Humanoid/WoW-1-Wan-14B-600k"
)
```

- 问题：core 层文件包含具体模型（LINGBOT/HELIOS/WOW/SANA/BERNINI/MIRA…）的 checkpoint 路径、demo prompt、测试 fixture 路径，且在 **import 时**做 `.exists()` 文件系统探测（网络盘上有延迟）；5018 行巨型文件同时承担"运行时引导"与"模型目录"两职责。模型目录属于 model zoo/registry 层。
- 影响：core 与具体模型/部署环境强耦合；import 慢且结果依赖磁盘状态（同一代码在不同机器 import 后常量不同）；文件不可维护。
- 建议：模型 spec 拆到 data/models 或 registry 配置；路径探测延迟到使用点。

### 主题 G：realtime / geometry / spatial_warp / nn / kernels 收尾

### [CC-35] P3 realtime.py / realtime_timing.py 契约代码质量良好（正面确认）
- 位置：`worldfoundry/core/realtime.py`、`worldfoundry/core/realtime_timing.py`
- 说明：两文件为纯 Python 数据契约与计时聚合，无 torch 依赖、无全局状态、边界校验完整（`RealtimeSpec.__post_init__`、`_finite_values` 过滤 NaN）。唯一注意点：`RealtimeTimingWindow` 非线程安全（`record`/`reset_interval` 并发会丢样本），当前单事件循环使用场景下可接受，建议在 docstring 标注。

### [CC-36] P2 Sparse3DCache 无容量上限，retrieve 每次全量 stack（会话级内存线性增长）
- 位置：`worldfoundry/core/spatial_warp.py:62-64,86-89,141`
- 证据：

```python
self._world_points: list[torch.Tensor] = []
...
def add_precomputed(self, *, points: torch.Tensor, latent_index: int, frame_id: int | None = None) -> None:
    self._world_points.append(points.detach())
...
points = torch.stack([value.to(device=device, dtype=torch.float32) for value in self._world_points])
```

- 问题：缓存只增不减（无 max_entries/无淘汰 API）；每帧的 `[B,H/4,W/4,3]` float32 世界点张量按其原始设备（通常 GPU）持有。`retrieve` 每次调用把**全部**历史帧 stack 成 `[N,B,H,W,3]` 大张量并与全部视角做矩阵乘（O(N) 显存+计算），随后贪心循环里每选一帧还有 2 次 `.item()` 同步（`spatial_warp.py:186-187`）。
- 影响：流式/交互式长会话（该缓存正是为 streaming 检索设计）中显存与检索延迟随生成帧数线性增长，最终 OOM 或帧率崩掉。
- 建议：加容量上限+LRU/时间戳淘汰（参考同仓 `memory/mosaic.py` 的 budget 设计）；retrieve 支持增量投影或对候选分块。

### [CC-37] P3 LitEma 每步 `.item()` 同步 + per-param Python 循环
- 位置：`worldfoundry/core/nn/ema.py:57-64`
- 证据：

```python
for key in m_param:
    if m_param[key].requires_grad:
        ...
        shadow.sub_(shadow - parameter, alpha=float(one_minus_decay.item()))
```

- 问题：`one_minus_decay` 是 GPU buffer 派生张量，`.item()` 在每个 training step 强制一次 device→host 同步；随后对每个参数单独发一个 `sub_` kernel（千级参数张量 = 千级小 kernel）。另外 `restore()` 的 `zip(self.collected_params, parameters)` 不校验长度，静默截断。
- 影响：EMA 更新在大模型上成为可测的 step 开销；`store/restore` 配对错误时静默错还原。
- 建议：decay 标量留在 CPU（python float）；用 `torch._foreach_lerp_`/`_foreach_sub_` 批量更新；`restore` 校验长度。

### [CC-38] P3 kernels/nn/acceleration/video_postprocess 组质量良好（正面确认）+ 两个小注意点
- 位置：`worldfoundry/core/kernels/__init__.py:19-24`、`worldfoundry/core/attention/__init__.py:8`、`worldfoundry/core/__init__.py:12`、`worldfoundry/core/kernels/triton_group_norm_silu.py:18-22`
- 说明（正面）：
  - `core/__init__`、`attention/__init__`、`kernels/__init__` 均用 lazy export map / `__getattr__`（`kernels/__init__.py:19-24`），`import worldfoundry.core` 不会拉起 triton/flash-attn/CUDA——包级 import 卫生是好的（个别模块内部仍有 import 副作用，见 CC-07/CC-25/CC-32）。
  - `kernels/registry.py` 的 dispatch 有显式 capability gate + pytorch fallback + `kernel_dispatch_report()` 可观测性；`kernels/capabilities.py` 的探测函数全部 `lru_cache` 且不在 import 时执行。
  - `nn/activation_checkpointing.py`、`acceleration/cache.py`（步进缓存，有 reset）、`video_postprocess.py`/`torchprofile.py`（纯契约/工具，校验完整）质量良好。
  - `core/log_filters.py` 虽是 import 副作用（安装 logging filter），但文档明确、作用面窄（降级 Inductor autotune 误报 ERROR），可接受。
- 注意点：
  - triton kernel 模块（如 `triton_group_norm_silu.py:20`）在 module level 调用 `configure_persistent_compile_cache(namespace=...)` 改写进程 env（TRITON_CACHE_DIR 等）——因 lazy dispatch 触发时机不确定，多线程下首次 dispatch 有竞态改 env 的理论风险，且这是 CC-02 分层违规的具体载体。
  - `camera_trajectory.py`/`world_explorer.py` 相机数学统一用 numpy float64 在 CPU 上算（冷路径、需要双精度），设计合理；`camera_pose.py:419-420` 的 `.cpu().numpy()` 仅在数据准备路径，可接受。

## 汇总

### 按严重度统计

| 严重度 | 数量 | 条目 |
| --- | --- | --- |
| P0 | 1 | CC-12 |
| P1 | 10 | CC-01、CC-02、CC-05、CC-06、CC-07、CC-13、CC-14、CC-25、CC-31、CC-32 |
| P2 | 17 | CC-03、CC-08、CC-09、CC-15、CC-16、CC-17、CC-18、CC-19、CC-20、CC-21、CC-22、CC-26、CC-27、CC-28、CC-33、CC-34、CC-36 |
| P3 | 10 | CC-04、CC-10、CC-11、CC-23、CC-24、CC-29、CC-30、CC-35（正面）、CC-37、CC-38（正面） |
| 合计 | 38 |（含 2 条正面确认）|

按主题分布：分布式正确性 11 条（含唯一 P0）、attention 后端选择 8 条、分层/import 卫生 5 条、vram/memory 6 条、全局状态与 monkey-patch 3 条（CC-13、CC-19、CC-32）、性能 3 条、其余为设备管理与数值。

### 本范围 Top 5 最重要问题

1. **[CC-12] P0** `generic_collectives.dist_init` 在 init 失败时吞掉异常并把本进程伪装成 `RANK=0/WORLD_SIZE=1` 继续运行——多机任务中失败 rank 静默变成"独立单进程"，写相同输出路径、跳过全部 collective，其余 rank 挂死在第一个 collective；这是数据损坏级缺陷，必须 fail fast。
2. **[CC-32] P1** `inference.py` 默认把 `F.scaled_dot_product_attention` 全局 monkey-patch 成自家 dispatch 包装（无恢复机制、对 vendored 模型与第三方库同样生效）——全进程 attention 语义被隐式改变，出现精度/性能问题时几乎不可定位；与 CC-13（patch `torch.cuda.set_stream`）、CC-19（patch `builtins.print`）同属一类系统性风格问题。
3. **[CC-31] P1** 同一个环境变量 `WORLDFOUNDRY_ATTENTION_BACKEND` 被 `backends.py` 与 `model_backends.py` 两套不兼容词汇表解析，一侧合法的取值（如 `flash_attention_2`）会让另一侧直接抛异常崩溃——同一配置项在同一进程内既可能生效又可能崩，属接口级缺陷。
4. **[CC-05/06/07] P1** attention 后端选择链名不副实：`auto` 实际只会选 torch SDPA（文档与 `attention_backend_report()` 展示的优先级链是假的）、显式请求的后端不可用时无警告静默降级到 SDPA、`dispatch.py` 又在 import 时 probe 后端触发 CUDA 初始化（违反自己注释的设计意图）——三者叠加使"用户以为在用 flash-attn，实际在跑 SDPA"成为默认现实且无日志可查。
5. **[CC-01/02] P1 + [CC-25] P1** 分层与 import 卫生的系统性破口：`inference.py` 顶层 import `worldfoundry.pipelines`、5 个 kernel 模块 import `worldfoundry.runtime`（core→上层反向依赖，循环依赖隐患）；`vram/memory.py` 在 import 时调用 `torch.cuda.current_device()` 初始化 CUDA 并固化 `gpu` 常量——core 作为最底层地基，其 import 行为直接决定所有上层进程（含 dataloader worker、无 GPU 的 CI）的启动正确性。

### 总体评价

core 计算层呈"新旧两层皮"结构：新写的契约/registry 类代码（realtime、video_postprocess、kernels registry、activation_checkpointing、包级 lazy export）质量明显较高，有校验、有可观测性、import 卫生良好；而从各模型仓/参考实现移植的代码（generic_collectives、metric_sync、vram/memory、LitEma、XFormers mask 处理）带入了大量已知反模式（吞异常、全局状态、import 副作用、monkey-patch、`.item()` 同步）。三个系统性建议：(1) 用 import-linter 固化 core 不得 import pipelines/runtime/training 的契约，并以 `WORLDFOUNDRY_*` 环境变量为唯一后端开关、单一解析器；(2) 全仓清理 module-level 的 CUDA 触碰与 monkey-patch，统一收敛到显式的 `initialize()` 入口；(3) 分布式 init/rank/collective 只保留一套实现（当前 `dist_init`/`get_rank`/`barrier` 至少三处定义、行为互不一致），失败一律 fail fast。
