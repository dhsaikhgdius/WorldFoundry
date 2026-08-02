# WorldFoundry CUDA/C++ 推理算子工程规格

> 状态：工程规格 v1；NAT-00 本地 scaffold/dispatcher smoke 已完成，wheel/CI 资格待补；任何公开模型算子和 `.cu` 仍受各自 promotion gate 约束
> 更新：2026-07-31
> 目标：明确“要写哪些 C++/CUDA、为什么写、接口是什么、接到哪里、何时停止”
> 原则：Torch 是语义真源；Triton/Inductor 是默认对照；native 扩展是可选 provider，不得成为主包 import 的硬依赖

本文是 native schema、ABI、shape 数学、错误语义和逐算子门槛的唯一真源；总体阶段/依赖由 [`inference_operator_optimization_plan.md`](./inference_operator_optimization_plan.md) 维护。环境与 world size 只来自 build/benchmark manifest，不把一次 shell 状态写成长期支持事实。

## 1. 冻结结论

首批实现只批准以下内容：

| ID | 内容 | 文件类型 | 当前决定 |
|---|---|---|---|
| N0 | dispatcher DSO、lazy loader、sidecar ABI manifest、build fingerprint | C++/Python/CMake，无公开模型 op、无 CUDA language | 立即做；所有后续 native op 的基础 |
| K1 | `causal_conv3d_cat_pad` functional + out | 1 个通用 `.cu` | 首个 exact CUDA 候选；先以真实 Wan VAE shape 过 promotion gate |
| K2a | `int8_per_token_quant` functional + out | 1 个 SM80 `.cu` | 条件项；先修正并冻结现有量化 reference |
| K2b | `int8_scaled_mm` functional + out | 1 个 SM80 CUTLASS `.cu` | 条件项；必须同时胜过现有 Triton、官方路径和 BF16 dense |
| K2c | INT8 `bias+GELU`、`residual+gate` epilogue | K2b 内模板或独立 `.cu` | K2b 模型级收益达标后才做 |

不属于首批、只能在对应硬件和 profile 到位后立项：

| ID | 内容 | 目标 | 当前决定 |
|---|---|---|---|
| K3 | LTX2 QK RMSNorm + split-RoPE | SM100+ | Blackwell + LTX trace 双重门槛；ABI 暂不冻结 |
| K4 | NVFP4/MXFP quant、scaled-MM、融合 epilogue | SM100 或 SM120，分别构建 | 先比较现有 `NVFP4Linear`、Triton 和 PyTorch scaled-MM |
| K5 | ring KV materialize/update | autoregressive/windowed | 先把 cache 改成 ring/block-table；只有 materialize copy 成为热点才写 |
| K6 | VAE tile blend/stitch | 长视频 VAE | 只有 tile overlap 的 allocation/copy 进入 top hotspot 才写 |

明确不自己重写：

- dense attention：继续使用 Torch SDPA、FlashAttention 等等价 provider；SageAttention/SageAttention3 是 approximate provider，只能显式启用。
- Hopper sparse attention：先接 `fastvideo-kernel` 外部 provider，不 fork ThunderKittens/TMA 源码。
- `layer_norm_scale_shift`、`rms_norm_scale_shift`、`residual_gate_add`、通用 QK norm/RoPE、SiLU/GLU、GroupNorm+SiLU：仓库已有 Torch/Triton 实现。
- Conv3D 计算本身：继续交给 cuDNN/Inductor；K1 只融合其前置数据搬运。
- merged QKV：通过权重 transform + 一次 `F.linear` + view/split 实现，不需要 CUDA kernel。
- QKV all-to-all 合包：仓库已有 `all_to_all_many` / `all_to_all_4d_many`，先优化 layout 和 workspace，不写 collective kernel。
- CUDA Graph、compile cache、step cache、token schedule：它们是 runtime/policy 工作，不是 CUDA kernel。

因此第一批 NAT-00 只有 **2 个 C++ 源文件 + Python/CMake/sidecar glue**，仅注册私有 `_build_info()` smoke op。NAT-01/K1 reference 与 profile 过门槛后才增加公开 schema/fake；K1 promotion 通过后才增加首个 `.cu`。K2 达标后才再增加 SM80 CUDA 源文件。

## 2. 已核对的仓库事实

### 2.1 已有公共 kernel

`worldfoundry/core/kernels/diffusion.py` 已公开：

- `silu_mul`
- `silu_and_mul`
- `group_norm_silu`
- `residual_gate_add`
- `layer_norm_scale_shift`
- `rms_norm_scale_shift`
- `hidden_qk_rmsnorm_rope_3d`
- `qk_rmsnorm_rope`

它们已有 Torch fallback、Triton candidate、资格判断、selection cache 和 dispatch report。2026-07-28 的静态盘点显示 diffusion network 直接调用仍较少；调用数量只是审计快照，真实 shape/频次由 P0 trace 决定。这里的首要工作是接线和 benchmark，不是再写 CUDA。

### 2.2 causal Conv3D 语义并不统一

canonical Wan 的 `models/autoencoders/wan/model.py::CausalConv3d.forward` 是：

```python
padding = list(self._padding)
if cache_x is not None and self._padding[4] > 0:
    cache_x = cache_x.to(x.device)
    x = torch.cat([cache_x, x], dim=2)
    padding[4] -= cache_x.shape[2]
x = F.pad(x, padding)  # 默认 constant-zero
return super().forward(x)
```

Cosmos2.5 和 Cosmos3 复用该 Wan VAE 类型，因此可自然命中 K1。其他 VAE 不能直接套用：

- HunyuanVideo 的 causal Conv3D 使用 `replicate` 等 `pad_mode`。
- LTX video VAE 通过首帧/尾帧 `repeat` 构造 temporal padding。
- Wan 的若干 reference/variant 文件有相同模式，但首批只改 canonical 真源，避免同步修改复制实现。

### 2.3 当前 INT8 不是 block-scale ABI

`worldfoundry/core/kernels/quantized_gemm.py` 与 Wan `variants/action_22.py::Int8Linear` 当前语义为：

- activation：展平为 `[M,K]`，每行一个 FP32 dequant scale，输出 INT8 `[M,K]`。
- weight：逻辑 `[N,K]`，每个 output channel 一个 FP32 dequant scale。
- GEMM：INT32 accumulate，再乘 `[M,1]` 与 `[1,N]` 两组 scale，输出 BF16/FP16。

所以 native provider 必须先复现 per-token/per-channel 语义。若以后改成 block-128 scale，那是新的量化格式、模型质量实验和 artifact 版本，不能伪装成同一个 backend。

### 2.4 多卡 QKV 已经有合包入口

以下代码已经把等 shape 的 Q/K/V 合成一次 collective：

- `worldfoundry/core/distributed/sequence_ops.py::all_to_all_many`
- `worldfoundry/core/distributed/sequence_parallel_runtime.py::all_to_all_4d_many`

它们目前使用 `stack/cat -> all_to_all -> unbind/split`。下一步应先让 merged QKV projection 直接产生适配 layout，或使用 runner workspace 消掉重复 allocation。只有 profiler 证明 pack/unpack copy 本身仍占显著比例，才考虑一个纯 layout CUDA op；NCCL collective 本身不由普通 CUDA copy kernel 替代。

### 2.5 最近开发快照与构建约束

- 2026-07-31 当前任务快照：1 × A100-SXM4-80GB（SM80）、Python 3.12.3、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、Torch CUDA/NVCC 12.8、driver 580.105.08、CXX11 ABI `True`。它只允许本地开发验证，不是 release support matrix。
- GPU 数/world size 属于 benchmark manifest；Torch/CUDA/compiler/ABI/SM target 属于 build manifest。两者都必须运行时生成，不能从本文抄常量。
- 当前仓库未发现锁定版本的 CUTLASS headers；K2 开工前必须 pin commit、license notice 和构建方式。
- 主项目声明 `torch>=2.7,<2.12`、Python `>=3.10,<3.14`。Torch C++ extension 不能假设跨所有这些 minor/CPython ABI 通用，native support matrix 必须比主包范围更窄且显式。

## 3. Native extension 边界

### 3.1 目录结构

建议独立 optional subproject：

```text
packages/worldfoundry-native-kernels/
  pyproject.toml
  CMakeLists.txt
  LICENSE
  cmake/
    write_build_manifest.cmake            # NAT-00
    WorldFoundryCudaArchitectures.cmake   # NAT-01 以后
    WorldFoundryTorchAbi.cmake            # NAT-01 以后
  csrc/
    registration.cpp                    # NAT-00 仅私有 _build_info schema
    build_info.cpp
    common/
      checks.h
      cuda_utils.cuh
      ops.h
    cuda/
      causal_conv3d_cat_pad_cuda.cu
    sm80/
      int8_per_token_quant_sm80.cu       # K2 达标后才加入
      int8_scaled_mm_sm80.cu             # K2 达标后才加入
    sm100/
      ltx2_qknorm_split_rope_sm100.cu    # 不在首批
    sm120/
      nvfp4_scaled_mm_sm120.cu           # 不在首批
    third_party/
      cutlass/                            # 只在 K2 被批准后，以固定 commit 引入
  python/worldfoundry_native_kernels/
    __init__.py
    _loader.py
    meta.py                              # NAT-01 首个公开 functional schema 时加入
    build_manifest.json                 # 构建生成的 sidecar，不手写运行环境
  checks/
    test_native_provider.py              # NAT-00 package/loader smoke
```

主包适配层：

```text
worldfoundry/core/kernels/native_provider.py
worldfoundry/core/kernels/causal_conv3d.py
worldfoundry/core/kernels/registry.py       # 复用并扩展现有 registry
tests/core/kernels/test_causal_conv3d_cat_pad.py
benchmarks/operators/causal_conv3d_cat_pad.py
benchmarks/operators/int8_linear.py
```

### 3.2 ABI epoch 与 pre-dlopen sidecar

NAT-00 冻结 `manifest_schema_version=1`、`operator_abi_version=1` 和私有 `worldfoundry_native::_build_info() -> str`。loader 必须先读取 wheel 内 sidecar，完成以下检查后才允许 `torch.ops.load_library()`：

- sidecar schema/operator ABI、build ID 和 source tree hash；所有必填字段先做类型、摘要格式和内部自洽性检查。
- 完整 PyTorch build version（包含 local suffix）、Torch CUDA、CXX11 ABI 是 pre-dlopen runtime hard gate。
- host compiler/libstdc++、NVCC（如有）、SASS/PTX targets 与编译 flags 必须记录；由 release qualification 和逐 device eligibility 校验，不在用户请求路径现场调用编译器。
- DSO 相对路径、SHA-256、是否链接 `libtorch_python`、已编译 capability 列表；DSO 不得逃逸 package root。

NAT-00 使用不含 pybind 的 dispatcher DSO，不链接 `libtorch_python`；若以后改变，必须把 CPython ABI 加入 wheel tag 和 sidecar。源码构建使用已资格验证的当前 Torch 环境和 `--no-build-isolation`，不得让 PEP 517 临时解析另一份浮动 Torch。sidecar 不匹配时禁止尝试 `dlopen`；内部 `_build_info()` 只用于加载后的双向一致性 smoke，不能代替 preflight。

进入 NAT-01/cache v2 前必须冻结 artifact identity 为至少 `(build_id, library_sha256, operator_abi_version)`。release `build_id` 必须纳入 source revision/tree、编译器、规范化后的实际 compile/link flags 与 libstdc++ 身份，且拒绝 `source_revision=unknown`；当前 NAT-00 本地 scaffold 的 build ID 只用于 smoke，不能直接作为跨 wheel 的持久 cache key。

### 3.3 加载与错误规则

- `import worldfoundry`、CPU CLI、文档构建不能加载 CUDA runtime 或 native `.so`。
- `inspectable` 只表示 sidecar/DSO 文件有效，不等于可执行；只有 runtime preflight 和显式 load 完成后才能报告 `available=true`，状态至少区分 absent、manifest-invalid、runtime-incompatible、load-failed、loaded。
- `native_provider` 第一次收到 eligible CUDA tensor 时才 lazy import 对应 module。
- corrupt/stale sidecar、import/load failure、Torch/CUDA ABI 不匹配、错误 SM、缺少 symbol 和显式 `KernelNotSupported` 可在执行前转成 `unavailable` capability，并保留 Torch/Triton fallback。
- OOM、illegal address、device assert、launch/capture failure、输入 contract `TORCH_CHECK` 错误必须原样抛出；不得在同一请求内尝试第二个 provider。missing symbol/`AttributeError` 只由 loader 翻译，kernel launch 层不做宽泛 `except`。
- 选择以 tensor 的实际 device 为准，不能缓存“进程 device 0”的结论到异构多 GPU。
- registry backend ID 使用 `cuda_ext`，不用 `native`。现有 `KernelRegistry.dispatch()` 把环境值 `native` 当作 Torch/native-PyTorch fallback 的兼容别名，复用该名字会导致扩展永远不被选择。
- 现有 autotune 流程一次只比较“一个 candidate 与 Torch fallback”，第一名 candidate 输给 Torch 后不会继续评测第二个 candidate。NAT-01 必须升级 cache v2：一条记录包含完整 provider set、各自 build ID/ABI、测量结果和唯一 winner；selection/quarantine key 包含 device profile、op ABI 与 native build ID。
- 普通线上请求只读取已经持久化的 N-way 选择；缺少记录时走安全 fallback，显式 benchmark/warmup 命令才允许竞速。
- forced `cuda_ext` 分为默认可降级与显式 strict 两种调用策略；doctor/CI 使用 strict，普通 `auto` 使用可降级。load 状态是进程级，SM/shape eligibility 是逐 device/signature 状态。

### 3.4 注册、functionalization 与 compile/export

公共调用只经过 `torch.ops.worldfoundry_native.*`：

```cpp
TORCH_LIBRARY_FRAGMENT(worldfoundry_native, m) {
  m.def("causal_conv3d_cat_pad(Tensor cache, Tensor x, int[] padding_after_cat) -> Tensor");
  m.def("causal_conv3d_cat_pad.out(Tensor cache, Tensor x, int[] padding_after_cat, *, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(worldfoundry_native, CUDA, m) {
  m.impl("causal_conv3d_cat_pad", &causal_conv3d_cat_pad_cuda);
  m.impl("causal_conv3d_cat_pad.out", &causal_conv3d_cat_pad_out_cuda);
}
```

out op 返回 `()`，Python wrapper 调用后再返回原 `out`；这样 AOTAutograd/Inductor functionalization 不需要处理 alias return。Python `meta.py` 在 schema DSO 成功加载后注册 fake/meta implementation，用输入 symbolic shape 与 padding 计算输出，不访问 tensor data；out fake 返回 `None`。

compile/export 规则固定为：只允许在 trace/capture **之前**完成 provider load、ABI 校验和离线 winner 解析。public wrapper 在 eager 与 compile 下都调用已经预选的 functional custom op；若没有选择记录或 fake 不可用，compile 前就选择 Torch reference。trace/capture 内禁止 import、dlopen、竞速、写 cache 或异常后现场 fallback。验收必须分别覆盖 public wrapper、direct op、`torch.library.opcheck`、FakeTensor、AOT eager、Inductor/fullgraph、export 和 dynamic symbolic shape；“direct op 能 trace”不能替代模型 wrapper 验收。

### 3.5 所有 op 的共同 C++ 约束

- `TORCH_CHECK` 覆盖 device、same-device、dtype、rank、shape、stride、layout、SM capability 和 out tensor contract；alignment 是逐 op fast-path 条件，不是所有 contiguous 输入的共同拒绝条件。
- 以 `x`/`x_q` 为 authoritative device：先 `c10::cuda::CUDAGuard`，再取得该 device 的 PyTorch current stream，allocation 与 launch 都在该 stream。
- launch 后调用 `C10_CUDA_KERNEL_LAUNCH_CHECK()`。
- capture-safe 路径禁止 `cudaDeviceSynchronize`、隐式 default stream、per-call `cudaMalloc/cudaFree`、同步 H2D metadata copy。
- functional op 可以通过 `at::empty` 分配；out op 不得换 storage、resize 或保存 Python/C++ 全局可变状态。
- inference-only public wrapper 遇到任一 `requires_grad=True` 输入时走可反传 Torch reference；direct op 必须同步报错，不能依赖未注册 Autograd kernel 的默认行为。
- exact op 不使用全局 `--use_fast_math`。
- 所有维度和 `numel` 算法用 64-bit，并显式检查溢出；kernel index 是否能降为 32-bit 由单次 launch 的资格条件决定。
- out 必须精确匹配 shape/dtype/device/contiguous layout，不 resize、不换 storage，并与所有输入无 overlap。runner workspace 归 request 所有、按 stream/device 隔离且可重入；禁止 module/global 单例 scratch。

## 4. K1：`causal_conv3d_cat_pad`

### 4.1 语义和 shape

输入：

- `cache`: contiguous CUDA `[N,C,Tc,H,W]`
- `x`: contiguous CUDA `[N,C,Tx,H,W]`
- `padding_after_cat`: `(pw_l,pw_r,ph_l,ph_r,pt_l,pt_r)`

要求 cache 与 x 的 `N/C/H/W`、dtype、device 相同。输出：

```text
[N,
 C,
 Tc + Tx + pt_l + pt_r,
 H + ph_l + ph_r,
 W + pw_l + pw_r]
```

逐元素语义严格等价于：

```python
torch.nn.functional.pad(
    torch.cat([cache, x], dim=2),
    padding_after_cat,
    mode="constant",
    value=0,
)
```

Wan wrapper 负责先执行 `padding_after_cat[4] -= cache.shape[2]`。该减法结果若为负数，表示 PyTorch crop 语义，首版 native 不接，直接 fallback。

padding 必须恰有 6 个整数；每项、每个输出维度和最终 `numel * element_size` 都做 64-bit checked arithmetic。functional/out 首版要求所有输入/output 维度非零；空 tensor 由 public wrapper 走 Torch。direct op 对 contract 外输入明确报错。

### 4.2 eligibility

首版支持：

- CUDA `is_contiguous(at::MemoryFormat::Contiguous)` NCTHW；允许非零 storage offset 和未满足 16-byte vector alignment 的 contiguous view，后者走 scalar/prologue/tail。
- FP16、BF16、FP32。
- 六个 padding 均为非负整数。
- constant zero padding。
- 非空正常 tensor；零尺寸先走 Torch，除非测试证明 native contract 完整。
- cache 与 x 都是只读，允许二者互相 alias；out 不得与 cache/x 的 storage range overlap。

首版不支持：

- channels-last-3d、任意 non-contiguous view。
- replicate/reflect/circular padding。
- negative padding/crop。
- cache 与 x 不同 device/dtype/空间 shape。
- autograd/training。

### 4.3 CUDA kernel 设计

采用输出驱动的一维 grid；每个 thread 或 vector lane 对应连续 W 方向元素：

```text
decode output linear index -> n,c,to,ho,wo
if wo/ph/to falls in any padding region:
    out = 0
else:
    ts = to - pt_l
    hs = ho - ph_l
    ws = wo - pw_l
    if ts < Tc:
        out = cache[n,c,ts,hs,ws]
    else:
        out = x[n,c,ts-Tc,hs,ws]
```

实现按两步 promotion：先写容易审计的 scalar correctness kernel，再根据 benchmark 增加 vector fast path。实现要点：

- W 为 contiguous 最内层，优先按 16-byte vector 处理；行首/尾和不对齐尾部走 scalar。
- 不单独 `memset` 输出再 copy，因为那会重新变成多个 memory pass。
- 不在 kernel 内做 Conv3D，也不创建中间 concat tensor。
- functional wrapper 只分配最终 padded tensor；out wrapper 复用 runner/VAE workspace。
- grid/block 参数以 memory bandwidth 为目标；不做复杂 shared-memory tiling。

### 4.4 接入位置

1. `worldfoundry/core/kernels/causal_conv3d.py` 提供 Torch reference、signature、eligibility 与 dispatch。
2. `worldfoundry/core/kernels/native_provider.py` 注册 native candidate 和 build fingerprint。
3. 修改 canonical `models/autoencoders/wan/model.py::CausalConv3d.forward` 调公共 wrapper，但完整保留现有 adapter 语义：`cache=None` 或 `_padding[4] <= 0` 仍走原 pad；先执行 `cache_x.to(x.device)`，再判 eligibility；dtype 不同仍保留原 `torch.cat` 报错；`pt_l - Tc < 0` 走 Torch crop；任一输入需要梯度时走 Torch reference/backward。
4. Cosmos2.5/3 通过复用 canonical Wan 类型受益，不添加 Cosmos-local 分支。
5. reference/variant VAE 只在确认仍是可达 production path 后逐个 consolidation；不批量搜索替换。

### 4.5 promotion gate

开始写 `.cu` 前，P0 trace 必须给出：调用次数、`N/C/T/H/W`、`Tc`、padding、dtype、总 GPU 时间和 peak temporary bytes。

合入 native candidate 必须同时满足：

- 所有 eligible 输入与 Torch reference bitwise 相等。
- top real shapes 的 median 至少 `1.25×`；p90 不得明显回退。
- Wan 或 Cosmos service VAE decode 至少改善 `3%`，或 peak allocated VRAM 有可复现的实质下降。
- 完整端到端不回退超过 `0.5%`。
- public wrapper/direct op 的 opcheck、FakeTensor、AOT eager、Inductor/fullgraph/export/dynamic shape，以及非默认 stream、CUDA Graph replay、同进程多 GPU 均通过。

如果 operator 快但完整 VAE 无收益，保留 benchmark branch/实验记录，不发布 wheel 中的死代码。

## 5. K2：A100 INT8 quant + scaled-MM

K2 是 approximate 技术，不进入 `exact` preset。先把现有量化语义从 family-local `Int8Linear` 提升为公共、可测试 contract，再决定是否写 CUTLASS。

### 5.1 PR-Q0：先冻结量化数学，不写 CUDA

QNT-00 冻结 `quant_schema_version="wf-int8-per-token-v1"`。canonical reference 使用 FP32、IEEE round-to-nearest-even 和对称 `[-127,127]` code range：

```text
x2d = reshape(x, [M,K])
amax[m] = max_k(abs(float(x2d[m,k])))
eps = torch.finfo(torch.float32).tiny
scale[m] = max(amax[m] / 127, eps)       # FP32 division，不改写为 reciprocal multiply
q[m,k] = clamp(round_rne(float(x2d[m,k]) / scale[m]), -127, 127)
# amax == 0 时 scale=eps 且 q 全 0
```

权重离线按 `[N,K]` 每行同样量化，输出 `weight_q[N,K]` 和 `weight_scale[N]`。

在冻结前必须处理：

- 当前 Triton activation quant 的加减 `0.5` 转换不是 RNE，必须在 QNT-00 修正或作为旧 provider 退出 v1 竞速。
- 当前 zero row 可能产生除零/NaN；v1 固定为 `q=0`、`scale=torch.finfo(float32).tiny`。
- NaN/Inf 不属于生产 contract；debug/reference wrapper 明确拒绝，release kernel 不做会触发 host sync 的逐调用 finite scan，其 direct-op 结果未定义且不得进入 selection cache。
- scale dtype 固定 FP32，不随 input dtype 改变。
- 同一 GPU 上 q codes 必须逐 bit 一致；scale 的 reference/native 也要求 bitwise 一致，因此不得使用 fast-math/reciprocal 近似。若真实编译器无法满足，必须在 QNT-00 重新冻结 ULP tolerance，不能由 provider 各自放宽。
- artifact 中记录 quantization schema version，避免以后 block-scale 权重被旧 runtime 误读。

只有 Torch reference、现有 Triton 和 family-local `Int8Linear` 统一到该 contract 后，才开始 K2a/K2b。

### 5.2 K2a schema

```text
int8_per_token_quant(Tensor x) -> (Tensor x_q, Tensor x_scale)
int8_per_token_quant.out(
    Tensor x,
    *,
    Tensor(a!) q_out,
    Tensor(b!) scale_out
) -> ()
```

- `x`: contiguous BF16/FP16 `[...,K]`
- `x_q`: same shape, INT8
- `x_scale`: FP32 `x.shape[:-1]`
- 1 CTA 处理一个或多个 row，FP32 max-abs reduction，随后 vectorized quant/store。
- zero row、rounding、clamp 必须与 PR-Q0 reference 一致。

### 5.3 K2b schema 与数学

```text
int8_scaled_mm(
    Tensor x_q,             # [M,K], int8
    Tensor x_scale,         # [M], float32
    Tensor weight_q,        # [N,K], int8
    Tensor weight_scale,    # [N], float32
    Tensor? bias,
    ScalarType out_dtype,
) -> Tensor                 # [M,N]

int8_scaled_mm.out(
    Tensor x_q,
    Tensor x_scale,
    Tensor weight_q,
    Tensor weight_scale,
    Tensor? bias,
    ScalarType out_dtype,
    *,
    Tensor(a!) out,
) -> ()
```

数学定义：

```text
acc[m,n] = sum_k int32(x_q[m,k]) * int32(weight_q[n,k])
y[m,n] = cast_out(acc[m,n] * x_scale[m] * weight_scale[n] + bias[n])
```

首版 eligibility：

- native raw op 固定接收 2D `[M,K]`；public quantized-linear wrapper 负责 flatten 原 `[...,K]` 并把 `[M,N]` restore 为 `[...,N]`。
- SM80 correctness target，首个 performance qualification 是 A100；其他 SM80 产品不得自动继承调优结论。BF16 或 FP16 output。
- `x_q` row-major `[M,K]`、`weight_q` row-major `[N,K]`。
- `M/N/K > 0`，Tensor Core correctness alignment 至少 `K % 32 == 0`、`N` 满足所选 tile 的明确对齐；`K % 128 == 0` 只是优先 tuned bucket。direct op 接受任意 int8 code，因此还要求 `K <= 131071`，保证最坏 `16384*K` 不溢出 INT32。
- x/weight/scale/bias/out 全部 same-device、contiguous；scale 精确为 FP32 `[M]`/`[N]`。bias 为 `None` 或 contiguous `[N]`，dtype 只允许 FP32 或 `out_dtype`。
- out 精确为 contiguous `[M,N]`、指定 BF16/FP16 dtype，不与任一输入 overlap。
- weight 与 scale 在 load transform 阶段预计算，不在 forward 重复量化。
- raw row-major `[N,K]` 是 v1 ABI；未来 CUTLASS opaque/prepacked weight 必须使用新的 artifact/schema version 和 op 名，不能在同一 ABI 下暗换 layout。

CUTLASS epilogue按 FP32 `acc -> *x_scale -> *weight_scale -> +bias -> cast_out` 顺序应用两组 scale和 bias，不能先产出 INT32 tensor 再额外 launch dequant kernel。这与当前 family-local “先 cast BF16、再加 BF16 bias”不同，QNT-00 必须把它视为 v1 语义迁移并重新做逐层/模型质量验收。

K2b 首版只允许经过验证的 zero-workspace CUTLASS algorithm。若胜出实现需要 workspace，ABI 必须先增加 caller-owned workspace/size-query 契约，并证明 request/stream 隔离和 CUDA Graph replay；不得在 op 内 per-call allocate。

### 5.4 K2c epilogue 顺序

K2c 的 ABI 尚未冻结。只有 K2a+K2b 已经带来模型级收益、且 profile 证明 epilogue 是剩余热点时，才按以下顺序设计独立 schema：

1. 第一层 bias+activation：先确认目标模型究竟使用 `GELU(tanh)` 还是当前 Triton 的 `x*sigmoid(1.702x)`，二者不得共用 op 名。
2. `int8_scaled_mm_residual_gate`：Wan FFN 第二层，将 bias、broadcast gate 与 residual add 融入 epilogue。
3. 只有真实模型需要且 shape 稳定时，再考虑 SiLU/GLU epilogue。

gate 的 broadcast/layout、bias dtype 和逐项算术顺序必须在立项时冻结。使用独立 op/schema，不使用一个含多个运行时 bool 的万能 kernel。

### 5.5 K2 benchmark 与停止条件

比较四条完整路径：

1. BF16 `F.linear` / cuBLASLt。
2. 现有 WorldFoundry Triton quant + GEMM。
3. 当前 PyTorch 能用的官方 scaled-MM 路径。
4. native SM80 quant + CUTLASS GEMM。

每个路径都计入 dynamic activation quant、output epilogue 和 workspace allocation，不只计 GEMM 核心。

门槛：

- 覆盖 shape trace 的 top-10 Wan FFN `(M,N,K)`，而非只测方阵。
- K2b 相对 Triton median 至少 `1.10×`。
- quant + linear 相对 Triton至少 `1.08×`。
- 目标 Wan denoiser 至少 `1.15×`，并通过逐层 cosine、逐步 latent drift 和完整生成质量预算。
- 若只有单个 shape 胜出，则保留 shape-specific candidate，不全局替换。
- 未达到模型级门槛时，不引入 CUTLASS 到 release dependency。

## 6. 后续条件算子

### 6.1 K3：LTX2 QK RMSNorm + split-RoPE

当前 LTX 实现的 `q_norm/k_norm` 和 `apply_split_rotary_emb` 是独立 Torch 路径；其 reshape/broadcast 语义与公共 3D RoPE 不完全相同。该 op 只有在以下条件同时成立时立项：

- 有 SM100/103 机器和持续 CI。
- LTX2 service trace 显示 QK norm + split-RoPE 是 top hotspot，且 Inductor/Triton 仍有明显 launch或中间 tensor 开销。
- checkpoint 的 norm axis、head layout、cos/sin broadcast 和 dtype 已固化。

在此之前不冻结 schema，避免照抄上游 ABI 后再为本项目 layout 做二次转换。

### 6.2 K4：Blackwell NVFP4/MXFP

SM100 与 SM120 分成两个 provider/wheel target。候选能力按递增顺序：

1. quantize only。
2. scaled-MM + scale epilogue。
3. bias + GELU/SiLU。
4. modulation + quant。
5. MM + residual-gate。

每一级都必须与 `worldfoundry/core/acceleration/nvfp4.py`、`triton_nvfp4.py` 和目标 PyTorch 官方路径竞速；只移植胜出的最小集合。不得把 LightX2V 统一编译为 SM120a 的产物用于 B100/B200。

### 6.3 K5：ring KV

现有 `BlockKVCache` steady-state 会 clone/copy 本地窗口。优先级顺序：

1. 将物理 roll 改为 write pointer + sink/ring logical layout。
2. 让 attention backend直接消费两个 segment 或 block table。
3. 若 backend 必须连续输入，复用固定 workspace materialize。
4. 只有 materialize 进入 top hotspot，才写 `kv_ring_materialize.out` CUDA copy kernel。

不要先写一个更快的“整窗 roll” kernel，因为它保留了 O(window) 每步搬运这一错误复杂度。

### 6.4 K6：VAE tile stitch

只有 profile 证明 `cat/pad/blend/copy_` 的 tile overlap 处理消耗显著，才设计：

```text
vae_tile_blend_accumulate.out(tile, weight, offset, accum, norm)
vae_tile_finalize.out(accum, norm, output)
```

首选先由 `torch.compile` 融合；native 版本需要证明减少 global-memory pass，而不是只减少 Python 行数。

## 7. 不需要 CUDA/C++ 的具体工作

| 工作 | 正确实现层 | 原因 |
|---|---|---|
| typed optimization plan | Python dataclass/composition | 策略、依赖、冲突不是 kernel |
| compile cache 接线 | `runtime/compile_cache.py` + loader | 已有实现，当前只是绕过 |
| CUDA Graph runner | Python/CUDA Graph API | 关键是稳定 storage、warmup、signature，不是新 `.cu` |
| exact attention selector | attention registry + benchmark cache | 复用官方/外部实现 |
| merged QKV/KV | post-load weight transform | 一个 linear 后 view/split 即可 |
| static cross K/V、RoPE cache | runner request state | 生命周期和 cache key 是核心 |
| QKV all-to-all 合包 | existing `all_to_all_many` + workspace | collective 已由 NCCL 实现 |
| AdaLN/residual/RoPE/activation | 现有 Triton + Inductor | 已有公共 kernel，先接入 |
| TeaCache/EasyCache/token pruning | Python schedule + tensor ops | 算法/质量门禁优先 |
| sparse metadata | CPU/GPU metadata cache | 先复用 provider，避免每层构建 |

## 8. PR 级实施清单

### 8.1 必须先完成的非 native 基础

| PR | 主要文件 | 交付物 | 退出条件 |
|---|---|---|---|
| OPT-00 | 扩展 `runtime/performance.py`、benchmark cases/compare | case/protocol、paired raw samples、完整 fingerprint/status | 现有 manifest round-trip；一个真实 smoke case 可复现 |
| OPT-01 | lifecycle observer、独立 shape trace | cold/first/steady、CUDA Event、VRAM、top-op summary | timed pass 无 profiler/hooks；默认行为不变 |
| OPT-02 | `core/model_loading/policy.py`、diffusion plan | typed policy、capability/effect/conflict、manifest | 默认配置行为不变，冲突在 build 时拒绝 |
| OPT-03 | loader + `runtime/compile_cache.py` + adapter region | whole-module 兼容迁移 + 真正 compile island | 一个 adapter region 可报告 variant/cold/break-even |
| OPT-04 | attention selector | equivalent backend per-shape cache | A100 上 Torch/FA2 可复现选择与回退；Sage 不进 exact |

### 8.2 Native 基础与 K1

| PR | 主要文件 | 交付物 | 依赖/退出条件 |
|---|---|---|---|
| NAT-00 | optional package shell | 纯 C++ dispatcher DSO、私有 `_build_info`、sidecar、pre-dlopen ABI 检查、CPU-safe lazy loader | 无 `.so`/坏 sidecar/错误 ABI 安全；dispatcher smoke 通过 |
| NAT-01 | public registration/meta/provider/registry | functional + `out -> ()` schema、fake、`cuda_ext` backend、cache-v2 N-way read path | opcheck/AOT/Inductor；test stub 验证选型及错误 ABI 降级 |
| NAT-02 | K1 Torch wrapper + benchmark | canonical reference、real-shape benchmark、promotion report | profile 证明值得写 CUDA，否则停止 K1 |
| NAT-03 | `causal_conv3d_cat_pad_cuda.cu` | functional/out、stream/graph safe kernel | parity、sanitizer、microbenchmark 全通过 |
| NAT-04 | canonical Wan VAE adapter | Wan/Cosmos 接入、pipeline A/B、manifest | VAE 与端到端门槛通过，否则默认禁用/撤出 wheel |

当前实现进度：NAT-00 的独立 package、纯 C++ dispatcher DSO、sidecar-first ABI 校验、thread-safe/terminal-failure-safe lazy loader、分阶段 provider 状态和本地 smoke tests 已落地；PEP 517 wheel 构建与跨 Python/Torch/平台 CI matrix 尚未形成发布资格，因此 NAT-01 不得把本地 smoke 当作 wheel release gate 已通过。wheel gate 还必须解包最终 artifact，验证 LICENSE、DSO/sidecar SHA 与 RECORD，并证明没有误打包 libtorch；任何 strip/auditwheel/ELF repair 若改变 DSO，必须重新生成 sidecar 与 wheel RECORD 后再验签。

### 8.3 A100 INT8

| PR | 主要文件 | 交付物 | 依赖/退出条件 |
|---|---|---|---|
| QNT-00 | public quant reference/tests | rounding、zero-row、scale layout、artifact version | Torch/Triton/family-local 三者一致 |
| QNT-01 | external benchmark adapter | FastVideo/官方/Triton/BF16 top-10 shape report | native CUTLASS 有明确可赢空间 |
| QNT-02 | K2a | per-token quant functional/out | parity、stream/graph、quant latency门槛 |
| QNT-03 | K2b | CUTLASS scaled-MM functional/out | GEMM 与 quant+linear 两级门槛通过 |
| QNT-04 | public quantized linear transform | weight artifact、layer include/exclude、fallback | Wan 单层/单 step 质量通过 |
| QNT-05 | K2c | GELU、residual-gate epilogue 候选 | 仅在 QNT-04 模型收益不足且 epilogue 是剩余热点时做 |
| QNT-06 | balanced preset | 完整 Wan A/B 与质量报告 | denoiser、端到端、显存、质量共同达标 |

### 8.4 硬件独立路线

- Ada：先测官方 FP8/现有 `Float8Linear`，只有 epilogue 明显受限才新增 SM89 native。
- Hopper：FA3/FP8/external sparse provider 并行推进；不等待 A100 INT8。
- SM100/103：LTX K3 与 NVFP4 分开验收。
- SM120/121：独立验证 SageAttention3 与 LightX2V low-precision 候选；不继承 SM100 结果。

## 9. 测试矩阵

### 9.1 每个 exact op

- supported dtype × representative/edge shape × padding/layout。
- bitwise op 使用 `torch.equal`；numerically-equivalent op 按 op/dtype fixture 定义 tolerance，不能使用全局 tolerance。
- non-contiguous、negative padding、wrong dtype/device 的 wrapper fallback。
- native direct call 的非法输入明确报错。
- functional `torch.library.opcheck` 全项；FakeTensor/meta、AOT eager、Inductor/fullgraph、export 和 dynamic symbolic shape。
- out schema 返回 `()`，验证 mutation/functionalization；Python wrapper 返回原 out，out pointer 在 replay 前后稳定。
- wrapper requires-grad fallback + backward；direct op requires-grad 明确报错。
- non-default stream、非 current device、同进程两张不同 GPU。
- CUDA Graph capture/replay 至少 100 次。
- 重复调用 1000 次无内存增长/race。
- `compute-sanitizer --tool memcheck`；共享内存/同步 kernel 增加 racecheck。

### 9.2 每个 approximate op

除上述工程测试外，再增加：

- quant codes/scales 与 canonical reference 对照。
- 有限输入的 error distribution、cosine 和逐层 amplification；NaN/Inf 只测试 debug/reference wrapper 的前置条件拒绝，release direct op 不要求未定义 parity。
- 单 block、单 denoise step 的 latent drift。
- layer/step schedule 与 dense boundary。
- 完整正式生成 benchmark 和人工样例。

### 9.3 构建/加载

- 子进程覆盖主包无 native wheel、损坏 `.so`、corrupt/stale sidecar、缺 symbol、错误 SM、错误完整 Torch build、错误 CXX11 ABI；sidecar 不匹配时断言未调用 `dlopen`。
- CPU-only/document import 不触发 CUDA 初始化；并发 lazy-load 只执行一次且所有线程得到同一状态。
- Python 3.10～3.13 中，只对 native support matrix 声明的 CPython/Torch 组合发布 wheel；其他组合明确 fallback。
- wheel 的 `build_manifest.json` 至少含 manifest/operator ABI、schema hash、build ID、source commit/tree hash、完整 Torch version、Torch CUDA、NVCC、host compiler/libstdc++、CXX11 ABI、DSO SHA-256、SM/SASS/PTX targets、CUTLASS commit、编译 flags和 `links_libtorch_python`。
- capture 前完成预选/加载；capture 内缺记录、load 或 fallback 必须失败。OOM/fatal CUDA error 的测试断言不执行第二条 Torch/provider 路径。

## 10. Benchmark 产物要求

每个 operator 结果至少保存：

```json
{
  "op": "causal_conv3d_cat_pad",
  "model": "wan",
  "stage": "vae_decode",
  "shape": {},
  "dtype": "bfloat16",
  "stride": [],
  "device": {},
  "software": {},
  "provider": "torch|inductor|triton|cuda_ext",
  "build_id": null,
  "library_sha256": null,
  "median_us": 0.0,
  "p10_us": 0.0,
  "p90_us": 0.0,
  "peak_allocated_bytes": 0,
  "parity": {},
  "eligible": true,
  "fallback_reason": null
}
```

INT8 另加 `(M,N,K)`、quant schema version、scale layout、quant time、GEMM time、epilogue time与质量 artifact ID。只有完整记录 provider/build ID 的结果才能进入 persistent selection cache。

## 11. Definition of Done

Native 工作完成不是“能 import `.so`”，而是：

- N0 的 optional package 在无 CUDA/无 wheel 环境安全降级；最终 wheel 解包校验通过且不 vendor libtorch。
- K1 有真实 Wan/Cosmos shape 证据、exact parity、compile/graph/stream/multi-GPU 测试和端到端收益。
- K2 若未过门槛，明确停在 benchmark provider，不把 CUTLASS 维护成本带入 release。
- 每个已发布 op 都能从 manifest 看见 requested、available、eligible、selected、build ID 和 fallback reason。
- 没有任何 model 文件直接 import native extension；所有调用经过 `core/kernels` 公共 wrapper/registry。
- 外部来源代码保留精确 commit、SPDX/copyright、修改说明和第三方 notice。
- unsupported hardware、Torch/CUDA ABI 或 shape 永远有 Torch/Triton fallback，不出现“配置成功但实际静默错误”或 import crash。
