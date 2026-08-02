# WorldFoundry 推理算子与运行时优化计划

> 状态：执行版 v1；NAT-00 本地骨架与 dispatcher smoke 已完成，F0 是当前主线；任何模型算子仍需 shape/profile promotion gate
> 更新：2026-07-31
> 范围：`worldfoundry/base_models/diffusion_model` 及其复用的 `worldfoundry/core` 推理基础设施
> 发布资格目标：A100/A800、L40S/RTX 4090、H100/H200、B100/B200 与 RTX 50 系列；每种架构只以对应真机结果声明支持
> 最近开发快照（非发布基线）：2026-07-31，1 × A100-SXM4-80GB、Python 3.12.3、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、Torch CUDA/NVCC 12.8、driver 580.105.08、CXX11 ABI `True`
> 参考仓库：flashdreams、FastGen、FastVideo、sglang、Sana、LightX2V
> CUDA/C++ 逐算子规格：[`cuda_cpp_operator_spec.md`](./cuda_cpp_operator_spec.md)  
> 训练与 RL post-training：[`training_rl_post_training_plan.md`](./training_rl_post_training_plan.md)

本文是阶段、依赖和验收的唯一真源；schema、ABI、shape 数学、错误语义和逐算子门槛只在 `cuda_cpp_operator_spec.md` 维护。环境信息必须由每次 benchmark/build manifest 动态采集，以上快照仅用于解释当前能本地验证什么。

## 1. 结论先行

WorldFoundry 当前最需要的不是再复制一批孤立 CUDA/Triton kernel，而是把已经存在的 attention dispatch、DiT 融合算子、FP8/NVFP4、CUDA Graph、compile cache、KV cache、稀疏 attention 和 token pruning 组织成一条可测量、可组合、可回退的 native diffusion 优化路径。

代码盘点得到的关键信号：

- `worldfoundry/core/attention` 已经有 Torch SDPA、FlashAttention 2/3、SageAttention/SageAttention3、xFormers、PISA、VSA、V-MoBA、SLA 等探测和 dispatch 基础。
- `worldfoundry/core/kernels` 已经有 SiLU/GLU、GroupNorm+SiLU、residual+gate、LayerNorm/RMSNorm+scale+shift、QK norm+RoPE、3D RoPE 等 Torch/Triton 双路径以及持久 autotune cache。
- `worldfoundry/core/acceleration` 已经有 FP8、NVFP4、固定步缓存、自适应残差缓存和 token pruning。
- `worldfoundry/runtime/compile_cache.py` 和 `worldfoundry/core/utils/cuda_graph.py` 已经存在，但 native diffusion loader 仍直接调用 `torch.compile(module)`，扩散模型侧没有 CUDA Graph 接入。
- 2026-07-28 的静态盘点中，13 个 diffusion network family 只有 15 个文件直接导入通用 kernel；这些数量是审计快照，不作为长期事实，P0 shape trace 才是排期依据。
- 同一快照中，通用融合调用集中在少数文件；这说明首要任务是可观测接线和真实命中率，而不是按文件搜索结果预判热点。
- attention 接入相对较好，约 62 个 diffusion network 文件使用 WorldFoundry attention，但 `auto` 的安全默认候选目前仅为 Torch；不能在没有 shape benchmark 的情况下直接把全局默认改成某个外部 kernel。
- 现有测试更偏功能/契约，尚未形成覆盖 operator、单步 denoiser 和完整 pipeline 的专用性能门禁。

因此建议按以下优先级推进：

1. 先建立 shape trace、微基准、阶段计时、精度对照和性能产物格式。
2. 建立 typed optimization plan，把已有能力真正接入 loader、assembler、runner 和模型 adapter。
3. 优先落地等价性保持路径：compile islands、稳定 shape、exact attention 选型、已有融合算子、静态 conditioning/KV/RoPE 缓存、稳态 CUDA Graph。
4. 再做硬件相关低精度：A100 以 BF16/INT8 候选为主；FP8 在 Hopper/Ada/Blackwell 验证；NVFP4 只在 Blackwell 验证。
5. 最后独立推进有损优化：step/cache、稀疏 attention、token pruning，并用模型级质量门禁管理。
6. 对 autoregressive/windowed 和多卡模型，再引入预分配循环 KV cache、打包 QKV 通信和通信计算重叠。

## 2. 目标、非目标与成功标准

### 2.1 目标

- 降低稳态端到端推理时间，重点关注 denoiser、attention、MLP、VAE decode 和多卡通信。
- 降低 kernel launch、临时 tensor、重复投影、重复 layout 转换和 host/device 同步开销。
- 在输入 shape 可稳定的阶段利用 compile 和 CUDA Graph，并限制 graph/compile variant 数量。
- 用统一策略描述硬件能力、模型能力、step/layer/stage schedule、冲突和回退原因。
- bitwise-exact、numerically-equivalent 和 approximate 三类路径严格分离；默认配置不因优化而静默改变生成语义。
- 每项优化都能回答：在哪种 GPU、模型、shape、dtype 下有效，收益是多少，精度代价是多少，失败时退回什么。

### 2.2 非目标

- 不在第一阶段重写所有模型网络。
- 不盲目搬运参考仓库的完整 runtime 或所有 kernel。
- 不用单个微基准结果替代端到端收益。
- 不在 A100 上把 FP8/NVFP4 当成近期主线。
- 不把稀疏 attention、缓存跳步或 token pruning 设为默认。
- 不用大量进程级环境变量代替 typed policy。

### 2.3 建议的阶段性 Go/No-Go 门槛

以下数字是研发门槛，不是最终性能承诺；P0 基准完成后应按模型和硬件校准。

| 类型 | 合入最低门槛 | 正确性要求 |
|---|---:|---|
| 单个等价融合算子 | 目标 shape 中位数至少 1.10×；无显著长尾回退 | 声明 bitwise 或逐算子 tolerance，并全部通过 |
| 等价模型改造 | denoiser 至少 1.10×，或端到端至少 1.05× | 固定 seed 的逐步 latent/最终输出通过预注册 tolerance |
| compile/CUDA Graph | 稳态端到端至少 1.05×；variant 数量有上限 | 无额外 graph break、无错误缓存、动态输入可靠回退 |
| 低精度 | 支持硬件上 denoiser 至少 1.15× | 通过预先声明的模型质量预算 |
| cache/sparse/prune | 端到端至少 1.30× 才值得承担复杂度 | 正式 benchmark 与产品样例均通过质量预算 |
| 多卡通信优化 | 目标并行度至少 1.10×，且扩展效率改善 | 各 rank 一致、无死锁、结果在并行误差范围内 |

所有研发门禁使用同机交错 paired A/B；开发阶段至少 5 对有效样本，release 至少 10 对，保存原始样本并报告 median、p10、p90、MAD/median 和 paired 95% bootstrap CI。任何优化若只改善微基准、绝对 stage 占比不足，或让任一主要端到端 case 回退超过 2%，应默认关闭或撤销。compile/graph 还必须报告 cold cost、break-even request 数、variant 数和峰值显存预算。

### 2.4 等价性等级

- `bitwise-exact`：复制、重排、零填充等无浮点归约变化的算子；例如 K1 对 eligible 输入必须逐 bit 相等。
- `numerically-equivalent`：compile、attention 或融合改变浮点运算/归约顺序，但通过 op-specific tolerance 和模型等价门禁。
- `approximate`：INT8/FP8/FP4、SageAttention/SageAttention3、稀疏 attention、cache 跳步和 token pruning；只能由 `balanced`/`aggressive` 或显式自定义 plan 启用。

## 3. 参考仓库带来的具体启示

| 参考仓库 | 可借鉴机制 | WorldFoundry 对应落点 | 注意事项 |
|---|---|---|---|
| flashdreams | compile wrapper、`max-autotune-no-cudagraphs`、显式 autotune drain、静态输入 CUDAGraph、稳态 cache 才 capture、预分配 KV、融合 3D RoPE | `runtime/compile_cache.py`、`core/utils/cuda_graph.py`、native runner、`core/attention/kvcache.py` | WorldFoundry 已有相似基础，不重复复制；重点补调用契约和稳态管理 |
| FastGen | 每层预分配 self/cross KV、显式 cache index、cond/negative cache 隔离 | autoregressive/windowed runner、BlockKVCache | 主要是训练仓库，只采纳 inference 中的 cache 数据结构思想 |
| FastVideo | attention metadata builder、VSA/VMoBA/SLA、共享 scratch、合并 untile index、QKV 打包后 all-to-all、INT8/NVFP4 kernel | attention registry、shape metadata cache、sequence parallel、可选 kernel adapter | 外部 CUDA 扩展需独立做 license、ABI、构建和支持 GPU 审查 |
| sglang | breakable/bucketed CUDA Graph、显式 warmup capture、diffusion kernel registry、compile islands、varlen pack/unpad、零拷贝 padding、通信优化 | graph policy、kernel registry、multimodal packing、distributed attention | 不把服务端全局状态模式直接移植到 instance-local native runtime |
| Sana | 合并 QKV、EasyCache、cache 控制流置于 compile 外、LTX 多融合 preset、step/layer selective FP4、dense boundary steps、typed technique composition | post-load transforms、cache schedule、LTX/Cosmos adapter、optimization composition | 其 fullopt 含 approximate 技术，需拆成单项消融，不能整体视为等价路径 |
| LightX2V | fused norm/modulation、量化循环 KV、sink+ring buffer、CPU pinned offload、按组件 attention、多种 SP/CFG topology | `core/kernels`、KV policy、distributed policy | backend 很多但硬件适配强；只接入能通过本项目基准的实现 |

借鉴原则：优先复用设计模式和接口思想。若需要移植具体源码，必须先记录来源 commit、许可证、修改说明、CUDA/PyTorch ABI 和目标架构；不直接 vendor 整个上游 runtime。

## 4. 总体架构设计

### 4.1 将 `RuntimePolicy` 扩展为可组合、可审计的优化计划

当前 `RuntimePolicy` 只有 device、dtype、attention、offload、quantization、compile bool 和无类型的 `options`。建议保留向后兼容入口，同时把已有公共类型扩展为嵌套 typed policy；不得在新目录再定义同名 `CompilePolicy`、capability 或 manifest：

```text
RuntimePolicy
├── placement / dtype / offload
├── attention: AttentionPolicy
├── compile: CompilePolicy
├── graph: GraphPolicy
├── fusion: FusionPolicy
├── quantization: QuantizationPolicy
├── cache: CachePolicy
├── token_pruning: TokenPruningPolicy
├── distributed: DistributedPolicy
└── observability: OptimizationTelemetryPolicy
```

plan 生命周期固定为 `resolve -> validate -> apply -> report`。resolved plan 是 runner/request scoped immutable value；环境变量只在 resolve 边界读取一次，模型 forward、attention dispatcher 和 kernel registry 不得继续把进程级环境变量当作实际请求状态。每个 optimization technique 至少声明：

- `technique_id` 和版本。
- `bitwise-exact`、`numerically-equivalent` 或 `approximate`。
- 运行阶段：load transform、build transform、warmup、per-step、post-step。
- 所需 capability：GPU 架构、dtype、shape/alignment、模型 seam、可选扩展。
- 影响的独占 seam，例如 attention backend、token set、step output、FFN precision、KV layout。
- 与其他 technique 的依赖和冲突。
- schedule 维度：model、stage、branch、step、layer、shape、device profile。
- 回退实现及回退原因。
- 精度等级和对应测试集合。

组合器必须在模型加载或 runner build 时拒绝冲突，而不是等到推理中途报错。例如：

- 同一层不能同时使用两个 attention backend。
- token pruning 与依赖固定 token grid 的 sparse metadata 必须有显式适配。
- 显式 CUDAGraph 与 Inductor 自带 cudagraph 不能重复控制同一 region。
- block offload 和 post-load weight fusion 的顺序必须固定为“加载权重 → fusion/quant transform → 安装 offload hook”。
- CUDA Graph 不能 capture 会发起动态 collective、更新 Python cache 决策或改变 tensor storage 的 region。

### 4.2 通用机制归 `core`，模型 seam 归 diffusion adapter

扩展现有真源，而不是建立平行系统。建议的未来目录不要求一次性创建完：

```text
worldfoundry/core/optimization/
  composition.py        # dependency/conflict validation
  schedules.py          # stage/step/layer/branch schedules
  telemetry.py          # 只编排 requested/effective/fallback 事件

worldfoundry/core/model_loading/policy.py  # RuntimePolicy 及 typed 子策略唯一真源
worldfoundry/runtime/compile_cache.py      # CompilePolicy 与 compile cache 唯一真源
worldfoundry/runtime/platforms/            # 硬件 capability 唯一真源
worldfoundry/runtime/performance.py        # PerformanceManifest/Fingerprint 唯一真源

worldfoundry/base_models/diffusion_model/optimizations/
  plan.py               # diffusion-facing plan resolution
  adapters/
    sana.py
    wan.py
    ltx.py
    cosmos.py
    hunyuan_video.py
  presets.py            # explicit exact / balanced / aggressive presets

benchmarks/operators/
  attention.py
  dit_fusions.py
  quantized_linear.py
  vae.py
  kv_cache.py
  shapes/

benchmarks/inference/
  native_diffusion.py
  compare.py
  cases/                  # 冻结 case；结果继续写 PerformanceManifest
```

模型网络只暴露稳定 seam，例如 block 列表、attention module、FFN linears、conditioning projector、VAE tile function；优化 manager 不应通过模型名字符串和脆弱的全局 monkey patch 猜测结构。

### 4.3 每次推理扩展现有 performance manifest

复用 `worldfoundry/runtime/performance.py::PerformanceManifest`、`OptimizationSnapshot` 和 `RuntimeFingerprint`；只升级 schema/extension，不创建第二种 optimization manifest。每次运行至少记录：

- `case_id`、protocol version、case/source digest、prompt-set digest、seed 集、解析后的 recipe/profile 值和 checkpoint revision/digest。
- WorldFoundry commit、dirty-tree 状态与 diff hash；PyTorch/Triton/CUDA/cuDNN/NCCL/driver、全部可见 GPU 与拓扑。
- 每层/每 stage 实际选用的 attention、fusion、quant、cache、graph 和 distributed backend。
- compile/graph signature、cache hit、fallback 原因、graph break/recompile 次数。
- equivalence level（bitwise/numerical/approximate）和 preset 名称。
- raw cold/warmup/steady paired samples、valid/invalid 状态与错误、stage latency、peak allocated/reserved VRAM 和最终质量评估 ID。

这样性能结果可复现，也能避免“配置写了 FP8，但实际一直 dense fallback”的假加速。

## 5. P0：先建立可信基准和 shape 证据

P0 分成两个不会互相阻塞的 gate：

- **P0a Foundation gate**：复用并扩展 `PerformanceManifest`，冻结 benchmark case/protocol，接入 no-op lifecycle observer，并完成 1 个可复现真实 smoke case。P0a 完成后即可启动有基准保护的 P1。
- **P0b Evidence lanes**：其余模型、service/stress shape 与 Ada/Hopper/Blackwell 数据按资源并行补齐；某硬件/模型只有拿到本 lane 的证据后才能声明资格，但不阻塞其他 lane。

### 5.1 端到端基准矩阵

每个 pilot model 至少定义三档请求：

- `smoke`：recipe 支持的最小实用 shape，8 个 denoise steps，用于快速 CI 和精度定位。
- `service`：catalog/profile 的默认产品 shape 与默认 steps，作为主要优化指标。
- `stress`：目标高分辨率、长时序或高 batch，用于显存和扩展性验证。

首批模型：

1. Sana/Sana-WM：较小模型，用于验证 policy、compile island、QKV merge 和 cache 组合流程。
2. Wan 2.1/2.2：代表性视频 DiT，用于 attention、AdaLN、FFN、TeaCache、VAE 和 Ulysses。
3. LTX：多 stage、多模态路径，用于 stage schedule、token/attention 优化和 tiled VAE。
4. Cosmos 3：大模型和 Blackwell 低精度目标；当前 A100 只跑 dense/compile 基准。
5. 一个 autoregressive/windowed recipe：用于 KV cache 和稳态 graph。

每个 case 必须有稳定 `case_id`，并冻结 recipe/profile 的唯一来源、checkpoint revision/digest、prompt-set digest、seed 集、解析后的像素/帧数/step 数、dtype、warmup 收敛规则和无效运行判定。benchmark 脚本不得从 recipe 与 catalog 二选一或另写一份常量；解析值及 source digest 必须固化进 manifest。

### 5.2 计时方法

- 端到端同时记录 cold start、首次请求、warm steady-state；不能把编译时间混入稳态，又不报告编译成本。
- 编译/autotune 路径至少 warmup 到 graph/compile variant 稳定；记录每次 warmup，而不是固定假设一次足够。
- 单算子用 CUDA Event；端到端用 wall clock，并在区间边界显式同步。
- 微基准每个 shape 至少 50 次测量；pipeline 开发门禁至少 5 对、release 至少 10 对交错 paired A/B，保留所有 raw samples。
- 独占 GPU，记录 GPU clock/power 状态和其他进程；同一 A/B 实验在同一节点交替执行。OOM、thermal throttle、外部进程干扰或配置/产物不一致必须标为 invalid，不能静默丢弃或计入结果。
- 记录 host-to-device、condition encoder、latent init、每个 denoise step、VAE decode、postprocess。
- shape/profile pass 与 timed pass 分离；`torch.profiler`、module hook 和 shape recorder 不得污染性能门禁。关键候选再用 Nsight Systems 看 launch/sync，用 Nsight Compute 看 occupancy、Tensor Core、HBM 和 fusion 是否真正生效。

### 5.3 自动采集真实 operator shapes

增加只在 profiling 模式启用的 shape recorder，采集：

- attention：B、heads、Q/K length、head dim、mask、GQA、dtype、layout、stage/step/layer。
- linear：M/N/K、bias、activation、权重 dtype、是否 FFN/QKV/out projection。
- norm/modulation：token 数、hidden dim、广播维度、连续性。
- VAE：N/C/T/H/W、kernel、stride、padding、tile 边界。
- KV cache：容量、写入长度、有效窗口、sink token、量化格式。
- 通信：tensor bytes、all-to-all/all-gather/reduce-scatter 耗时和等待时间。

聚合后只保留 shape 频次和元数据，不落盘真实 prompt/latent 内容。

### 5.4 P0a Foundation 交付物与退出条件

- `benchmarks/operators` 和 `benchmarks/inference` 可独立运行，一条命令原子输出现有 schema 的 JSON + Markdown summary；失败运行也保存 status/error。
- no-op observer 不改变默认输出；一个冻结的 Sana 或 Wan smoke case 完成至少 5 对有效 baseline，第二次运行可比较，不兼容 fingerprint 默认拒绝比较。
- timed pass 不带 profiler/hooks；trace pass 只保存聚合 shape/频次和 top-op/allocation/layout/graph-break 信息，不保存 prompt/latent 内容。
- manifest 能说明实际可见 GPU、软件栈、dirty tree、case/checkpoint/prompt digest、raw samples 与实际 backend。
- 每个后续优化 PR 同时增加 correctness test 和最小 regression comparison；性能治理从 P0 开始，不等到 P6。

### 5.5 P0b Evidence lane 退出条件

- 当前可见设备只决定本次能产出哪条 evidence，不写进源码或 case；world size 从 manifest 获取。
- Wan、Sana、LTX 逐步补齐 smoke/service/stress；Ada/Hopper/Blackwell 使用同一 protocol 在对应真机执行。
- 微基准 MAD/median 目标小于 3%，端到端目标小于 5%；超出时先修环境。
- P1/P2 的模型工作项必须对应到 profile 热点或重复开销；没有证据的 kernel 只进 backlog。

## 6. P1：统一接线、compile islands 与 exact attention

### 6.1 修复 compile 接入断点

当前 `NativeModuleLoader` 直接 `torch.compile(module)`。把它改为 `compile_module_cached` 只解决 cache/option 一致性，**不等于 compile island**；迁移层不得继续把整个未知 module 当成推荐编译区域。`RuntimePolicy.compile` 从 bool 兼容升级为已有 `CompilePolicy` 后，`ModuleLoadSpec`/模型 adapter 还必须显式声明稳定 region，由 runner 对 region 使用 `compile_callable_cached` 或等价 wrapper。

默认策略：

- 优先编译稳定的 transformer block loop、单个 block、FFN 和 tiled VAE region，而非整条 pipeline。
- cache 决策、Python 状态更新、动态 pack/unpack、checkpoint/offload hook 和 distributed collective 留在 compile island 外。
- 使用显式 CUDAGraph 时，Inductor mode 优先 `max-autotune-no-cudagraphs`；不使用显式 graph 时再比较 `reduce-overhead`、`default` 和 `max-autotune`。
- compile key 必须包含 model revision、region ID/version、device fingerprint、dtype、shape bucket 和完整 compile options。
- 每个 region 的 variant 数设置上限，建议初始 16；整个 runner 建议初始 32。超过上限改走 eager，不无限编译。
- warmup 阶段显式等待 compile/autotune 完成；服务请求过程中不得突然 capture 新 graph。
- 统计 graph break 和 recompile reason；出现 data-dependent shape 时先缩小 island，不用 `suppress_errors` 掩盖。

### 6.2 exact attention 不采用盲目全局优先级

保持 Torch SDPA 为安全 fallback，但新增离线或显式 warmup 的 numerically-equivalent attention selector：

- 候选：Torch SDPA 内部 cuDNN/Flash/Efficient backend、经过本项目 tolerance 验证的 FlashAttention 2、FlashAttention 3 和 xFormers。
- selector key：GPU/SM、dtype、Q/K length、head dim、head count、GQA、mask、layout、causal、是否 sequence parallel。
- 先做 capability 和数值校验，再测 latency/peak memory；选择结果写入现有 persistent selection cache。
- 线上请求只查表，不在真实请求中做多 backend 竞速。
- capability 不满足、显式 unsupported 或可恢复加载错误可在执行前选择 SDPA 并记录原因。OOM、illegal address、device assert、launch/capture failure 不在同一请求内吞掉后重试；只有完整请求边界能恢复 RNG/scheduler/cache state 时才允许调用方显式 retry。
- SageAttention/SageAttention3、PISA、sparse/quantized attention 不进入 exact selector；它们是 approximate provider，只能由 `balanced`/`aggressive` policy 单独启用。

当前 A100 exact 首轮只比较 Torch SDPA 与 FlashAttention 2；Ada 比较 Torch/FA2/xFormers，Hopper 比较 Torch/FA2/FA3/xFormers。Sage/FP8/PISA 另走 approximate lane，FlashAttention 3、SageAttention3/NVFP4 不在 SM80 上尝试。

### 6.3 建立稳定 shape 和静态 buffer 契约

- 每个 runner 明确哪些维度可 bucket：batch、latent T/H/W、text length、CFG branch、stage。
- 文本 padding、varlen pack/unpack、RoPE index 和 sparse metadata 在 warmup/build 阶段预计算并缓存。
- scratch buffer 从 runner workspace 获取，不在每个 block/step 重复分配。
- 缓存 key 必须包含所有影响结果或 shape 的值，不能只使用 model name。
- 对不规则请求使用最近合法 bucket + mask，或走 eager；不得错误重放旧 graph。

### 6.4 P1 退出条件

- native diffusion 的迁移入口不再绕过持久 compile cache，且 whole-module compile 不被误报为 island。
- P0a pilot 至少有一个 adapter 声明的稳定 compile region，能报告 region ID、variant、cold cost、break-even 和 fallback；其余 pilot 在对应 evidence lane 补齐。
- exact attention selector 的结果可复现、可清空、可禁用。
- 默认 exact preset 在固定 seed 下通过等价测试。
- 每种架构只在其 service evidence lane 达标后获得对应资格；缺少 Ada/Hopper runner 不阻塞 A100 代码进展，也不能提前宣称支持。

## 7. P2：扩大等价融合和静态计算复用

### 7.1 优先复用已有 kernel

按 profile 热点逐模型把以下公共算子接到稳定 seam：

1. `layer_norm_scale_shift` / `rms_norm_scale_shift`：AdaLN/AdaRMSNorm 调制。
2. `residual_gate_add`：attention/FFN 输出、gate 和 residual 合并。
3. `qk_rmsnorm_rope` / `hidden_qk_rmsnorm_rope_3d`：QK norm 与 RoPE。
4. `silu_mul` / `silu_and_mul`：SwiGLU/GEGLU 类 FFN。
5. `group_norm_silu`：VAE/residual block。

接入时必须比较三条路径：纯 eager Torch、`torch.compile` 自动融合、显式 Triton kernel。若 Inductor 已生成更优融合，就不强制显式 kernel。

每个 public fused op 需要：

- Torch reference 和 dtype/shape/stride/alignment eligibility。
- 对 non-contiguous、奇数 hidden dim、广播 modulation 的明确行为。
- backward 非目标时明确 inference-only，而不是静默产生错误梯度。
- 单测、随机 property test、真实 shape benchmark 和 fallback test。
- kernel dispatch report 能显示命中率和退回原因。

### 7.2 post-load projection fusion

借鉴 Sana 的 merged QKV，但以通用 build transform 实现：

- 当 Q/K/V 输入一致、bias/输出维度和 checkpoint 语义允许时，合成一个 `F.linear`，再 view/split。
- cross-attention K/V 可合并；若 prompt conditioning 在所有 denoise steps 不变，进一步缓存投影后的 K/V。
- weight fusion 必须发生在 checkpoint 完整加载之后、offload hook 安装之前。
- 保留 state-dict provenance，保存/导出时能还原或明确只支持 inference artifact。
- 对 LoRA/adapter：若运行时仍需动态切换，不能永久 merge；若 LoRA 已烘焙进权重，则重新构造 fused weight。
- GQA/MQA、QK norm、不同 bias 和量化 linear 必须单独校验，不能只按模块名匹配。

### 7.3 静态 conditioning、RoPE 和 metadata 缓存

在同一请求的 denoise steps 间复用：

- text/image/audio/action conditioning encoder 输出。
- cross-attention projected K/V。
- 不随 step 改变的 RoPE frequency/index。
- varlen pack/unpack index、padding mask、tile/untile index。
- stage 内固定的 sparse block metadata 和通信 layout。

缓存 key 至少包含 checkpoint/adapter revision、conditioning tensor identity 或内容 hash、dtype/device、shape、CFG branch、stage 和相关 policy。输入变化、LoRA 切换、模型权重 mutation 或 runner reset 时必须失效。

### 7.4 VAE 专项

只有当 profile 证明 VAE 占比较高时再推进：

- compile tile decode region，固定 tile shape 并预分配 overlap buffer。
- 评估 causal Conv3D pad、GroupNorm+SiLU、residual add 等融合。
- 清理 tile loop 中 `.item()`、CPU shape 计算和频繁同步。
- 评估 channels-last/3D memory format，但以完整 decode 收益和数值一致为准。
- 长视频 decode 优先降低峰值显存和重叠拷贝，再追求单 kernel 峰值。

### 7.5 模型落地顺序

- Sana：验证 merged QKV、compile boundary 和已有 fastlinear kernel 与公共 registry 的关系。
- Wan：扩大 AdaLN、residual gate、QK norm+3D RoPE 覆盖，并清理重复实现。
- HunyuanVideo/GammaWorld：已有公共算子调用，可作为回归样板。
- LTX：按 stage 接入 dual modulation、FFN、attention 和 VAE fusion。
- Cosmos：先完成 dense exact 路径，再为后续低精度准备 layer seam。

### 7.6 P2 退出条件

- 所有接入模型的融合命中率、fallback 原因可观察。
- 至少 Wan 和 Sana 完成 exact preset 的端到端 A/B。
- 不新增 family-local 通用 Triton kernel；确属通用能力必须放到 `core/kernels`。
- 重复 TeaCache、norm/activation/RoPE 实现有明确 consolidation 清单，不在同一路径维护两个真源。

## 8. P3：硬件感知的低精度线性层

低精度是独立 opt-in technique，不能复用 `dtype` 一个字段含糊表示。

### 8.1 硬件矩阵

| 硬件 | 首选研究路径 | 不应尝试 |
|---|---|---|
| A100/SM80（当前可开发） | BF16/FP16、可能的成熟 INT8 weight/activation 路径、exact attention、compile | native FP8、NVFP4 |
| L40S/SM89 | BF16、FP8、FlashAttention2；Sage 仅 approximate lane | Hopper 专属 FA3 |
| H100/H200/SM90 | BF16、FP8、FA3、compile/graph；PISA/TMA 仅 approximate lane | NVFP4 |
| B100/B200/SM100/103 | BF16、FP8、NVFP4、Blackwell attention | 未明确编译目标的扩展 |
| RTX 50/SM120/121 | BF16、FP8/NVFP4 候选、SageAttention3 目标依扩展而定 | 把数据中心 Blackwell capability 直接等同消费卡 |

### 8.2 FP8

- 先使用已有 `Float8Linear` 做离线资格测试，验证真实走 `_scaled_mm` 而非 dense fallback。
- 优先替换计算占比高、误差敏感度较低的 FFN `gate/up/down`，随后才评估 QKV/out projection。
- 排除 input/output projection、VAE、condition encoder、极小 M/N/K 和不满足 alignment 的层。
- 比较动态 activation quantization 成本与 GEMM 收益；热点允许时增加预计算 weight scale 或预量化 artifact。
- calibration 期间保留 dense weight 便于逐层 A/B；production artifact 验证后可选择去掉 dense fallback，避免双份权重显存。
- 支持 step/layer schedule：首尾 step、stage 边界或高敏感层保持 BF16。

### 8.3 NVFP4

- 仅当 `kernel_device_profile` 明确支持目标 Blackwell 且 CUDA/PyTorch ABI 通过时启用。
- 第一目标仍是 FFN，沿用“首尾 step dense、中间 step FP4”的风险控制方式。
- 先逐层测量 cosine/error amplification，再跑完整生成质量；不能仅以单层 MSE 决定。
- 检查 `NVFP4Linear` 的动态 activation quantization、scale layout 和 alignment 是否与目标 GEMM 匹配。
- calibration 和 production 的 dense fallback 保留策略必须写入 manifest。

### 8.4 可选 INT8 外部 kernel

FastVideo/TurboDiffusion 的 INT8 GEMM 只作为 optional provider：

- 先完成许可证、wheel/build、SM80 支持和 PyTorch ABI 审查。
- 通过 WorldFoundry kernel registry 适配，不让模型直接 import 外部 package。
- 有 Torch/dense fallback，且 import 探测不得导致进程启动崩溃。
- 如果 A100 service shape 的 end-to-end 收益低于门槛，则不承担维护成本。

### 8.5 P3 退出条件

- 每种格式都有 hardware eligibility、逐层清单、实际 backend report 和质量报告。
- 不支持硬件直接在构建期拒绝或明确 fallback，不允许“成功启用但实际全 dense”。
- 组合矩阵至少覆盖 quant × compile、quant × attention、quant × offload。
- 正式 preset 默认仍为 BF16 exact；低精度 preset 名称显式包含精度等级。

## 9. P4：有损 cache、稀疏 attention 和 token pruning

### 9.1 统一 step/cache 基础设施

目前通用 `FixedStepCache`、`AdaptiveResidualCache` 和 Wan-specific TeaCache/runner cache 存在重复。先统一生命周期和事件模型，再加新算法：

- 每次 request/reset 都创建隔离 state，不使用进程全局状态。
- cache policy 声明 warmup steps、允许跳过的 step/layer/stage、误差阈值和强制 refresh schedule。
- cache decision 的 Python 控制流放在 compile island 外；真正的 block loop 保持稳定。
- cond/uncond/negative CFG 分支分别存储状态，但必须用一致且经过验证的决策规则。
- 如果算法使用共同 residual 修正 CFG，必须显式实现，不能让某个分支复用过期 residual。
- prompt/control/image/action/geometry 条件变化时强制失效。
- cache hit 并不等于速度收益；必须记录跳过 FLOPs、额外比较开销和实际 step latency。

候选顺序：固定 schedule cache → TeaCache/AdaptiveResidual → EasyCache 类在线变化率预测。每种算法单独消融后才能组合。

### 9.2 稀疏 attention

- 先使用已有 PISA/piecewise attention 建立 WorldFoundry 自有质量和性能基线。
- VSA、VMoBA、SLA、SageSLA 等外部 provider 通过 attention registry 接入，不直接散落到模型代码。
- metadata key 包含 T/H/W、text token layout、stage、step bucket、layer、sparsity、backend 和 sequence-parallel shard。
- metadata、tile/reverse/untile index 与 scratch buffer 预计算和复用。
- 默认采用 dense boundary：早期/末期 step、高变化 layer、stage 切换和不支持 shape 使用 exact dense attention。
- schedule 必须是 model adapter 声明的能力，不允许用一个全局 sparsity 覆盖所有模型。
- 稀疏 attention 与 CFG、multimodal prefix、causal mask、GQA 和 SP 分别测试。

### 9.3 Token pruning

- 第一阶段只在 token 重要性和恢复语义明确的模型/stage 使用。
- 保留原始 grid 坐标和稳定 restore index；恢复后顺序必须与未剪枝路径一致。
- text/audio/control token 默认禁止 pruning；只处理明确的视觉 latent token。
- sequence parallel 下各 rank 必须对保留集合达成一致，或使用可证明正确的局部分片协议。
- 与 RoPE、sparse metadata、residual cache 和 skip connection 逐对验证。
- 采用 step schedule：高噪声或模型已验证的中间阶段 pruning，边界 step dense。

### 9.4 质量门禁

有损优化不能只看输出视频是否“肉眼差不多”。每个 model/preset 在实验开始前登记质量预算：

- 固定 prompt/seed 的 latent trajectory、PSNR/SSIM/LPIPS 或适合该模型的感知指标。
- VBench、SANA-WM Bench 或该 recipe 已登记的正式 benchmark。
- 运动、时序一致性、文本遵循、物体稳定性、控制信号遵循的分项指标。
- 至少一组人工盲评样例，覆盖快速运动、镜头运动、细纹理、文字、多人/多物体和长时序。
- 不只看平均分；任一关键分项越过预算即 No-Go。

### 9.5 P4 退出条件

- 每种有损 technique 能单独启停，有独立 A/B 和质量报告。
- `balanced` preset 只包含通过预算的组合；`aggressive` 明确标注风险。
- 默认 exact preset 不因安装了某个可选 package 而自动进入有损路径。
- 组合优化按单项 → 两两组合 → 完整 preset 递增验证，禁止直接只测 fullopt。

## 10. P5：autoregressive KV cache 与多卡数据移动

### 10.1 预分配、循环和量化 KV cache

结合 flashdreams、FastGen 和 LightX2V 的共同模式，扩展已有 `BlockKVCache`：

- self-attention K/V 按最大 window/capacity 一次预分配，使用写指针或 block table 更新，不做每步 `cat`。
- static cross-attention K/V 在 conditioning 确定后只计算一次。
- sink token + circular/ring region；窗口滑动只更新索引，不物理 `roll` 整个 tensor。
- cond/uncond/negative、不同 stage 和不同 request 使用隔离 cache namespace。
- filling phase 保持 eager；shape 和指针稳定的 steady phase 才进入 CUDA Graph。
- graph replay 使用 `copy_` 更新预分配 buffer，不能替换 storage。
- 长窗口显存成为瓶颈后，再评估 KIVI 类 INT8/INT4 KV；量化 block、scale 和 sink token 需要独立精度测试。
- CPU offload 只对超长历史评估，使用 pinned buffer 和 GPU staging；先证明 PCIe/NVLink 传输没有抵消收益。

### 10.2 Sequence parallel / CFG parallel

多卡工作只在拥有对应 2/4/8 GPU runner 时验证；当前 1×A100 开发环境只能实现和单测拓扑无关的部分。取得 runner 后优先做：

- Q/K/V 在 all-to-all 前合并打包，减少三次 collective 和中间 contiguous。
- 比较 Ulysses、ring 和 hybrid 的通信量、head divisibility、长序列扩展效率。
- 异步 all-to-all 与 projection/FFN 重叠，但必须用 event/stream 证明真实 overlap。
- 对所有 rank 相同的 text prefix/suffix，评估复制小文本段、只分片视觉 token，避免冗余通信。
- CFG branch 并行与 sequence parallel 做拓扑搜索；离线选型，线上查 profile。
- collective 和 Python process-group 控制流保持在 compile/graph region 外；capture 前完成 NCCL 初始化和 warmup。
- 记录每个 collective bytes、duration、wait 和 rank skew，不能只报告总吞吐。

### 10.3 P5 退出条件

- autoregressive steady phase 无逐步分配增长，cache 指针和 graph signature 稳定。
- 2/4/8 GPU scaling curve 完整，报告 latency 和 parallel efficiency。
- 目标多卡 service shape 达到 Go/No-Go 门槛；小 shape 允许自动回退单卡或较低并行度。
- kill/retry、不同 rank 输入错误和 graph fallback 不会造成 collective 死锁。

## 11. G0–G6：贯穿全程的性能 CI、发布和回归治理

治理不是最后一个阶段：P0a 建 schema/CPU test/比较器，每个优化 PR 增加相应 GPU smoke，nightly 与 release 矩阵再随硬件 runner 逐步扩展。

### 11.1 测试层级

1. CPU/普通 PR：policy composition、capability、conflict、fallback、cache lifecycle、shape key 单测。
2. GPU PR smoke：少量真实 shape 的 operator parity 和短 diffusion request。
3. nightly Tier 1：A100、L40S/RTX4090、H100/H200 的完整 operator matrix 和 Wan/Sana/LTX service baseline；A100/Hopper 还需 2/4/8 GPU scaling。
4. hardware-qualified nightly：Ada/Hopper 的 FP8、Hopper 的 FA3/PISA，以及 Blackwell 的 NVFP4/MXFP/SageAttention3。
5. release qualification：exact、balanced、aggressive preset 的正式质量 benchmark 和长稳态运行。

### 11.2 性能回归规则

- benchmark artifact 绑定 commit、dirty-tree/diff hash、case/protocol digest、hardware/software fingerprint，不跨不兼容 fingerprint 直接比较。
- 用 paired raw samples、MAD 与置信区间判断，不用单次最低值。
- exact preset 若端到端回退超过 5% 则阻断；2%～5% 进入人工复核。
- approximate preset 同时检查速度和质量，任何一侧失败都不发布。
- 可选扩展构建失败不得破坏 baseline import；但明确请求该 backend 时应返回可操作的错误。

### 11.3 对外配置

建议最终提供：

- `exact`：只包含数值等价技术，默认。
- `exact-fast`：更激进的 compile/graph/exact backend，但仍不启用量化或跳步。
- `balanced`：经过模型级质量门禁的低精度/cache/sparse 组合。
- `aggressive`：最高速度，显式声明可能的质量变化。
- 自定义 typed plan：高级用户可覆盖单项 schedule。

每次运行都打印一行摘要并可输出完整 manifest，例如“requested FP8，SM80 不支持，实际 BF16 dense”，避免隐式降级造成误判。

## 12. 精度验证细则

### 12.1 逐算子

- 使用 FP32 Torch reference，BF16/FP16 输入同时比较绝对误差、相对误差、cosine 和 NaN/Inf。
- tolerance 按 dtype 和 op 建 fixture；不使用一个全局 tolerance。
- 覆盖连续/非连续、奇数尺寸、广播、短/长序列、极值和空 mask。
- attention 额外覆盖 GQA、custom scale、fully masked rows、causal 和 varlen。
- quantized op 记录误差分布而非只看最大值。

### 12.2 单 block 与单 step

- 相同权重/输入比较每个 block 输出，定位误差第一次放大的层。
- 固定 scheduler state 比较一次完整 denoiser step，包括 CFG combine。
- cache 命中和强制 refresh 分别测试；错误 cache key 用 mutation test 验证能失效。

### 12.3 完整生成

- 固定模型 revision、prompt、negative prompt、seed、scheduler、steps、shape。
- exact 路径记录逐步 latent drift 和最终像素/视频差异。
- approximate 路径跑正式 benchmark、感知指标和人工盲评。
- 多卡与单卡比较，区分正常浮点归约顺序差异与算法错误。

## 13. 风险与对策

| 风险 | 典型后果 | 对策 |
|---|---|---|
| 全模型 compile | graph break、编译爆炸、首请求极慢 | compile islands、signature bucket、variant cap、显式 warmup |
| graph key 不完整 | 错误重放旧输入或旧 cache | key 包含 tensor signature 和非 tensor 常量；状态变化时 reset |
| pointer 不稳定 | CUDAGraph replay 崩溃或静默错误 | 预分配、`copy_`、禁止 capture 中换 storage |
| 融合与 offload 顺序错误 | 参数占位符/shape 被破坏 | 统一 load transform 生命周期并写顺序测试 |
| 外部 kernel ABI 不匹配 | import crash、非法指令 | side-effect-free probe、独立 wheel matrix、registry fallback |
| auto backend 过度激进 | 某些 shape 变慢或 OOM | serving 只读预选记录；OOM/fatal CUDA error 原样抛出，不在请求中换 backend 重试 |
| 低精度双份权重 | 显存反而增大 | calibration 和 production artifact 分离 |
| cache 污染 CFG 分支 | 伪影、guidance 放大误差 | branch namespace、共同决策/残差规则、强制 refresh |
| sparse/prune metadata 失配 | token 顺序错误或越界 | 完整 shape/stage/step/SP key，恢复顺序 property test |
| 多卡 compile/graph | collective hang | collective 留在稳定边界外，分 rank timeout 和失败测试 |
| 优化组合冲突 | 单项快、组合错或更慢 | effect/seam conflict checker，单项和两两消融 |
| 只看微基准 | 总体无收益 | 端到端 Go/No-Go 是最终门槛 |

## 14. 建议的 PR/实施顺序

每个 PR 保持一项可独立验证的能力：

1. 扩展现有 `PerformanceManifest`，冻结 case/protocol、paired compare、status/error 和完整 fingerprint；建立 CPU regression test。
2. 给标准 runner 增加默认 no-op、instance-local lifecycle observer；shape trace 使用独立 untimed pass，完成一个真实 smoke baseline。
3. 引入 runner-scoped resolved typed plan 与 capability/effect/conflict 模型，打通 requested/effective/fallback report，默认行为不变。
4. `NativeModuleLoader` 接入 `compile_module_cached` 作为兼容迁移；同时由 adapter 声明 region，完成一个真正的 block/loop compile island。
5. numerically-equivalent attention per-shape benchmark/selection 和 persistent cache；Sage/PISA 不进入 exact lane。
6. 把已有 norm/modulation/residual/RoPE/activation kernel 扩展到 Wan 和 Sana，并逐项 A/B。
7. 通用 merged QKV/KV build transform 与静态 conditioning projection cache。
8. 稳态 CUDA Graph runner：静态 buffer、warmup、signature bucket、eager fallback。
9. Wan/Sana exact-fast preset，并完成 A100、Ada、Hopper 各自的端到端验收。
10. VAE 与多卡 QKV packing/collective profiling，根据热点决定是否实施。
11. Hopper FP8、Blackwell NVFP4 的独立硬件资格和 model schedule。
12. 统一 step cache，再依次接入 sparse attention 和 token pruning。
13. autoregressive circular/quantized KV cache 与 steady-state graph。
14. 扩展 nightly hardware matrix、正式 quality gate 和 release preset；基础性能 CI 已从第 1 个 PR 开始。

## 15. 第一轮实验矩阵

禁止一次同时打开所有优化。建议对每个 pilot 按以下顺序保存 A/B：

| 实验 | 相对基线的唯一变量 | 目的 |
|---|---|---|
| A | BF16 eager exact baseline | 建立真基线 |
| B | compile island | 测 compile 单项收益和冷启动成本 |
| C | exact attention selector | 测 attention backend 单项收益 |
| D | 公共融合算子 | 测 launch/allocation 降低 |
| E | merged QKV + static K/V/RoPE | 测重复计算和 layout 降低 |
| F | steady-state CUDA Graph | 测 launch/CPU overhead |
| G | B+C+D+E+F | 形成 `exact-fast`，检查组合非线性 |
| H | 低精度 | 只在支持硬件，形成单项质量曲线 |
| I | step/cache | 单独测 speed/quality curve |
| J | sparse attention | 单独测 layer/step sparsity curve |
| K | token pruning | 单独测 keep-ratio curve |
| L | 通过门禁的 H/I/J/K 组合 | 候选 `balanced/aggressive` |

每个实验都输出：端到端、denoiser/step、VAE、peak VRAM、kernel 数、compile/graph 信息、实际 backend、逐步误差和质量分数。

## 16. 暂不优先的候选

以下机制可进入 backlog，但只有 profile 命中后才实施：

- 新写通用 scale-shift、通用 QK norm/RoPE、varlen pack/pad 等 native kernel；WorldFoundry 已有 Torch/Triton 路径，先扩大调用覆盖并与 Inductor 比较。`causal_conv3d_cat_pad` 是本条的唯一首批候选，但仍需通过配套 CUDA/C++ 规格中的 promotion gate。
- 为 HunyuanVideo 或 LTX 直接复用 Wan 的 causal cat+zero-pad kernel；HunyuanVideo 使用 replicate pad，LTX 使用首/尾帧 repeat，语义不同，必须另立证据和接口。
- 大量引入 LightX2V 的硬件专用 FP4/MXFP kernel。
- 把 FastVideo 全套 sparse backend 作为强依赖。
- 对所有模型做统一 50% token pruning。
- 在每次线上请求里动态 autotune attention。
- 同时开启 Inductor cudagraph 和 WorldFoundry 显式 CUDAGraph。
- 为了局部 microbenchmark 删除所有 dense fallback。

## 17. 最终完成定义

本优化项目完成的标准不是“加入了若干 Triton 文件”，而是：

- native diffusion 的所有优化都通过一个 typed、可组合、可冲突检查的计划生效。
- exact 默认路径在支持模型上无精度回退，并能稳定报告实际命中的 backend。
- A100、Hopper、Blackwell 各有明确能力矩阵；不支持能力不会伪装成已启用。
- Wan、Sana、LTX、Cosmos 和一个 autoregressive 模型都有可复现 baseline、优化结果和质量报告。
- compile、graph、attention、fusion、quant、cache、sparse、prune、distributed 的单项贡献可通过消融说明。
- performance CI 能发现显著回归，fallback 和运行 manifest 足以定位原因。
- 最终 speedup 只基于端到端、固定配置、同硬件 A/B 数据发布，并同时公布冷启动、稳态、显存和质量变化。

## 18. 当前执行板

| Lane | 当前切片 | 可开始条件 | 完成证据 |
|---|---|---|---|
| F0 | benchmark contract + lifecycle observer + 1 个真实 smoke case | 立即 | paired raw JSON、Markdown、round-trip/失败产物测试 |
| P1 | resolved typed plan | F0 schema 稳定 | 默认行为零变化；requested/effective/fallback 可审计 |
| L1 | 首个 adapter-declared compile island | F0 baseline + typed plan | region/variant/cold/break-even/等价报告 |
| NAT-00 | 纯 C++ dispatcher wheel、sidecar ABI manifest、lazy provider（本地骨架已验证） | 可与 F0 并行；不含模型算子 | 无 wheel/坏 ABI/CPU import/dispatcher load smoke；最终 wheel SHA/LICENSE/no-vendored-libtorch 与 CI matrix 待补 |
| NAT-01/K1 | schema/fake/reference/shape benchmark | NAT-00 + F0 trace contract | promotion report；未达门槛即停止 |
| K1 CUDA | `.cu` 与 Wan adapter | K1 promotion 通过 | bitwise、stream/graph/compile、VAE/E2E 收益 |

NAT-00 已落地可选 package、sidecar-first loader、主包 provider adapter 与本地纯 C++ dispatcher smoke；它不预判任何热点，也不改变默认推理。当前转入 F0 基准契约主线，并行补 NAT-00 的 PEP 517 wheel/CI 资格矩阵。当前只可见 1 张 A100，因此本地结果只能形成 SM80 单卡开发证据，不能替代 2/4/8 卡或其他架构资格。

## 19. Native lane 摘要（非算子规格）

逐算子 schema、ABI、shape 数学、构建/加载、错误分类、测试矩阵和 promotion gate 只在 [`cuda_cpp_operator_spec.md`](./cuda_cpp_operator_spec.md) 维护。本节只保留与总路线有关的决定：

- NAT-00 可以与 F0 并行：先证明纯 dispatcher DSO、sidecar manifest、pre-dlopen ABI 校验、thread-safe lazy load 和无 wheel 安全降级；不注册公开模型算子，不启用 CUDA language。
- K1 是首个 bitwise-exact CUDA 候选，但只有真实 Wan/Cosmos VAE trace 过 promotion gate 后才写 `.cu` 或修改 canonical adapter。
- A100 INT8、Ada/Hopper FP8、Hopper sparse 和 Blackwell FP4/MXFP 是独立 hardware evidence lane；不存在“在当前机器交叉编译成功即已支持”。
- SageAttention/SageAttention3 属于 approximate provider，不进入 `exact`/`exact-fast`；外部 dense/sparse attention 继续通过 attention adapter 接入，不复制整套 runtime。
- 所有普通请求只读取离线/显式 warmup 产生的 provider selection；缺记录走安全 Torch 路径，不在首请求竞速。
- 没有 wheel、manifest/ABI 不匹配、缺 symbol 或 unsupported SM 时允许在执行前降级；OOM、illegal address、device assert、launch/capture failure 原样抛出。
- 主包安装不运行 NVCC；source build 必须使用已资格验证的当前 Torch 环境并关闭 build isolation，预构建 wheel 按明确的 Torch/CUDA/CXX11 ABI/架构矩阵发布。

Native 当前顺序固定为 `NAT-00 -> NAT-01/K1 reference+trace -> promotion decision -> K1 CUDA -> model adapter`。K2 及以后只有各自 gate 通过才进入实现。
