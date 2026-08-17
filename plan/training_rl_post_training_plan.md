# WorldFoundry 原生 Training / RL / Post-Training 实施计划

> 更新时间：2026-08-10
> 执行所有者：`worldfoundry-native`
> 当前状态：常用蒸馏、UniRL 算法面、点名 video RL、Agentic tool-use RL、HTTP scorer、Ray rollout/权重同步均已进入 WorldFoundry 原生执行面；整体 production DoD 尚未达成
> 下一阶段重心：为新增路径补发布权重、真实 scorer 权重、GPU Ray、2+ rank、质量与性能证据

## 0. 结论与边界

WorldFoundry 要实现的是自己的训练基础设施，不是把官方仓库包装成后端。

本轮点名的基础模型中，存在可执行作者训练代码的 LTX-Video/LTX-2、Cosmos Predict2.5/Cosmos3、LVDM、DynamiCrafter 与 T2V-Turbo 均已有至少一条明确选择的作者子路径进入 WorldFoundry 原生 training stack；这不表示穷举同一仓库的所有历史 trainer/profile。LongCat-Video 与 Step-Video 当前官方 HEAD 仍没有公开 trainer，因此明确保持未支持，而不是根据报告猜出一条训练路径。

1. model forward、data/corruption、rollout、reward、loss、backward、optimizer、scheduler、EMA、distributed state、checkpoint、export 都由 WorldFoundry 执行；
2. 作者仓库用于核对架构、公式、训练配置和更新顺序；正常训练进程不启动、导入或代理作者 trainer；
3. 许可证允许且工程收益明确时，可以移植作者模型组件或局部算法实现，并在 `THIRD-PARTY-NOTICES` 和源文件头保留 attribution；
4. 官方提供 trainer 的算法，以固定源码和发布配置作为实现依据；官方未提供 trainer 的算法只能称为 report-derived implementation，不能称为官方复现；
5. recipe 只保留会改变执行行为或会阻断错误执行的字段；不增加 provenance registry、reference-trace registry、repository-scope CLI、只写不读的治理 metadata；
6. checkpoint identity 使用可读的配置、文件清单和状态字段直接比较；角色独立性、冻结状态、optimizer parameter ownership、resume state 都是功能性门禁，必须保留；
7. GPU 数由实际 process group 和并行拓扑推导，不固定为 8；上游脚本中的 8 卡只是一份发布 profile；
8. 单卡、任意合法 world size、多节点使用同一算法状态机；不能为某个固定卡数另写训练 loop；
9. 公共 attention/cache、gradient 与 atomic write 等能力复用 `worldfoundry/core`；training 不重复建设一套基础组件；
10. code/schema/class 名称不使用 `version 1`、`v1` 一类编号式命名；
11. 单元测试、toy E2E、真权重短跑、分布式短跑、长程质量是不同证据，文档必须分别表述。

当前不再以增加算法目录数量作为目标。新的算法只有在出现明确的产品模型、作者训练代码、许可证边界和真实验证资源时才进入 scope；工程资源优先用于把已支持路径推进到真模型和分布式证据。

## 1. “已实现”的严格含义

后续状态使用以下分层，避免把局部公式测试等同于完整训练正确性。

| 层级 | 要求 | 能证明什么 |
|---|---|---|
| Contract | strict recipe、typed batch/model seam、unknown-field rejection | 输入和角色边界明确 |
| Math | 直接执行公式、gradient boundary、reduction、cadence 测试 | 局部训练数学成立 |
| Native stack | builder、optimizer、engine、session、checkpoint state | WorldFoundry 状态机可以闭环 |
| Model boundary | 真 model class/materializer、checkpoint/config audit | 算法可以接到具体模型 family |
| Real short run | 固定真权重与数据完成 forward/backward/update/resume | 当前硬件和资产上实际可运行 |
| Distributed gate | 2+ rank 更新、uneven batch、resume、failure tests | 当前 topology 上的并行语义成立 |
| Quality/performance | 长程训练、held-out 指标、质量/延迟/显存基线 | 可以对外声明训练能力 |

“Native stack 完成”不自动表示“真模型完成”；“有作者 trainer”也不表示 WorldFoundry 已与作者完整 optimizer update 对齐。

## 2. 当前实现快照

### 2.1 Distillation

| Family | WorldFoundry 当前执行面 | 模型边界 | 当前仍缺 |
|---|---|---|---|
| Wan DMD | strict recipe、few-step rollout、real/fake score、双 optimizer、session、DCP/export；FastVideo profile 使用 interval 末尾 student update 与双角色 accumulation=8；修正后 Wan2.1 1.3B update、exact resume 与 fresh PEFT reload 真权重 gate 已通过 | Wan2.1 1.3B 已接线 | 2+ GPU、full tuning、长程质量 |
| SANA SiD | report-derived strict recipe、few-step detached prefix、student/teacher/fake-score、双 optimizer、session、DCP/export | SANA-Sprint 600M local-Diffusers materializer、cache/prompt-only loader、CLI/config 与 real-roundtrip gate script 已接线 | 该组合没有 SANA 作者 trainer，只能作为 experimental reimplementation；仍需真权重、2+ GPU、质量/显存 gate |
| Cosmos Predict2.5 DMD2 | 作者 T2V discriminator loop 的四步 TrigFlow→rectified-flow 网格、student/guidance 交替更新、共享 adversarial score input、三层 feature discriminator、双 optimizer/scheduler、session、DCP/export；正式 raw-video precompute 使用有种子的随机连续 93 帧和 direct resize `704×1280`，latent token budget 为 `24×88×160=337920` | Cosmos Predict2.5 2B 原生模型、正/负文本与视频 cache、FP32 student/guidance/discriminator master parameter + recipe BF16 autocast、single/FSDP2 materializer；torch AdamW 保持作者 FP32-master 数学/精度契约，不声明 NVIDIA fused kernel bit parity | 官方真权重 update/resume/export、2+ GPU、质量门禁 |
| 通用 DMD2 / SGMD / DFD | 各自 recipe、公式、角色、engine、session、checkpoint | model-neutral adapter seam | 除 Cosmos Predict2.5 DMD2 外，仍需逐 family 真模型 materializer 与 real gate |
| SCM-LADD | TrigFlow/consistency、student/teacher/discriminator、G/D 交替、双 optimizer；generator 使用 constant-with-warmup 5000，discriminator 不创建 scheduler；student 全图 FP32 attention，teacher 与 discriminator frozen backbone 保持关闭；scheduler/session/DCP 已接线 | SANA 语义 seam | 官方 SANA-Sprint 权重短跑与质量对照 |
| Progressive distillation | DDIM 两步教师目标、逐阶段减半、LR/EMA/stage checkpoint | model-neutral prediction seam | 具体 image/video family materializer 与真实多阶段训练 |
| Latent Consistency Model | DDIM pair、boundary scaling、guidance embedding、EMA、engine/session | model-neutral prediction seam | 官方 LCM 配置的真权重更新对照 |
| Scale-wise Distillation | 多尺度 schedule、DMD/GAN/MMD、fresh fake updates、SD3 prediction/critic adapter | SD3 execution adapter | 官方 SD3.5 资产短跑、分布式和质量基线 |
| rCM / Causal-rCM | continuous/discrete consistency、DMD joint loss、JVP seam、TF/SF causal stack、session | bidirectional 与 causal adapter seam | 真 Wan/Cosmos 模型 gate、JVP kernel 性能与多卡 |
| Causal ODE / Causal Consistency | 独立 recipe、pair/schedule、objective、engine、session | causal adapter seam | 官方 Causal-Forcing 资产与完整三阶段验证 |
| Self-Forcing | 自生成 causal rollout、KV/cache commit、gradient truncation、DMD session | native Wan causal chunk adapter | 发布权重短跑、长上下文 drift、SP/CP |
| Self-Gradient-Forcing | noisy-context commit、teacher-forcing replay、DMD engine/session | native Wan causal adapter | 作者配置真权重 gate 与长程质量 |
| Diagonal / Adaptive Video | diagonal block rollout、motion head/EMA、adaptive regression、DMD execution | causal/model-neutral seams | 对应作者资产与 production materializer |
| AnyFlow | FAR/双向 pretrain 与 on-policy 四种 recipe；FlowMap、central difference、DMD、fresh fake-score、官方 constant-with-warmup、pretrain/on-policy EMA、EMA export、同步随机决策、双 optimizer、session/DCP | native AnyFlow Wan graph；FAR 与双向 1.3B materializer；两个官方 1.3B 已完成 load/forward，双向 pretrain 已完成 update/DCP exact resume | FAR update、两条 on-policy 真权重门禁、2+ GPU、14B、质量/速度 |
| SenseFlow | strict recipe；IDA、ISG、DMD、adversarial objective；student/teacher/fake-score/discriminator；三 optimizer/session/DCP | model-neutral flow adapter seam | SD3.5/FLUX materializer、官方权重短跑、多卡/质量 |
| Reward-Forcing | strict recipe；21-frame causal rollout、EMA-Sink、local attention、Re-DMD reward weighting、双 optimizer/session/DCP | native causal/Wan seams | 官方模型与 reward 资产短跑、流式质量/速度 |
| Adversarial Diffusion Distillation | strict recipe；pixel decoder/feature discriminator、SDS 或 exponential target、R1、D→G 更新、双 optimizer/session/DCP | generic image adapter seam | SDXL model materializer、真权重训练、官方不可得细节的敏感性实验 |
| T2V-Turbo | 作者 float32 scaled-linear schedule、DDIM-50 网格、top-k 20、CFG 5–15、pseudo-Huber consistency、同一 student stop-gradient target、16 FPS/16×320×512 cache contract、DCP exact resume；按作者实现覆盖 UNet 内 Linear/Conv2d/Conv3d 的 extended LoRA，Linear/Conv2d dropout=0.1、Conv3d dropout=0、rank cap 与 scale=1；训练与 fresh-base inference 共用确定性的 frozen guidance projection；导出作者 inference loader 可消费的有序 `unet_lora.pt` | VideoCrafter2 UNet 原生 materializer；FP32 student master、FP32 frozen teacher storage、CUDA FP16 teacher forward、BF16 student compute、FSDP | 固定作者仓库实际发布了 HPSv2/ViCLIP/InternVideo2 reward wrapper，但对应外部权重和 AdamW8bit 运行链尚未接入；正式 preset 主动限定为 consistency-math profile，不声明 released reward run parity；仍缺真权重、2+ GPU、质量门禁 |
| DiffusionOPD / MOPD | strict recipe、单域 batch、student on-policy rollout、同一轨迹 teacher/student replay、ODE mean matching 或共享方差 KL、完整 teacher-cycle accumulation、session、DCP exact resume 与 export；loss 不消费 reward/advantage | model-neutral `FlowPredictionAdapter`，各 teacher 独立 checkpoint 并由 domain 选择 | SD3.5 与 teacher LoRA materializer、CLI profile、发布权重/GPU/多卡/质量 gate |

`post_training/distillation/causal` 和 `post_training/distillation/consistency` 是被多种算法执行路径消费的共享数学/contract，不是治理占位目录。

### 2.2 RL 与 preference optimization

| Family | WorldFoundry 当前执行面 | 当前仍缺 |
|---|---|---|
| Flow-GRPO | Flow-SDE rollout/replay、group advantage、clipped objective、reference KL、revision/session；默认 Wan profile 使用 UniRL 的 14-step shift-5 schedule、前半 SDE、group 16、replay anchor、2 个互斥 optimizer partition 和 global population std；Wan2.1 1.3B + VideoAlign 单卡真权重 update/DCP exact resume 已通过 | VideoAlign 是 WorldFoundry reward 选择而非 UniRL 的 VideoPickScore；仍缺 2+ GPU、长程质量与 reward-hacking gate |
| Flow-DPPO | old-mean divergence mask、KL-ADV、共享 flow-policy stack | 同上 |
| DANCE-GRPO | constant-diffusion transition、同组噪声/奖励归一化、独立 recipe/session | 作者发布 profile 的真模型对照 |
| MixGRPO | progressive sliding window、window cadence、完整 recipe/session | 作者发布 profile 的真模型对照 |
| GRPO-Guard | differentiable drift bias、冻结 old anchor | 真模型/奖励 gate |
| Bagel Flow-UniGRPO | conditional mixing 与算法专属 objective/session | Bagel model materializer |
| DiffusionNFT | terminal clean-latent collection、forward-process NFT、old policy refresh | Wan single-device 路径之后的 FSDP2 与质量 |
| Diffusion-DPO | offline chosen/rejected pair、current/reference loss | production dataset 与 model materializer |
| DDRL | PPO-style diffusion ratio、reference mean、data regularizer | Cosmos Predict2.5 rollout/reward/model materializer |
| token GRPO/GSPO/DPPO/DRPO/CPPO | packed-token trajectory/replay/partition engine，各算法独立 reduction；Qwen3-4B 原生 materializer、官方 Hermes chat template/`<\|im_end\|>` turn boundary、local/HTTP reward、CLI/config；五种 grouped objective 均通过 model-level update smoke | Qwen3 发布权重短跑、LoRA/FSDP learner、正式数据/reward 与质量 gate |
| token PPO | rollout old log-prob/value、terminal reward scatter、packed GAE、actor asymmetric clipping、FP32 clipped critic loss、multiple update epochs、session、DCP exact resume 与 actor-critic export；Qwen3-4B-Base policy + FP32 value head、CLI/config 已接线 | Qwen3 发布权重短跑、MathVerify 等正式 reward、LoRA/FSDP learner 与质量 gate |
| Agentic tool-use RL | 多轮 message/tool-call/tool-result、同 turn 并行 tool calls、逐 sibling 隔离、至少两个成功 sibling 才更新、dropout-free replay、local/Ray rollout、trajectory/local/HTTP reward、packed-token learner、prompt cursor/rollout/RNG/optimizer exact resume 与 export；Qwen3-4B calculator profile 可从 recipe/CLI 构建 | 当前 Qwen learner 是 full tuning、single process；仍缺正式 search/visit 环境、发布权重任务短跑、分布式 learner、质量与安全评测 |

点名 video RL 已接入同一原生 Flow policy 生命周期：

- Wan2.2 A14B：high/low-noise 双 expert、boundary 路由、独立 policy/reference checkpoint、rollout old log-prob、LoRA update/resume/export；
- HunyuanVideo 1.0/1.5：各自 conditioning/model adapter；1.5 固定作者 profile 的 guidance=0、token-refiner LoRA target；
- LTX-2：9 帧、group 16、LoRA alpha 256、video SDE + audio ODE、rollout old log-prob；
- LTX-2.3：33 帧、group 4、LoRA alpha 64、joint AV SDE、replay old log-prob 与 video/audio joint log-prob；当前只开放 reference KL 为零的 Flow-GRPO，未定义的 DPPO/KL/NFT 组合直接拒绝；
- LTX-2.3 AV profile：native joint AV decode 后，将 video/audio artifact 交给 HTTP scorer，按 UniRL 发布配置使用 VideoPickScore 0.5 + CLAP 0.5。

Wan2.2、HunyuanVideo 1.0/1.5、LTX-2 与 LTX-2.3 video-only parity preset 均请求 WorldFoundry 原生 HTTP service 的 `videopickscore`，不再用 VideoAlign 冒充 UniRL 发布 reward；服务可由 `train-reward-service` 从 strict YAML/JSON 启动，VideoAlign 仍是可选 reward。上述 preset 的 global prompt batch 分别固定为 48、8、48、8、8；运行时按 active world size 计算 rank-local batch，并要求整除以保证每个 FSDP rank 的 collective 次数一致。操作者可按实际 topology 覆盖 global batch，因此算法没有固定 8 卡 contract；sampler 的 drop/pad 尾部也按完整 logical global batch 对齐，不会发出 partial rollout。

“发布 profile”只限定已经逐项核对的 scheduler、SDE window、group/replay、optimizer、activation checkpoint、global batch 和 reward 数值，不扩大成未验证的全栈等价声明。HunyuanVideo 1.0 的本仓原生 fused `to_qkv` 只允许共享 LoRA 输入低秩因子，而 UniRL/diffusers profile 对 q/k/v 分别注入 LoRA；两者不是严格的 adapter parameterization parity。HunyuanVideo 1.5 的 target mapping 也仍需发布 checkpoint 实跑确认。

以上新增 video profile 已覆盖 family-specific strict recipe、模型 materializer、conditioning 与 rollout/replay 数学；通用 Flow run 已覆盖 update→DCP resume→周期/显式 export，Wan2.2 的正式 Ray recipe 另覆盖完整 tiny materializer 生命周期，LTX-2.3 joint AV 覆盖两 actor 分片合并→replay→backward。不能把这些组合证据写成“每个 family 都已用发布权重跑完整生命周期”，对应长分辨率训练和质量复现仍未完成。

所有 RL 算法都必须有自己的行为 contract；共享 rollout/engine primitive 不等于把不同算法压成同一个名字。

### 2.3 Rollout、权重同步与 scorer 基础设施

| 能力 | 当前执行面 | 证据与边界 |
|---|---|---|
| Ray DevicePool | placement-group based device slab；`external` trainer + rollout，以及 actor-hosted trainer/rollout/reward 的 separate 或同 bundle 不同 fractional slot；原生 worker 的 CPU/CUDA 设备直接由 recipe accelerator resource 决定 | 真实 CPU-Ray actor 测试覆盖两种 placement；正式 Qwen profile 分别提供 actor+separate 与 actor+colocate。`external` 模式只拥有 rollout slab，和调用方 trainer 的物理隔离由调用方负责；多节点按每组 placement group 分配，不声明整个 pool 的单次全局 gang reservation |
| 权重同步 | replicated full/LoRA tensor selection、bounded bucket、revision gate、所有 receiver 先 validate 再顺序 commit；validation/pre-commit 失败会清空 staged update；支持 controller fan-out 和 trainer actor 内直接 fan-out | receiver commit 后没有跨 worker rollback，因此不声明分布式事务；controller-local DTensor materialization保留，actor-hosted FSDP/DTensor source 尚未实现并显式拒绝；正式 Qwen actor materializer当前只开放 full tuning，LoRA 的 Ray 证据来自 runtime/toy 与 video external 路径 |
| Flow / Agentic Ray rollout | 完整 group 不拆分、结果恢复原顺序、每个 policy revision 同步后 rollout；Agentic sibling 并发与失败隔离；Flow run 实际消费 `export_every_steps` | Agentic 已完成 external+separate 以及正式 actor+separate/colocate CPU-Ray E2E；共置路径覆盖 update、checkpoint、uninterrupted 对 split-resume 的 policy/rollout state 精确比较与 export。video 目前是 single learner + external+separate：Wan2.2 正式 recipe 覆盖 update/resume/周期 export，LTX-2.3 joint AV 覆盖两 actor interleaved shard 的 audio trajectory/means/scales 合并、joint replay 与 backward；video actor/FSDP2 Ray 明确拒绝 |
| HTTP scorer | typed request/result、batched client、FastAPI service/registry、partial-result policy、Flow/DiffusionNFT terminal artifact adapter、Agentic trajectory adapter；原生 lazy VideoPickScore/CLAP 与 transcript correctness/`tool-success` 已接 registry；`train-reward-service --config ...` 可启动 AV、Agentic 或混合 sidecar | 本地 HTTP round-trip、批次顺序与逐样本 invalid 语义、strict service config、CLI wiring、首帧 PickScore 归一化和 CLAP mono/48 kHz 预处理已覆盖；Agentic scorer 直接读取每个请求声明的 expected-answer 字段和 required-tool 契约，不包含 LLM judge；当前环境未执行真实 PickScore/CLAP 权重、默认 TorchAudio kernel、GPU sidecar 或部署吞吐 |

### 2.4 基础训练 model family

| Family | WorldFoundry 当前执行面 | 官方依据与当前边界 |
|---|---|---|
| LTX-Video / LTX-2 / LTX-2.3 | 原生 raw-video 与 T5/Gemma text cache；正式 precompute 使用作者 trainer 的头部连续帧和 floor-resize 后 center crop；sequence-length-aware shifted logit-normal、uniform mixture、首帧 conditioning、LTX-2 causal position/stretch 语义和 legacy discrete-timestep profile；按样本 mask-normalize loss；只向当前 video forward 实际消费的 `attn1/attn2` 注入 LoRA；LTX-2/2.3 使用 FP32 master parameter + BF16 compute；官方 LinearLR、single/FSDP2、DCP，以及官方/ComfyUI 可读的单文件 BF16 LoRA export | 分别依据固定的 legacy LTX-Video-Trainer 与 LTX-2 trainer；正式配置覆盖 legacy 2B 与 LTX-2.3 22B。这里的基础训练路径仍是 joint AV trainer 的 video-only I2V/T2V 子路径；RL 路径另有 native joint AV trajectory/decode/remote reward，但两者都不声明完整 joint AV base-training parity。已覆盖真实本仓模型图 forward/backward 与 world-size-one FSDP2 lifecycle，仍缺发布权重完整 update、2+ GPU、长程质量/性能 |
| Cosmos Predict2 / Predict2.5 | 原生 raw-video/text cache；正式 Predict2.5 precompute 按 seed/sample 选择随机连续 93 帧并 direct resize 到 `704×1280`；velocity flow matching、作者 conditional-frame mix 与 CFG dropout、attention/MLP LoRA/full tuning；Predict2.5 使用 pretrained base、FP32 trainable master parameter + recipe BF16 autocast、作者 AdamW/LambdaCosine 和 PowerEMA；single/FSDP2、DCP、EMA export | 依据固定 Cosmos Predict2.5 trainer；正式配置为 Predict2.5 2B LoRA。torch AdamW 对齐作者 FP32-master 的数学与精度契约，不声明 NVIDIA fused optimizer kernel 的 bit parity；已覆盖本仓原生模型图、scheduler/EMA cadence 与 exact resume，仍缺发布权重 update、2+ GPU、质量/性能 |
| Cosmos3 Nano vision SFT | 原生 video/token cache；Waver sampler、256px shift 3、T2V/I2V/V2V task mix、CFG dropout、conditioned-frame loss mask；只更新作者配置的 `moe_gen/time_embedder/vae2llm/llm2vae` 参数组；full activation checkpoint、AdamW、LambdaCosine、PowerEMA、single/FSDP2、DCP/EMA export | 依据固定 Cosmos Framework `vision_sft_nano.toml` 和 trainer；已完成真实 tiny Cosmos3 图 forward/backward、block recompute、exact resume 及 world-size-one FSDP2 CUDA export。当前正式配置是固定 49 帧的 video-only 子路径，不声明官方 variable-length/45056-token packing、action/audio 或联合模态训练；仍缺发布权重 update、2+ GPU、质量/性能，`torch.compile` 目前关闭 |
| LVDM short unconditional | 移植官方 short UNet；epsilon target、sqrt-linear beta、uniform timestep、L1、AdamW、按 train batch 更新的 LitEma；single/FSDP2、DCP exact resume、EMA full-model export | 对固定官方源码的 tiny forward/state 做过精确对照。当前正式路径只消费预计算 latent；没有把本仓 compact/toy VideoAE 当成官方等价实现，仍缺官方 raw-video cache producer、真权重更新与 2+ GPU |
| DynamiCrafter 512/1024 I2V | 官方 trainable UNet wrapper + image projector；v-prediction、zero-terminal-SNR、dynamic rescale、hybrid condition/dropout；single/FSDP2、DCP/full export；FP32 master + CUDA FP16 autocast/GradScaler | 对齐作者 Lightning `precision: 16` 的 master/compute 语义，GradScaler 进入 DCP。当前只提供预计算 tensor importer，仍缺官方 WebVid/VAE/OpenCLIP 数据生产链、真权重更新与 2+ GPU |
| LongCat-Video | 不提供 training recipe/materializer | 本轮固定官方 HEAD 只有模型、权重与 inference/demo；报告提到 GRPO/蒸馏不等于发布 trainer。没有可核对的作者训练 loop，因此不推测或伪造集成 |
| Step-Video-T2V / TI2V | 不提供 training recipe/materializer | 本轮固定官方 HEAD 公开模型、权重与 inference；报告描述 Flow Matching、DPO/step distillation，但没有对应作者 trainer/config。没有可核对的执行细节，因此不推测或伪造集成 |

### 2.5 阶段判断

常用 diffusion/flow distillation、当前 UniRL public algorithm surface、点名 video/Agentic RL 与 disaggregated rollout 基础设施已经形成原生执行面，因此“继续收集更多算法”阶段结束。现有 model-neutral stack 会保留并测试，但不追求 model × algorithm 笛卡尔积。下一阶段只为有明确使用优先级的组合实现 production materializer、CLI 和真实门禁。

## 3. 官方训练语义基线

### 3.1 本阶段重点来源

| 技术 | 固定 source | License | 作者是否发布训练代码 | WorldFoundry 定位 |
|---|---|---|---|---|
| AnyFlow | `NVlabs/AnyFlow@d2acf7373a45173082ec47eb16553a373b10f856` | Apache-2.0 | 是；FAR/双向、pretrain/on-policy 四条路径 | 移植必要模型图，重建 native recipe/engine/session |
| SenseFlow | `XingtongGe/SenseFlow@fafc81b7334eaf7ccf4ced296c25faa412b477ea` | Apache-2.0 | 是；SDXL、SD3.5、FLUX trainers | 依据可执行 trainer 固定 IDA/ISG/DMD/GAN 语义 |
| Reward-Forcing | `JaydenLyh/Reward-Forcing@4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66` | Apache-2.0 | 是 | 依据 trainer 固定 causal rollout、Re-DMD、EMA-Sink/cadence |
| ADD | `Stability-AI/generative-models@e6f0e36f5e856d9651c597d75aed13ae7298d03b` 与 ADD report | repo MIT；模型条款独立 | 固定提交中没有 ADD 专用 trainer | 只能称 report-derived；不宣称官方 trainer parity |

AnyFlow 作者 README 中的 `--nproc_per_node=8` 是示例，README 同时明确该参数应设置为实际 GPU 数。WorldFoundry 不把 8 写入算法 contract。

### 3.2 AnyFlow 权重边界

WorldFoundry 的 native materializer 固定以下两份 1.3B 权重身份：

- `nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers@915af337434035df8545797ecc910d79fa78cf29`；
- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers@4c2ec05c7fa4dbafbca131ad32430905c7ff2974`。

这两份模型权重使用 NVIDIA One-Way Noncommercial License（NSCLv1），不可因为 AnyFlow 源码是 Apache-2.0 就推断权重可以商用。权重由操作者单独下载，不能随 WorldFoundry 源码重新分发。

### 3.3 其他已使用来源

| 技术 | 主要作者来源 | 训练代码状态 | 使用方式 |
|---|---|---|---|
| LTX-Video / LTX-2 | `Lightricks/LTX-Video-Trainer@e055182fa36dba6f48eb0919aef09d277da30fbd`、`Lightricks/LTX-2@4f8905737aac86a554637cac86c178877a39c744` | 有 | legacy 与 LTX-2/2.3 timestep、conditioning、位置、flow target、LoRA、optimizer/scheduler 语义 |
| Cosmos Predict2.5 | `nvidia-cosmos/cosmos-predict2.5@a2c298b0a3df3778b973fe65e9e58877b292d8a7` | 有；SFT 与 DMD2 | Predict SFT、conditional-frame/CFG、LoRA、PowerEMA，以及 2B T2V discriminator DMD2 loop |
| Cosmos3 | `NVIDIA/cosmos-framework@4155d61d14b14e05a8cafe2bd796d090fcb5f145` | 有；Nano vision SFT | Waver/task mix、selected-key tuning、optimizer/scheduler、activation checkpoint 与 PowerEMA；不外推 action/audio |
| Progressive Distillation | `google-research/google-research/ddpm_distillation` | 有 | 两步 DDIM target、cosine log-SNR、stage halving |
| Latent Consistency Model | `luosiallen/latent-consistency-model` | 有 | DDIM pair、boundary scaling、guidance embedding、EMA |
| Scale-wise Distillation | `yandex-research/swd` | 有 | SD3.5 schedule、scale boundaries、DMD/GAN/MMD |
| rCM / Causal-rCM | `NVlabs/rcm@ed3cb14dd936f92cdc9f9381af7369991509b41f` | 有 | TrigFlow、JVP consistency、DMD joint cadence、causal roles |
| Self-Forcing | `guandeh17/Self-Forcing` | 有 | self rollout、cache commit、gradient truncation |
| Causal ODE/CD | `thu-ml/Causal-Forcing` | 有 | causal ODE/consistency initialization 与 asymmetric DMD 分阶段语义 |
| SANA SCM-LADD | `NVlabs/Sana@6298508fcb511762a11c42cff45b2fc9fd930325` | 有 | SANA/Sprint model 与 SCM-LADD 训练语义 |
| SANA-Sprint SiD | SiD / Few-step SiD reports 与本仓公式测试；`NVlabs/Sana@6298508fcb511762a11c42cff45b2fc9fd930325` 仅提供模型资产 | SANA 固定提交没有 SiD trainer/profile | SANA 组合是 WorldFoundry experimental reimplementation，不声明作者 trainer parity |
| LVDM short | `YingqingHe/LVDM@d251dccfbf6352826f5c5681abd86e87ed7e6371` | 有 | short UNet、DDPM objective、optimizer 与 per-batch EMA |
| DynamiCrafter | `Doubiiu/DynamiCrafter@859021927d8e0f8eb4d91d16f86711b8c25a2023` | 有 | v-prediction、ZTSNR、dynamic rescale、hybrid conditioning、Lightning FP16 precision |
| T2V-Turbo | `Ji4chenLi/t2v-turbo@eaae323f10d136796a33e8f5304ed50e40def570` | consistency 与 reward-feedback trainer、HPSv2/ViCLIP/InternVideo2 wrapper 均有；reward 权重是外部资产 | 独立实现 consistency 公式与 extended LoRA/export；当前正式 profile 是 consistency-math scope，不声明 reward/AdamW8bit run parity |
| LongCat-Video / Step-Video | `meituan-longcat/LongCat-Video`、`stepfun-ai/Step-Video-T2V`、`stepfun-ai/Step-Video-TI2V` 本轮固定 HEAD | 无可执行 trainer | 只列为未支持边界，不依据 report 猜测训练实现 |
| Wan DMD | `hao-ai-lab/FastVideo@1b2b2a0161bc6b3b80158d1fa6380a051c6530c7` | 有 | few-step schedule、三角色、双 optimizer、cadence |
| UniRL algorithms/runtime | `Tencent-Hunyuan/UniRL` | 有；FlowGRPO/FlowDPPO、GRPO/GSPO/PPO/CPPO/DPPO/DRPO、DiffusionNFT、DiffusionOPD、Bagel Flow-UniGRPO、Agentic 与 Ray rollout/runtime | 逐算法核对 objective/reduction/cadence；模型与 runtime 以 WorldFoundry 原生 contract 重新组合，不导入上游 trainer |
| Qwen3 CausalLM / tool use | `QwenLM/Qwen3`、`Qwen/Qwen3-4B`、`Qwen/Qwen3-4B-Base` | 官方提供模型、chat template 与 function-calling 文档；RL loop 来自 UniRL | 使用官方 Hermes template、`<\|im_end\|>` assistant turn boundary、并行 tool call 语义与公开 checkpoint；不导入 Qwen 或 UniRL trainer runtime |
| 其他 Flow RL | `yifan123/flow_grpo`、`verl-project/verl-omni` | 有 | 用于核对历史 transition/window 和 token/diffusion 分布式差异，不作为 runtime backend |

如果 paper、README 和 executable trainer 不一致，recipe 必须选择一个明确的可执行 profile，并在测试中固定差异；不能混合多个来源拼出一个无人发布的“官方默认值”。

Flow-GRPO 当前明确对齐 `Tencent-Hunyuan/UniRL@f70f8c9a44772446244dd72bbc13d74c5af160fd` 的 Flow-SDE 与更新语义。默认 Wan profile 固定为 14-step、shift 5、前半 7 个 SDE transition、group 16、FP16 trajectory、replay old-log-prob、2 个互斥分区 update，以及逐组中心化后除以全 batch population std。固定公式对照覆盖 transition next latent、mean、scale、log-prob、sparse/window timestep、group advantage、PPO clipping 和 old-log-prob freeze。reward 明确使用 VideoAlign，而不是 UniRL 发布 profile 的 VideoPickScore，因此只声明 policy-update parity，不声明完整实验 profile parity。旧版 `yifan123/flow_grpo@879042cf5707f8b90daa98d147d7deac2317c5da` 的线性 diffusion coefficient、前缀 train window 和 inner-epoch accumulation 已与当前 UniRL 分叉；WorldFoundry 不宣称同时逐行复现这两个版本。省略 `sde_window` 时采用 UniRL 的非重叠、到末尾停住默认值；重叠和 rollback 必须显式配置。

Wan2.2、HunyuanVideo 1.0/1.5、LTX-2/2.3 的发布配置分别固定其 scheduler、SDE window、group/replay 和 conditioning 语义，不能用 Wan 默认值覆盖。LTX-2.3 joint AV 使用 video/audio 两条 transition 的联合 log-prob；其 AV reward profile 对应 UniRL 的 VideoPickScore/CLAP 组合，但 WorldFoundry 通过自有 HTTP scorer contract 调用，不导入 UniRL reward runtime。

Qwen3 路径将模型协议与 RL 数学分开核对：chat rendering 和 tool-call parsing 依据 Qwen3 官方 Hermes 文档；grouped token policy、PPO/GAE、old-policy anchor、optimizer cadence 与 Agentic sibling lifecycle 依据上述固定 UniRL 源码。Agentic preset 使用公开可解析的 `Qwen/Qwen3-4B` 和本地 calculator 任务，PPO 使用 `Qwen/Qwen3-4B-Base`；它们证明 WorldFoundry 原生 software path，不冒充 UniRL deep-research 或 DAPO-Math 的完整发布实验，因为正式 search/visit、LLM judge/MathVerify、发布 batch geometry 和多卡 topology 尚未执行。

DiffusionOPD 当前依据 UniRL 的可执行 algorithm 与 multi-teacher SD3 profile 实现：student 在自己的轨迹上 rollout，domain 选择唯一 frozen teacher，teacher 在同一 state/step 上 replay；`add_kl_coefficient=false` 使用 `0.5 * delta²`，开启时才除以共享 SDE transition variance。每次 optimizer commit 必须覆盖完整且均衡的 teacher cycle。它是 teacher-anchored distillation，不是 reward RL。

SANA SCM-LADD 当前明确对齐 `NVlabs/Sana@6298508fcb511762a11c42cff45b2fc9fd930325` 的可执行训练配置：generator 使用 5000-step constant warmup，discriminator 不使用 scheduler；student linear attention 使用 FP32 数学，teacher 以及复用 teacher feature backbone 的 discriminator 保持关闭。scheduler 只在 generator optimizer commit 后推进，并随 optimizer、phase 和 progress 一起进入 DCP 恢复状态。

### 3.4 不能复制的边界

`tianweiy/DMD2` 源码使用 CC-BY-NC-SA-4.0。acknowledgement 不能替代授权，因此 WorldFoundry 只依据论文重新实现公式；在单独许可证复核前，不复制其源码。

## 4. 单一 native execution architecture

```text
PostTrainingRecipe + model-family contract
                 |
                 v
WorldFoundry materializer / caller-owned typed model seam
                 |
        +--------+-------------------+
        |                            |
        v                            v
local rollout                 Ray DevicePool rollout
                              separate / actor-colocated
        |                            |
        +-------------+--------------+
                      v
native decode -> local / Ray / HTTP scorer -> scalarization
                      |
                      v
algorithm-owned replay/objective -> WorldFoundry optimizer
                      |
                      v
single / DDP / FSDP2 -> session -> DCP resume -> export
```

teacher/reference/decoder/reward 是 frozen role；student、critic、fake-score、discriminator 是算法声明的 trainable role。上游仓库不在这张运行图中。必要的许可证兼容组件复制到 WorldFoundry model graph 后，由本地 materializer 加载、本地 adapter 执行。

## 5. Recipe 与角色门禁

### 5.1 通用规则

- `execution_owner` 必须是 `worldfoundry-native`；
- unknown field fail closed；
- algorithm 字段必须被 objective、engine、builder 或 session 消费；
- 未配置的 scheduler/EMA 不创建假状态；
- 每个 trainable role 有独立 optimizer spec；
- role checkpoint reference 必须非空，并与 materialized checkpoint identity 一致；
- frozen role 任一参数可训练或收到 gradient 时立即失败；
- trainable roles 共享 module object、parameter storage 或 optimizer parameter 时立即失败；
- recipe optimizer topology 必须和算法所需角色完全一致。

### 5.2 多 optimizer 更新顺序

更新顺序是算法语义，不是实现细节：

- Wan DMD：外层 iteration 从 1 计数，第 `interval`、`2 * interval`… 次才提交 student；FastVideo 发布 profile 的 interval=5、student/fake-score accumulation 均为 8，student 提交后再用新 student 更新 fake-score；
- ADD：先 discriminator，再 generator；这是 WorldFoundry 的显式 adversarial update contract，不把报告未披露的精确 cadence 标成官方行为；
- AnyFlow on-policy：student batch 与 fresh fake-score batch 分离，每个 fake update 消费新的 batch；
- SenseFlow：fake-score、discriminator、generator 按 TTUR/cadence 显式推进；
- Reward-Forcing：fake-score 每 logical iteration 更新，student 依据发布 cadence 更新；
- 任一 optimizer 已提交后出现异常，engine 进入 poisoned state，只能从上一完整 checkpoint 恢复。

### 5.3 功能性状态校验

以下 identity 会影响正确性，因此不是治理 metadata：

- model/checkpoint config 与 tensor inventory；
- 完整 recipe 行为字段；
- dataset sample identity 与 cache 文件清单；
- role checkpoint 的 repository/revision 或本地路径与文件清单；
- optimizer/scheduler/EMA state；
- distributed topology；
- loader position 与 RNG；
- rollout/policy revision。

这些值按结构直接比较，不折叠成摘要字符串。这样错误报告能指出具体字段或 tensor，训练路径也不会为了审计标签额外计算摘要。

不再生成或读取“本次是不是官方 trainer”“参考仓库列表”“论文 URL registry”一类不影响执行的运行时文件。

## 6. 重点蒸馏路径

### 6.1 AnyFlow

必须保持四条独立 recipe：

1. FAR pretraining；
2. bidirectional pretraining；
3. FAR on-policy distillation；
4. bidirectional on-policy distillation。

共同语义：

- FlowMap 的 source time、destination time 与 delta-time conditioning；
- central-difference consistency target；
- diffusion/consistency/velocity mixture ratios；
- on-policy fresh student rollout；
- real-score 与 fake-score 独立角色；
- fake-score 使用独立 fresh rollout/batch；
- inference-step choice 和梯度区间选择在 data-parallel group 内同步；
- EMA 只在 student optimizer commit 后更新；
- FAR/双向 pretrain 使用 LR warmup 1000、EMA decay 0.999/warmup 1000；两条 on-policy 使用 LR warmup 0、EMA decay 0.99/warmup 200；导出默认临时应用 EMA 后恢复 live 参数；
- FAR 的 temporal chunk partition、full/compressed patch geometry 和 long-context ratio 进入 recipe 与 model config gate；
- world size 不改变算法概率、cadence 或 batch ownership。

近期 DoD：

- **已完成**：两个 1.3B 官方 checkpoint config/load/forward gate；
- **部分完成**：双向 pretrain 已完成有限 loss/backward/update；FAR update 与 on-policy 真权重更新仍待执行；
- **已完成（双向 pretrain）**：checkpoint immediate resume 和 continuous resume；
- 2+ rank synchronized choice、uneven local weight、参数同步；
- 2/4/8/16/50 steps 的 held-out quality/latency 曲线。

### 6.2 SenseFlow

必须保留四个独立角色：student、teacher、fake-score、discriminator。执行面覆盖：

- segment schedule 与 backward simulation；
- implicit distribution alignment（IDA）；
- intra-segment guidance（ISG）；
- DMD distribution gradient；
- adversarial feature loss；
- SD3.5 Large、SD3.5 Medium、FLUX 发布 profile 的 TTUR、guidance、IDA decay 与 loss weights；
- paper MSE 与发布代码 Charbonnier 的差异通过显式 recipe 字段表达；
- 三 optimizer、scheduler、RNG 与 progress 进入 checkpoint。

近期 DoD：接入 SD3.5 和 FLUX model materializer，分别完成发布配置真权重短跑，再验证任意 world size 的 loss denominator 和参数同步。

### 6.3 Reward-Forcing

发布 profile 的行为必须完整消费：

- 4-step shifted schedule；
- 21 training frames、3 frames/block；
- 9-frame local attention 与 3-frame EMA-Sink；
- same-step-across-blocks 选择；
- reward-weighted distribution matching；
- teacher CFG、reward beta、EMA decay/start、logical 1:5 cadence；
- student/fake-score 各自 optimizer 和 gradient accumulation；
- causal cache 来自 `worldfoundry/core`，不复制一套训练专用缓存。

近期 DoD：固定官方 ODE initialization、Reward-Forcing student 与 VideoReward 资产，完成 rollout→reward→fake-score→student→resume 真权重闭环。

### 6.4 Adversarial Diffusion Distillation

由于作者没有发布 ADD 专用 trainer，WorldFoundry 只实现报告中能直接确认的部分：

- 四步 student noise schedule；
- frozen teacher score distillation；
- decoder 与 frozen feature network；
- feature discriminator 与 adversarial generator loss；
- exponential weighting，以及 recipe 显式提供完整权重时的 SDS 路径；
- R1 regularization；
- discriminator 与 generator 两个 optimizer。

任何报告未披露的 NFSD、数据、augmentation、精确 discriminator 配置都不能伪装成官方默认。需要通过独立消融和真模型实验选择 WorldFoundry profile，并明确标为自有配置。

### 6.5 Progressive / LCM / SwD / rCM / causal family

- Progressive：每个 stage 将 teacher step 数减半，teacher 从上个已提交 student/EMA 产生；stage reset、LR anneal、EMA 与 export boundary 可恢复；
- LCM：teacher DDIM pair、guidance embedding、boundary condition、consistency target、EMA 完整执行；
- SwD：每个 solver interval 绑定空间尺度；DMD/GAN/MMD 的启用、权重、feature blocks 和 fresh fake update 数均为行为字段；
- rCM：continuous/discrete consistency 与 DMD joint objective 分离，JVP capability 未满足时 fail closed；
- Causal-rCM：teacher-forcing consistency 与 self-forcing DMD 使用独立 teacher/score roles，attention block pattern 复用 core；
- Causal ODE/CD：作为 DMD 前的独立 initialization stage，不塞进一个 DMD boolean；
- Self-Forcing/SGF/Diagonal：cache commit、rollout window、gradient truncation 和 context-noise 语义进入 checkpointable state。

### 6.6 SANA SCM-LADD

- generator 与 discriminator 严格交替更新；只有 generator optimizer commit 推进 student scheduler；
- generator 固定使用发布配置的 5000-step constant warmup，discriminator 不创建无意义的 scheduler 状态；
- student 图显式开启 FP32 linear attention，teacher 与 discriminator 的 frozen teacher backbone 显式关闭；
- scheduler、两个 optimizer、G/D phase、progress 与数据位置进入同一 DCP 边界；
- tiny 真 SANA 图已覆盖 loss/backward、G-only scheduler cadence 和 scheduler exact resume；下一证据层是官方 SANA-Sprint 权重短跑。

### 6.7 DiffusionOPD / MOPD

- rollout 永远来自当前 student；teacher 只 replay student 已访问的 state，不生成另一条 teacher trajectory；
- batch 必须是单 domain，domain 直接选择独立 frozen teacher/checkpoint/CFG；
- `add_kl_coefficient=false` 仍使用极小非零 eta 记录 replay transition，但 objective 是不除方差的 mean matching；
- 开启 KL coefficient 时使用同一 transition strategy 的 scale，teacher target 在 FP32 中 detach；
- optimizer accumulation 必须覆盖完整且均衡的 teacher cycle；reward 与 advantage 不进入 loss；
- 当前完成 model-neutral run/update/resume/export，SD3.5 teacher LoRA materializer 是下一证据层。

## 7. RL 正确性 contract

### 7.1 Flow transition 与 replay

- stochastic rollout 和 train replay 调同一纯 transition 函数；
- trajectory 保存 latent states、sigma schedule、old log-prob/mean、strategy identity、policy revision；
- `eta == 0` 是 ODE，不能进入需要 likelihood 的 policy objective；
- rollout-anchor 与 replay-anchor 分开，只有 replay-anchor 要求首步 ratio 为一；
- stale policy revision、schedule drift、transition drift 均拒绝更新。

### 7.2 Advantage 与 objective

不同算法保留自己的 denominator：population variance/std、sample std、global std 不能互换。Reward vector 在 evaluator 边界不静默求和；scalarization、invalid policy、reference KL 和 clip schedule 都是显式训练状态。

### 7.3 Video family contract

- Wan2.2 high/low-noise experts 的 checkpoint、LoRA target 和 boundary routing 独立；reference role 从自己的 checkpoint materialize，不能复用 policy override；
- HunyuanVideo original/1.5 分别使用自己的 conditioning、guidance 和 tuning contract；
- LTX-2 video SDE 与 audio ODE 分开，LTX-2.3 joint AV 才把两条 transition 的 log-prob 按 typed latent geometry 合并；
- decoder 接收 terminal native artifact；LTX-2.3 AV reward 同时携带 video、audio 和 sample rate，不能只传 video tensor；
- 未定义联合 transition mean 的 LTX-2.3 DPPO/reference-KL/NFT 组合不靠猜测开启。

### 7.4 Token PPO

- rollout 时固定 old actor log-prob 和 pre-update critic value；
- terminal reward、per-token reward 与 bootstrap value 明确散射后计算 packed GAE；
- actor 使用 asymmetric clipped surrogate，critic clipping 与 value loss 始终在 FP32；
- update epochs、microbatch/reduction、entropy/value coefficients 与 policy revision 都进入 strict spec/state；
- actor、critic、optimizer、loader、rollout position、RNG、progress 进入同一 checkpoint boundary。

### 7.5 Agentic 与 remote reward

- 每个 sibling 独立执行多轮 assistant→tool→tool-result；单个 sibling 失败不终止其他 sibling；
- 每组至少两个成功 sibling 才形成可归一化的 policy batch，失败 sibling 不参与 reward 与 advantage；
- rollout sampled token、old log-prob、完整 transcript 和 tool result 保持对齐；training replay 关闭 dropout；
- local 与 Ray rollout 使用相同 typed trajectory，Ray 在每个新 policy revision rollout 前同步权重；
- HTTP scorer 只返回命名 reward vector；`correctness` 对终局文本与请求 conditions 中声明的 expected answer 做规范化精确匹配，`tool-success` 核对完整 transcript 中 required tool 的 call/result 与 `tool_failed`；答案错误或工具执行失败是有效零分，缺少目标或 transcript 元数据才是 invalid，scalarization 仍由 trainer 侧执行。

### 7.6 Algorithm completeness

共享 `flow_policy` 或 `token_policy` engine 只提供公共执行原语。每个公开算法名必须另外具备：

- strict recipe；
- 独立 objective/reduction；
- 作者 cadence；
- algorithm-specific state；
- 直接公式测试；
- 至少一个真实 model × algorithm gate 后，才能标为 release-supported 或用于对外能力声明；在此之前的配置只代表可执行候选 profile。

## 8. 可扩展分布式设计

```text
world_size = dp_replicate * dp_shard * cp * tp
```

### 8.1 不变量

- topology 从实际 process group 推导；
- multi-rank trainable module 必须由 DDP/FSDP2/DTensor 同步；
- local mean loss 用 global denominator 正确缩放，支持 uneven local batch；
- 需要同组决策的 timestep、step budget、rollout choice 由 process group broadcast；
- 不需要共享的 augmentation/noise 使用 checkpointable rank-local RNG；
- rank 0 只协调输出，不拥有唯一算法状态；
- exact resume 默认要求相同 topology；model reshard 与 data/RNG exact resume 是不同 capability；
- 上游使用 8/32/64 张卡的发布实验不改变 WorldFoundry contract。

### 8.2 Ray rollout execution plane

- `TrainerBinding.EXTERNAL` 不创建空 trainer actor，也不宣称 trainer/rollout 共置；Ray 只拥有 dedicated rollout/reward slab；
- `TrainerBinding.ACTOR` 创建真实 trainer group；separate 使用不同 placement bundle，colocate 使用同一 bundle 的不同 fractional slot；
- runtime 的 replicated trainer actor 支持 full/LoRA 分桶同步；所有 rollout receiver 先完成 shape/name/revision validation，再逐个 commit，commit 后失败没有跨 actor rollback；
- actor-hosted FSDP/DTensor source 尚未实现所需的全 trainer-rank collective materialization，因此显式拒绝；controller-local distributed synchronizer 仍要求所有 rank 参与；
- Agentic 的正式 Qwen recipe 已分别接通 actor-hosted separate 与 colocate；external+separate 仍是 caller-owned trainer API。generic video Flow 已接 external+separate，并为 Wan2.2/LTX-2.3 提供正式 profile；video actor trainer 与 per-rank FSDP2 Ray materializer尚未实现并主动拒绝；
- CPU-Ray 证明调度与状态语义，不外推 GPU affinity、NCCL、跨节点带宽或吞吐。

### 8.3 验证顺序

1. CPU/toy single process；
2. single GPU；
3. world-size-one FSDP2；
4. 2+ GPU DDP/FSDP2；
5. HSDP；
6. context/sequence parallel；
7. tensor parallel；
8. multi-node；
9. rollout/training disaggregation。

每一步都要分别通过数值、resume、failure injection 和性能门禁，不能由前一层自动推断。

## 9. Checkpoint 与故障语义

完整 checkpoint 至少包含：

- 所有 trainable model 和 optimizer；
- scheduler、EMA、grad scaler；
- algorithm engine phase/cadence/counters；
- session progress；
- loader/sampler；
- objective/rollout RNG；
- Python/CPU/CUDA RNG；
- reward scalarizer 或 adaptive statistics；
- recipe/model/data/cache/topology identity。

写入流程使用 staging、tensor snapshot、manifest、`_SUCCESS` 与 atomic latest pointer。已有 step 不覆盖；恢复时直接核对必需状态、文件大小和 tensor inventory。

边界规则：

- optimizer step 前失败可清梯度后重试；
- 任一 optimizer 已提交后失败，engine poisoned；
- checkpoint 只允许落在完整 algorithm commit boundary；
- on-policy 算法若将来支持 mid-trajectory resume，必须保存完整 trajectory/reward/advantage/anchor/cursor。

## 10. 正确性验证

### 10.1 每个官方 trainer-backed 技术

1. 固定作者 commit 和 license；
2. 读取作者 trainer、model implementation、发布 config、paper/report；
3. 列出源码与论文差异；
4. 将所有行为选择放入 strict recipe；
5. 使用小 tensor 直接执行作者公式和 WorldFoundry 公式；
6. 验证 forward、loss、gradient isolation、update order；
7. toy E2E 与 exact resume；
8. 真 checkpoint short run；
9. 条件允许时在隔离环境运行作者 trainer，比较同输入完整 optimizer update；
10. 多卡、长程质量和性能。

作者 trainer 只在隔离验证环境运行，不成为 production runtime dependency。

### 10.2 ADD 一类 report-derived 技术

1. 只实现报告明确给出的方程与结构；
2. 不给未披露配置贴“官方”标签；
3. 对不确定部分使用显式 WorldFoundry recipe；
4. 做消融、稳定性和质量对照；
5. 文档永久保留 report-derived 标签，除非作者后来发布可核对 trainer。

### 10.3 必跑命令族

- targeted algorithm formula/runtime/recipe tests；
- `tests/training` 全量回归；
- Ruff；
- `compileall`；
- public facade/lazy-import boundary tests；
- single-process distributed denominator tests；
- 条件允许时运行真实 GPU 和 2+ rank gates。

不在文档里写一个会迅速过期的固定测试数量；发布时记录具体 commit、命令、环境和结果。

## 11. 性能路线

正确性门禁后逐项启用：

### 11.1 低风险

- BF16 model compute + FP32 trajectory/reduction；
- fused AdamW；
- activation checkpoint；
- stateful prefetch/pinned memory；
- latent/text cache；
- bucketed batches；
- frozen role no-grad/eval；
- async DCP immutable staging。

### 11.2 中风险

- `torch.compile`；
- selective activation checkpoint；
- FlashAttention/SDPA/FlexAttention；
- CPU/NVMe optimizer/checkpoint offload；
- async reward batching；
- HSDP、CP/SP；
- dedicated Ray rollout workers；
- rollout/decode/reward pipeline overlap。

### 11.3 高风险

- native sparse video attention；
- custom Flow-SDE/log-prob CUDA kernel；
- JVP attention kernel；
- FP8/FP4；
- GPU/multi-node actor-colocated rollout 与 sharded trainer weight materialization；
- causal cache kernel；
- quantization-aware DMD。

每项记录 forward/backward/update parity、samples/s、peak memory、communication time、reward latency；不只记录“能跑”。

## 12. 文件组织

```text
worldfoundry/training/
  api/                         # functional batch/model/objective contracts
  data/                        # manifest/cache/bucket/stateful loader
  distributed/
    parallel.py fsdp.py        # current topology/DDP/FSDP2
    ray_runtime.py             # placement groups and actor groups
    rollout_runtime.py         # external/actor trainer binding
    flow_rollout.py            # group-preserving remote Flow sampler
    weight_sync.py             # full/LoRA staged revision fan-out
  models/                      # training model materializers/adapters
  engine/                      # model-family run lifecycle and actor-local video policy factories
  post_training/
    shared/                    # optimizer/distributed/checkpoint primitives
    distillation/
      dmd/
      dmd2/ sid/ sgmd/ dfd/
      progressive/ latent_consistency/ scale_wise/
      rcm/ causal_ode/ causal_consistency/
      self_forcing/ self_gradient_forcing/
      adaptive_video/ diagonal/
      anyflow/ senseflow/ reward_forcing/
      adversarial_diffusion/
      t2v_turbo/ diffusion_opd/
    agentic/                    # multi-turn tool environment and local/external/actor-hosted Ray run
    causal_lm/
      qwen3/                    # model/value head, Hermes codec and native materializer
    rl/
      transitions/
      rollout_strategies/
      algorithms/               # flow/token algorithms, including token_ppo
    rewards/
      http/                     # typed scorer client/service/registry
      scorers/                  # VideoPickScore/CLAP/Agentic scorers and strict service config
  checkpoint/
  recipes/
    post_training/rollout.py   # local/Ray placement and weight-transfer contract
  tuning/

worldfoundry/base_models/
  .../wan/variants/anyflow.py  # attributed native AnyFlow model graph

tests/training/
  fixtures/source_formulas/
  fixtures/recipes/
```

公共 facade 保持 lazy import；distillation leaf 不互相导入对方 runtime；model graph 不导入 training engine；training 不导入 synthesis trainer。

## 13. 已有真实证据与未完成项

### 13.1 已有真实证据

仓库外 gate report
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/wan21-real-dmd-gate-20260730f/gate_result.json`
记录了一次 Wan2.1 T2V 1.3B、单张 A100-SXM4-80GB 的 native DMD update、DCP immediate/continuous resume 与 fresh-base PEFT reload。但该报告早于本轮 FastVideo cadence 审计：当时 student 在第 1 次 iteration 更新且 accumulation=1，不能作为当前修正后发布 profile 的真权重证据，只保留为历史工程记录，并由下述修正后报告取代。

仓库外 gate report
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/wan21-real-dmd-gate-20260809-official-resume-oracle/gate_result.json`
记录了修正后 FastVideo profile 的 Wan2.1 T2V 1.3B 单卡门禁：student 与 fake-score 的 gradient accumulation 均为 8，前 4 次 logical iteration 只更新 fake-score，第 5 次提交一次 student 更新后再更新 fake-score。student 的 240 个 LoRA tensor 与 fake-score 的 480 个 LoRA tensor 全部发生变化；step-5 checkpoint immediate resume、从同一 live step-5 boundary 分叉的 step-6 continuation、以及 fresh-base student PEFT reload 都逐 tensor/状态精确一致。continuation 对照明确复用同一个训练分支，而不是比较两次独立 CUDA 初始化。峰值 CUDA allocation 为 14,506,540,544 bytes，运行 138.5 秒。该报告取代上一段历史报告作为当前单卡 profile 证据，但不外推到 full tuning、2+ GPU 或长程质量。

仓库外 gate report
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/wan21-real-flow-grpo-gate-20260801f/gate_result.json`
记录了一次 Wan2.1 T2V 1.3B、固定 Qwen2-VL/VideoReward、64×64×5、group size 2、单张 A100-SXM4-80GB 的 native Flow-GRPO rollout→VAE decode→VideoAlign→replay→LoRA update→DCP immediate resume→PEFT export。240 个 LoRA 参数张量发生变化；恢复时 480 个 trainable-state 张量以及 engine、progress、data cursor、generator state 精确一致；峰值 CUDA allocation 为 10,905,873,920 bytes，运行 209.1 秒。该证据不外推到其他分辨率、group size、reward、拓扑或训练时长。

仓库外报告
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/anyflow-real-far-gate-20260809a/summary.json`
记录了 AnyFlow FAR 官方 1.3B checkpoint 在单张 A100 上的 native load 与 BF16 finite forward：1,422,147,136 个参数，输入输出均为 `[1,16,9,4,4]`，峰值显存约 2.67 GiB。

仓库外报告
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/anyflow-real-bidirectional-gate-20260809a/summary.json`
记录了 AnyFlow 双向官方 1.3B checkpoint 的 native load、两次真实 optimizer update、DCP step-1 恢复和 resumed step-2。连续与恢复后的 loss 都是 `2.4771006107330322`；model、optimizer、engine、progress 与 RNG 状态直接逐字段/逐 tensor 比较一致，98,304 个被跟踪参数元素发生变化，峰值显存约 10.67 GiB。该证据只覆盖双向 pretrain 的单卡短跑。

本仓软件生命周期 gate 另覆盖以下新增执行面，但证据层级仍是 CPU/tiny/native graph：

- Wan2.2、HunyuanVideo 1.0/1.5、LTX-2/2.3 的 strict profile、family conditioning 与 rollout/replay；model-neutral Flow run 覆盖 optimizer update、DCP split-resume 与周期/显式 export，Wan2.2 正式 Ray materializer另覆盖完整 tiny lifecycle；
- LTX-2.3 joint AV decode、VideoPickScore/CLAP HTTP request 与 reward scalarization；两 actor CPU-Ray 测试覆盖 interleaved shard 重排后 audio trajectory/transition means/scales、joint replay old-log-prob 与 backward；native scorer 使用 fake model 校验官方公式/预处理，service config 与可启动 CLI 已覆盖，但未执行真实 scorer 权重；
- Agentic 多轮 tool-use 的 local sibling rollout、packed-token update、原生 HTTP correctness/required-tool reward、DCP exact resume 与 export；HTTP round-trip 覆盖批处理顺序、失败零分和缺失契约 invalid；真实 CPU-Ray external+separate E2E 覆盖 full/LoRA、连续 revision、多轮工具调用、单 sibling 失败隔离、old-log-prob replay 与 optimizer commit；正式 Qwen actor+separate/colocate recipe 均完成真实 update/placement，共置路径额外覆盖 uninterrupted 对 split-resume 的 policy/rollout state 精确一致与 export；
- Qwen3 grouped token policy 与 token PPO 的 model/value-head update、官方 Hermes turn boundary、FP32 value clipping、GAE、resume 与 fresh-module strict export reload；
- DiffusionOPD 的 complete multi-teacher cycle、同轨迹 replay、update/resume/export；
- Ray DevicePool 的 CPU actor separate/colocate，以及 replicated full/LoRA staged revision weight sync；正式 GPU profile已进入配置，但尚未在 GPU Ray allocation 上执行。

本轮收口后已执行 `tests/training` 全量回归、目标功能组合回归、Ruff lint、目标文件 format check、`compileall`、public facade/import boundary、禁用模式扫描与 `git diff --check`；不以固定测试数量代替上述证据边界。

### 13.2 不能据此声称

- SANA SCM-LADD、SANA SiD、AnyFlow FAR/on-policy、SenseFlow、Reward-Forcing、ADD 或其他新增 distillation 已完成真权重训练；
- 当前 SANA-Sprint SiD 与 SANA 作者 trainer 对齐；固定 SANA 提交没有该 trainer/profile；
- 任一新增算法已与作者完整 optimizer update 数值一致；
- 2+ GPU post-training 已完成；
- 14B、full tuning、multi-node、长程质量已完成；
- LVDM/DynamiCrafter/T2V-Turbo 已完成正式数据预处理、真权重或 2+ GPU 训练；
- LTX、Cosmos Predict2.5 或 Cosmos3 已完成发布权重的完整 optimizer update、2+ GPU 或长程质量验证；
- Wan2.2、HunyuanVideo 或 LTX-2/2.3 的新增 RL profile 已完成发布权重、正式分辨率、真实 scorer、GPU Ray 或 2+ rank 训练；
- Qwen3 Agentic/PPO 的 model/value-head、tokenizer/codec、recipe/CLI、calculator correctness/required-tool HTTP reward 和 software lifecycle，不能据此声称已完成发布权重任务短跑、LoRA/FSDP 或多卡 learner、正式 search/visit/MathVerify/LLM-judge 数据与 reward、质量或线上稳定性证据；
- Ray CPU actor 测试已经证明 GPU affinity、NCCL、multi-node 性能，或 actor-hosted FSDP/DTensor 权重同步；
- video Ray actor-hosted/colocate learner、per-rank FSDP2 runtime，或 external trainer 的自动物理隔离；当前只支持 external+separate，并由调用方保证 trainer 与 rollout 的物理隔离；
- DiffusionOPD 已有 SD3.5 teacher LoRA materializer、CLI、发布权重或质量对齐；
- compact LVDM VideoAE 与作者 `aemodules3d` state/forward 等价；
- LongCat/StepVideo 的 report-derived 公式已经等价于作者未公开的 trainer；
- 软件单测可以替代发布质量或性能测试。

## 14. 下一阶段执行顺序

1. **已完成**：收口 AnyFlow、SenseFlow、Reward-Forcing、ADD 的 shared registration、lazy facade、角色/checkpoint 门禁与蒸馏全回归；常用蒸馏算法 scope 冻结；
2. **已完成**：Wan Flow-GRPO 的 rollout→decode→VideoAlign→replay→update→DCP exact resume 真权重门禁；
3. **已完成**：重新执行修正后的 Wan DMD FastVideo profile 真权重 update/resume/fresh PEFT reload gate；
4. **已完成（软件执行面）**：Wan2.2、HunyuanVideo 1.0/1.5、LTX-2/2.3 Flow-RL，LTX-2.3 AV HTTP reward，Agentic local/Ray tool-use RL，token PPO，DiffusionOPD，以及 Ray DevicePool/full-LoRA sync；
5. 使用 recipe 已固定的 Wan2.2、HunyuanVideo 1.0/1.5、LTX-2/2.3 发布 checkpoint 与 VideoPickScore/CLAP scorer，逐条执行真实权重、正式分辨率 update、DCP split-resume、fresh-process export reload；
6. **已完成（软件执行面）**：Qwen3 CausalLM/value head、Hermes tokenizer/codec、calculator 工具环境、prompt manifest、grouped Agentic/PPO recipe 与 CLI；下一步改用正式 search/visit/MathVerify 任务执行发布权重短跑，再补 LoRA/FSDP 和多卡 learner；
7. 在 GPU Ray allocation 上验证 Qwen actor separate/colocate 与 video external rollout；Qwen 的单 replicated actor 软件路径已完成，下一步先补多 rank source collective materialization，再允许 actor-hosted FSDP/DTensor weight sync；
8. 为 DiffusionOPD 接 SD3.5 student 与三份冻结 teacher LoRA，跑完整 task-cycle update/resume 和 held-out per-domain 质量；
9. 使用官方 SANA-Sprint 权重执行 SCM-LADD 的 generator/discriminator update、DCP split-resume 与 student export reload；SiD 保持 experimental reimplementation 标签；
10. **部分完成**：两个固定 AnyFlow 1.3B checkpoint 已完成 config/load/forward，双向 pretrain 已完成真权重 update/resume；继续补 FAR update 与一条 on-policy 真权重路径；
11. 使用固定官方 checkpoint 对 LTX-Video/LTX-2.3 基础训练、Cosmos Predict2.5 SFT/DMD2、Cosmos3 Nano vision SFT、LVDM short、DynamiCrafter 512/1024 和 T2V-Turbo consistency-math profile 执行真权重 update/resume/export；LongCat、StepVideo 在作者未发布 trainer 前不补写；
12. 在实际 2+ GPU allocation 上验证 arbitrary world size、uneven local batch、参数同步与 DCP；建立 held-out quality、reward-hacking、long-horizon drift、latency、显存和吞吐 benchmark；
13. 只有上述真实门禁稳定后，再推进 sparse attention、JVP kernel、compile、async reward、HSDP/CP/TP 与多节点性能；不要求所有算法逐一接满所有模型。

完成标准不是“目录和类都存在”，也不是穷举 model × algorithm 笛卡尔积；而是每个正式声明支持的组合都能从 strict recipe 构建 WorldFoundry-owned stack，完成真权重 update/resume，并在声明过的 topology、质量和性能门禁上留下可复核证据。

## 15. 企业级 infra 产品差距（2026-08-14 评估）

以上各节覆盖训练正确性纪律：证据分层、license 边界、fail-closed contract、原子 checkpoint 在同类项目中已属高标准。但“正确的训练栈”不等于“企业可用的 infra 产品”。以下差距按影响排序，均以当前仓库实际状态核对过，不重复既有章节已列的真权重/多卡缺口。

### 15.1 可观测性执行面缺失（最高优先级）

- `worldfoundry/training` 全包没有结构化 logging，也没有任何实验跟踪集成；训练进程对操作者是黑盒，只有结束后的 `run.json`/gate report；
- post-training session 的 `event_sink` 是唯一遥测口，但 CLI `post-train` 不接线任何 sink，正式运行不落 metrics 流；基础训练的 `MetricWriter` jsonl 只覆盖 `SingleDeviceTrainingSession`；
- 没有 NaN/loss 发散检测、显存水位、吞吐（samples/s）等运行时信号，长程训练的故障只能事后从 checkpoint 边界推断。

DoD：统一 run 事件 schema（现有 jsonl 事件已是雏形），CLI 默认落盘；提供可选 wandb/tensorboard exporter（保持核心零依赖）；engine 层加 per-step 有限性与显存/吞吐采样，越界时进入与 poisoned-state 一致的 fail-closed 路径。

### 15.2 CI 不执行训练测试

`tests/training` 约 4.8 万行、数百项测试只能人工执行；`.github/workflows/ci.yml` 仅运行 docs 构建、Ruff、compileall 与 CLI 检查。第 10.3 节的“必跑命令族”没有自动化载体，回归窗口完全依赖人工纪律。

DoD：CPU 可跑子集进 PR CI；GPU runner 每夜执行 tiny/short-run 门禁；性能基线（吞吐/显存）纳入夜间回归并对回退报警；发布分支要求全量 `tests/training` 通过记录。

### 15.3 作业编排与自动容错

当前编排边界是 torchrun + Ray DevicePool。没有集群调度器（Slurm/K8s）提交与配额集成、没有监督进程实现 checkpoint 自动重启、没有抢占/节点故障的自动摘除。poisoned-state 语义已定义，但恢复动作完全人工。

DoD：提供 launcher 层（至少 Slurm 脚本模板 + K8s Job spec）；崩溃后从 latest pointer 自动 resume 的 supervisor；rank 心跳与 NCCL 超时的诊断输出。

### 15.4 环境身份不进 resume identity

checkpoint identity 覆盖 recipe/model/data/cache/topology，但不含 torch/CUDA/驱动版本与容器镜像 digest。跨环境 resume 时数值语义可能漂移而不 fail closed；gate report 的环境信息目前靠人工记录。

DoD：identity 增加 environment 段（torch/CUDA 版本、镜像 digest、关键库版本），默认精确匹配，允许显式降级为警告。

### 15.5 训练中质量回路

held-out 质量、reward-hacking 是第 12/13 节的 gate 名称，但没有 in-loop 执行面：无周期 held-out 评估、无早停、无 reward 分布漂移检测工具。RL 路径的 reward-hacking 只能在训练结束后人工发现。

DoD：session 支持可选 eval cadence（复用 evaluation 包的 benchmark runner）；reward 统计漂移（均值/方差/组内退化）进入 event 流并可配置阈值中断。

### 15.6 产品面与代码面的比例

recipe 层注册约 41 个算法 spec，CLI `post-train` 只能启动约 14 个；其余是 library-only 候选，且多数停在 CPU/tiny 证据层。每个候选都占用回归与维护成本。

DoD：显式三层标签（release-supported / candidate / experimental）进入文档与 CLI 错误信息；支持矩阵从 recipe registry 自动生成，避免文档漂移；candidate 层冻结新增功能，只随共享原语演进。

### 15.7 其余差距（按序推进）

- **API 稳定性**：facade 导出 500+ 符号，无 public/experimental 分层、无弃用政策与版本化；
- **数据规模化**：数据面以本地预计算 cache 为中心，缺对象存储/流式读取、cache GC 与配额策略；
- **安全与合规运营**：license 纪律已达标，但缺 secrets 管理约定（HF token 等）、导出 artifact 的签名/访问控制，以及 RL 后模型的安全评测门禁（目前仅 Agentic 一处提及）；
- **runbook**：训练文档只有单页 guide，缺故障目录（OOM、NCCL 超时、DCP 恢复失败、cache 校验失败的处置步骤）。

执行顺序建议：15.1/15.2/15.4 先行——它们直接决定既有真权重门禁的可信度与可复现性；15.3/15.5 随 2+ GPU 与长程训练（第 14 节第 12 步）同批落地；15.6/15.7 在下一次对外声明支持范围前完成。
