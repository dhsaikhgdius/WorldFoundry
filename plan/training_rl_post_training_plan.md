# WorldFoundry 原生 Training / RL / Post-Training 实施计划

> 更新时间：2026-08-03
> 执行所有者：`worldfoundry-native`
> 当前状态：常用蒸馏算法的 strict contract 和 native stack 集成已达到阶段目标
> 下一阶段重心：SANA SiD/AnyFlow 真权重、2+ GPU、质量与性能

## 0. 结论与边界

WorldFoundry 要实现的是自己的训练基础设施，不是把官方仓库包装成后端。

1. model forward、data/corruption、rollout、reward、loss、backward、optimizer、scheduler、EMA、distributed state、checkpoint、export 都由 WorldFoundry 执行；
2. 作者仓库用于核对架构、公式、训练配置和更新顺序；正常训练进程不启动、导入或代理作者 trainer；
3. 许可证允许且工程收益明确时，可以移植作者模型组件或局部算法实现，并在 `THIRD-PARTY-NOTICES` 和源文件头保留 attribution；
4. 官方提供 trainer 的算法，以固定源码和发布配置作为实现依据；官方未提供 trainer 的算法只能称为 report-derived implementation，不能称为官方复现；
5. recipe 只保留会改变执行行为或会阻断错误执行的字段；不增加 provenance registry、reference-trace registry、repository-scope CLI、只写不读的治理 metadata；
6. checkpoint identity、角色独立性、冻结状态、optimizer parameter ownership、resume identity 都是功能性门禁，必须保留；
7. GPU 数由实际 process group 和并行拓扑推导，不固定为 8；上游脚本中的 8 卡只是一份发布 profile；
8. 单卡、任意合法 world size、多节点使用同一算法状态机；不能为某个固定卡数另写训练 loop；
9. 公共 attention/cache、gradient、I/O integrity、path confinement、atomic write 等能力复用 `worldfoundry/core`；
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
| Wan DMD | strict recipe、few-step rollout、real/fake score、双 optimizer、session、DCP/export | Wan2.1 1.3B 已接线 | 2+ GPU、full tuning、长程质量 |
| SANA SiD | strict recipe、few-step detached prefix、student/teacher/fake-score、双 optimizer、session、DCP/export | SANA-Sprint 600M local-Diffusers materializer、cache/prompt-only loader、CLI/config 与 real-roundtrip gate script 已接线 | 执行并留存真权重报告、2+ GPU、质量/显存 gate |
| DMD2 / SGMD / DFD | 各自 recipe、公式、角色、engine、session、checkpoint | model-neutral adapter seam | 逐 family 真模型 materializer 与 real gate |
| SCM-LADD | TrigFlow/consistency、student/teacher/discriminator、双 optimizer、session | SANA 语义 seam | 官方 SANA-Sprint 权重短跑与质量对照 |
| Progressive distillation | DDIM 两步教师目标、逐阶段减半、LR/EMA/stage checkpoint | model-neutral prediction seam | 具体 image/video family materializer 与真实多阶段训练 |
| Latent Consistency Model | DDIM pair、boundary scaling、guidance embedding、EMA、engine/session | model-neutral prediction seam | 官方 LCM 配置的真权重更新对照 |
| Scale-wise Distillation | 多尺度 schedule、DMD/GAN/MMD、fresh fake updates、SD3 prediction/critic adapter | SD3 execution adapter | 官方 SD3.5 资产短跑、分布式和质量基线 |
| rCM / Causal-rCM | continuous/discrete consistency、DMD joint loss、JVP seam、TF/SF causal stack、session | bidirectional 与 causal adapter seam | 真 Wan/Cosmos 模型 gate、JVP kernel 性能与多卡 |
| Causal ODE / Causal Consistency | 独立 recipe、pair/schedule、objective、engine、session | causal adapter seam | 官方 Causal-Forcing 资产与完整三阶段验证 |
| Self-Forcing | 自生成 causal rollout、KV/cache commit、gradient truncation、DMD session | native Wan causal chunk adapter | 发布权重短跑、长上下文 drift、SP/CP |
| Self-Gradient-Forcing | noisy-context commit、teacher-forcing replay、DMD engine/session | native Wan causal adapter | 作者配置真权重 gate 与长程质量 |
| Diagonal / Adaptive Video | diagonal block rollout、motion head/EMA、adaptive regression、DMD execution | causal/model-neutral seams | 对应作者资产与 production materializer |
| AnyFlow | FAR/双向 pretrain 与 on-policy 四种 recipe；FlowMap、central difference、DMD、fresh fake-score、EMA、同步随机决策、双 optimizer、session/DCP | native AnyFlow Wan graph；FAR 与双向 1.3B materializer | 真权重训练短跑、2+ GPU、14B、质量/速度 |
| SenseFlow | strict recipe；IDA、ISG、DMD、adversarial objective；student/teacher/fake-score/discriminator；三 optimizer/session/DCP | model-neutral flow adapter seam | SD3.5/FLUX materializer、官方权重短跑、多卡/质量 |
| Reward-Forcing | strict recipe；21-frame causal rollout、EMA-Sink、local attention、Re-DMD reward weighting、双 optimizer/session/DCP | native causal/Wan seams | 官方模型与 reward 资产短跑、流式质量/速度 |
| Adversarial Diffusion Distillation | strict recipe；pixel decoder/feature discriminator、SDS 或 exponential target、R1、D→G 更新、双 optimizer/session/DCP | generic image adapter seam | SDXL model materializer、真权重训练、官方不可得细节的敏感性实验 |

`post_training/distillation/causal` 和 `post_training/distillation/consistency` 是被多种算法执行路径消费的共享数学/contract，不是治理占位目录。

### 2.2 RL 与 preference optimization

| Family | WorldFoundry 当前执行面 | 当前仍缺 |
|---|---|---|
| Flow-GRPO | Flow-SDE rollout/replay、group advantage、clipped objective、reference KL、revision/session；Wan2.1 1.3B + VideoAlign 单卡真权重 update/DCP exact resume 已通过 | 2+ GPU、长程质量与 reward-hacking gate |
| Flow-DPPO | old-mean divergence mask、KL-ADV、共享 flow-policy stack | 同上 |
| DANCE-GRPO | constant-diffusion transition、同组噪声/奖励归一化、独立 recipe/session | 作者发布 profile 的真模型对照 |
| MixGRPO | progressive sliding window、window cadence、完整 recipe/session | 作者发布 profile 的真模型对照 |
| GRPO-Guard | differentiable drift bias、冻结 old anchor | 真模型/奖励 gate |
| Bagel Flow-UniGRPO | conditional mixing 与算法专属 objective/session | Bagel model materializer |
| DiffusionNFT | terminal clean-latent collection、forward-process NFT、old policy refresh | Wan single-device 路径之后的 FSDP2 与质量 |
| Diffusion-DPO | offline chosen/rejected pair、current/reference loss | production dataset 与 model materializer |
| DDRL | PPO-style diffusion ratio、reference mean、data regularizer | Cosmos Predict2.5 rollout/reward/model materializer |
| token GRPO/GSPO/DPPO/DRPO/CPPO | packed-token trajectory/replay/partition engine，各算法独立 reduction | 真实 autoregressive model rollout/temperature adapter |

所有 RL 算法都必须有自己的行为 contract；共享 rollout/engine primitive 不等于把不同算法压成同一个名字。

### 2.3 阶段判断

常用 diffusion/flow distillation 的 recipe、数学、engine/session 和 checkpoint seam 已形成完整的原生框架，因此“继续收集更多算法”阶段结束。现有 model-neutral stack 会保留并测试，但并不要求为每个算法补齐所有模型 family。下一阶段只为有明确使用优先级的组合实现 production materializer 和真实门禁。

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
| Progressive Distillation | `google-research/google-research/ddpm_distillation` | 有 | 两步 DDIM target、cosine log-SNR、stage halving |
| Latent Consistency Model | `luosiallen/latent-consistency-model` | 有 | DDIM pair、boundary scaling、guidance embedding、EMA |
| Scale-wise Distillation | `yandex-research/swd` | 有 | SD3.5 schedule、scale boundaries、DMD/GAN/MMD |
| rCM / Causal-rCM | `NVlabs/rcm@ed3cb14dd936f92cdc9f9381af7369991509b41f` | 有 | TrigFlow、JVP consistency、DMD joint cadence、causal roles |
| Self-Forcing | `guandeh17/Self-Forcing` | 有 | self rollout、cache commit、gradient truncation |
| Causal ODE/CD | `thu-ml/Causal-Forcing` | 有 | causal ODE/consistency initialization 与 asymmetric DMD 分阶段语义 |
| SANA SCM-LADD | `NVlabs/Sana@6298508fcb511762a11c42cff45b2fc9fd930325` | 有 | SANA/Sprint model 与 SCM-LADD 训练语义 |
| Wan DMD | `hao-ai-lab/FastVideo@1b2b2a0161bc6b3b80158d1fa6380a051c6530c7` | 有 | few-step schedule、三角色、双 optimizer、cadence |
| Flow RL | `Tencent-Hunyuan/UniRL`、`yifan123/flow_grpo`、`verl-project/verl-omni` | 有 | transition、replay、advantage、policy objectives |

如果 paper、README 和 executable trainer 不一致，recipe 必须选择一个明确的可执行 profile，并在测试中固定差异；不能混合多个来源拼出一个无人发布的“官方默认值”。

Flow-GRPO 当前明确对齐 `Tencent-Hunyuan/UniRL@1f595a426bd5795b6edf093a48c0983c9e1673e1` 的 Flow-SDE 与更新语义。固定公式对照覆盖 transition next latent、mean、scale、log-prob、sparse/window timestep、group advantage、PPO clipping 和 old-log-prob freeze。旧版 `yifan123/flow_grpo@879042cf5707f8b90daa98d147d7deac2317c5da` 的线性 diffusion coefficient、前缀 train window 和 inner-epoch accumulation 已与当前 UniRL 分叉；WorldFoundry 不宣称同时逐行复现这两个版本。省略 `sde_window` 时采用 UniRL 的非重叠、到末尾停住默认值；重叠和 rollback 必须显式配置。

### 3.4 不能复制的边界

`tianweiy/DMD2` 源码使用 CC-BY-NC-SA-4.0。acknowledgement 不能替代授权，因此 WorldFoundry 只依据论文重新实现公式；在单独许可证复核前，不复制其源码。

## 4. 单一 native execution architecture

```text
PostTrainingRecipe
        |
        v
strict algorithm spec + role checkpoint identities
        |
        v
WorldFoundry model materializer / adapter
        |
        +-------- frozen roles: teacher / real-score / reference / decoder
        |
        +-------- trainable roles: student / fake-score / discriminator
        |
        v
algorithm-owned objective + WorldFoundry-owned optimizer engine
        |
        v
native distributed context
single process / DDP / FSDP2 / HSDP / CP / TP
        |
        v
native session + DCP + export + metrics
```

上游仓库不在这张运行图中。必要的 Apache-2.0 模型组件复制到 WorldFoundry 的 model graph 后，由本地 materializer 加载，由本地 adapter 执行。

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

- Wan DMD：generator cadence 到达时先提交 student，再用新 student 更新 fake-score；
- ADD：先 discriminator，再 generator；这是 WorldFoundry 的显式 adversarial update contract，不把报告未披露的精确 cadence 标成官方行为；
- AnyFlow on-policy：student batch 与 fresh fake-score batch 分离，每个 fake update 消费新的 batch；
- SenseFlow：fake-score、discriminator、generator 按 TTUR/cadence 显式推进；
- Reward-Forcing：fake-score 每 logical iteration 更新，student 依据发布 cadence 更新；
- 任一 optimizer 已提交后出现异常，engine 进入 poisoned state，只能从上一完整 checkpoint 恢复。

### 5.3 功能性身份校验

以下 identity 会影响正确性，因此不是治理 metadata：

- model/checkpoint config 与 tensor identity；
- recipe digest；
- dataset/cache identity；
- role checkpoint identity；
- optimizer/scheduler/EMA state；
- distributed topology；
- loader position 与 RNG；
- rollout/policy revision。

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
- FAR 的 temporal chunk partition、full/compressed patch geometry 和 long-context ratio 进入 recipe 与 model config gate；
- world size 不改变算法概率、cadence 或 batch ownership。

近期 DoD：

- 两个 1.3B 官方 checkpoint config/load/forward gate；
- pretrain 与 on-policy 各一次有限 loss/backward/update；
- checkpoint immediate resume 和 continuous resume；
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

## 7. RL 正确性 contract

### 7.1 Flow transition 与 replay

- stochastic rollout 和 train replay 调同一纯 transition 函数；
- trajectory 保存 latent states、sigma schedule、old log-prob/mean、strategy identity、policy revision；
- `eta == 0` 是 ODE，不能进入需要 likelihood 的 policy objective；
- rollout-anchor 与 replay-anchor 分开，只有 replay-anchor 要求首步 ratio 为一；
- stale policy revision、schedule drift、transition drift 均拒绝更新。

### 7.2 Advantage 与 objective

不同算法保留自己的 denominator：population variance/std、sample std、global std 不能互换。Reward vector 在 evaluator 边界不静默求和；scalarization、invalid policy、reference KL 和 clip schedule 都是显式训练状态。

### 7.3 Algorithm completeness

共享 `flow_policy` 或 `token_policy` engine 只提供公共执行原语。每个公开算法名必须另外具备：

- strict recipe；
- 独立 objective/reduction；
- 作者 cadence；
- algorithm-specific state；
- 直接公式测试；
- 至少一个真实 model × algorithm gate 后，才能进入正式配置目录。

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

### 8.2 验证顺序

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

写入流程使用 staging、immutable tensor snapshot、manifest、`_SUCCESS` 与 atomic latest pointer。已有 step 不覆盖；incomplete/tampered/symlink/path escape fail closed。

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
- rollout/decode/reward pipeline overlap。

### 11.3 高风险

- native sparse video attention；
- custom Flow-SDE/log-prob CUDA kernel；
- JVP attention kernel；
- FP8/FP4；
- remote rollout workers；
- causal cache kernel；
- quantization-aware DMD。

每项记录 forward/backward/update parity、samples/s、peak memory、communication time、reward latency；不只记录“能跑”。

## 12. 文件组织

```text
worldfoundry/training/
  api/                         # functional batch/model/objective contracts
  data/                        # manifest/cache/bucket/stateful loader
  distributed/                 # topology/DDP/FSDP2/HSDP/CP/TP
  models/                      # training model materializers/adapters
  engine/                      # model-family run lifecycle
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
    rl/
      transitions/
      rollout_strategies/
      algorithms/
    rewards/
  checkpoint/
  recipes/
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
记录了一次 Wan2.1 T2V 1.3B、单张 A100-SXM4-80GB 的 native DMD update、DCP immediate/continuous resume 与 fresh-base PEFT reload。该证据只覆盖这一个 model × algorithm × hardware profile。

仓库外 gate report
`/mnt/cpfsB/yangboxue/visual_generation/juanxi/.cache/worldfoundry/wan21-real-flow-grpo-gate-20260801f/gate_result.json`
记录了一次 Wan2.1 T2V 1.3B、固定 Qwen2-VL/VideoReward、64×64×5、group size 2、单张 A100-SXM4-80GB 的 native Flow-GRPO rollout→VAE decode→VideoAlign→replay→LoRA update→DCP immediate resume→PEFT export。240 个 LoRA 参数张量发生变化；恢复时 480 个 trainable-state 张量以及 engine、progress、data cursor、generator state 精确一致；峰值 CUDA allocation 为 10,905,873,920 bytes，运行 209.1 秒。该证据不外推到其他分辨率、group size、reward、拓扑或训练时长。

AnyFlow 已用缩小配置的真实 native AnyFlow Wan graph 在 A100 上完成 differentiable forward/backward/optimizer update；这只证明本地模型图与 materializer 的 CUDA 边界，不等价于两个官方 1.3B checkpoint 的真权重训练。

### 13.2 不能据此声称

- SANA SiD、AnyFlow 两个官方 1.3B checkpoint、SenseFlow、Reward-Forcing、ADD 或其他新增 distillation 已完成真权重训练；
- 任一新增算法已与作者完整 optimizer update 数值一致；
- 2+ GPU post-training 已完成；
- 14B、full tuning、multi-node、长程质量已完成；
- 软件单测可以替代发布质量或性能测试。

## 14. 下一阶段执行顺序

1. **已完成**：收口 AnyFlow、SenseFlow、Reward-Forcing、ADD 的 shared registration、lazy facade、角色/checkpoint 门禁与蒸馏全回归；常用蒸馏算法 scope 冻结；
2. **已完成**：Wan Flow-GRPO 的 rollout→decode→VideoAlign→replay→update→DCP exact resume 真权重门禁；
3. 用固定 SANA-Sprint 600M local-Diffusers 资产完成 SiD update、DCP split-resume 与 fresh-process student export reload，保存 gate report；
4. 用两个固定 AnyFlow 1.3B checkpoint 完成 config/load/forward，并选择一条最高优先级路径跑真权重 update/resume；
5. 在实际 2+ GPU allocation 上验证 SANA SiD、Wan DMD 和 Wan Flow-GRPO 的 arbitrary world size、uneven local batch、参数同步与 DCP；
6. 建立固定 prompt/seed 的 held-out quality、reward-hacking、long-horizon drift、latency、显存和吞吐 benchmark；
7. 根据产品模型优先级，从 SenseFlow、Reward-Forcing、ADD、SwD、rCM、causal/self-forcing 中选择下一条 production materializer；不要求所有算法逐一接满所有模型；
8. 只有在上述真实门禁稳定后，再推进 sparse attention、JVP kernel、compile、async reward、HSDP/CP/TP 和多节点性能；
9. DDRL、Diffusion-DPO、token policy 等路径在出现明确 production model/data 需求时再接 materializer，不作为当前阻塞项。

完成标准不是“目录和类都存在”，也不是穷举 model × algorithm 笛卡尔积；而是每个正式声明支持的组合都能从 strict recipe 构建 WorldFoundry-owned stack，完成真权重 update/resume，并在声明过的 topology、质量和性能门禁上留下可复核证据。
