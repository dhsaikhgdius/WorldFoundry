# training 引擎层修复日志（对应评审报告 plan/code_review/08_training_engine.md）

> 修复人：infra 修复 agent；日期：2026-08-14。
> 范围约束：只改 worldfoundry/training/{engine,distributed,checkpoint,data,optimizers}/、optimization.py、ema.py、state_comparison.py、training/`__init__.py`；api/models/objectives/post_training/recipes/tuning/safety 由另一 agent 负责。
> 验证约定：每条修复附 py_compile + import 冒烟 + 新建纯 CPU 单测（tests/training/test_training_engine_fix_*.py，共 6 文件 25 例全部通过）+ 相关既有测试。
> 环境限制：本机无 pypi，`torchdata`/`peft`/`ray` 不可安装——依赖它们的既有测试在修复前后同样 skip/fail（错误点均在 `data/loader.py:106-108` 或 importorskip，先于本次改动的代码路径），非本次改动引入。

## 已修复

### [TE-01] P1 FSDP2 单 rank 非有限 loss 本地 raise → 集合一致判定

- 改动：`worldfoundry/training/engine/fsdp.py:152-159`
  - 原逻辑 `if not torch.isfinite(loss).all(): raise FloatingPointError`：loss 有限性依赖本 rank 数据，rank 间可不一致；先 raise 的 rank 放弃后续 FSDP2 backward 集合通信，幸存 rank 卡死在 reduce-scatter 直到 NCCL watchdog 超时。
  - 修复后语义：每 rank 先算本地 `isfinite` 布尔，转 float32 走既有 `_reduced(..., ReduceOp.MIN)`（全 DP 组 all_reduce），任一 rank 非有限 → 所有 rank 同步得到 0 → 所有 rank 同一 microbatch 位置一致 raise。fail-fast 语义保留，僵死路径消除。
- 数值行为：无变化。loss/梯度计算不动；仅把"谁 raise"从局部决定改为集合一致决定。每个 microbatch 增加一次 1 元素 all_reduce（与该文件既有 metrics reduce 同量级）。
- 验证：
  - py_compile + `import worldfoundry.training.engine.fsdp` 冒烟通过。
  - 新增 `tests/training/test_training_engine_fix_te01_collective_loss_check.py`（1 例）：gloo 双进程 spawn，场景 A rank1 loss=NaN、rank0 有限 → 断言两 rank 集合判定一致为非有限（双方都会 raise，无单侧退出）；场景 B 双方有限 → 判定一致为有限。通过。
- 风险：低。all_reduce 在所有 rank 的相同代码位置执行，无错位死锁；单进程（未初始化 dist）路径 `_reduced` 原样返回，等价旧行为。

### [TE-02] P2 clip_grad_norm_(error_if_nonfinite=True) 短路 GradScaler 自愈 → 标准 skip-step 模式

- 改动：`worldfoundry/training/engine/single_device.py:156-187,214-236,354-381`（train_step + train_accumulation），`worldfoundry/training/engine/fsdp.py:203-266`（FSDP2 accumulation，继承同一 `_step_optimizer`）
  - 修复前时序：`unscale_ → clip(error_if_nonfinite=True) → scaler.step → scaler.update`。fp16 溢出时 clip 先 raise，scaler.step/update 永远执行不到，动态降 scale 自愈机制死路，训练直接终止（引擎标记 poisoned）。
  - 修复后时序（仅当存在**真 `torch.amp.GradScaler` 且 enabled**，经新增 `_active_amp_grad_scaler()` 判定）：`unscale_ → clip(error_if_nonfinite=False) → scaler.step（内部按 found_inf 跳步）→ scaler.update（跳步时按 backoff 降 scale）`。跳步检测：`update()` 前后 `get_scale()` 严格下降 ⇔ 本步被跳（GradScaler 强制 backoff_factor<1 且仅在 found_inf 时应用）。FSDP2 下 DTensor 梯度的 found_inf 会跨 mesh 全局归约，所有 rank 一致跳步/降 scale。
  - 保持不变的路径：无 scaler（bf16/fp32）→ `error_if_nonfinite=True` 原样 fail-stop；鸭子类型测试 scaler、disabled 真 scaler → 同旧语义（`_active_amp_grad_scaler()` 返回 None）。
  - grad-norm 记录语义（显式约定）：正常步记录 `metrics["grad_norm"]`；被跳步不记录 grad_norm（非有限值会被 durable metrics 的 canonical JSON 拒绝），改记 `metrics["optimizer_step_skipped"]=True`，并把 `TrainStepResult.skipped=True` 传给结果（`.skipped` 无既有下游消费者，rg 全仓确认）。
  - 跳步时 `global_step` 仍 +1、`optimizer_step_end`/EMA 回调仍触发（与 torch AMP 惯例一致：跳步是"空 optimizer step"）；引擎不 poisoned，下一步自愈。
- **数值行为变化（显式标注）**：仅 fp16+真 GradScaler 且梯度溢出的路径——旧行为=训练崩溃，新行为=跳过该步、scale 减半、继续训练（torch AMP 标准语义）。溢出步的非有限 grad-norm 不再产生（也不再进 metrics）。所有非溢出步逐位不变；bf16/无 scaler/假 scaler 路径逐位不变。
- 验证：
  - py_compile + import 冒烟通过。
  - 新增 `tests/training/test_training_engine_fix_te02_scaler_skip.py`（5 例）：自定义 autograd Function 注入 inf 梯度——真 scaler 单步/累积路径不抛错、参数不动、scale 下降、无 grad_norm、有 skipped 标志、global_step 递增、引擎未 poisoned、下一步正常参数更新（自愈闭环）；无 scaler 路径仍 fail-stop；鸭子类型 scaler、disabled 真 scaler 仍 fail-stop。全部通过。
  - 既有 `tests/training/test_single_device_engine.py` 全量通过（含用 `_RecordingGradScaler` 假 scaler 的用例，旧语义被显式保住）。
- 风险：中低。跳步检测依赖"scale 严格下降 ⇔ 跳步"这一 GradScaler 公开不变量（growth 只在连续 growth_interval 个正常步后发生，不会与 backoff 同步出现）；fp16 溢出从崩溃改为跳步是行为变化本身即修复意图。

### [TE-05] P2 checkpoint 无保留策略 + 孤儿 staging 累积 → keep-last-N（显式开启）+ 孤儿清理（默认开启）

- 改动：`worldfoundry/training/checkpoint/checkpointer.py`
  - **keep-last-N（默认关闭=现状不删）**：构造参数 `keep_last: int | None = None` 或 env `WORLDFOUNDRY_TRAINING_CHECKPOINT_KEEP_LAST`（int≥1，非法值构造期 fail-fast）。清理在 `finalize_staged_checkpoint` 的 rank0 块内、新 checkpoint **完整落盘之后**执行（os.replace + root fsync + latest.json 更新之后、结尾 `_barrier()` 之前——barrier 保证所有 rank 等清理完成后才继续）。只删同时具备 `_SUCCESS`+manifest 的已提交 `step-{8,}` 目录（fullmatch 本 run 命名模式），按 step 数保留最新 N 个；刚提交的 checkpoint 无条件保留（防御 resume 自旧步后 root 里存在更高 step 的分叉残留时被 N 规则挤出）；未提交/不匹配目录永不触碰；rmtree 失败仅告警不中断训练。
  - **孤儿 staging 清理（默认开启）**：`.step-{8,}.<32hex>.staging` fullmatch 目录在**本实例首次 save() 前**由 rank0 删除 + barrier 配套（save 是集合调用，barrier 对称）。刻意不在 `__init__` 清理：load-only checkpointer（如 `sessions/single_device.py:358` 的 resume 加载）可能指向另一个仍在运行的 run 的 root，init 清理会误删其在写 staging；而 save 的目标 root 归本 run 独占（并发写同 root 本就不受支持，final 提交会 FileExistsError），此时的 staging 必为崩溃残留。关闭开关：构造参数 `clean_orphaned_staging=False` 或 env `WORLDFOUNDRY_TRAINING_CHECKPOINT_CLEAN_STAGING=0`。
  - 引入模块级 `logging.getLogger(__name__)`：删除类操作必须留痕（孤儿删除记 warning——指示此前有崩溃；保留清理记 info）。training/ 此前零 logging 属现状而非硬约定（core/cli/evaluation 共 97 文件用同一模式），评审报告 TE-14 亦明确要求补日志。
- 数值行为：无变化（纯文件生命周期管理，默认行为=现状）。
- 验证：
  - py_compile + import 冒烟通过。
  - 新增 `tests/training/test_training_engine_fix_te05_checkpoint_retention.py`（10 例，真实 CPU DCP 同步/异步 save，鸭子类型 engine/loader 绕开 torchdata）：默认不删；keep_last=2 连存 4 个只留最新 2 且不碰未提交/不匹配目录、幸存者 inspect/load 完好；异步 finalize 后同样生效且旧 checkpoint 在新 checkpoint 提交前存活；"刚提交步号低于历史残留"时不删刚提交项；env 开启生效；keep_last/env 非法值 fail-fast；孤儿 staging 首存前删除（构造本身不删）、非匹配目录/文件不动；两种关闭开关 + env 非法值 fail-fast。全部通过。
  - 既有 `tests/training/test_rcm_checkpoint.py`（真实走 save/finalize 路径）通过；`test_training_checkpoint.py` 因缺 torchdata 跳过（环境限制，见页首）。
- 风险：低。默认路径行为不变；开启 keep_last 后按名字 fullmatch+已提交双重过滤，rank0-only+barrier 与该文件既有协调模式一致。已知取舍：开启 keep_last 后被删除的旧 step 无法再被 `--resume` 指名加载（保留策略的固有语义，属用户显式 opt-in）。

### [TE-08] P2 PowerEMA 按名静默跳过不匹配参数 → 先严格匹配后更新

- 改动：`worldfoundry/training/ema.py:75-117`
  - 新增 `_tracked_parameters(model, action)`：以构造期捕获的 `named_parameters()` 名字集为准，若任一 shadow 找不到同名活参数（模块被容器包裹/PEFT 注入/FSDP1 扁平化导致名字加前缀）→ `RuntimeError` 列出缺失数与样例名。**验证先于任何变更**：`forward`/`copy_to` 都先完成全量匹配再更新/导出，失败时 shadow 与 `num_updates` 保持原状（无半更新状态）。
  - 修复前语义：不匹配参数被 `continue` 静默跳过——EMA 悄悄空转，导出的"EMA 权重"停留在初始化附近，错误只会在最终评估时以精度形式暴露。
- 数值行为：匹配完好的现役路径（cosmos 系：EMA 在 fully_shard/PEFT 之后、以同一 `adapter.trainable_module` 构造并调用，FSDP2 原地改造不改参数名，`video_flow.py:294,391` 已核实）逐位不变；仅在"本来就静默失效"的错误配置上从无声空转变为 fail-fast。`store/restore` 按位置索引工作，不受影响；`_shadow_names` 为构造期快照、resume 按相同模型重建，checkpoint 兼容性不变。
- 验证：
  - py_compile + import 冒烟通过。
  - 新增 `tests/training/test_training_engine_fix_te08_power_ema_matching.py`（4 例）：名字匹配时更新/导出数值正确（iteration 0 beta=0 shadow==param）；构造期冻结参数不入 shadow、后续 forward 不误报；`nn.Sequential` 重包裹后 forward/copy_to 报错且 `num_updates` 未动（验证先于变更）；参数消失场景报 "1 of 2 tracked"。全部通过。
  - 既有 `test_cosmos_ema_training.py` 因缺 peft 跳过（环境限制）。
- 风险：低。报错仅出现在参数集/名字集与构造期不一致的场景，该场景旧行为是静默错误。

### [TE-10] P2 `ema_update` 默认 "microbatch" 致梯度累积下 EMA 计数膨胀 → 默认改 "optimizer-step"，lvdm 显式保留

- 改动：`worldfoundry/training/engine/video_flow.py`（两个共享 builder 的默认值 `"microbatch"` → `"optimizer-step"`，docstring 写明两种模式语义）；`worldfoundry/training/engine/lvdm/sft.py`（两个 builder 显式传 `ema_update="microbatch"` + 注释：作者对齐 Lightning `on_train_batch_end` 语义，既有 lvdm EMA 测试钉死该行为）。
  - 修复前：默认 microbatch 下梯度累积 N 微批 → EMA 每个 optimizer step 被调用 N 次，其中 N-1 次吸收未变参数，且 `num_updates` 膨胀 N 倍，扭曲 PowerEMA/LitEma 的按计数 decay 调度。
- **数值行为变化（显式标注）**：只影响"使用共享 builder + 未显式传 `ema_update` + 配置了 EMA + accumulation>1"的组合。核查现役调用方：cosmos（`cosmos/sft.py`）已显式传 `"optimizer-step"`——不变；lvdm（`lvdm/sft.py`）原依赖默认值 microbatch——本次显式钉住 microbatch，行为逐位不变；无其他调用方（rg 确认）。即**本次无任何现役配置的数值发生变化**，仅未来新家族的默认值语义更正确（shadows 每 optimizer step 恰好吸收一次，decay 计数 = 优化步数）。
- 验证：
  - py_compile + `import worldfoundry.training.engine.video_flow, worldfoundry.training.engine.lvdm.sft` 冒烟通过。
  - 新增 `tests/training/test_training_engine_fix_te10_ema_update_default.py`（3 例）：钉住两个 builder 的新默认值为 `"optimizer-step"`；计数 EMA 在 accumulation=2 下 optimizer-step 模式恰更新 1 次/步、microbatch 模式恰更新 2 次/步（引擎回调层验证，绕开 torchdata）。全部通过。
  - 既有 `test_lvdm_ema_training.py` 因缺 torchdata 失败于 `loader.py:108`（环境限制，修复前同样失败，见页首）。
- 风险：低。现役家族全部显式钉住；默认值变化只作用于未来代码。

### [TE-11] P1 wan/sana SFT 会话忽略 `recipe.scheduler` → 接线 build_lr_scheduler

- 改动：`worldfoundry/training/engine/wan/sft.py`、`worldfoundry/training/engine/sana/sft.py`（各 2 个 builder，共 4 处）
  - 修复前：配置了 warmup/cosine 的 recipe 被静默忽略，恒按常数 LR 训练，无任何告警；且 scheduler 状态不进 checkpoint。
  - 修复后（照抄 `video_flow.py` 的既有正确模式）：optimizer 构造后 `lr_scheduler = build_lr_scheduler(optimizer, recipe.scheduler)`；engine 侧 `optimizer_step_end=None if lr_scheduler is None else lr_scheduler.step`（每 optimizer step 后步进，与累积边界对齐，见报告 TE-04 正向确认）；session 侧传 `lr_scheduler=lr_scheduler` 使其进入 `TrainingState` 的 optional_state（checkpoint/resume 完整）。
- **数值行为变化（显式标注）**：仅"recipe 配置了 scheduler 的 wan/sana 会话"——LR 从错误的常数变为按配置调度（修复意图本身）。未配置 scheduler 的 recipe（`recipe.scheduler=None` → `build_lr_scheduler` 返回 None → `optimizer_step_end=None`）逐位不变。
- 验证：
  - py_compile + import 冒烟通过（wan/sana 两模块）。
  - 新增 `tests/training/test_training_engine_fix_te11_wan_sana_scheduler.py`（2 例）：wan 单卡会话配 linear scheduler 跑 2 步后 optimizer LR 精确等于调度值、不配 scheduler LR 恒为 base。sana builder 与 wan 结构同构（同一模式同一改法），由 import 冒烟 + py_compile 覆盖。通过（测试内 monkeypatch loader 工厂绕开 torchdata、tuning.mode="full" 绕开 peft，均为环境限制的最小替身，不触及被测的 scheduler 链路）。
  - 既有 `test_wan_session.py` 因缺 peft 跳过（环境限制）。
- 风险：低。接线模式与 video_flow/cosmos 现役路径完全一致；optional_state presence 校验保证新旧 checkpoint 的 scheduler 有无不匹配时 fail-fast 而非静默。

### [TE-14.1] P3 ray_runtime shutdown 静默吞 actor close 异常 → 保留 best-effort 语义 + warning 日志

- 改动：`worldfoundry/training/distributed/ray_runtime.py:497-507`
  - `ray.get(close_refs)` 的 `except Exception: pass` → `logger.warning(..., exc_info=True)` 后继续 kill/清理。shutdown 仍 best-effort（不因 close 失败中断资源回收），但 rollout 角色 close 失败（可能意味着权重同步/状态丢失）不再无痕。
- 数值行为：无变化。
- 验证：py_compile + `import worldfoundry.training.distributed.ray_runtime` 冒烟通过；既有 `test_ray_training_runtime.py` 因缺 ray 跳过（环境限制）。shutdown 路径无 ray 不可执行，日志分支属直读改动。
- 风险：极低。

### [TE-16.2/16.3] P3 复现链路两处文档缺口（低风险顺手项）

- 改动：
  - `worldfoundry/training/checkpoint/state.py`：`TrainingProgress.gradient_accumulation_phase` 加注释——恒为 0、字段保留为 schema 双保险（`__post_init__` 拒绝损坏/手改 checkpoint 的非 0 值），回应"死字段需注释说明或删除"（选注释：删除会破坏既有 checkpoint 的 progress schema 集合相等校验）。
  - `worldfoundry/training/engine/sessions/single_device.py` `run()` docstring：写明 `max_steps` 是"本次调用再跑的步数"而非全局训练总步数，resume 后在恢复的 global step 之上追加（manifest 记录 `initial_global_step`）。
- 数值行为：无变化（纯注释/docstring）。
- 验证：py_compile + import 冒烟通过。
- 风险：无。

## Deferred（含原因与方案）

### [TE-12] P2 wan/sana 手工复制 video_flow 通用 builder（约 2×300 行重复）

- 按任务边界"结构性重构：写方案，不执行"显式 deferred。
- 方案：wan/sana 的 4 个 builder 收敛到 `video_flow.build_cached_video_flow_{single_device,fsdp2}_session`，家族差异（attention 后端 env、caption dropout、cache 契约校验、tuning factory）经既有的 `tuning_factory`/`validate_cache_contract`/`ema_factory` 注入点表达；先为两家族补齐"行为快照"测试（本次 TE-11 的 scheduler 测试即第一块拼图），再做等价替换。本次 TE-11 已消除两处最危险的行为漂移（scheduler 接线），降低了重构前的风险敞口。

### [TE-03] P3 每步 cuda.synchronize + 计时 all_reduce + 逐行 fsync

- 原因：纯吞吐问题，非正确性；改为按 `log_every_steps` 周期同步/落盘会改变 metrics.jsonl 的行粒度与容灾语义（宕机丢最近 N 步指标），需要 owner 对可观测性/持久性取舍拍板，不属于"低风险且清晰"。
- 方案：计时同步与 loss host 同步按 log 周期做；`MetricWriter.write` 的 `os.fsync` 改为周期性或仅 checkpoint 前；保持 rank 间量纲一致（all_reduce 只在记录步执行）。

### [TE-06]（=TE-16.1）P3 NumPy 全局 RNG 不入 checkpoint

- 原因：`state.py` 的 runtime schema 用**集合相等**校验字段（`set(local_runtime) != runtime_expected` 即拒绝），加 `numpy_random_state` 字段会让**所有既有 checkpoint 无法 resume**。需要 schema 版本化或双向兼容读取，正确性收益目前为零（主训练链路无 np.random 消费，评审确认"防御性缺口"），破坏面却是全量存量 checkpoint——不划算，deferred。
- 方案：待下次不得不动 runtime schema 时（真正的破坏性变更窗口）一并加入，或引入 `schema` 字段版本协商后向后兼容地可选读取。

### [TE-07] P3 session 异常路径 finish_pending_checkpoint 内含 barrier 的挂死风险

- 原因：正确修法需要区分"全 rank 异常"与"部分 rank 异常"，而这本身又需要集合通信（同样可能挂死）；python ProcessGroup barrier 无逐调用超时参数，"带超时的 barrier"需要额外线程或 monitored_barrier（gloo-only）。任何轻率改动都可能把"异常时尽力保存 checkpoint"改坏。本次 TE-01 已把最常见的部分-rank 异常源（数据相关的非有限 loss）改为全 rank 一致 raise，实际挂死窗口已显著收窄。综合评估 deferred。
- 方案：异常处理器中仅本 rank `future.result(timeout=...)` 等待落盘线程、不做 finalize（不进入含 barrier 的提交路径），把 staging 留给下次启动的孤儿清理（本次 TE-05 已默认启用）；正常路径 finalize 语义不变。该方案丢弃的是"异常时那份未提交 checkpoint"，换取确定性快速失败——需 owner 确认取舍。

### [TE-09] P3 PowerEMA 每步 2-3 次 `.item()` host 同步

- 原因：把 `num_updates`/`iteration_shift`/`exponent` 从 GPU buffer 改为 python int 会改变 `state_dict` 载荷形态（buffer 序列化 → 需自定义 state_dict/load_state_dict），破坏既有 EMA checkpoint 兼容性；属性能清理且量级小（评审原文"量级不大，属清理项"）。
- 方案：保留 buffer 作为持久化载体，运行期用 python int 镜像计数、仅在 state_dict/load_state_dict 时同步 buffer；或等 TE-06 的 schema 版本化窗口一并处理。

### [TE-13] P3 训练/推理三套 FSDP 与两套 DCP 并存、core 版缺 inference-only 标注

- 原因：评审建议的落点（给 `core/distributed/fsdp2_sharding.py`、`core/distributed/block_fsdp.py` 加 docstring 标注 + 长期收敛）在 `worldfoundry/core/`，超出本任务允许改动范围（只许改 training 子树）。
- 方案：由 core owner 在两个 `shard_model` docstring 标注"inference/vendored-only，训练一律用 `training/distributed.apply_fsdp2`（其 fallback 分支静默退化单卡，用于训练是灾难）"；长期将 vendored 推理路径收敛到审计版。

### [TE-14.2/14.3/14.4] P3 weight_sync 串行 gather、anyflow DDP find_unused_parameters/no_sync、video_rollout 私有导入

- 原因：14.2（bucket 化重叠 gather）是性能优化，需在真实多卡 RL 同步负载下验证，无法在本机（无 ray、单 GPU）验证正确性；14.3 评审确认"数学仍正确，多付 N-1 次 allreduce"，纯性能，且 `find_unused_parameters` 的关闭需要逐 role 验证计算图无条件分支；14.4 把 `_wan22_checkpoints` 转公有 API 涉及 `video_policy` 模块导出面的命名决策，属外观整理。均 deferred。
- 方案：14.2 按 `build_weight_buckets` 既有分桶做流水线 gather+发送；14.3 在 anyflow roles 上试关 `find_unused_parameters` 并在累积期加 `no_sync()`，用既有 anyflow 集成测试回归；14.4 在 video_policy 导出 `wan22_checkpoints()` 公有函数后替换导入。

### 无需行动（评审正向确认项）

- TE-04（accumulation 边界与 scheduler 回调时序正确）、TE-15（数据管线质量高）、TE-06/07/14/16 的正向部分：确认无缺陷，不动。

## 验证汇总

- 新增测试：6 文件 25 例全部通过（合并运行 `25 passed in 41.81s`）：
  - `test_training_engine_fix_te01_collective_loss_check.py`（1 例，gloo 双进程）
  - `test_training_engine_fix_te02_scaler_skip.py`（5 例）
  - `test_training_engine_fix_te05_checkpoint_retention.py`（10 例，真实 CPU DCP 同步+异步 save）
  - `test_training_engine_fix_te08_power_ema_matching.py`（4 例）
  - `test_training_engine_fix_te10_ema_update_default.py`（3 例）
  - `test_training_engine_fix_te11_wan_sana_scheduler.py`（2 例）
- 全部改动文件 py_compile + import 冒烟通过：`engine/single_device.py`、`engine/fsdp.py`、`engine/wan/sft.py`、`engine/sana/sft.py`、`engine/video_flow.py`、`engine/lvdm/sft.py`、`ema.py`、`checkpoint/checkpointer.py`、`checkpoint/state.py`、`distributed/ray_runtime.py`、`engine/sessions/single_device.py`。
- 相关既有 CPU 测试（时间盒内批量运行）：`test_single_device_engine.py`、`test_training_optimization.py`、`test_rcm_checkpoint.py`、`test_video_flow_session.py`（可运行部分）等 **30 通过**；5 skip（peft/ray 缺失）；7 fail 全部定位为 `data/loader.py:106-108` 的 torchdata 缺失（`test_video_flow_session.py` 1 例、`test_training_session.py` 3 例、`test_lvdm_ema_training.py` 3 例），错误发生在 loader 构建期、先于任何本次改动的代码路径，修复前同样失败，属既有环境限制而非回归。
