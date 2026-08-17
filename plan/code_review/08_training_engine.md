# training 引擎层评审（engine/distributed/checkpoint/data/optimizers）

> 状态：已完成（评审员：infra 代码评审 agent，日期：2026-08-14）

## 评审范围与方法

- 范围：
  - `worldfoundry/training/engine/`（含 sessions/、wan/、sana/、cosmos/、ltx/、hunyuan_video/、wan22/、lvdm/、dynamicrafter/、anyflow/ 子目录，约 15.3k 行）
  - `worldfoundry/training/distributed/`（fsdp/parallel/ray_runtime/rollout_runtime/weight_sync/flow_rollout，约 2.3k 行）
  - `worldfoundry/training/checkpoint/`（checkpointer/state/staging/stateful/artifacts/errors，约 1.1k 行）
  - `worldfoundry/training/data/`（loader/sampler/dataset/video_dataset/video_cache/manifest 等，约 5.9k 行）
  - `worldfoundry/training/optimization.py`、`optimizers/came.py`、`ema.py`、`state_comparison.py`、`__init__.py`
- 方法：核心训练循环（engine/single_device.py、engine/sessions/single_device.py、engine/fsdp.py、video_policy.py、video_flow.py）、分布式与 checkpoint 全部精读；data 管线精读 loader/sampler/video_dataset/video_cache；engine 各模型子目录抽样精读（wan/sft、sana/sft、cosmos/dmd2 等）验证共性问题。
- 所有发现均给出 `路径:行号` 与代码摘录证据。

## 发现（按主题分组）

### 主题 A：训练循环与混合精度

#### [TE-01] P1 FSDP2 下单 rank 非有限 loss 直接抛异常，其余 rank 卡死在集合通信
- 位置：`worldfoundry/training/engine/fsdp.py:152-153`（对照 `engine/single_device.py:183-184`）
- 证据：

```152:161:worldfoundry/training/engine/fsdp.py
                    if not bool(torch.isfinite(result.loss.detach()).all()):
                        raise FloatingPointError("non-finite FSDP2 training loss")
                    reported_denominator = result.metrics.get("loss_denominator")
                    ...
                    gradient_weight = denominator / global_denominator * self.data_parallel_size
                    self._backward(result.loss * gradient_weight)
```

- 问题：loss 的有限性检查是**每 rank 本地判定**（数据相关，各 rank 天然可能不一致）。若仅 rank k 出现 NaN/inf，rank k 抛 `FloatingPointError` 跳过 backward，而其余 rank 继续进入 `_backward`，其中 FSDP2 的 reduce-scatter 集合通信会等待 rank k 参与——rank k 已经离开训练循环，结果是全作业挂死直到 NCCL timeout（默认 30 分钟，`parallel.py:176` 默认 1800s）。分母检查（`fsdp.py:126-128`）先做了 all_reduce 所以是一致的，唯独 loss 检查没有做"any-rank 非有限"聚合。
- 影响：大规模训练中一条坏样本/数值尖刺即可让整个作业无诊断信息地僵死半小时后被 watchdog 杀掉，浪费全部 GPU 时间且难以定位。
- 建议：将有限性判定改为集合一致决策（如把 `isfinite` 标志放进已有的 denominator all_reduce，或 all_reduce(MIN) 一个 finite 标志），全体 rank 一致地跳过/报错；或去掉本地 raise，依赖 clip 阶段的全局 grad-norm 检查（`core/gradient.py:66-67` 的 `full_tensor()` 已是全局一致值）。

#### [TE-02] P2 fp16 GradScaler 的动态降 scale 机制被 clip 的 error_if_nonfinite 短路，溢出即永久性训练中止
- 位置：`worldfoundry/training/engine/single_device.py:186-194`、`engine/fsdp.py:197-204`
- 证据：

```186:194:worldfoundry/training/engine/single_device.py
            self._backward(result.loss)
            self._unscale_gradients()
            grad_norm = clip_grad_norm_(
                self.parameters,
                float("inf") if self.max_grad_norm is None else self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self._mark_optimizer_step_started()
            self._step_optimizer()
```

- 问题：fp16+GradScaler 的标准流程是：溢出时 `scaler.step` 跳过该步、`scaler.update()` 降低 scale，训练自动恢复。这里在 `unscale_` 之后、`scaler.step` 之前先调用 `clip_grad_norm_(error_if_nonfinite=True)`——梯度一旦溢出（fp16 下属常规事件）就直接抛 `RuntimeError`，`_abort_training_step` 清梯度后向上抛；`scaler.update()` 永远不会执行，scale 不会下调。调用方若重试同一批次将无限重复溢出。GradScaler 实际上退化为纯 scale 乘除，丧失了动态 loss scaling 的自愈能力，fp16 路径形同"一次溢出就死"。
- 影响：fp16 训练路径在实际负载下几乎不可用（bf16 不受影响）。结合 TE-01，FSDP2+fp16 下还会变成集合通信挂死。
- 建议：溢出时走 skip-step 路径：检测到非有限梯度→跳过 `optimizer.step`、执行 `scaler.update()`、返回带 `skipped` 标志的结果；只有连续 N 次溢出才升级为致命错误。若刻意选择 fail-stop 语义，应在文档/构造函数中禁用 fp16 或明示该限制。

#### [TE-03] P3 每步强制 `cuda.synchronize` + 计时 all_reduce + fsync，吞吐敏感场景开销可观
- 位置：`worldfoundry/training/engine/sessions/single_device.py:429-432`、`sessions/single_device.py:145-150`、`sessions/io.py:60-63`
- 证据：

```429:432:worldfoundry/training/engine/sessions/single_device.py
                if self.engine.device.type == "cuda":
                    torch.cuda.synchronize(self.engine.device)
                step_seconds = self._maximum_across_ranks(time.perf_counter() - step_started)
                loss = float(result.loss.detach())
```

```60:63:worldfoundry/training/engine/sessions/io.py
    def write(self, value: Mapping[str, object]) -> None:
        self._handle.write(canonical_json(json_value(value)) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
```

- 问题：每个 optimizer step 做（1）全设备 `cuda.synchronize`，（2）float64 all_reduce 求 step 时间最大值，（3）`float(result.loss)` host 同步，（4）rank0 上 metrics.jsonl 逐行 `os.fsync`（本仓库跑在 CPFS 网络盘上，fsync 延迟不低）。引擎层已刻意用 tensor 保存 metrics 避免同步（`single_device.py:201-203` 注释可见意图），却在 session 层每步全部同步回来。
- 影响：小 step（LoRA、小分辨率）场景下 CPU-GPU 流水线被打断，吞吐下降；网络盘 fsync 在每步几毫秒到几十毫秒不等。
- 建议：计时同步与 metrics 落盘改为每 N 步（如 log_every_steps）；fsync 至多按周期或仅 checkpoint 前执行。

#### [TE-04] P3 accumulation 边界设计正确、lr scheduler 通过 optimizer_step_end 回调步进（正向确认）
- 位置：`worldfoundry/training/engine/single_device.py:279`、`engine/fsdp.py:139-143,160-161`
- 证据：

```139:143:worldfoundry/training/engine/fsdp.py
                for index, (prepared, denominator) in enumerate(zip(prepared_batches, denominators)):
                    final_microbatch = index + 1 == len(prepared_batches)
                    if accumulating_without_sync:
                        root.set_requires_gradient_sync(final_microbatch)
                        root.set_reshard_after_backward(final_microbatch)
```

- 说明（非缺陷）：梯度累积按 token 加权（`loss * denominator/total`），FSDP2 下乘 `data_parallel_size` 校正 mean-reduce，数学正确；非最终 microbatch 关闭梯度同步与 reshard，符合 FSDP2 最佳实践；grad clip 在 unscale 之后、step 之前，顺序正确。scheduler/EMA 经 `optimizer_step_end` 每 optimizer step 恰好步进一次，accumulation 中间不会误步进。异常后引擎"poisoned"防半步状态复用（`single_device.py:104-139`），设计良好。

### 主题 B：checkpoint 与 resume

#### [TE-05] P2 checkpoint 无保留/清理策略，全量 DCP 检查点无限累积
- 位置：`worldfoundry/training/checkpoint/checkpointer.py:109-156`；配置面 `worldfoundry/training/recipes/spec.py:388-403`
- 证据：

```388:395:worldfoundry/training/recipes/spec.py
    save_every_steps: int = 0
    ...
            "save_every_steps",
            _positive_int(self.save_every_steps, field_name="checkpoint.save_every_steps", allow_zero=True),
```

- 问题：`TrainingCheckpointer` 只会新增 `step-XXXXXXXX` 目录（`_paths` 若已存在直接报错），没有任何 keep-last-N / keep-every-M 的裁剪逻辑，recipe schema 中也没有 retention 字段。每个检查点包含全量模型+优化器状态（AdamW 下约 3-4 倍模型体积，多 optimizer 的 post-training 更大）。另外崩溃残留的 `.step-*.{token}.staging` 目录也无人清理（`checkpointer.py:104-106` 只检查不删除）。
- 影响：长训练按 `save_every_steps` 周期落盘会线性吃满磁盘/配额（14B 模型 + AdamW 每个检查点 >100GB），在共享 CPFS 上是现实的运维风险。
- 建议：加 `checkpoint.keep_last_n`（rank0 在 commit 成功后删除旧目录，删除前校验 latest.json 指向新目录）；启动时清理孤儿 staging 目录。

#### [TE-06] P3 resume 完整性覆盖全面（正向确认），但未保存 NumPy RNG
- 位置：`worldfoundry/training/checkpoint/state.py:233-245`
- 证据：

```236:245:worldfoundry/training/checkpoint/state.py
        return {
            "schema": TRAINING_RUNTIME_STATE_SCHEMA,
            "engine": engine_state,
            "progress": self.progress.state_dict(),
            "dataloader": serialized_dataloader,
            "torch_cpu_rng_state": torch.get_rng_state().clone(),
            "torch_cuda_rng_states": cuda_states,
            "objective_generator_state": self.objective_generator.get_state().clone(),
            "python_random_state": random.getstate(),
        }
```

- 说明：model/optimizer(多个)/engine step/dataloader/lr_scheduler/EMA/grad_scaler/torch CPU+CUDA RNG/objective generator/python random 全部按 rank 保存，world_size、identity（recipe+data+环境+并行拓扑）严格校验，`gradient_accumulation_phase != 0` 拒绝保存（`state.py:84-85`），这是我见过较完整的 exact-resume 实现。唯一缺口：`numpy.random` 全局状态未入 checkpoint——若任何数据增广/objective 用 np.random（当前主链路未见使用，属防御性缺口），resume 后将不可复现。建议顺手补一项或在文档声明"np.random 禁用"。

#### [TE-07] P3 原子提交与 rank 协调正确（正向确认）；异常路径上的 barrier 有挂死风险
- 位置：`worldfoundry/training/checkpoint/checkpointer.py:179-224`；`engine/sessions/single_device.py:464-470`
- 证据：

```210:214:worldfoundry/training/checkpoint/checkpointer.py
            sync_directory(staging_path)
            os.replace(staging_path, final_path)
            sync_directory(self.root)
```

```464:470:worldfoundry/training/engine/sessions/single_device.py
        except Exception as error:
            checkpoint_error: Exception | None = None
            if pending_checkpoint is not None:
                try:
                    finish_pending_checkpoint()
```

- 说明：staging 目录写完 → 写 manifest+`_SUCCESS` → fsync 目录 → `os.replace` 原子改名 → 更新 `latest.json`，前后各一个 barrier，rank0-only 写文件配套 barrier 齐全，写一半崩溃只会留下 staging 目录、不会产生半个"已提交"检查点；`inspect` 校验 `_SUCCESS`+manifest+逐文件尺寸。设计正确。风险点：session 异常处理器中 `finish_pending_checkpoint()` → `finalize_staged_checkpoint` 内部 `_barrier()`（checkpointer.py:179）——若异常只发生在部分 rank（最常见的 OOM/坏样本场景），其余 rank 不会进入该 handler，此 barrier 将挂到 NCCL 超时，把"快速失败"变成"半小时僵死"。建议异常路径不做集合操作，只在本 rank 尽力 `future.result()` 后放弃 finalize，或使用带超时的 barrier。

### 主题 C：EMA

#### [TE-08] P2 PowerEMA 按参数名静默跳过不匹配参数，包装顺序变化会导致 EMA 静默失效
- 位置：`worldfoundry/training/ema.py:81-87`（`copy_to` 同样模式 :93-99）
- 证据：

```81:87:worldfoundry/training/ema.py
        for name, parameter in model.named_parameters():
            shadow_name = self._shadow_names.get(name)
            if shadow_name is None:
                continue
            shadow = get_local_tensor_if_dtensor(getattr(self, shadow_name))
            source = get_local_tensor_if_dtensor(parameter).detach()
            shadow.mul_(beta).add_(source, alpha=1.0 - beta)
```

- 问题：`_shadow_names` 在构造时按 `named_parameters()` 名字建映射，`forward`/`copy_to` 对名字不匹配的参数**静默 continue**。若 EMA 在模型被再包装（如外面套一层容器 module、PEFT 注入、FSDP1 flatten）之后名字加了前缀，更新会部分或全部空转，且没有任何计数/报错。构造在 FSDP2 之后时 shadow 以 DTensor 分片存储（`register_buffer(parameter.detach()...clone())` 保留 DTensor），`get_local_tensor_if_dtensor` 处理正确；但"名字必须逐字相同"这一前置条件完全靠调用方自觉。
- 影响：一旦发生（重构模块层级、换包装顺序），EMA 导出权重≈初始权重的错误只有在最终评测崩坏时才会被发现。
- 建议：`forward` 结束时断言"本次匹配的参数数 == shadow 数"，不匹配立即报错；或在构造时记录参数 id 映射而非名字。

#### [TE-10] P2 `ema_update` 默认 "microbatch"：梯度累积期间 EMA 反复吸收未更新的参数并使计数膨胀
- 位置：`worldfoundry/training/engine/video_flow.py:269,349`（默认值）、`video_flow.py:204-209`（回调挂载）、`engine/single_device.py:279-281`（触发点）；受影响调用方 `engine/lvdm/sft.py:123-135,151-164`（未显式覆盖）
- 证据：

```267:269:worldfoundry/training/engine/video_flow.py
    ema_factory: VideoEmaFactory | None = None,
    export_ema: bool = False,
    ema_update: VideoEmaUpdate = "microbatch",
```

```279:281:worldfoundry/training/engine/single_device.py
                self._backward(result.loss * (denominator / total_denominator))
                if index + 1 < len(prepared_batches) and self.train_batch_end is not None:
                    self.train_batch_end()
```

- 问题：`train_batch_end` 在每个中间 microbatch 的 backward 后触发，此时 `optimizer.step()` 尚未执行、参数完全没变；累积 N 个 microbatch 时 EMA 每个 optimizer step 被调用 N 次（N-1 次吸收旧参数 + 1 次吸收新参数）。对 LVDM 的 `LitEma(use_num_upates=True)` 与 `PowerEMA`（其 docstring 明确写 "Update shadows after one optimizer step"，`ema.py:77`）来说，`num_updates` 膨胀 N 倍会扭曲 decay 调度曲线，EMA 权重偏向陈旧参数。cosmos 显式传了 `ema_update="optimizer-step"`（`cosmos/sft.py:356,389`）逃过一劫；lvdm 用默认值。此默认值可能是为复刻 Lightning `on_train_batch_end` 语义，但作为共享 builder 的默认值对新家族是陷阱。
- 影响：`gradient_accumulation_steps > 1` 的 LVDM（及未来沿用默认值的家族）EMA 调度错误；accumulation=1 时无影响。
- 建议：默认改为 `"optimizer-step"`；若需 author-parity 保留 microbatch 模式，要求家族显式选择并在文档标注其语义。

#### [TE-09] P3 PowerEMA 每步两次 `.item()` host 同步
- 位置：`worldfoundry/training/ema.py:70,79`
- 证据：

```75:80:worldfoundry/training/ema.py
    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        """Update shadows after one optimizer step."""

        iteration = int(self.num_updates.item())
        beta = self.beta(iteration)
```

- 问题：`num_updates`/`iteration_shift` 存为 GPU buffer，每次 update 触发 `.item()` 同步（`beta()` 内又一次 `self.iteration_shift.item()`、`self.exponent.item()`）。计数器完全可以是 python int + `state_dict` 序列化。
- 影响：小步长训练每步 2-3 次额外 host-device 往返；量级不大，属清理项。

### 主题 D：调度器接线与家族级重复实现

#### [TE-11] P1 wan/sana SFT 会话完全忽略 `recipe.scheduler`——配置了调度器的 recipe 静默按常数 LR 训练
- 位置：`worldfoundry/training/engine/wan/sft.py:162-185`、`engine/sana/sft.py:221-236,380-397`；对照正确实现 `engine/video_flow.py:296-304`
- 证据（wan 的 session 构建全程无 scheduler，grep `scheduler` 在 wan/sft.py、sana/sft.py 均零命中）：

```170:177:worldfoundry/training/engine/wan/sft.py
    objective = _flow_objective(recipe, adapter)
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=recipe.optimizer.max_grad_norm,
        autocast_dtype=None if expected_dtype is torch.float32 else expected_dtype,
    )
```

```296:299:worldfoundry/training/engine/video_flow.py
    lr_scheduler = build_lr_scheduler(
        optimizer,
        recipe.scheduler if isinstance(recipe, TrainingRecipe) else None,
    )
```

- 问题：`TrainingRecipe.scheduler` 是公开可配字段（`recipes/spec.py:443`），`build_wan_*_session` / `build_sana_*_session` 既不调用 `build_lr_scheduler`，也不在 `_validate_recipe_for_wan`（`wan/sft.py:54-92`）里拒绝配置了 scheduler 的 recipe。用户在 wan2.1/sana recipe 里写了 warmup+cosine，实际训练全程恒定 LR，且 metrics.jsonl 里的 `learning_rates` 会如实显示常数——但没有任何报错或告警指出配置被丢弃。session 的 `_resume_identity` 还把 `recipe.get("scheduler")` 写进 resume identity（`sessions/single_device.py:195`），进一步暗示 scheduler 生效了。
- 影响：静默错误训练。warmup 缺失可能直接训崩（大模型 SFT 对 warmup 敏感），且用户从产物上难以发现原因。
- 建议：短期在 `_validate_recipe_for_wan`/sana 校验中 `if recipe.scheduler is not None: raise`；正解是并入 TE-12 的通用 builder，天然获得 scheduler 接线。

#### [TE-12] P2 wan/sana 手工复制 video_flow.py 通用 builder（约 2×300 行重复），行为漂移已经发生
- 位置：`worldfoundry/training/engine/wan/sft.py:135-307`、`engine/sana/sft.py:272-449`；通用实现 `engine/video_flow.py:258-434`
- 证据：ltx/cosmos/dynamicrafter/lvdm 都经 `build_cached_video_flow_*_session` 构建（`ltx/sft.py:256,284`、`cosmos/sft.py:346,378`、`dynamicrafter/sft.py:131,156`、`lvdm/sft.py:123,151`），而 wan/sana 直接手写 `SingleDeviceTrainingSession(...)` / `FSDP2TrainingSession(...)`，重复了 validate→tuning→optimizer→engine→loader→session 全流程和 `data_identity` 组装。
- 问题：同一逻辑三份实现（video_flow 通用版、wan 版、sana 版），已产生真实漂移：wan/sana 丢了 scheduler（TE-11）、丢了 `ema_factory`/`export_ema` 能力、`data_identity` 字段集不一致（wan 多 `token_sampler.sample_ids`，sana 无 token_sampler）。后续在通用版修 bug（如 TE-10 默认值）不会惠及 wan/sana。
- 影响：维护成本、修复遗漏、跨家族行为不一致。
- 建议：wan/sana 迁移到 `build_cached_video_flow_*_session`（wan 的专用 cache loader 可经 `consumed_data_options`/自定义 loader 钩子注入），删除手写路径。

#### [TE-13] P3 与 core 层的并行实现：训练 FSDP2/DCP 与 core/distributed、core/checkpoint 各自为政
- 位置：`worldfoundry/training/distributed/fsdp.py`（审计版 fully_shard 应用）vs `worldfoundry/core/distributed/fsdp2_sharding.py:25-40`、`core/distributed/block_fsdp.py:12-36`（vendored Wan 团队 shard_model）；`worldfoundry/training/checkpoint/`（训练 DCP）vs `worldfoundry/core/checkpoint/dcp.py`（"inference-only" DCP 加载）
- 证据：

```25:31:worldfoundry/core/distributed/fsdp2_sharding.py
def shard_model(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32):
    if fully_shard is None or MixedPrecisionPolicy is None:
        model.to(param_dtype)
        if torch.cuda.is_available():
            model.to(torch.device("cuda", torch.cuda.current_device()))
        return model
```

- 问题：仓库内至少三套 FSDP 包装实现（core 两套 vendored + training 一套原生）与两套 DCP 读写栈并存。training 版明显是治理后的正主（参数名审计、mesh 校验、precision island），core 版是推理/移植遗留。方向上无大问题，但没有任何文档声明"训练一律用 training/distributed，core 版仅限 vendored 推理路径"，新代码容易误引 core 版（其 fallback 分支在无 FSDP2 时静默退化为单卡 `.to(cuda)`，用于训练会是灾难）。
- 建议：在 core 两个 shard_model 上加 docstring/命名标注 inference-only；长期将 vendored 推理路径也收敛到 training/distributed 的审计版或 core 统一实现。

### 主题 E：分布式运行时（ray/weight_sync/rollout）

#### [TE-14] P3 分布式协调整体规范（正向确认），零散小problem
- 位置：`worldfoundry/training/distributed/parallel.py:169-229`、`distributed/fsdp.py:139-298`、`engine/artifacts.py:27-51`
- 说明（正向）：
  - FSDP2 应用顺序正确：先注入/冻结 adapter → `apply_fsdp2`（block 深度优先、root reshard=False）→ 之后才建 optimizer（`fsdp.py:149-154` docstring 明示），并断言参数名/可训练集合未变（`fsdp.py:273-278`）。
  - rank0-only 写文件均有配套协调：run 目录创建用"rank0 创建+broadcast 结果+barrier"（`artifacts.py:40-50`），导出用"全 rank 集合 gather + rank0 落盘 + broadcast 结果"，失败也会广播，不会半死锁。
  - `DistributedTrainingContext` 只在自己创建 process group 时销毁它（`parallel.py:226-229`）。
  - weight_sync 的 `materialize_weight_tensors` 明确文档 "每个 rank 都必须调用（full_tensor 是集合操作）"（`weight_sync.py:103-107`），按 sorted 名字迭代保证集合顺序一致。
- 小问题：
  1. `ray_runtime.py:494-497` shutdown 里 `except Exception: pass` 静默吞掉 actor close 失败（best-effort 清理可接受，但至少应留一行日志）。
  2. `weight_sync.py:118-123` 对全模型逐 tensor 顺序 `full_tensor()` + `.to("cpu")`，14B 级模型每次 revision 同步串行 gather，无 bucket 化重叠；RL 高频同步下是可测的开销。
  3. `engine/anyflow/roles.py:118-124` DDP 用 `find_unused_parameters=True`（每步全图扫描开销），且累积期间未见 `no_sync()`（数学仍正确，多付 N-1 次 allreduce）。
  4. `engine/video_rollout.py:54` 从 `video_policy` 导入私有符号 `_wan22_checkpoints`，模块边界不洁。

### 主题 F：数据管线

#### [TE-15] P3 数据管线整体质量高（正向确认）：确定性 shuffle、worker seed、fail-fast 损坏处理
- 位置：`worldfoundry/training/data/sampler.py:180-227`、`data/latent_token_sampler.py:130-172`、`data/loader.py:104-134`、`data/video_dataset.py:169-170,250-267`、`data/video_cache.py:427-442`
- 说明（正向）：
  - `DeterministicDistributedSampler` 用无依赖 splitmix64 洗牌，`__iter__` 耗尽后自动进入下一 epoch 并重新排列（`sampler.py:211-221`），从结构上消灭了"忘调 `set_epoch` 导致每个 epoch 同序"的经典 bug；resume 校验 `next_sample_id` 逐样本对齐。
  - `LatentTokenBatchSampler` 按 bucket 同质 + token 预算组 microbatch，rank 分配带 `(rank+position)%world_size` 轮转去相关（`latent_token_sampler.py:166-169`），pad/drop 尾部策略全局对齐——各 rank token 数差异由引擎的全局加权正确处理，闭环成立。
  - worker seed 按 `shuffle_seed + rank` 派生并喂给 StatefulDataLoader 的 generator（`loader.py:110-111`），worker 间自动加 worker_id 偏移。
  - 损坏样本 fail-fast：解码几何/帧数/fps 与 manifest 不符、像素含 NaN 一律 raise（`video_dataset.py:169-170,237-267`），没有静默跳过 → 无 epoch 长度漂移；cache 对象写入是 temp+fsync+`os.replace` 原子提交（`video_cache.py:428-440`），读取校验字节数与描述符。
  - 预取内存受 `prefetch_factor` 显式控制且必须搭配 num_workers（`loader.py:89-92`），无隐藏无界队列。
- 备注：fail-fast 策略的代价是单个坏样本会停整个作业（FSDP2 下叠加 TE-01 变成挂死）；对大规模爬取数据建议在 cache 预计算阶段过滤，而这正是本仓库的架构（训练只读已验证 cache），逻辑自洽。

### 主题 G：可复现性、死代码与其它

#### [TE-16] P3 复现链路完整（正向确认）+ 三处小缺口
- 位置：`worldfoundry/training/engine/sessions/single_device.py:169-209,363-368`、`checkpoint/state.py:66-135`
- 说明（正向）：resume identity 覆盖 recipe/model/tuning/optimizer/scheduler/runtime/distributed/data/seed 派生规则/并行拓扑/torch+CUDA 版本+算力（`sessions/single_device.py:176-209`），checkpoint 拒绝任何 identity 漂移；seed 派生 `base+rank`，fixed_corruption 模式支持逐步重置。
- 缺口：
  1. NumPy 全局 RNG 不在 checkpoint 内（见 TE-06）；`set_seed_everywhere` 初始化时会设 numpy（`core/utils/torch_utils.py:107-116`），resume 后不恢复。
  2. `TrainingProgress.gradient_accumulation_phase` 恒为 0：`record_step` 从不修改它，`__post_init__` 又拒绝非 0（`checkpoint/state.py:84-85,87-97`）——字段是死的，仅作双保险占位；建议注释说明或删除。
  3. `SingleDeviceTrainingSession.run(max_steps=...)` 在 resume 后语义是"再跑 max_steps 步"而非"总步数到 max_steps"（`sessions/single_device.py:405`，`initial_global_step` 仅记录在 manifest），与常见框架（总步数）相反，容易误用；建议文档明示或加 `total_steps` 别名。

## 汇总

### 其他核查结论（无独立发现）

- 导入卫生：`training/` 全树 grep 无任何 `worldfoundry.evaluation` / `worldfoundry.studio` 导入，分层干净。
- bare except：全树无 `except:`；`except BaseException` 共 9 处均为"标记引擎 poisoned 后 re-raise"模式（如 `engine/single_device.py:135,217,373`），不吞错。唯一静默吞错为 `ray_runtime.py:496-497`（已记入 TE-14）。

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0 | 0 | — |
| P1 | 2 | TE-01, TE-11 |
| P2 | 5 | TE-02, TE-05, TE-08, TE-10, TE-12 |
| P3 | 9 | TE-03, TE-04, TE-06, TE-07, TE-09, TE-13, TE-14, TE-15, TE-16 |

（P3 中 TE-04/06/07/14/15/16 为正向确认附带小缺口，非缺陷项。）

### Top 5

1. **[TE-01] P1** FSDP2 引擎对非有限 loss 做每 rank 本地 raise，单 rank NaN 会让其余 rank 挂死在 reduce-scatter 集合通信上直到 NCCL 超时——应改为 all-reduce 一致决策。
2. **[TE-11] P1** wan/sana 的 SFT 会话构建完全不接 `recipe.scheduler`，配置了 warmup/cosine 的 recipe 静默按常数 LR 训练，且无任何告警。
3. **[TE-10] P2** 共享 builder 的 `ema_update` 默认 `"microbatch"`：梯度累积下 EMA 每个 optimizer step 被调用 N 次（N-1 次吸收未变参数），`num_updates` 膨胀扭曲 decay 调度（lvdm 现役中招，cosmos 显式规避）。
4. **[TE-02] P2** `clip_grad_norm_(error_if_nonfinite=True)` 位于 `scaler.step/update` 之前，fp16 溢出直接抛错终止训练，GradScaler 的动态降 scale 自愈机制永远无法触发。
5. **[TE-05] P2** checkpoint 无 keep-last-N 保留策略也不清理孤儿 staging 目录，全量 DCP（模型+优化器）按周期无限累积，共享 CPFS 上磁盘爆炸是现实风险。
