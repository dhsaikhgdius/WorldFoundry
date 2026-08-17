# training 配方层评审（api/models/objectives/post_training/recipes/tuning/safety）

> 状态：已完成（评审人：infra 代码评审 agent；日期：2026-08-14；共 23 条发现：P1×2、P2×5、P3×16，其中 6 条为正面核验记录）

## 评审范围与方法

- 范围（约 415 个 py 文件，~7.5 万行）：
  - `worldfoundry/training/api/`（2 文件，346 行）：全部精读
  - `worldfoundry/training/objectives/`（3 文件，914 行）：全部精读
  - `worldfoundry/training/tuning/`（4 文件，1079 行）：全部精读
  - `worldfoundry/training/safety/`（2 文件，375 行）：全部精读
  - `worldfoundry/training/recipes/`（~7k 行）：spec.py / recipe.py / common.py / rollout.py 精读，algorithms/ 36 个配方精读 8 个代表 + rg 全量扫描
  - `worldfoundry/training/models/`（14 文件，5047 行）：精读 6-8 个代表 + rg 全量扫描
  - `worldfoundry/training/post_training/`（~62k 行）：rl/ 核心框架（contracts/run/batching/trajectory/rewards/transitions/rollout_strategies/objectives）与 shared/、rewards/、agentic/ 精读；rl/algorithms 精读 token_policy、flow_grpo、dance_grpo 等代表；distillation/（42k 行、24 个子包）精读 dmd、self_forcing、rcm 等代表 + rg 全量扫描
- 方法：先 Read 确认证据，每条发现附 `路径:行号` 与代码摘录；rg 做横向一致性扫描（bare except、越层 import、顶层重依赖、seed、复制粘贴漂移）。
- 严重度：P0=损坏/危险；P1=严重设计缺陷；P2=应修复；P3=改进建议。

## 总体印象

这一半区的整体工程质量显著高于常见研究代码库：api 契约无 torch 依赖、loss 全部走显式 numerator/denominator 分解（支持变长 token 的分布式精确加权）、RL 引擎有 poison 语义与 anchor 一致性校验、checkpoint/export 有 schema 级严格校验、安全过滤 fail-closed。主要问题集中在：ray 生命周期的失败路径清理、根 recipe 解析器的集中式 if/elif 漂移风险、以及大量结构相似子包（distillation 24 个、models 13 个）之间的复制粘贴模板。

## 发现（按主题分组）

### 主题 A：API 设计（training/api）

#### [TR-1] P3 api 契约整体设计良好，但 `TensorLike = Any` 放弃了静态类型保护

- 位置：`worldfoundry/training/api/contracts.py:16`
- 证据：

```16:17:worldfoundry/training/api/contracts.py
TensorLike: TypeAlias = Any
TensorTree: TypeAlias = TensorLike | Mapping[str, TensorLike]
```

- 问题：契约刻意不 import torch（docstring line 1-6 说明是为了让基础包无 torch 也能审查 recipe），代价是所有张量字段静态类型为 `Any`，mypy/pyright 无法捕获误传。运行时通过 `_shape()` duck-typing 校验兜底（contracts.py:29-39），批维、mask 广播、sample_weights 形状全部有显式检查，实际风险低。
- 影响：仅影响静态检查体验；运行时契约是完备的。
- 建议：可用 `typing.Protocol`（带 `shape` 属性的 `SupportsShape`）替代 `Any`，零运行时开销拿回部分静态检查。

#### [TR-2] P3 `TrainModelAdapter`/`TrainingObjective` 协议清晰、职责边界（"model owns packing, objective owns corruption/reduction"）明确——正面评价

- 位置：`worldfoundry/training/api/contracts.py:293-315`
- 证据：

```293:304:worldfoundry/training/api/contracts.py
@runtime_checkable
class TrainModelAdapter(Protocol):
    """Model-owned conditioning/packing seam; it does not own loss math."""

    prediction_type: str
    trainable_module: object
    lora_target_preset: str | None
    fsdp_block_classes: tuple[type, ...]
```

- 说明：`TrainingBatch → PreparedBatch → ObjectiveBatch → TrainStepResult` 四段式数据流 + 两个 Protocol 的接缝设计，与 engine 层（engine/ 内各家 sft.py 消费该契约）边界单向清晰。models/ 13 个适配器全部遵守（rg 验证 `prepare_batch`/`forward_train` 一致）。无发现越层反向依赖。

### 主题 B：objectives（loss 正确性 / 时间步采样）

#### [TR-3] P3 flow_matching 与 classic_diffusion 的 torch 导入策略不一致，lazy-import 设计被 `__init__` 击穿

- 位置：`worldfoundry/training/objectives/classic_diffusion.py:16`、`worldfoundry/training/objectives/flow_matching.py:19-24`、`worldfoundry/training/objectives/__init__.py:3-23`
- 证据：

```19:24:worldfoundry/training/objectives/flow_matching.py
def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("flow-matching training requires the 'train-core' extra (PyTorch)") from error
    return torch
```

```16:17:worldfoundry/training/objectives/classic_diffusion.py
import torch
from torch import Tensor
```

- 问题：flow_matching.py 精心做了 torch 延迟导入（配合 api 层"无 torch 可审查 recipe"的设计目标），但 classic_diffusion.py 顶层 `import torch`，且 `objectives/__init__.py` 同时 re-export 两者——任何 `import worldfoundry.training.objectives` 都会拉起 torch，flow_matching 的延迟导入变成纯粹的代码噪音。
- 影响：无 torch 环境下 `from worldfoundry.training.objectives import flow_shift_sigmas` 会失败，与 flow_matching 的设计意图矛盾。
- 建议：要么 classic_diffusion 也走 `_require_torch()`（并把 `__init__` 改成惰性 `__getattr__`，models/__init__.py:74-80 已有现成模式），要么删掉 flow_matching 里的延迟导入假设。

#### [TR-4] P3 loss 实现正确性核验通过（正面记录，含数值稳定性证据）

- 位置：`worldfoundry/training/objectives/flow_matching.py:166-223`、`worldfoundry/training/objectives/classic_diffusion.py:166-189`
- 证据（fp32 上采样 + 显式分子/分母 + 除零防护）：

```205:217:worldfoundry/training/objectives/flow_matching.py
    flat_squared = squared.reshape(batch_size, -1)
    flat_effective = effective.reshape(batch_size, -1)
    per_numerator = (flat_squared * flat_effective).sum(dim=1)
    per_denominator = flat_effective.sum(dim=1)
    denominator = per_denominator.sum()
    if not bool(denominator.detach() > 0):
        raise ValueError("flow-matching loss has no positive-weight elements")
    numerator = per_numerator.sum()
    per_sample = torch.where(
        per_denominator > 0,
        per_numerator / per_denominator.clamp_min(1.0e-12),
        torch.zeros_like(per_numerator),
    )
```

- 核验点：(1) reduction 是"权重和作分母"而非"rank 内均值再平均"，`loss_denominator()`/`prepared_loss_denominator()`（flow_matching.py:462-506）允许 engine 在 backward 前做全局分母 all-reduce——变长 token 分布式加权正确；(2) mask 校验非负 + 有限（`_expanded_mask`，flow_matching.py:148-163）；(3) 全零权重 fail-fast 而非 NaN；(4) `flow_shift_sigmas`（:72-82）分母 `1+(shift-1)*sigma` 在 shift>0、sigma∈[0,1] 下恒正。
- 时间步采样：uniform/logit-normal/waver 三种（flow_matching.py:319-331）。waver 采样公式 `1 - u - 1.29*(cos²(πu/2) - 1 + u)` 已验证单调映射 [0,1)→(0,1]，f'(u) = -2.29 + 2.026·sin(πu) < 0 恒成立，不会越界；离散模式下 `torch.floor(unit*steps).clamp_(0, steps-1)`（:340）对 unit=1.0 边界有防护。
- 一个需要适配器作者注意的语义点：连续模式下 `ObjectiveBatch.timesteps` 是未 shift 的 base sigma（:335），离散模式下是整数 index（:340）——语义随配置改变。当前 wan/sana/cosmos 适配器都只消费 `batch.sigmas`（如 `wan.py:465` `model_timesteps = sigmas * self.model_timestep_scale`），无实际 bug，但该字段语义值得在 docstring 标注。

#### [TR-5] P3 classic_diffusion 每个 step 重复做 `self.sigmas.to(device)` 传输

- 位置：`worldfoundry/training/objectives/classic_diffusion.py:269`
- 证据：

```265:270:worldfoundry/training/objectives/classic_diffusion.py
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=noisy,
            target=target,
            sigmas=self.sigmas.to(device=clean.device).gather(0, timesteps),
```

- 问题：`self.alphas/self.sigmas` 常驻 CPU（构造时未指定 device，:206-216），每次 `corrupt()` 调用（每 step）都触发 H2D copy；`extract_schedule`（:57-61）里 `values.to(device=timesteps.device)` 同样。
- 影响：每步几 KB 的同步拷贝，对吞吐影响可忽略但属于免费可拿的优化；schedule 表建议首次使用时按 device 缓存。

### 主题 C：models/（适配器）

#### [TR-6] P2 五份适配器重复同一组私有 helper，已出现签名漂移

- 位置：`worldfoundry/training/models/wan.py:27-63`、`sana.py:31-67`、`cosmos.py:37-67`、`hunyuan_video.py:70-88`、`ltx.py:44`
- 证据（rg 汇总）：

```text
models/cosmos.py:37:def _component_module(component: object, *names: str) -> nn.Module | None:
models/hunyuan_video.py:70:def _component_module(component: object | None, *names: str) -> nn.Module | None:
models/sana.py:31:def _component_module(component: object, *names: str) -> nn.Module | None:
models/wan.py:27:def _component_module(component: object, *names: str) -> nn.Module | None:
models/cosmos.py:63:def _merge(destination: ..., *, owner: str) -> None:
models/wan.py:54:def _merge_without_overwrite(destination: ..., *, source_name: str) -> None:
```

- 问题：`_component_module`/`_module_device_dtype`/`_freeze`/`_merge*` 在 ≥5 个文件中逐字复制，且已经开始漂移（hunyuan 版签名接受 `None`、cosmos 版改名 `_merge` 换参数名 `owner`）。这正是 recipes/models 体系"复制粘贴漂移"的教科书案例——将来修一个 bug（例如 `_module_device_dtype` 对全 buffer 模块的 dtype 推断）需要记得改五处。
- 影响：维护性；漂移后行为不一致难以发现。
- 建议：提一个 `models/_shared.py`（私有模块不进公共 API），五个文件各删 40 行。
- 同时记录正面项：适配器质量本身很高——wan.py 的几何校验（`_validate_pixel_geometry` :189-202，温压缩/patch 整除检查）、缓存 latent 与 metadata 几何交叉验证（:339-351）、context 形状+有限性检查（:250-260）、`build_cached_wan_train_adapter` 对多余组件 fail-fast（:519-522）都是防御到位的写法。

#### [TR-7] P3 LoRA preset 字符串契约分散：`tuning/peft.py` 仅认识 2 个 preset，其余 5 个 preset 的审计逻辑散在 engine 层

- 位置：`worldfoundry/training/tuning/peft.py:120-122`；对照 `models/ltx.py:84`、`models/cosmos.py:426,530`、`models/wan22.py:102`、`models/hunyuan_video.py:137`
- 证据：

```120:122:worldfoundry/training/tuning/peft.py
    normalized = str(preset).strip().lower().replace("_", "-")
    if normalized not in {SANA_ATTENTION, WAN_ATTENTION}:
        raise ValueError(f"unsupported LoRA target preset: {preset!r}")
```

```84:84:worldfoundry/training/models/ltx.py
    lora_target_preset = "ltx-attention"
```

- 问题：`audit_lora_targets` 只支持 `sana-attention`/`wan-attention`；`ltx-attention`、`cosmos-predict-attention-mlp`、`cosmos3-generation-attention`、`wan22` 双专家、hunyuan 动态 preset 都绕过它、由各自 engine 构造 `LoraTargetAudit` 后调 `apply_peft_lora_with_audit`（engine/ltx/sft.py:229 等）。功能上无 bug（audit 对象化的设计使然），但"哪个 preset 在哪审计"没有单一注册点：`load_peft_adapter(expected_preset=...)`（peft.py:438-482）作为公开恢复 API 只能校验 2 个 preset，给 ltx/cosmos 的 LoRA 工件传 `expected_preset` 会直接 ValueError。
- 影响：恢复/合并路径对 3 个模型族不可用（除非绕过校验参数），API 面与能力不对齐。
- 建议：提供 preset→audit 构造函数的注册表（模型族在导入时注册），`audit_lora_targets` 查表分发。

### 主题 D：tuning（LoRA/PEFT/全量导出）

#### [TR-8] P3 tuning 层实现质量高（正面记录）：审计先行、原子写、加载/合并双向校验

- 位置：`worldfoundry/training/tuning/peft.py`、`full_model.py`
- 证据（注入后逐项对账）：

```246:266:worldfoundry/training/tuning/peft.py
    targeted = tuple(str(name) for name in getattr(peft_model, "targeted_module_names", ()))
    ...
    unexpected = tuple(
        name for name in trainable if "lora_" not in name and not any(marker in f".{name}." for marker in allowed_saved)
    )
    if unexpected:
        raise RuntimeError(f"PEFT left unexpected base parameters trainable: {unexpected}")
    trainable_count = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    expected_count = audit.expected_trainable_parameters(int(rank))
    if not save_modules and trainable_count != expected_count:
        raise RuntimeError(f"PEFT trainable parameter count {trainable_count} does not match audited {expected_count}")
```

- 核验点：save 走 tmpdir + `os.replace` 原子提交（peft.py:321-342，full_model.py:185-241 还加了 fsync + 目录 sync）；merge 后残留 `lora_` 参数检查（peft.py:519-521）；full model 载入前 manifest/shape/dtype 全量对账（full_model.py:412-441）+ 符号链接拒绝（:259-261）。`merge_peft_adapter` 对 PEFT 0.20 的 config 兼容 hack（:495-518）有注释说明动机且异常路径恢复原 config。
- 唯一小问题：`load_peft_adapter` 在 `worldfoundry/` 内无任何调用方（rg 全库仅 tuning 自身与 `__init__` re-export），属于"只写不用"的公共 API，见 TR-7 的 preset 校验缺口——真正的推理侧合并走的是 `base_models/diffusion_model/models/denoisers/wan.py:126` 的 `audit_lora_targets + merge_peft_adapter` 路径。建议给该函数补集成测试或标注实验性。

### 主题 E：safety（失败关闭 vs 失败开放）

#### [TR-9] P2 安全过滤本体 fail-closed 且证据链完整，但强制点在数据预处理阶段，SFT 引擎不复核审计边车

- 位置：`worldfoundry/training/safety/shieldgemma.py:254-305`、`worldfoundry/training/data/prompt_audits.py:93-111`；接线点 rg 汇总：`training/data/{sana_precompute,wan/training_cache,ltx/training_cache,cosmos/training_cache,video_precompute,rollout_audits}.py`
- 证据（拒绝截断、拒绝非有限 logits、超限即抛）：

```267:281:worldfoundry/training/safety/shieldgemma.py
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[1]) > self.max_input_tokens:
            raise ValueError("ShieldGemma input exceeds max_input_tokens; filtering refuses to truncate content")
        ...
        selected = logits[:, -1, list(self.yes_no_tokens)].float()
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("ShieldGemma returned non-finite Yes/No logits")
```

- 核验点（好的一面）：checkpoint 按 revision+文件字节数 pin 死（:24-48）；`PromptSafetyAudit.from_mapping` 重放时重算派生字段并拒绝不一致（:145-189）；`UnsafeTrainingPromptError` 走异常而非返回值，无静默放行分支；审计边车 `select_for_manifest` 会校验 prompt 文本逐字一致 + manifest 的 `prompt_safe` 标志 + 模型 revision（prompt_audits.py:104-110）。
- 绕过路径（问题所在）：审计只在 cache/precompute 工具链强制（`engine/wan/cache.py`、`engine/sana/cache.py` 等会调 `select_for_manifest`）。直接把未审计的 manifest/pixel 数据喂给 SFT 引擎（`data.cache` 为空的原始视频路径）时，训练主循环没有任何"必须携带审计"的门槛——安全性完全依赖操作者走推荐工具链。ShieldGemma 本身也只做 prompt-only 审计（文件 docstring 就声明了范围），不覆盖图像/视频内容。
- 影响：许可证合规风险（SANA 600M 的 license 明确要求输入过滤）依赖流程纪律而非代码强制。
- 建议：在 engine 构建 dataloader 处对 `sample.safety.get("prompt_safe") is not True` 的样本 fail-closed（至少对 sana/wan 这两个有许可证要求的族），或在 recipe 中加 `data.require_prompt_audit` 默认 true。
- 修正（细读后收窄范围）：库层缓存链路两端其实都有强制——构建端 `data/wan/training_cache.py:63`、`data/cosmos/training_cache.py:146`、`data/ltx/training_cache.py:108` 都过 `validate_video_prompt_audits`（`video_precompute.py:106` 对 `prompt_safe is not True` 直接抛错），消费端 `engine/wan/cache.py:204`、`engine/sana/cache.py:99` 加载 cache 时还会复核 `sample.safety` + `provenance.safety_audit` 双标志。真正的缺口只剩：(a) `PromptAuditSet.select_for_manifest` 的调用点在 CLI 层（`cli/training_commands/handlers/cache.py:111`），绕开 CLI 直接用库函数构造 manifest 时 `safety` 字段可以手写伪造，库层只校验标志位不校验审计文件签名；(b) 审计仅覆盖 prompt 文本，不覆盖视频/图像内容本身。综合评级维持 P2 但性质是"信任边界文档化不足"而非"没有强制"。

### 主题 F：recipe 体系（配置组织 / 默认值 / 漂移 / 版本化）

#### [TR-10] P1 根 recipe 的算法-优化器兼容性校验是 120 行 isinstance 链，每加一个算法都要改这个中心文件

- 位置：`worldfoundry/training/recipes/post_training/recipe.py:195-317`
- 证据（节选，实际覆盖约 18 个算法 spec）：

```222:236:worldfoundry/training/recipes/post_training/recipe.py
        elif isinstance(self.algorithm, AdversarialDiffusionAlgorithmSpec):
            if self.discriminator_optimizer is None:
                raise ValueError("adversarial diffusion distillation requires discriminator_optimizer")
            ...
        elif isinstance(self.algorithm, SenseFlowAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("SenseFlow requires fake_score_optimizer")
            if self.discriminator_optimizer is None:
                raise ValueError("SenseFlow requires discriminator_optimizer")
```

- 问题：每个算法需要哪些辅助优化器（fake_score/guidance/discriminator）本质上是算法 spec 自己的属性，却写死在 `PostTrainingRecipe.__post_init__` 的 if/elif 链里。同文件 :352-398 的算法 payload 解析倒是用了 dict 注册表（`algorithm_parsers.get(algorithm_type)`），两种风格并存说明作者知道注册表模式，只是校验没跟上。24 个 distillation 算法 + 9 个 RL 算法还在扩张，这条链是全仓库合并冲突和漂移遗漏的高发点（新算法忘记加分支时落到 :313-317 的兜底 elif，报错语义不对）。
- 影响：可扩展性/可维护性；新增算法必须动核心 recipe 文件。
- 建议：在 `FlowPolicyAlgorithmSpec`/各 distillation spec 基类上声明 `required_auxiliary_optimizers: frozenset[str]`，`__post_init__` 统一按声明校验。

#### [TR-11] P3 recipe 体系正面记录：纯 dataclass + 严格 schema，无 hydra 魔法，但没有配置版本号

- 位置：`worldfoundry/training/recipes/spec.py`（全文）、`recipes/post_training/recipe.py:483-500`
- 说明：配置组织是"frozen dataclass + `from_file`（JSON/YAML）+ `__post_init__` 全量校验 + 未知键拒绝"，无 hydra/omegaconf 依赖，默认值全部内联在字段定义处、可审计性好；`TrainingRecipe.to_dict` 支持往返。checkpoint 侧用 canonical JSON 的 identity 绑定（`checkpoint/state.py:307-310`）防止换配方续训。
- 缺口：recipe 文件本身没有 `schema_version` 字段——字段增删后旧 recipe 文件的失败模式是"unknown key"硬错误（可接受）或默认值静默变化（不可接受，例如未来改 `shuffle_seed` 默认值会静默改变旧文件语义）。建议加 `recipe_version` 并在解析时记录。
- 另有小重复：`spec.py` 的 `_plain` 与 `post_training/common.py` 的 `plain_data` 逻辑重复；`_positive_int` 一类小校验函数在 recipes 树内有 ≥4 份拷贝。

### 主题 G：post_training / RL（ray 生命周期、数据流、算法正确性）

#### [TR-12] P1 `RayDevicePool.setup()` 失败路径泄漏 placement group 与 ray 会话，`shutdown()` 因 `self._ray is None` 直接早退

- 位置：`worldfoundry/training/distributed/ray_runtime.py:394-415`、`:488-490`
- 证据：

```405:415:worldfoundry/training/distributed/ray_runtime.py
        while remaining:
            ...
            group = placement_group([dict(bundle) for _ in range(node_devices)], strategy="STRICT_PACK")
            groups.append(group)
            remaining -= node_devices
        ray.get([group.ready() for group in groups])
        self._ray = ray
        self._placement_groups = tuple(groups)
```

```488:490:worldfoundry/training/distributed/ray_runtime.py
    def shutdown(self) -> None:
        if self._ray is None:
            return
```

- 问题：placement group 在 :410 已创建、`self._started_ray` 在 :400 可能已置 True，但 `self._ray` 要到 :414（`ray.get` 成功后）才赋值。若 :413 的 `ray.get` 抛错（集群故障、K8s pod 驱逐等），`shutdown()` 看到 `self._ray is None` 直接 return——placement group 永不 remove、本进程启动的 ray 也不 shutdown。`__enter__`（:512-514）里 setup 抛错时 `__exit__` 同样不会执行。
- 影响：失败恢复场景下集群资源泄漏；反复重试会把集群占满。
- 建议：`setup()` 内部 try/except，失败时先 remove 已创建的 group、`ray.shutdown()`（若 `_started_ray`）再抛。

#### [TR-13] P2 `placement_group.ready()` 无超时：资源永远不满足时 setup 无限挂起

- 位置：`worldfoundry/training/distributed/ray_runtime.py:413`
- 证据：`ray.get([group.ready() for group in groups])` 未传 `timeout`。
- 问题：STRICT_PACK 策略下若单节点凑不齐 `devices_per_node` 个 bundle（配置写错、集群缩容），`ready()` 永远 pending，训练进程无任何日志地挂死——排障成本远高于一次超时报错。
- 建议：`ray.get(..., timeout=config.placement_timeout_seconds)`，超时后带上 `ray.util.placement_group_table()` 诊断信息抛错。

#### [TR-14] P2 `shutdown()` 静默吞掉 actor 关闭错误；worker 无失败恢复策略

- 位置：`worldfoundry/training/distributed/ray_runtime.py:493-497`、`:447,474`（actor 创建处无 `max_restarts`）
- 证据：

```493:497:worldfoundry/training/distributed/ray_runtime.py
        if close_refs:
            try:
                ray.get(close_refs)
            except Exception:
                pass
```

- 问题：(1) actor `close()` 阶段的异常被 `pass` 吞掉且无日志——若 close 里有 NCCL destroy/文件落盘，失败了也无从知晓；这是本半区极少数真正的静默吞错点。(2) `remote_worker.options(**resources)` 未设置 `max_restarts`/`max_task_retries`，任何 rollout actor 死亡（OOM 最常见）会在下一次 `ray.get` 时以 RayActorError 炸掉整个训练 run；配合 checkpoint 恢复可以接受，但多小时 RL run 中单 actor OOM 即全体重启的代价应当是显式声明的设计决策。
- 建议：吞错处至少 `logger.warning(..., exc_info=True)`；在 `RayDevicePoolConfig` 中暴露 actor 重启策略（默认 0 保持 crash-fast，允许按 role 覆盖）。

#### [TR-15] P3 agentic Ray rollout：失败 sibling 的错误信息被静默丢弃，只留 sample_id

- 位置：`worldfoundry/training/post_training/agentic/remote.py:174-179`（worker 侧捕获并封装 error 字符串）、`:319-330`（trainer 侧只取 trajectory，`result.error` 无人消费）
- 证据：

```319:326:worldfoundry/training/post_training/agentic/remote.py
        successful_counts = Counter(
            result.trajectory.request.group_id for result in results if result.trajectory is not None
        )
        trajectories = tuple(
            result.trajectory
            for result in results
            if result.trajectory is not None and successful_counts[result.trajectory.request.group_id] >= 2
        )
```

- 问题：worker 精心把异常转成 `RayAgenticSampleResult.error`（隔离单样本失败、保住其余 sibling，这个设计本身好），但 trainer 侧组装 `AgenticTrajectory` 时只记录 `failed_sample_ids`，error 字符串从未打日志也未进 metrics——环境挂了 50% 的 rollout 时只能看到样本变少，看不到为什么。
- 建议：把 `result.error` 聚合进 trajectory metadata 或按 (错误类型 → 计数) 打 warning 日志。

#### [TR-16] P3 flow-policy 运行时注册表隐含"子类必须排在父类前"的顺序约束，无注释无测试保护

- 位置：`worldfoundry/training/post_training/rl/algorithms/flow_policy/runtime.py:56-106`；对照 `recipes/post_training/algorithms/mix_grpo.py:18`、`dance_grpo.py:21`
- 证据：

```103:106:worldfoundry/training/post_training/rl/algorithms/flow_policy/runtime.py
    for runtime in _FLOW_POLICY_RUNTIMES:
        if isinstance(algorithm, runtime.algorithm_type):
            return runtime
    raise TypeError(f"unsupported native flow-policy algorithm: {type(algorithm).__name__}")
```

- 问题：`DanceGRPOAlgorithmSpec(FlowGRPOAlgorithmSpec)`、`MixGRPOAlgorithmSpec(FlowGRPOAlgorithmSpec)` 是子类，注册表按首个 `isinstance` 命中返回——当前 Dance/Mix 恰好排在 FlowGRPO 前面才正确。有人按字母序重排该元组（很自然的 refactor）会让 Dance/Mix 静默解析到 Flow-GRPO 引擎，训练行为改变且无报错。
- 建议：改用 `type(algorithm) is runtime.algorithm_type` 精确匹配（spec 都是叶子类），或在元组旁注释顺序不变量并加一条"子类先于父类"的单元测试。

#### [TR-17] P3 RL 数学与数据流核验通过（正面记录）

- 位置：`post_training/rl/transitions/flow_sde.py`、`rl/objectives/group_advantages.py`、`shared/accumulation.py`、`shared/distributed.py`、`rl/algorithms/flow_policy/engine.py`、`rl/algorithms/token_policy/objectives/*`、`rl/reduction.py`
- 核验点：
  - SDE 转移：`gaussian_transition_log_prob` 全程 fp32、scale 非正/非有限直接抛错（flow_sde.py:71-72）、常数项 `0.5*log(2π)` 显式（:73-77）；`eta<1e-7` 时 log_prob 显式置 None 而非计算退化值（:154-155）。
  - 优势归一：四种模式（population/sample-std、组内/全局）实现与命名一致；epsilon 必须为正（group_advantages.py:70-71）、组大小 <2 拒绝（:86）、分母统一 `std + epsilon` 无除零路径（:127-134）；DP 版用 all-reduce 统计量而非收集样本（`normalize_data_parallel_grouped_advantages`，:152-177），与 `shared/distributed.py` 的 `global_standard_deviation` 两段式 all-reduce 一致。
  - PPO 裁剪：token GRPO 的 `clipped_policy_objective`（common.py:59-99）ratio 溢出显式 `FloatingPointError`；GSPO 序列 log-ratio `clamp(max=10)` 防 exp 溢出（gspo.py:65）；clip 上下界支持 clip-higher 且范围校验 `0<=lower<1`。
  - 精确 reduction：`reduce_token_losses` 保留分子/分母（reduction.py:60-83），与 `shared/accumulation.py` 的"backward 前 all-reduce 全局分母 + `check_reported_weight` 复核申报值"（accumulation.py:66-79）闭环——变长序列 + 梯度累积 + DP 三者叠加时的加权是数学精确的，这是很多 RLHF 框架都做错的点。
  - 引擎完整性：`NativeFlowPolicyEngine` 有 poison 语义（失败后拒绝继续 step）、replay anchor 一致性审计、`_audit_distributed_backward_calls` 保证各 rank backward 次数一致（engine.py，之前精读确认）。
  - reward 流：`DecodedTerminalRewardAdapter` 解码终端 latent → `RewardEvaluator`（HTTP 或本地）→ `WeightedRewardScalarizer` 默认 `invalid_policy="reject"`（NaN reward 直接拒样本而非污染优势）。HTTP client 有超时+指数退避重试（client.py:67-79），service 端 `fail_fast=True` 默认。
- 结论：RL 侧核心数学未发现正确性缺陷。

#### [TR-18] P3 reward HTTP 服务默认绑定 0.0.0.0 且无鉴权

- 位置：`worldfoundry/training/post_training/rewards/http/service.py:216-226`
- 证据：

```216:221:worldfoundry/training/post_training/rewards/http/service.py
def serve_reward_service(
    service: NativeRewardService,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
```

- 问题：`/score` 接口无任何认证，默认监听所有网卡。共享集群上任何能路由到该节点的进程都能提交任意打分请求（占用 GPU reward 模型）或探测 `/rewards` 列表。
- 影响：多租户集群上的资源滥用面；不泄漏权重，风险有限。
- 建议：默认 `127.0.0.1`，文档指导通过显式参数暴露；或加共享 token 校验。

### 主题 H：distillation（24 个算法子包）

#### [TR-19] P2 19 个 distillation engine 平行复制同一套 train_step/poison/state_dict 骨架（约 7.5k 行中 ~40% 是模板）

- 位置：`worldfoundry/training/post_training/distillation/*/engine.py`（19 个文件，共 7566 行）
- 证据（结构对比）：

```text
$ rg -c "_poisoned" */engine.py   # 19 个 engine 每个恰好 5 处 poison 样板
dfd/engine.py:5  causal_ode/engine.py:5  sid/engine.py:5  sgmd/engine.py:5 ...
$ rg -n "def " dmd/engine.py rcm/engine.py sid/engine.py
# 三者函数骨架同构：_finite_loss / train_step / state_dict / load_state_dict / _expected_*_optimizer_steps
```

- 问题：每个 engine 都手写：poison 标志与 5 处检查、`_finite_loss`（逐字重复 19 份）、微批循环 + `accumulation_context` + `declared_loss_weight/global_denominator` 编排、`state_dict/load_state_dict` schema 校验、双优化器交替相位计算（`_expected_student_optimizer_steps` 等）。共享数学已下沉到 `shared/accumulation.py`（好），但 train_step 编排层没有骨架基类/组合器。dmd vs dmd2 归一化后 diff 仍有 464/856 行差异，说明这是"平行演化"而非纯复制——但也正因如此，某处骨架 bug（如 poison 检查漏了某条路径）修复时无法机械地同步 19 份。
- 影响：维护成本随算法数线性增长；骨架级修复容易漏。
- 建议：不必强行抽象 train_step 本体（各算法确实不同），但 `_finite_loss`、poison 守卫、state schema 校验、交替相位计数这四件事可以下沉到 `shared/` 的小工具（各 ~20 行），能删约 1.5k 行。

#### [TR-20] P3 DMD 分布匹配梯度用 `nan_to_num` 静默清洗非有限值，且归一化 epsilon 默认为 0

- 位置：`worldfoundry/training/post_training/distillation/dmd/objective.py:166-177`、`:78`
- 证据：

```174:177:worldfoundry/training/post_training/distillation/dmd/objective.py
    normalizer = denominator.clamp_min(epsilon) if epsilon > 0 else denominator
    gradient = (fake_score_clean.float() - real_score_clean.float()) / normalizer
    gradient = torch.nan_to_num(gradient)
    return gradient, denominator
```

- 问题：`normalization_epsilon` 默认 0.0（DMDConfig:78），退化情形（生成分布与教师输出几乎重合）下 `denominator→0`，除法产生 inf/NaN 后被 `nan_to_num` 映射为 0 或 ±3.4e38 的巨值——与官方 DMD2 实现一致，但与本仓库其余部分"非有限即抛 FloatingPointError"的哲学（如 token_policy common.py:82-83、flow_sde 的全程 finite 检查）相悖，且 3.4e38 的梯度目标进入 `dmd_proxy_loss` 的 MSE 会平方溢出成 inf loss，届时报错点距离真因已隔两层。
- 影响：数值异常的首报点漂移，排障困难；正常训练不受影响。
- 建议：`nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)` 并在 metrics 里记录清洗计数，或默认 `normalization_epsilon=1e-6`。

### 主题 I：错误处理与导入卫生（横向扫描）

#### [TR-21] P3 横向扫描结论：越层 import 为零、顶层重依赖为零、broad except 绝大多数是"清理后重抛"的正当模式

- 扫描方法：`rg "from worldfoundry\.(evaluation|studio|cli)" training/` → 0 命中；`rg "^(import|from) (ray|transformers|torch)" recipes/ api/` → 0 命中（api/recipes 层完全无重依赖，objectives/models/post_training 顶层 import torch 属合理，ray 全部走 `import_module("ray")` 或函数内 import）；`rg "except (Exception|BaseException)" training/` → 约 30 处，逐一抽查分类：
  - 清理后重抛（正当）：`tuning/peft.py:340`（删临时目录后 `raise`）、`tuning/full_model.py:238`（同）、`shared/accumulation.py:88`（恢复 FSDP 同步标志后 `raise`）、`agentic/remote.py:421`（`runtime.shutdown()` 后 `raise`）、各 engine 的 poison-then-reraise。
  - 失败隔离并显式携带错误（正当）：`agentic/rollout.py:151`、`agentic/remote.py:174`（转成 result.error，但见 TR-15 的下游丢失问题）。
  - 真正的静默吞错：仅 `distributed/ray_runtime.py:496-497`（见 TR-14）。
- 结论：错误处理纪律整体优秀，无 bare except。

### 主题 J：可复现性（seed / 超参记录 / 精确续训）

#### [TR-22] P3 可复现性正面记录：checkpoint 保存每 rank 全量 RNG 状态、identity 绑定配方，精确续训是真实现而非口号

- 位置：`worldfoundry/training/checkpoint/state.py:233-245`、`:319-321`、`:435-439`；`engine/sessions/single_device.py:363-368`
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

- 核验点：torch CPU/全 CUDA 设备/objective generator/python random 四路 RNG 全量保存并按 rank 恢复；恢复时强制 world_size 一致（:319-321，"exact data/RNG resume requires the same world size"——诚实地把不支持的场景拒掉而不是近似恢复）；identity 用 canonical JSON 绑定 recipe/data/model/runtime，换配方续训直接 `TrainingCheckpointCompatibilityError`；只允许在 optimizer-step 边界存档（state.py:84-85）。session 侧 seed 按 `seed+rank` 派生并写入 run manifest（single_device.py:363-368 + `_base_manifest(seed=...)`）。
- 小缺口：seed 不在 recipe 文件里而是 `session.run(seed=...)` 参数（CLI 传入）——recipe 文件单独不能完全确定一次 run，需要连同 manifest 一起归档；`recipes/spec.py:161` 的 `shuffle_seed: int = 42` 只管数据顺序。属可接受的设计，记录备查。

### 主题 K：死代码与可测试性

#### [TR-23] P3 可测试性核验（正面）：数学模块纯函数化、Protocol 假件接缝、tests/training 覆盖密集

- 位置：`tests/training/`（按算法组织的 formulas/runtime/distributed 三层测试，如 `test_adversarial_diffusion_formulas.py`）；`tests/training/test_peft_tuning.py:202-274`（`load_peft_adapter` 有 save→load 往返测试，修正此前"无调用方即死代码"的猜测：它是面向下游的公开 API，生产内无调用方但测试覆盖完整，推理侧合并走 `base_models/diffusion_model/models/denoisers/wan.py:126` 的 `audit_lora_targets + merge_peft_adapter`）。
- 说明：可测试性设计整体好——纯函数化的数学模块（flow_sde、group_advantages、reduction）无状态可直测；Protocol 接缝（`FlowPredictionAdapter`、`TrajectoryRewardAdapter`、`WeightedLossAdapter`）允许注入假件；engine 的 state_dict schema 便于快照测试。
- 顺带的分层观察：`base_models/diffusion_model/models/denoisers/wan.py:126` 存在 base_models → training.tuning 的反向依赖（函数级延迟 import，无循环风险）。training 半区自身对外依赖单向干净（见 TR-21），但这个接缝意味着"training 可整体裁剪"的假设不成立，建议在分层文档中显式登记。

## 汇总

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0 | 0 | — |
| P1 | 2 | TR-10, TR-12 |
| P2 | 5 | TR-6, TR-9, TR-13, TR-14, TR-19 |
| P3 | 16 | TR-1~5, TR-7, TR-8, TR-11, TR-15~18, TR-20~23 |

（其中 TR-2/4/8/17/22 及部分条目为正面核验记录，计入 P3 备查。）

### Top 5

1. **TR-12 (P1)** `RayDevicePool.setup()` 失败路径泄漏 placement group 与 ray 会话：`self._ray` 赋值晚于资源创建，`shutdown()` 见 None 早退，`__enter__` 抛错时 `__exit__` 不执行——失败恢复场景下集群资源被反复占满。
2. **TR-10 (P1)** 根 recipe 用 120 行 isinstance 链集中校验 18+ 个算法的优化器组合，算法自身的需求没有声明在算法 spec 上，是扩张中的算法库（33 个算法）最主要的漂移与合并冲突点。
3. **TR-19 (P2)** 19 个 distillation engine 平行复制 train_step/poison/state-schema 骨架（7.5k 行中约 40% 模板），骨架级缺陷无法一次修复。
4. **TR-9 (P2)** 安全过滤 fail-closed 且缓存链路两端强制，但 `select_for_manifest` 的强制点在 CLI 层，绕开 CLI 手写 manifest `safety` 标志即可跳过审计；审计仅覆盖 prompt 不覆盖像素内容——信任边界应文档化或下沉。
5. **TR-13/14 (P2)** ray 生命周期细节：placement group `ready()` 无超时（资源不满足时无日志挂死）+ actor close 异常被 `except: pass` 吞掉 + 无 actor 重启策略声明。

### 总体评价

该半区代码质量在同类训练框架中属上游：契约层无重依赖、loss 全线显式分子/分母（变长+累积+DP 叠加时数学精确）、RL 数学（SDE 转移、优势归一、PPO 裁剪、精确 reduction）逐项核验无误、checkpoint 精确续训与 fail-closed 安全过滤是真实现。主要债务是规模性的：算法数量（24 distillation + 9 RL）已超出"每个算法一份手写骨架 + 中心 recipe 手工校验"这套组织方式的承载力，以及 ray 生命周期在失败路径上的清理完整性。
