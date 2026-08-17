# training 配方层修复日志（对应评审报告 plan/code_review/09_training_recipes.md）

> 修复人：infra 修复 agent；日期：2026-08-14。
> 范围约束：只改 worldfoundry/training/{api,models,objectives,post_training,recipes,tuning,safety}；distributed/、engine/、checkpoint/、data/ 等由另一 agent 负责。
> 验证约定：每条修复附 py_compile + import 冒烟 + 单测/既有测试证据。

## 已修复

### [TR-16] P3 flow-policy 运行时注册表 isinstance 顺序陷阱 → 精确类型匹配

- 改动：`worldfoundry/training/post_training/rl/algorithms/flow_policy/runtime.py`
  - `resolve_flow_policy_algorithm_runtime` 中 `isinstance(algorithm, runtime.algorithm_type)` 改为 `type(algorithm) is runtime.algorithm_type`，并加注释说明不变量。
- 独立确认：DanceGRPO/MixGRPO 确为 FlowGRPO 子类（dance_grpo.py:21、mix_grpo.py:18），注册表 6 项全部是解析器构造的叶子类；全仓唯一调用方 `flow_policy/builder.py:126` 传入 `recipe.algorithm`（由 recipe.py 解析器注册表构造，恒为叶子类）；tests/ 无这些 spec 的子类。
- 行为变化说明：未注册的子类此前静默解析到父类引擎，现在抛 `TypeError`（fail-fast，符合修复意图）；对所有现存调用路径行为不变。
- 验证：
  - `python -m py_compile` 通过；`PYTHONPATH=. python -c "from ...runtime import resolve_flow_policy_algorithm_runtime"` 通过。
  - 新增 `tests/training/test_training_recipes_fix_flow_policy_runtime.py`（8 例，含 6 个注册 spec 各自解析到自身运行时、子类独立解析、未注册子类拒绝），全部通过。
- 风险：低。仅在"传入未注册子类"这一原本就是错误的路径上改变行为。

### [TR-15] P3 agentic Ray rollout 失败 sibling 错误信息被静默丢弃 → 聚合暴露

- 改动：`worldfoundry/training/post_training/agentic/remote.py`
  - 新增模块级 `_summarize_rollout_errors(error_counts, limit=5)`：错误字符串按 (消息 → 计数) 排序渲染，超限截断。
  - `RayAgenticRolloutAdapter.__init__` 新增诊断属性 `last_rollout_error_counts: dict[str, int]`（沿用类内 `last_sync_report` 的可观测属性模式）；每次 `rollout_agentic` 用本轮 worker `result.error` 聚合刷新（全部成功时为空 dict）。
  - 全部 sibling 不可训练时的 `RuntimeError` 消息追加聚合的失败原因摘要（此前只有 "produced no trainable sibling group"，看不到为什么）。
- 独立确认：worker 侧 `rollout_sample` 确实把异常转成 `RayAgenticSampleResult.error`（remote.py:174-179），trainer 侧组装时只消费 `trajectory`，`error` 无任何下游（rg 全仓 `\.error` 确认）。
- 设计取舍：报告建议 "进 trajectory metadata 或打 warning 日志"。核实发现 `worldfoundry/training/` 全树零 logging 依赖（刻意约定，rg `import logging|get_logger` 0 命中），且 `AgenticTrajectory` 是 frozen 契约、经 packed 往返（`agentic_trajectory_from_packed`）无法携带 error 字段而不扩散契约变更。故选择"adapter 诊断属性 + 全失败异常携带摘要"，不引入 logging、不动契约。
- 数值行为：无变化（纯诊断信息）。
- 验证：
  - py_compile + import 冒烟通过。
  - 新增 `tests/training/test_training_recipes_fix_agentic_remote_errors.py`（4 例：部分失败记录并在下轮成功后清空、全失败异常含 2x/1x 聚合、摘要截断）。
  - 连同既有 `tests/training/post_training/agentic/test_agentic_training.py` 共 13 例全部通过。
- 风险：低。异常消息变长（追加分号小节），未发现对该消息全文断言的测试（既有断言用子串 match）。

### [TR-20] P3 DMD 分布匹配梯度 `nan_to_num` 隐式产生 ±3.4e38 目标 → 显式 0 映射

- 改动：`worldfoundry/training/post_training/distillation/dmd/objective.py:176`
  - `torch.nan_to_num(gradient)` → `torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)`，附注释说明动机。
- 独立确认：`normalization_epsilon` 默认 0.0（DMDConfig:78），退化情形（`generated == guided_real` 逐位相同）分母为 0，除法产生 inf；默认 `nan_to_num` 把 posinf 映射为 float32 max（3.4e38），进入 `dmd_proxy_loss` 的 MSE 平方后 inf，报错点在两层之外的 engine `_finite_loss`。
- **数值行为变化（显式标注）**：仅在 `normalization_epsilon == 0` 且分母恰为 0 的退化路径上，非有限梯度元素从 ±3.4e38 变为 0（退化元素贡献零更新、loss 保持有限，而非在 engine 层以 inf loss 崩溃）。NaN（0/0）路径新旧都映射为 0，不变。正常训练路径（分母 > 0）逐位不变。此为报告建议的首选修法。
- 保守性说明：报告同时提出的两个备选（默认 epsilon 改 1e-6、metrics 记录清洗计数）未采纳——前者改变 config 契约与正常路径数值；后者需要改函数签名（第二调用方 reward_forcing/objective.py:205 会扩散），且既有 `dmd_normalizer` metric 已可观测分母趋零。
- 相邻未动项（备查）：同模式 `nan_to_num((a-b)/normalizer)` 还存在于 dmd2/math.py:75、scale_wise/math.py:107、senseflow/math.py:227、anyflow/math.py:239、diagonal/math.py:85,97——报告未点名（多数默认 epsilon > 0），遵循"只修报告明确指出的缺陷"不顺手改。
- 验证：
  - py_compile 通过。
  - 新增 `tests/training/test_training_recipes_fix_dmd_gradient.py`（3 例：正常路径与手算逐位一致、全零分母产出零梯度且 proxy loss 有限为 0、per-sample 模式只清洗退化样本），通过。
  - 触及该函数的既有测试 `test_post_training_math.py`、`test_source_formulas.py`、`test_reward_forcing_formulas.py`（33 例）、`test_dmd_accumulation.py`（4 例）全部通过。
- 风险：低。变化仅限实践中测度近零的退化路径，且方向是"更安全的中性值"。

### [TR-3] P3 objectives 包 `__init__` 击穿 flow_matching 的 torch 延迟导入 → 惰性 facade

- 改动：`worldfoundry/training/objectives/__init__.py` 重写为惰性 `__getattr__` facade（照抄 `training/__init__.py` 与 `models/__init__.py` 的既有模式），17 个 re-export 符号与原 `__all__` 完全一致。
- 独立确认：flow_matching.py 顶层无 torch（`_require_torch()` 惰性）；classic_diffusion.py 顶层 `import torch`；原 `__init__` 同时急切导入两者。全仓 rg `training\.objectives`：所有消费方要么 `from ... import <符号>`（走 `__getattr__`，兼容）、要么直接导子模块（不受影响）；无 "导包后访问 `objectives.classic_diffusion` 属性" 的用法。
- 选择说明：报告给出的另一半建议（classic_diffusion 也全面 lazy 化）未做——该文件模块级 `ConditioningBuilder` 类型别名、`isinstance(x, Tensor)` 运行时检查均需 torch，收益仅是"无 torch 环境也能导 classic 模块本身"（该模块功能本来就需要 torch），改动/收益比差。
- 数值行为：无变化（纯导入结构）。
- 验证：
  - py_compile；子进程冒烟：`from worldfoundry.training.objectives import flow_shift_sigmas` 后 `'torch' not in sys.modules` 且 classic_diffusion 未加载；随后导 `ClassicDiffusionObjective` 正常拉起 torch。star-import 17 符号齐全。
  - 新增 `tests/training/test_training_recipes_fix_objectives.py::test_flow_matching_symbols_import_without_torch_or_classic_diffusion`（子进程隔离验证）。
  - 既有 `test_flow_matching.py`、`test_source_formulas.py`（11 例）与 lvdm/dynamicrafter 集成测试通过。
- 风险：低。惰性 facade 是仓库既有模式；唯一语义差异是"导包不再急切加载子模块"，已确认无调用方依赖。

### [TR-4] P3（文档）`ObjectiveBatch.timesteps` 语义随配置漂移 → docstring 标注

- 改动：`worldfoundry/training/api/contracts.py` `ObjectiveBatch` 类 docstring 增补 5 行：timesteps 为 objective 自定义语义（连续 flow=未 shift base sigma、离散 flow=整数索引、classic=DDPM 索引），适配器取有效噪声水平必须消费 `sigmas`。
- 独立确认：flow_matching.py:333-343（连续分支 `timesteps = base_sigmas`、离散分支 `torch.floor(...).to(long)`）、classic_diffusion.py corrupt（整数 timesteps）与措辞逐一核对。
- 验证：py_compile + `tests/training/test_contracts.py`（4 例）通过。纯 docstring，无行为变化。

### [TR-5] P3 classic_diffusion 每 step 重复 H2D 拷贝 schedule 表 → 按 device 缓存

- 改动：`worldfoundry/training/objectives/classic_diffusion.py`
  - `ClassicDiffusionObjective.__init__` 增加 `_device_schedules: dict[torch.device, tuple[Tensor, Tensor]]`，以构造设备（CPU）自身表作种子；新增 `_schedules_for_device()`。
  - `corrupt()` 中 `extract_schedule(self.alphas/self.sigmas, ...)` 与 `self.sigmas.to(device=...).gather(...)` 改为使用缓存表（CPU 路径命中种子条目，零拷贝；CUDA 路径首次调用后复用）。
- 独立确认：`self.alphas/self.sigmas` 构造后只读（rg 全仓无外部赋值）；`extract_schedule` 内部 `.to(device=同设备)` 为 no-op，公共函数签名未动。
- 数值行为：无变化（同值张量的设备拷贝，gather 结果逐位一致）。
- 验证：
  - py_compile；新增测试 `test_classic_diffusion_corrupt_is_deterministic_and_reuses_schedule_tables`（同种子两次 corrupt 逐位一致、CPU 缓存零拷贝命中、输出与未缓存公式逐位一致），通过。
  - 既有 `test_lvdm_training_integration.py`、`test_dynamicrafter_training_integration.py` 通过。
- 风险：低。缓存 dict 随 objective 实例生命周期存在，每设备一份 fp32 表（~KB 级）。

### [TR-18] P3 reward HTTP 服务默认绑定 0.0.0.0 无鉴权 → 默认回环 + docstring 指引

- 改动：`worldfoundry/training/post_training/rewards/http/service.py`
  - `serve_reward_service` 默认 `host` 从 `"0.0.0.0"` 改为 `"127.0.0.1"`，docstring 说明 `/score` 无鉴权、跨节点需显式传 `host="0.0.0.0"` 并做网络层限制。
- **行为变化（显式标注）**：直接调用该库函数且依赖隐式默认绑定所有网卡的外部代码，现在默认只绑回环，需显式传参。仓库内核查：唯一生产调用方 `cli/training_commands/handlers/reward_service.py:27` 始终显式传 `host=config.host`（CLI 行为不变）；测试全部 monkeypatch。
- 相邻未动项（备查）：`rewards/scorers/service_config.py:92` 的 CLI 配置默认 `server.get("host", "0.0.0.0")` 未改——报告未点名，且改它会改变现有部署（省略 `server.host` 的配置文件）的跨节点可达性，属部署行为决策，列 deferred 提请 owner 决定。
- 验证：py_compile；签名冒烟 `host: str = '127.0.0.1'`；既有 `tests/training/test_reward_service_cli.py`（8 例）通过。
- 风险：低-中（对外部直接调用者是显式行为变化，方向为安全默认；报告建议的原文修法）。

### [TR-6] P2 五份适配器逐字复制私有 helper 且已漂移 → 提取 `models/_shared.py`

- 改动：
  - 新建 `worldfoundry/training/models/_shared.py`（不进 models facade 公共 API）：`component_module`（取签名并集 `object | None`，对非 None 调用方行为逐字节一致）、`module_device_dtype`（五份逐字一致，原样提取）、`freeze_module`（wan/sana 早退式与 cosmos if 式行为等价，取早退式）、`merge_without_overwrite(destination, source, *, source_name, family)`（family 参数保留各族错误消息逐字不变）。
  - `wan.py`/`sana.py`/`cosmos.py`：删除 4 个本地 def，改为别名导入（`component_module as _component_module` 等），`_merge*` 两处调用点改为显式传 `family="Wan"/"SANA"/"Cosmos"`（cosmos 的 `owner=` 关键字并入 `source_name=`，消息不变）。
  - `hunyuan_video.py`：删除 2 个本地 def，别名导入。
  - `ltx.py`：删除 2 个本地 def，别名导入（`component_module as _module_from_component` 保留其历史名）。
- 独立确认：rg 全仓（含 tests/）无任何外部对这些私有 helper 的导入；五份 `_module_device_dtype` 逐字一致；hunyuan 版 `component: object | None` 与其余版本对 None 输入行为一致（getattr(None,...) → None）；错误消息模板仅 family 词不同。
- 数值/行为变化：无。错误消息逐字保留（"conditioner.shared collides with encoded Wan/SANA/Cosmos conditioning keys: [...]"）。
- 验证：
  - 五文件 + `_shared.py` py_compile；六个适配器类（含 wan22 间接依赖 wan）import 冒烟通过；rg 确认 models/ 下旧本地 def 零残留。
  - 新增 `tests/training/test_training_recipes_fix_models_shared.py`（6 例：None 组件、属性查找、buffer-only/int-buffer dtype 回退、freeze、三族消息逐字校验），通过。
  - 既有测试：`test_wan_adapter.py`、`test_cosmos_training.py`、`test_ltx_flow_policy.py` 通过（35 passed, 7 skipped）。
  - 环境限制（预存，与本次改动无关，均为 ModuleNotFoundError）：`test_sana_adapter.py` 收集失败缺 `transformers`（发生于 base_models 导入链）；`test_ltx_training.py::test_ltx_session_reuses_shared_video_flow_engine_and_updates` 失败缺 `torchdata`（`data/loader.py:108` 主动抛"requires the 'train-core' TorchData dependency"）；`test_hunyuan_video_rl.py` 收集失败缺 `ftfy`。本机 pypi 不可用无法安装；`python -c "import torchdata/transformers"` 直接复现同错。hunyuan/sana 适配器改动通过 import 冒烟 + 与 wan/cosmos 完全同源的共享实现覆盖。
- 风险：低。纯等价提取；漂移点（签名/参数名）已在共享版收敛为并集语义。

### [TR-10] P2 recipe `__post_init__` 120 行 isinstance 辅助优化器校验链 → spec 声明式规则

- 改动：
  - 新建 `worldfoundry/training/recipes/post_training/algorithms/auxiliary_optimizers.py`（纯 stdlib，满足 recipes 层"无执行面导入"架构约束）：`AuxiliaryOptimizerRule`（frozen dataclass：optimizers 元组 + required + 原文消息）、`requires_auxiliary`/`forbids_auxiliary` 构造器、`DEFAULT_AUXILIARY_OPTIMIZER_RULES`（逐个拒绝三个辅助优化器，即原链的兜底 elif 分支）、`resolve_auxiliary_optimizer_rules`（未声明的 spec 落默认）、`validate_auxiliary_optimizers`（按声明顺序应用，抛原文 ValueError）。
  - 17 个 spec 类新增 `auxiliary_optimizer_rules()` 方法声明自身约束（dmd、adaptive_video、anyflow×4、adversarial_diffusion、reward_forcing、senseflow、rcm×2、dmd2、dfd、diagonal、scm_ladd、scale_wise、self_forcing、self_gradient_forcing、sid、sgmd、ddrl）；条件性规则（rCM `dmd_loss_scale>0`、DFD `adversarial_enabled`、scale-wise `dmd_enabled`）在方法内按实例状态构造，f-string 消息（DMD/adaptive/self-forcing 系的 `{self.type} ...`）原样保留。
  - `recipe.py` `__post_init__`：124 行 isinstance 链替换为一次 `validate_auxiliary_optimizers(self.algorithm, ...)` 调用 + 注释。顶部 spec 类导入全部保留（`__all__` 再导出 + 架构测试 `from ...recipe import DMDAlgorithmSpec` 依赖该导入面）。
- 独立确认：
  - 原链 21 个分支逐条抄录消息与顺序（requires 先于 forbids、SenseFlow 两个 requires 的先后、DDRL/默认分支 fake→guidance→disc 的 elif 顺序），声明序列与原链逐分支等价。
  - 继承关系核查：链中点名的 spec 类互不继承（`CausalRCMAlgorithmSpec` 独立于 `RCMAlgorithmSpec`）；FlowGRPO/Token 系子类不在链中、无声明方法 → 落默认拒绝分支，与原链行为一致；方法继承语义与 isinstance 对子类的匹配语义在现存类型图上等价。
  - `AdaptiveVideoAlgorithmSpec` 的 forbid 消息原文是 `f"{self.algorithm.type} only accepts ..."`（type="adaptive-video-distillation" 带连字符），已按原文保留（初稿误用空格版，自查修正）。
  - 架构约束：新模块仅导 stdlib；`ADAPTIVE_VIDEO_ALGORITHM_FIELDS = frozenset(__dataclass_fields__)` 等字段集不受新增方法影响（方法非 dataclass 字段）；`plain_data` 序列化不受影响（规则对象不存实例字段）。
- 数值/行为变化：无。所有校验条件、触发顺序、错误消息逐字保留；唯一结构变化是新算法 spec 今后自带声明而非改中心链。
- 验证：
  - 19 个改动文件 py_compile + recipe/auxiliary_optimizers import 冒烟通过。
  - 新增 `tests/training/test_training_recipes_fix_auxiliary_optimizer_rules.py`（60 例）：原链全部 21 分支的触发消息逐字断言（含 rCM 有/无 DMD、DFD 有/无 GAN、scale-wise DMD/MMD-only 条件分支、DDRL 与默认分支三消息、各 OK 组合）、未声明 spec 落默认规则、规则构造器拒绝未知优化器名/空消息、recipe 端到端（dmd 缺 fake 报 "DMD requires fake_score_optimizer"、多给 guidance 报 "dmd only accepts fake_score_optimizer"）。全部通过。
  - 既有测试：`tests/training/recipes/`（4 例，含 AST 架构约束 + 序列化 round-trip）、`test_post_training_recipe.py` + `test_rcm_recipe.py` + `test_senseflow_recipe.py` + `test_recipe.py` + `test_rollout_recipe.py`（41 例）全部通过。
  - `test_validation_script_recipes.py` 2 例失败为预存环境缺 `ftfy`（base_models 编码器导入链，与本改动无关，与 TR-6 记录的同一环境限制）。
- 风险：低-中（触及 33 个算法共用的校验入口，但等价性有 60 例逐字消息断言 + 41 例既有测试背书）。方法分派 vs isinstance 的唯一语义差异在"spec 子类"场景：子类继承父类声明（原链行为相同）、新增独立 spec 未声明时落默认拒绝（原链也落默认）。

### [TR-9] P2 安全过滤强制点在缓存工具链、审计只校验完整性不校验来源 → 信任边界文档化

- 改动（docstring only，无行为变化）：
  - `worldfoundry/training/safety/shieldgemma.py` 模块 docstring 增补 "Trust boundary" 小节，写明三条边界：(1) 强制点在缓存工具链（构建端 `validate_video_prompt_audits` / sana_precompute 的 `prompt_safe` 检查，消费端 wan/sana cache loader 双标志复核），不走审计缓存链的训练路径无任何安全门；(2) 审计是完整性校验而非来源认证——无签名，绕开 CLI（`cli/training_commands/handlers/cache.py` 接线 `select_for_manifest`）手写 manifest `safety` 块可通过标志检查，诚实产出审计是 manifest 作者的义务；(3) 覆盖范围仅 prompt 文本，不覆盖像素/视频内容。
  - `PromptSafetyAudit` 类 docstring：说明 `from_mapping` 重算派生字段能拒绝篡改的 safe/blocked_categories，但无法证明概率来自真实前向。
  - `safety/__init__.py` 包 docstring：一句话范围声明 + 指向模块 docstring。
- 独立确认：报告所列强制点逐一 rg 复核仍属实（`data/wan/training_cache.py:63`、`data/ltx/training_cache.py:108`、`data/cosmos/training_cache.py:146`、`video_precompute.py:106`、`engine/wan/cache.py:204`、`engine/sana/cache.py:99`、`cli/.../cache.py:111`）。
- 范围说明：报告的备选修法（engine dataloader 处 fail-closed 或 recipe 加 `data.require_prompt_audit`）需要改 `training/data/` 与 `engine/`，超出本次授权边界（data/engine 明确排除），且属于部署行为变更；按报告修正后的定性（"信任边界文档化不足"）选 docstring 修法，代码强制列 deferred。
- 验证：py_compile + import 冒烟通过；`pytest -k "shieldgemma or safety or prompt_audit"` 9 passed, 27 skipped（skip 为预存 GPU/依赖门）。
- 风险：零（纯文档）。

## Deferred

### [TR-19] P2 distillation/RL 骨架抽取 —— 可执行方案（本次不实施）

**现状定量（2026-08-14 复核，行号以当前代码为准）**

- 20 个 `post_training/distillation/*/engine.py` 共 7571 行（报告写 19 个/7566 行，现多一个）。
- poison 样板：15 个 engine 各恰好 5 处 `_poisoned` 触点（anyflow 双 engine 10 处）——`__init__` 置位、`train_step` 入口拒绝、optimizer step 后失败置毒、`state_dict` 拒绝、`load_state_dict` 复位。
- finite-loss 守卫：≥10 份同构 def（dmd 叫 `_finite_scalar_loss`，其余叫 `_finite_loss`）。漂移实证：rcm 版消息是 "objective must return"（其余是 "loss adapter must return"）且漏 `.all()`（标量下等价，但正是"平行演化无法机械同步"的证据）。
- 每 engine 手写：`*_ENGINE_STATE_SCHEMA` 字符串 + `state_dict/load_state_dict` 字段集校验 + `_expected_*_optimizer_steps` 交替相位函数。
- 已有的好底子（抽取落点，不新建顶层包）：`post_training/shared/accumulation.py`（微批数学，已共享）、`shared/validation.py`（`non_negative_int`/`positive_float`/`validate_stateful_or_none`）、`shared/distributed.py`。

**抽取目标（四件小工具，报告原文建议，预计净删 ~1.5k 行）**

1. finite-loss 守卫 → `shared/validation.py` 新增 `finite_scalar_loss(result, *, result_type, role, source="loss adapter")`：isinstance 检查 + 单元素张量检查 + isfinite 检查。engine 侧保留一行别名（照 TR-6 的别名导入手法）；`source` 参数保留 rcm 的 "objective" 措辞，消息逐字不变。
2. poison/commit 状态机 → `shared/commit_guard.py` 新建 `CommitGuard(engine_name)`：`require_idle(action)`（train_step/state_dict 入口）、`begin()/commit()`、`poison_if(optimizer_step_started)`、`reset()`（load_state_dict）。异常消息模板 `f"{engine_name} engine has a partially committed iteration; restore the last checkpoint"` 参数化各 engine 名。
3. state 头校验 → `shared/validation.py` 新增 `validate_state_header(state, *, schema, fields)`：Mapping 类型检查 + 字段集全等 + schema 相等；counter 的语义校验（cadence 一致性）留在各 engine。
4. 交替相位 → `shared/cadence.py`：`expected_interval_steps(completed_iterations, interval)` 与 `validate_alternating_counters(...)`。

**迁移顺序（风险递增分波，每波独立可评审/可回滚，一波一个 PR）**

- 波 0（前置，必须先行）：为 poison/state/cadence 语义写 behavior-pinning 测试，用假 module/optimizer 驱动 3 个代表 engine（dmd=双优化器交替、senseflow=三优化器、sid=对抗可选）：optimizer step 后注入失败 → poisoned → 后续 train_step/state_dict 拒绝且消息逐字断言；load_state_dict 复位；计数不一致拒绝。迁移前后这些测试必须不改一字地通过。
- 波 1：finite-loss 守卫（纯函数、无状态，20 文件机械替换，风险最低）。
- 波 2：state 头校验（无行为分支，字段集不变）。
- 波 3：cadence 相位函数。
- 波 4：poison 状态机（唯一有状态的抽取，最后做；每 engine 5 触点逐一替换）。
- 明确不做：train_step 本体/微批编排循环的基类化。各算法角色数与优化器拓扑不同（报告核实 dmd vs dmd2 归一化后仍差 464/856 行），强行抽象会制造继承迷宫——这是报告建议的原文边界（"不必强行抽象 train_step 本体"）。

**验证方法（每波执行）**

- 结构：`rg "def _finite_loss|def _finite_scalar_loss" distillation/*/engine.py` → 0（各波对应各自的归零断言）；py_compile + import 冒烟全 20 engine。
- 行为：波 0 pinning 测试 + 既有 `tests/training/test_*_formulas.py`、`test_dmd_accumulation.py`、各 runtime/session 测试全绿；异常消息逐字断言（照本次 TR-10 的 60 例消息断言模式）。
- 覆盖缺口预检：迁移某 engine 前先 `pytest --collect-only -q tests/training | rg <算法名>` 确认存在直接测试；没有的先补 smoke（构造→一步→checkpoint 往返）再迁移。

**RL 侧（9 算法）组织**

- 本次已完成两步：TR-10 优化器兼容性下放 spec 声明（消中心链）；TR-16 flow_policy 运行时注册表精确匹配（消顺序陷阱）。RL 侧骨架债务比 distillation 轻：flow 系共用 flow_policy 运行时 + 注册表，token 系共用 token_policy spec 继承。
- 可选波 5：`recipe.py:from_mapping` 的 41 项手写 parser 表 → 各 `algorithms/*.py` 模块尾部自注册（`ALGORITHM_PARSERS[type] = parse_*`，`algorithms/__init__.py` 聚合导出），消除"加算法必改 recipe.py"的最后一处。低风险（recipe.py 顶层本就 import 全部算法模块，注册不改变导入集），需 golden round-trip 测试背书（已有 `test_recipe_serialization_round_trip_is_stable_across_file_split`）。

**本次不实施的理由**：触达 20 engine × 4 类触点 ≈ 100+ 处替换；波 0 的 pinning 测试是硬前置（现缺）；与另一 agent 的 engine/distributed 半区改动窗口存在合并冲突风险；收益是维护性而非正确性——按"正确性第一、宁可少修"原则列 deferred。

### 其它 deferred 项（含原因）

1. **TR-12/13/14（ray 生命周期，P1+P2+P2）**：位置全在 `worldfoundry/training/distributed/ray_runtime.py`（setup 失败泄漏 placement group `:394-415`、`ready()` 无超时 `:413`、shutdown 吞错+无重启策略 `:493-497`）——`distributed/` 明确排除在本次授权外（另一 agent 负责），未动。本 agent 范围内的 ray 失败路径已独立核查完毕：`agentic/remote.py:428-447`（`setup_ray_agentic_rollout` 失败 → `runtime.shutdown()` 后 `raise`）、`causal_lm/qwen3/materializer.py` 三条构建路径（`:593-628` actor-hosted setup 失败 shutdown、`:695-708` external 失败关闭已建 closeables、`:712-738` materialize 失败逆序 close/shutdown 后 raise）均有清理且不吞错；TR-15 的错误信息丢弃已修。
2. **TR-1（P3，api `TensorLike = Any`）**：建议改 `SupportsShape` Protocol。未做：纯静态类型收益（报告确认运行时契约完备、实际风险低），但 `TensorLike/TensorTree` 被 13 个适配器与 engine/ 消费，engine/ 不在本次可改范围，本机也无法跑 mypy/pyright 验证不会引入新告警——改了无法验证，违背"修一条验一条"。
3. **TR-7（P3，LoRA preset 契约分散）**：建议 preset→audit 注册表。未做：ltx/cosmos/wan22/hunyuan 的 `LoraTargetAudit` 构造在 engine 层（`engine/ltx/sft.py:229` 等），注册接线必须改 engine/（排除范围）；只在 tuning/ 单侧加注册表而无注册方是死代码。受影响 API `load_peft_adapter(expected_preset=...)` 在 worldfoundry/ 内无生产调用方（报告 TR-23 核实），无现实故障。留给拥有 engine/ 权限的批次一并做。
4. **TR-18 相邻（reward service CLI 配置默认 host）**：`rewards/scorers/service_config.py:92` 的 `server.get("host", "0.0.0.0")` 未改——省略 `server.host` 的现有部署会因此失去跨节点可达性，属部署行为决策且报告未点名该处；库函数默认值已按报告收紧（TR-18 已修）。
5. **TR-20 相邻（同模式 nan_to_num）**：`dmd2/math.py:75`、`scale_wise/math.py:107`、`senseflow/math.py:227`、`anyflow/math.py:239`、`diagonal/math.py:85,97` 存在同构 `nan_to_num((a-b)/norm)`——报告未点名（多数路径默认 epsilon>0 不可达），遵循"不顺手优化数学代码"。
6. **TR-9 代码级强制**：engine dataloader 对 `prompt_safe is not True` fail-closed 或 recipe 增加 `data.require_prompt_audit`——需要 data/、engine/、recipes spec 三方联动，前两者超出授权；本次已完成信任边界文档化（见已修复清单）。
7. **TR-11 附注（recipe 无版本号）**：正面记录附带建议；schema 字符串已存在（`worldfoundry-post-training`），加版本号属契约演进决策，未动。

### 无需行动的正面核验记录

TR-2（api 协议边界）、TR-4 主体（loss 正确性；其 docstring 附注已修，见已修复清单）、TR-8（tuning 质量）、TR-17（RL 数学）、TR-21（导入/异常纪律）、TR-22（可复现性）、TR-23（可测试性）。

## 验证汇总

- 每条修复均有：py_compile + import 冒烟 + 新增单测/既有测试运行记录（见各条目）。
- 新增测试文件（6 个，均纯 CPU，命名 `test_training_recipes_fix_*.py`）：flow_policy_runtime（8 例）、agentic_remote_errors（4 例）、dmd_gradient（3 例）、objectives（2 例）、models_shared（6 例）、auxiliary_optimizer_rules（60 例）——共 83 例全部通过。
- 既有测试（本次触达面）：recipes 目录 4 例、recipe 系列 41 例、safety 系列 9 例、post_training 数学/累积 37 例、wan/cosmos/ltx 适配器 35 passed + 7 skipped、contracts 4 例、flow_matching/lvdm/dynamicrafter 11+ 例——全绿。
- 预存环境失败（与改动无关，pypi 不可用无法装依赖）：缺 `transformers`（test_sana_adapter）、缺 `torchdata`（test_ltx_training 单例）、缺 `ftfy`（test_hunyuan_video_rl、test_validation_script_recipes）——均为收集期/构造期 ModuleNotFoundError，`python -c "import <pkg>"` 可直接复现。
