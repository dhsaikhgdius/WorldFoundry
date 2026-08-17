# pipelines 层修复日志

> 修复人：infra 修复 agent；日期：2026-08-14
> 对应评审报告：`plan/code_review/06_pipelines.md`
> 约束：只改 `worldfoundry/pipelines/`；无 GPU 端到端验证条件；pypi 不可用（不装新依赖）。
> 验证手段：`PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile`、`python -m compileall`、`PYTHONPATH=. python -c "import ..."`、`rg` 全仓引用核查、可收集的 CPU pytest、无依赖 stub 的语义 smoke test。

## 已修复

### [PL-10] vggt `stream()` CLI 分支生成器内 `return 值` 丢值
- 文件：`worldfoundry/pipelines/vggt/pipeline_vggt.py`
- 改动：CLI 分支改为 `yield self.run_two_stage_3dgs_stream_cli(...)` + 裸 `return`（单元素生成器），docstring 同步更新；报告推荐方案之一。
- 验证：`py_compile` 通过；`import` 链仅在缺 `huggingface_hub` 第三方依赖处失败（环境无该包，属预期）。
- 风险：迭代该生成器才会执行 CLI（生成器语义固有）；与修复前"拿到生成器但 return 值必丢"相比严格改善。

### [PL-23] 死 import 与死函数删除（rg 全仓验证零引用）
- `vggt/pipeline_vggt.py`：删除未使用的 `fetchPly`（dataset_readers）与整个 `flash_world.render.gaussian_render` import（消除 flash_world 渲染栈的 import 负担）；`storePly` 保留（:423 使用）。
- `cut3r/pipeline_cut3r.py`：同样删除未使用的 `flash_world.render.gaussian_render` import（报告未点名，但与 vggt 完全同型，rg 验证本文件零使用）。
- `matrix_game/pipeline_matrix_game_2.py`：删除必崩死函数 `tensor_to_pil`（对 Tensor 调 `.astype`，全仓零调用），连带删除仅被它使用的 `numpy`/`cv2` import。
- 验证：`rg "fetchPly|gaussian_render|tensor_to_pil"` 确认 pipelines/test/tests/test_stream 无残留引用（representations 内部的同名使用是它们自己的 import，不受影响）；各文件 `py_compile` 通过。

### [PL-20] 死参数清理（部分；rg 验证调用方）
- `wan/pipeline_wan_2p5.py`：删除 `__call__` 的 `save_content`/`output_dir` 死参数（体内零引用；`test/test_wan_2p5.py` 的 `output_dir` 是本地变量不传入 pipeline；catalog 无引用）。
- `vggt/pipeline_vggt.py`：删除 `run_two_stage_3dgs_video`/`run_stage2_3dgs_video_from_reconstruction` 的 `camera_trajectory` 死参数（体内零引用；studio catalog 的 vggt `call_params` 不含该键，catalog 中出现的 `camera_trajectory` 属 recammaster）。
- mirror 的 `save_colmap` 见 Deferred（是 catalog/脚本/synthesis 的活契约，不能删）。
- 验证：`py_compile`；`rg` 全仓（含 test/、docs/、catalog）确认无调用方传这些参数给对应 pipeline。

### [PL-16] CWD 相对默认输出路径改显式 artifact 根（报告列名的 5 处全部完成）
- `vggt/pipeline_vggt.py`：新增模块级 `_default_output_dir()`（`worldfoundry.core.io.artifact_root_path()` 下），替换 6 处 `"./vggt_output"`/`"./vggt_stream_output"` 默认值；签名默认值改为 `None`、调用时解析（避免 import 时冻结 env）。
- `hunyuan_world/pipeline_hunyuan_mirror.py`：默认 `output_path` 改为 `artifact_root_path()/"hunyuan_mirror"`；`mkdir` 从 `__init__` 延迟到 `save_results` 实际写出时。
- `wan/pipeline_wan_2p5.py`：`./output/wan25` 随死参数 `output_dir` 一并删除（PL-20）。
- `video_official/pipeline_official_video.py`：`tmp/pipeline_eval/{model_id}.mp4` → `artifact_root_path()/"pipeline_eval"/...`。
- `bernini/pipeline_bernini.py`：同上。
- 验证：`rg` 确认 `test/test_vggt.py`、`test_stream/test_vggt_stream.py` 均自带 output_dir 传参；studio/评测路径均显式传 output_dir/output_path；两文件 `PYTHONPATH=. import` 通过。
- 风险：裸调不传 output 的用户产物落点从 CWD 变为 artifact 根（`WORLDFOUNDRY_ARTIFACT_DIR` 或默认缓存根）——这是本次修复的目标行为。
- 备注：报告未点名但同型的默认值还散布在 cut3r（`./cut3r_output`×4）、pi3（`./pi3_output`、`./loger_output`）、lingbot_map、infinite_vggt、vggt_omega、depth_anything（`./vis_depth`）、worldlabs（`./output/worldlabs_assets`）、voyager（`./output/hunyuan_world_voyager/...`）；修法与本条相同（`artifact_root_path()` + 延迟解析），因超出报告列名范围且量大，留待下一轮批量处理。

### [PL-15] skyseg.onnx 下载改显式 checkpoint 根
- 文件：`worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py`
- 改动：`skyseg.onnx` 的存在性检查与下载目标从进程 CWD 改为 `checkpoint_root_path("skyseg", "skyseg.onnx")`；下载日志改 logger。
- 验证：`py_compile`；`rg skyseg` 确认无其它 CWD 相对引用。
- 风险：已有用户 CWD 里的旧 `skyseg.onnx` 不再被复用，会在 checkpoint 根重新下载一次（一次性成本）；并发写竞态依旧存在（原子重命名带锁属 core 下载工具的改进，超出本层）。

### [PL-07] wan_2p5 失败路径 print → logging
- 文件：`worldfoundry/pipelines/wan/pipeline_wan_2p5.py`
- 改动：成功/失败分支的 3 处 print 改为 `logger.info`/`logger.error`（含无 response 对象的分支）；保留"失败返回 `video_url=None`"的现有语义。
- 说明：报告建议失败抛 `RuntimeError`——属对外行为变更（批量评测调用方可能依赖 None 语义），列入 Deferred。

### [PL-08] depth_anything v1/v2 批处理吞样本改为可观测跳过
- 文件：`pipeline_depth_anything_v1.py`、`pipeline_depth_anything_v2.py`
- 改动：图像分支 `print+continue` → `logger.warning(..., exc_info=True)` 后 continue；视频分支零输出的 `except Exception: continue` → `logger.warning("Skipping unreadable video %s", ..., exc_info=True)` 后 continue。跳过行为本身保留（严格模式抛异常属行为变更，见 Deferred 备注）。
- 验证：两文件 `py_compile` 通过。

### [PL-18] 实例状态跨调用泄漏（两处均修）
- `hunyuan_world/pipeline_hunyuan_mirror.py`：`__call__` 用 `try/finally` 保存并恢复 `self.output_path`，消除单次调用改写实例路径导致的跨调用泄漏。
- `videocrafter/base.py`：`__call__` 的 per-call 覆盖（`num_frames`→`frames`、`num_inference_steps`→`ddim_steps` 等）此前永久写进共享 `synthesis_model.runtime_kwargs` 与 `generator` 属性；现改为记录原值（含"原本不存在"哨兵）、`try/finally` 中恢复：generator 属性仅在 generator 对象未被 predict 内部重建时恢复（`is` 判等），`runtime_kwargs` 按键恢复/删除。保留块本身的原因：synthesis 层 `_prediction_runtime_overrides` 的别名表映射不到 videocrafter runtime 的 `ddim_steps` 构造参数，删除该块会静默丢弃 steps 覆盖。
- 验证：`py_compile`；用 FakeSynthesis/FakeGenerator 的 CPU smoke test 断言"调用内生效 + 调用后恢复"均通过（`OVERRIDE_RESTORE_OK`）。
- 残留：videocrafter 惰性路径下若 generator 在本次 predict 内部才首次构建，构建用的是覆盖后的 runtime_kwargs，恢复 runtime_kwargs 后该 generator 实例仍带覆盖值（下次无参调用继承）。彻底修复需 predict 后主动置 `generator=None` 强制重建（代价是每次覆盖后重载权重）或 synthesis 层支持 per-call 参数透传——后者超出本层，见 Deferred PL-01/PL-02。

### [PL-24 + PL-02 轮询循环 print] print/emoji → logging（18 个文件，交互式 REPL 除外）
- 已转换（全部 `logging.getLogger(__name__)` 模块级 logger）：
  - hosted-API 轮询循环状态打印（状态变化时 `logger.info`）：`kling/pipeline_kling_api.py`、`wan/pipeline_wan_2p6.py`、`wan/pipeline_wan_2p7.py`、`luma/pipeline_luma_ray2.py`、`minimax/pipeline_hailuo_2p3.py`、`runway/pipeline_runway_gen4p5.py`、`worldlabs/pipeline_worldlabs.py`；`sora/pipeline_sora2.py` 的 `sys.stdout.write` 进度条整体替换为状态变化日志（含进度百分比），删除 `sys` import 与进度条渲染代码。
  - 加载/推理信息：`hunyuan_world/pipeline_hunyuan_mirror.py`（14 处，emoji 移除）、`kling/pipeline_astra.py`（6 处）、`hunyuan_world/pipeline_hunyuan_world_voyager.py`（3 处，其中 video_length 自动调整改 `logger.warning`）、`lingbot_world/pipeline_lingbot_world.py`（2 处）、`matrix_game/pipeline_matrix_game_2.py`（2 处）、`hunyuan_world/pipeline_hunyuan_worldplay.py`（video_length 不一致改 `logger.warning`）、`pi3/pipeline_loger.py`（1 处）、`cut3r/pipeline_cut3r.py`（深度图跳过改 `logger.warning`）、`wan/pipeline_wan_2p5.py`（见 PL-07）、`depth_anything` v1/v2（见 PL-08）。
- 保留：`vggt/pipeline_vggt.py` 的 20 处 print 全部位于 `run_two_stage_3dgs_stream_cli` 交互式 REPL（与 `input()` 成对的用户界面输出），按报告"进度类输出仅在 CLI 层"的精神保留为 stdout。
- 验证：`rg -c "print\(" worldfoundry/pipelines/` 仅剩 vggt CLI 的 20 处；全部触及文件 `py_compile` 通过。
- 备注：voyager `:378` 的 `LOCAL_RANK` 检查现仅门控视频保存（分布式主进程写盘），不再涉及打印，属正当逻辑，保留。

### [PL-05 缓解] video_official/bernini 对 evaluation 的 import 移入 TYPE_CHECKING
- 文件：`video_official/pipeline_official_video.py`、`bernini/pipeline_bernini.py`
- 改动：`PipelineInvocation` 仅用作 `run_pipeline_invocation` 的类型注解（方法体只做属性访问），两文件均有 `from __future__ import annotations`，故移入 `if TYPE_CHECKING:` 后运行期不再拉起 evaluation 包。
- 验证：`PYTHONPATH=. python -c "import ..."` 两模块导入成功（此前会级联 import evaluation）。
- 说明：报告建议的"`PipelineInvocation` 下沉到 core"是跨包搬家，本轮禁止，见 Deferred。

### [PL-06 缓解] cut3r 对 studio 的 import 改为函数内惰性
- 文件：`cut3r/official_runtime.py`
- 改动：`depth_to_world_points` 的 import 从模块顶层移入唯一使用点 `build_rerun_recording`（该函数本就对缺 `rerun` 依赖优雅降级返回 None）；无 studio 的裁剪部署现在可以 import cut3r，只有真正构建 rerun 可视化时才需要 studio。
- 验证：`py_compile`；`rg` 确认模块内唯一使用点在函数体内。
- 说明：报告建议的"函数迁至 core 几何模块"是跨包搬家，见 Deferred。

### [PL-13] world_model `__init__.py` 移除跨家族 re-export
- 文件：`worldfoundry/pipelines/world_model/__init__.py`
- 改动：删除 `from ..dreamx_world import DreamXWorld5BARPipeline, DreamXWorld5BCamPipeline` 与 `__all__` 中对应两项。
- 验证：`rg` 全仓（test/、tests/、docs/、data bindings）确认所有调用方都从 `worldfoundry.pipelines.dreamx_world` 直接 import，无一经由 `world_model`；`import worldfoundry.pipelines.world_model` 通过（36 个导出）。

### [PL-25] 命名与文档卫生（低风险子项）
- `hunyuan_world/pipeline_hunyuan_mirror.py`：import 之后的游离字符串合并为真正的模块 docstring。
- `wan/pipeline_wan_2p5.py`：删除过时注释"这里面包含了sora2，veo3 和wan2.5"，模块 docstring 改为准确描述。
- `pi3/pipeline_loger.py`：模块 docstring 注明 LoGeR 是模型名而非 logging 工具（防误改）。
- `hunyuan_world/pipeline_hunyuan_worldplay.py`：删除 `save_pretrained` 空 stub（`pass` + 中文 TODO）；rg 全仓确认无调用方（`training/tuning/peft.py` 的 `save_pretrained` 针对 `application.model` 而非 pipeline）。
- mirror 的 `save_pretrained` 用 `torch.save` 序列化纯 dict：改 JSON 属磁盘格式变更，见 Deferred。
- mirror `save_results` docstring 标注 `save_colmap` 当前仅建目录、无 COLMAP 导出实现（配合 PL-20 的保留决定）。

## Deferred（风险大于收益 / 跨包 / 需要 GPU 端到端验证）

### [PL-01] `stream()` 三种实现语义互斥 → 契约统一为迭代器
- 原因：`PipelineABC.stream`、`NativeVisualDiffusionPipeline.stream` 与各手写 stream 的返回类型（值 vs 生成器）是对外契约，统一包装会改变全部调用方（studio realtime、test_stream/ 下 40+ 脚本）的消费方式，无法在无 GPU 环境回归。
- 建议方案：在 `pipeline_utils.py` 定义 `stream()` 必须返回 `Iterator[Any]` 的契约；基类提供 `_ensure_iterator()`（值→单元素生成器）；分三步走：先加 wrapper 不改子类（兼容期），再逐家族迁移，最后移除兼容层。验证：test_stream/ 全量 + studio realtime 手动冒烟。

### [PL-02] hosted-API 基类抽取（10 文件模板收敛）
- 原因：结构重构，涉及 `wan_2p5/2p6/2p7、kling、sora、veo、luma、minimax、runway、worldlabs` 的 `__init__/api_init/process/__call__/poll/extract/download` 全链路；行为细节各家已漂移（kling `_extract_video_url` 查 9 个候选键 vs wan_2p5 只查 1 个；失败有的抛有的返 None），强行统一必然改变部分家族对外行为，且无法对 10 个真实 API 做回归。
- 建议方案：新增 `worldfoundry/pipelines/api_pipeline.py` 定义 `HostedApiPipeline(PipelineABC)`：模板方法 `_submit/_poll(status_extractor)/_extract_url/_download`；子类只声明 `ENDPOINT_DEFAULT/_API_KEY_ENV/payload 组装`。迁移顺序：先 wan_2p6/2p7（同源最像）→ luma/minimax/runway → kling/sora/worldlabs（提取器差异大）→ wan_2p5（失败语义特殊）。每迁一家跑对应 `test/test_*.py` 手动 API 冒烟（需真实 key）。本轮已先把 8 份轮询循环的 print 统一为状态变化 logger（见上），减小后续 diff。

### [PL-03] API 包装 `__init__` 不调 `super().__init__()`
- 原因：`PipelineABC.__init__` 会设置 `model_id/memory_module/options/operators/device` 等状态并可能触发组件装配逻辑，给 10+ 个手写 `__init__` 补 `super().__init__(...)` 需要逐家核对参数映射与副作用（有的家族根本没有 memory/operator 概念），改错即运行期行为变化；当前靠 `_uses_component_contract()` 返回 False 是稳定的。
- 建议方案：与 PL-02 的基类抽取合并做（`HostedApiPipeline.__init__` 统一调 super 并显式声明属性），native 家族则在 PL-09 归拢时处理；过渡期可在 `PipelineABC.__init_subclass__` 加"缺关键属性时警告"的软校验。

### [PL-04] `_call_component_pipeline` 的 `processed["actions"]` 隐式四键契约
- 原因：加 TypedDict/显式校验属基类契约变更，会影响全部走组件契约的声明式 pipeline（~50 个类），校验过严可能把现在能跑的 operator 打断；收益是报错更早，风险是误伤。
- 建议方案：在 `pipeline_utils.py` 定义 `ProcessedInputs(TypedDict)`；`_call_component_pipeline` 在取键前做 `missing = {"prompt","images","video","actions"} - processed.keys()`，缺键时抛带 operator 类名与修复提示的 `TypeError`。CPU 单测可全覆盖（伪 operator），可在下一轮安全落地——留给有时间跑全量 tests/pipelines 的轮次。

### [PL-05] `PipelineInvocation` 下沉 core（跨包搬家）
- 原因：目标位置在 `core/`（或独立协议模块），本轮禁止跨包移动。已用 TYPE_CHECKING 消除运行期反向依赖（见已修复）。
- 建议方案：`worldfoundry/core/contracts/invocation.py` 新建 dataclass（字段：`prompt/image/video/output_path/pipeline_kwargs/operator_kwargs/request`）；`evaluation.models.pipelines.invocation` 改为 re-export 兼容别名；pipelines 侧 TYPE_CHECKING import 换到 core 路径；一个 deprecation 周期后删旧路径。验证：`rg PipelineInvocation` 全仓改点 + `pytest tests/pipelines test/eval_core`。

### [PL-06] `depth_to_world_points` 迁 core（跨包搬家）
- 原因：函数体在 `studio/visualization/core/geometry.py`，目标位置 `core/`；本轮禁止跨包移动。已改惰性 import 消除 import 时依赖（见已修复）。
- 建议方案：函数（纯 numpy 几何，无 studio 依赖）移至 `core/utils/geometry.py`（或 `core/io/geometry.py`），studio 原位置留 re-export；cut3r 改 import core 路径。验证：`rg depth_to_world_points` 全仓 + studio 可视化冒烟。

### [PL-07 残余] hosted-API 失败改抛异常
- 原因：`video_url=None` 的静默降级是当前对外契约，改抛 `RuntimeError` 会改变批量评测的失败流转方式（样本从"成功但空产物"变为"异常中断"），需要评测侧配合改错误处理后一起上。
- 建议方案：与 PL-02 基类合并：基类 `__call__` 提供 `on_failure="raise"|"none"` 参数，默认先 `"none"` 保持兼容，评测側切换后翻转默认值。

### [PL-09] 8 个 native 家族归拢 `NativeVisualDiffusionPipeline`
- 原因：大型结构重构（cosmos×4、gen3c、hunyuan_video、vchitect、t2v_turbo），涉及 dtype/offload 策略、`DiffusionRequest` 组装、负种子语义、checkpoint 角色探测的逐家差异梳理；无 GPU 无法验证任何一家的端到端产物一致性。
- 建议方案：以 wan2.1 声明式子类为模板：每家改为继承 `NativeVisualDiffusionPipeline` 并只声明 `MODEL_ID/OWNER/ACCEPTS_IMAGES/REQUEST_INPUT_DEFAULTS/_checkpoint_overrides` 钩子；先做 cosmos_predict2（报告已确认复刻最完整），diff 期间保留旧实现为 `_legacy` 引用对照；每家迁移后需 GPU 机器上跑对应 `test/test_*.py` 与固定 seed 的产物 hash 对照。8 份负种子习语随归拢自然消失。

### [PL-11] vggt 1012 行巨类拆分 + REPL 出库
- 原因：REPL 移到 cli 层、几何移到 representations/operators 均为跨包搬家；`_apply_interaction_to_camera` 对未知 token 从静默改抛 `ValueError` 是行为变更（现有脚本可能传大小写混合 token 依赖静默容忍）。
- 建议方案：交互 token→相机增量映射表迁 `operators/vggt_operator.py`；`run_two_stage_3dgs_stream_cli` 整体迁 `worldfoundry/cli/`（或 studio 的交互面），pipeline 保留纯函数式 `run_two_stage_3dgs_video`；未知 token 先 `logger.warning`（一个版本）再改抛错。验证：`test_stream/test_vggt_stream.py` + CLI 手动冒烟。

### [PL-12] `lyra_utils` 收敛 core
- 原因：`working_directory/load_pil_image/materialize_image_input/build_subprocess_env` 的规范位置在 core（跨包）；且 `synthesis/visual_generation/lyra_2/runtime.py` 反向 import 的修复必须同步改 synthesis（不许改）。lyra 版 `load_pil_image` 与 core 版行为已漂移（恰好一张 vs first_sequence_item），直接换会变行为。
- 建议方案：core 侧确认 `core/utils/image_utils.py` 的实现覆盖 lyra 语义（加 `require_single=True` 开关）；`lyra_utils` 改薄壳 re-export + DeprecationWarning；synthesis 反向 import 改指 core；最后 `lyra_utils` 只留权重布局探测。验证：`pytest tests/` 中 lyra/helios/video_official 相关 + lyra1/lyra2 GPU 冒烟。

### [PL-14] `lyra_utils.project_root()` 源码检出假设
- 原因：虽然文件在 pipelines/ 内可改，但 `prepare_lyra1_checkpoint_root` 的软链布局、`build_subprocess_env` 的 PYTHONPATH 注入都建立在该根路径上，换成 `checkpoint_root_path()` 后 lyra1/lyra2 子进程运行时的权重解析路径全变，无 GPU/权重环境无法验证不破坏现网源码部署。
- 建议方案：`project_root()` 改为：优先 `WORLDFOUNDRY_CACHE_DIR`/`checkpoint_root_path()`，找不到 `pyproject.toml` 时抛 `RuntimeError`（禁止 `parents[4]` 兜底静默指错）；`cache/runtime` 软链根迁 `core` cache API。需在有 lyra 权重的机器上跑 lyra1/lyra2 全流程后合入。

### [PL-17] 进程级全局状态修改（chdir/sys.path/env/torch 日志级别）
- 原因：`lyra_utils` 的 `os.chdir` 上下文/`sys.path.insert`/`NVTE_FUSED_ATTN` 与 vendored 官方 runtime 的工作方式深度耦合（官方代码内部用相对路径），直接去掉会破坏 lyra 子流程；matrix_game_2 压低 torch 日志级别是官方 runtime 噪声的权宜（去掉后评测日志被 dynamo 刷屏）。均为"改了必须 GPU 验证"项。
- 建议方案：lyra 的 chdir/sys.path 限制在子进程内（`build_subprocess_env` 已是正确方向——把污染留在 subprocess，主进程零污染，需审计现存主进程内 chdir 调用点并迁移）；matrix_game 日志压制改为可选参数并默认关闭、文档注明；`NVTE_FUSED_ATTN` 改为 subprocess env 注入而非主进程 `os.environ`。

### [PL-19] 子包 `__init__.py` 三种风格统一
- 原因：为 ~55 个目录补 re-export `__init__.py` 是大面积机械改动，影响 import 路径解析顺序与打包（`pyproject.toml` 的失效排除项清理也不在本轮许可范围）；收益是工具友好，风险是任何一个 re-export 引入意外的 import 时副作用。
- 建议方案：定规范（家族包一律显式 `__init__.py` + `__all__` re-export 公开 pipeline 类；惰性家族用 `__getattr__` 延迟）；脚本化生成 + `python -c "import worldfoundry.pipelines.<pkg>"` 全量冒烟；同步清理 pyproject 排除项。

### [PL-20 残余] mirror `save_colmap` 参数
- 原因：rg 证实是 studio catalog、`test/test_hunyuan_mirror.py`、synthesis 侧共同引用的活契约；真正实现 COLMAP 导出或删参数都超出零风险范围。
- 现状：已在 `save_results` docstring 标注"当前仅创建 `sparse/0` 目录、无 COLMAP 导出"。
- 建议方案：或实现（用 representation 的相机参数写 `cameras.bin/images.bin`），或走 deprecation：参数保留一版本、传入时 `logger.warning`，catalog 同步移除后删除。

### [PL-21] 统一 `unload()`/realtime 协议
- 原因：`PipelineABC` 加方法是全局契约扩展；realtime 四件套（matrix_game_2/3、worldplay、helios）抽 Protocol 涉及多家族同步改动，显存释放行为无 GPU 无法验证。
- 建议方案：`PipelineABC.unload()` 默认实现（drop `synthesis_model/operator/memory_module` 引用 + `torch.cuda.empty_cache()`，子类覆写补充）；`worldfoundry/pipelines/realtime_protocol.py` 定义 `RealtimeSessionProtocol`（`prepare/configure/stream/reset` 四方法签名），四家族声明实现；studio 切换模型处接 `unload()`。GPU 机器上用 `nvidia-smi` 验证释放。

### [PL-22] pipeline 层直接触碰模型组件缺 `inference_mode` 保护
- 原因：matrix_game_2 的 `vae.encode`/`clip.encode_video` 包 `torch.inference_mode()` 理论无害，但 inference_mode 张量禁止后续 in-place/requires_grad 操作，若下游（官方 runtime 内部）有此类操作会当场报错——无 GPU 跑不了该链路，无法排除。
- 建议方案：GPU 机器上给 `pipeline_matrix_game_2.py:106-111` 加 `torch.inference_mode()` 后跑 `test/test_matrix_game_2.py` 与 realtime 流各一轮；mirror 的逐帧 GPU→CPU 落盘改批量 `.cpu()` 后循环写盘（同样需实测显存/速度）。

### [PL-25 残余] mirror `save_pretrained` 的 `torch.save` 配置改 JSON
- 原因：磁盘产物格式变更（`pipeline_config.pt` → json），仓内无读取方但无法排除外部依赖该格式。
- 建议方案：写 `pipeline_config.json` 的同时保留读侧兼容（如未来加 `from_pretrained` 读取逻辑时两格式都认）；或确认无外部消费者后直接切换。

## 验证汇总

- **编译**：`python -m compileall -q worldfoundry/pipelines/` 全树通过（exit 0）；每个改动文件单独 `py_compile` 通过。
- **导入**：`worldfoundry.pipelines.video_official.pipeline_official_video`、`worldfoundry.pipelines.bernini.pipeline_bernini`（不再拉起 evaluation）、`worldfoundry.pipelines.world_model`（36 导出）均 import 成功；`vggt` 链在缺 `huggingface_hub` 处按预期 ImportError（环境无该包，记录备案）。
- **引用核查**：所有删除符号（`fetchPly`/`gaussian_render`/`tensor_to_pil`/`save_content`/`output_dir`/`camera_trajectory`/worldplay `save_pretrained`/world_model 的 DreamX re-export）均 `rg` 全仓（含 test/、tests/、docs/、data/ bindings、字符串引用）零残留调用方。
- **语义 smoke test**：videocrafter 覆盖恢复逻辑用无依赖 stub 验证"调用内生效 + 调用后恢复"（PASS）。
- **pytest（CPU，时间盒 3 分钟/批）**：
  - `tests/synthesis/test_hunyuan_worldplay_realtime.py` + `tests/synthesis/test_lingbot_world_realtime.py`：7 passed, 1 skipped（GPU skip）。
  - `tests/pipelines/test_conditioning_path_forwarding.py`：passed。
  - `test/eval_core/test_dreamx_world_dispatch.py`：1 failed——断言 catalog 元数据 `runtime_status == "ready"` 而数据文件值为 `native_checkpoint_runtime_a100_workspace_validated`，属 data/ 目录元数据与测试期望不一致，与本轮改动无关（本轮未触碰任何 data 文件；该测试 import 路径也不经过 world_model re-export）。
  - `test/eval_core/test_model_runtime_layering.py`：10 passed, 2 failed——失败均为 `load_runtime_environment_profiles() got an unexpected keyword argument 'legacy_root'`，测试与 evaluation 层 API 签名不匹配的存量问题，与本轮改动无关。
  - `test/test_*.py`（kling/wan/luma 等）经检查是需要真实 API key 的示例脚本（`main()` 形式，非 pytest 用例），跳过并记录。
