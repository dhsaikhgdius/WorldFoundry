# synthesis/representations 树集成卫生修复日志

> 修复人：vendored-integration 修复 agent（synthesis/representations 分册）；日期：2026-08-14
> 对应评审报告：`plan/code_review/11_vendored_integration.md`（负责 [VI-18]、[VI-23] 的 representations 子集、[VI-3]、[VI-4] 的 synthesis 子集、[VI-21] 第 1 部分）
> 约束：只改 `worldfoundry/synthesis/` 与 `worldfoundry/representations/`；不触碰 pyproject.toml / MANIFEST.in / core / evaluation / pipelines / tests / docs（另一并发 session 在修其它模块，编辑前逐一 `stat -c '%y'` 核对 mtime，全部为 2026-07 旧文件，无并发冲突）。
> 验证手段：`PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile`（12 个改动文件全部通过）；可达依赖链的模块做真实 `import` 测试；AST 断言（monkey-patch 不再在模块顶层执行）；`rg` 全仓引用核查改动符号的所有调用方；临时 .pth 文件对中央加载器安全/回退双路径做真实 round-trip。环境限制：系统 python3 有 torch 2.7，无 diffusers/accelerate/transformers/huggingface_hub/timm，pypi 不可用，故 framepack/depth_anything 等模块只能做 py_compile + AST 级验证（详见各条）。

## 已修复

### [VI-18] P1 framepack 在 import 时全局覆写 `torch.nn.LayerNorm.forward`、diffusers 归一化类与 accelerate
- 文件：`worldfoundry/synthesis/visual_generation/framepack/framepack_runtime/diffusers_helper/model.py`、`.../framepack_runtime/inference.py`
- 改动：
  - model.py：原第 23-69 行 6 处模块顶层全局赋值（`accelerate.accelerator.convert_outputs_to_fp32`、diffusers `LayerNorm.forward`、`torch.nn.LayerNorm.forward`、`FP32LayerNorm.forward`、`RMSNorm.forward`、diffusers `AdaLayerNormContinuous.forward`）全部移入显式函数 `enable_framepack_global_patches()`，用模块级 `_GLOBAL_PATCHES_APPLIED` 标志保证幂等；补丁函数体逐字节保留。函数 docstring 明确声明进程全局影响面与"有意不可逆"（上游推理栈从构造到采样都假设该语义，中途还原会让已构造模块状态不一致；不可接受污染时应独立进程运行 framepack）。全部改动带 `# Modified by WorldFoundry:` 标记。
  - 关键细节：上游从 diffusers import 的 `AdaLayerNormContinuous` 在文件后部（原 541 行）被同名本地类遮蔽；补丁若延迟到调用期执行、按模块全局名解析会错打到本地类。改为 import 别名 `_DiffusersAdaLayerNormContinuous` 捕获 diffusers 原类，函数体内对别名打补丁，保证与原顶层执行语义完全一致。
  - 调用点（`rg` 确认 `diffusers_helper.model` 全仓唯一 import 方是 `framepack_runtime/inference.py:29`）：(a) inference.py（官方 runtime 入口，由 `worldfoundry_runner.py` 文本替换后 exec，本身即"运行 framepack"的动作）在 import 块后、任何模型构造前显式调用；(b) `HunyuanVideoTransformer3DModelPacked.__init__` 顶部再调用一次（模型构造期兜底，幂等无副作用），保证任何直接构造该模型的路径行为与上游一致。
  - `worldfoundry_runner.py` 未改；核对其 `_patched_script` 的 3 个文本替换锚点（HF_HOME 正则、`from_pretrained('lllyasviel/FramePackI2V_HY'`、`outputs_folder = './outputs/'`）在改后 inference.py 中全部仍命中。
- 验证：两文件 `py_compile` 通过；AST 断言脚本确认模块顶层零 `*.forward =`/`convert_outputs_to_fp32 =` 赋值、6 处赋值全部位于 `enable_framepack_global_patches` 函数体内且函数带 docstring。任务书要求的运行时断言 `import 后 torch.nn.LayerNorm.forward is ref` 因本环境无 diffusers/accelerate（`python3 -c "import diffusers"` 失败）无法执行，以"顶层无补丁赋值"的 AST 证据替代——模块 import 只执行顶层语句，顶层无赋值即不可能改写 torch 全局。
- 风险：`import inference`（而非 `import diffusers_helper.model`）仍会激活补丁——但 inference.py 是在模块顶层加载全套模型权重的官方入口脚本，import 它等价于启动 framepack runtime，属预期行为并已在注释中说明。补丁本身语义未变，framepack 实际运行路径行为与修复前逐字节一致。

### [VI-23] representations 子集：4 个 wrapper 裸 `torch.load(..., weights_only=False)` 收敛到中央安全加载器
- 文件：`worldfoundry/representations/depth_generation/depth_anything/depth_anything_v2_representation.py`（原 183 行）、`depth_anything_v1_representation.py`（原 106 行）、`point_clouds_generation/flash_world/flash_world_representation.py`（原 404 行）、`point_clouds_generation/lingbot_map/lingbot_map_representation.py`（原 242 行）
- 改动：先核对既有范式（`pi3/loger_representation.py:144`、`vggt/infinite_vggt_representation.py:83` 均 `from worldfoundry.core.model_loading import load_torch_checkpoint`）与中央加载器真实签名（`core/model_loading/file.py:173`：`load_torch_checkpoint(path, *, map_location="cpu", weights_only=True, allow_unsafe_pickle_fallback=False, **kwargs)`，回退仅在 weights-only UnpicklingError 时触发）。4 个文件逐一确认加载对象均**纯作 state_dict 消费**（v1/v2 解 `model`/`state_dict` 外层后 `load_state_dict`；flash_world 取 `["transformer"]`/`["recon_decoder"]` 两个子 state_dict；lingbot_map 取 `.get("model", checkpoint)`），故统一改为 `load_torch_checkpoint(..., weights_only=True, allow_unsafe_pickle_fallback=True)`：默认安全反序列化，仅对安全路径失败的旧式包装 checkpoint（如带 argparse.Namespace 的训练存档）显式回退，与修复前无兼容性回归、且严格优于无条件 `weights_only=False`。连带删除 3 处仅为压 `weights_only` FutureWarning 而存在的 `warnings.catch_warnings` 包裹及 3 个因此不再使用的 `import warnings`（representations 为仓库自写 wrapper 层，非 vendored 上游文件）。map_location 保持各自原值（lingbot_map 为 `device or "cpu"`）。
- 验证：4 文件 `py_compile` 通过；`rg` 确认 4 文件内零残留 `torch.load(`/`warnings`；lingbot_map 依赖链完整、真实 `import` 通过；其余 3 个 import 仅因环境缺 `huggingface_hub`（修复前就在模块顶层 import 的既有第三方依赖）失败，与本改动无关。另用临时 .pth 做真实 round-trip：纯 tensor dict 走安全路径成功、带 Namespace 的包装 checkpoint 触发回退后成功，证实所选参数组合在 torch 2.7 下行为符合预期。
- 风险：恶意构造"故意让 weights-only 失败"的文件仍可经回退执行 pickle——与修复前的无条件 unsafe 等价、不新增攻击面，且回退现在显式、可 grep、可由中央加载器统一收紧（正是报告建议的收敛点）。

### [VI-3] P2 wonderjourney/wonderworld 用 `parents[4]`/`parents[5]` 数目录层级伸手 base_models
- 文件：`worldfoundry/synthesis/visual_generation/wonderjourney/wonderjourney_runtime/run.py`（原 18-20 行）、`wonderworld/wonderworld_runtime/util/midas_utils.py`（原 4-8 行）
- 改动：删除两处 `sys.path.insert(0, ...depth 目录)`（[VI-2] 型强制置顶抢占）与裸 `from midas... import`，改为包内绝对 import：`from worldfoundry.base_models.three_dimensions.depth.midas.model_loader import load_model` / `...midas.transforms import NormalizeImage`；连带删除各自不再使用的 `import sys`（及 midas_utils.py 的 `from pathlib import Path`）。带 `# Modified by WorldFoundry:` 说明。
- 验证：先 `ls` 确认 `depth/midas/` 含 model_loader.py/transforms.py；`__init__.py` 链检查发现 `midas/` 缺 `__init__.py`（PEP 420 命名空间包，[VI-7] 已记录），按任务书要求实测：`python3 -c "import worldfoundry.base_models.three_dimensions.depth.midas.model_loader"` 解析成功（报错止于缺 `timm` 第三方依赖——旧 sys.path 方式同样会在此失败，证明包路径解析本身可用）；`transforms` 依赖链完整、绝对 import 完全成功。`rg` 确认 midas 内部全用相对 import（`from .dpt_depth import ...`），无裸 `import midas` 残留；两个 runtime 全树无其它文件依赖被删的 depth sys.path 条目（depth/ 下其它顶层名如 unidepth/moge 零引用）。两文件 `py_compile` 通过。
- 风险：midas 是命名空间包，若未来打包策略改用 classic `find_packages`，wheel 会缺失该目录——这是 [VI-7]/[VI-17] 的既有问题，本修复不加剧（反而消除了对磁盘相对布局的隐式契约）。

### [VI-4] P2 synthesis 子集：CWD 相对 `sys.path.append('.'|'..')` 清理
- 文件与逐条处置：
  - `ac3d/ac3d_runtime/inference/cli_demo_camera.py`（原第 3 行 `sys.path.append('..')`）：该脚本需要 `ac3d_runtime/` 在 sys.path 上（裸 import `cogvideo_controlnet`/`controlnet_pipeline`/`inference.*`，`ls` 确认均在 ac3d_runtime 根下）。wrapper 路径（`ac3d/runtime.py` 的 `_run_official` + 子进程 PYTHONPATH）已保证该目录在 path 上，`'..'` 只会按 CWD 注入垃圾路径。改为锚定 `Path(__file__).resolve().parents[1]`（= ac3d_runtime 根）的**非抢占 append + 去重守卫**，任意 CWD 下脚本模式（文件尾部有 argparse `__main__` 块）继续可用。
  - `open_sora_plan/open_sora_plan_runtime/opensora/models/frame_interpolation/interpolation.py`（原第 14 行 `sys.path.append('.')`）：**证实完全不必要后直接删除**——该文件其余 import 全部是 `worldfoundry.base_models...amt` 包内绝对路径，唯一的动态加载 `build_from_cfg`（`amt/utils/build_utils.py`，只读核查）已把 `networks.*` 规范化为 worldfoundry 绝对前缀；连带删除因此不再使用的 `import sys`。
  - `open_sora_plan/.../opensora/sample/rec_image.py`（原第 2 行 `sys.path.append(".")`）：脚本裸 import `opensora`，`'.'` 仅当 CWD 恰为 open_sora_plan_runtime 时有效。改为锚定 `parents[2]`（= open_sora_plan_runtime 根）的非抢占 append + 去重守卫。
  - 三处均带 `# Modified by WorldFoundry:` 说明。
- 验证：三文件 `py_compile` 通过；脚本实测两处锚定路径分别解析到 `.../ac3d/ac3d_runtime`（含 cogvideo_controlnet.py 与 inference/__init__.py）与 `.../open_sora_plan_runtime`（含 opensora/）；`rg` 确认 interpolation.py/rec_image.py 无 in-repo import 方（纯脚本模式工具），改动不影响任何调用链。
- 风险：rec_image/cli_demo_camera 仍向 sys.path 追加一条目录（脚本模式运行所必需），但已从"CWD 时序函数"变为锚定绝对路径、append 末尾不抢占、重复时跳过，[VI-14] 型顶层名冲突面不因本改动扩大。

### [VI-21] 第 1 部分 P2 `CredentialedSynthesis` 游离于 `BaseSynthesis` ABC 之外
- 文件：`worldfoundry/synthesis/visual_generation/api_video_client.py`
- 改动：`CredentialedSynthesis` 改为继承 `BaseSynthesis`（`from ..base_synthesis import BaseSynthesis`，无循环 import：base_synthesis 仅依赖可选 torch），`__init__` 补 `super().__init__()`（基类为空 pass，行为不变）。契约映射写入类 docstring：`api_init` 保留既有 **classmethod 工厂**语义（覆盖基类实例方法声明；`rg` 确认全部 10 个 pipeline 调用点均为 `Vendor.api_init(endpoint=..., api_key=..., logger=..., **kwargs)` 的类上关键字调用，语义不受影响，工厂 docstring 补充了与基类的差异说明）；`from_pretrained` 继承基类 `NotImplementedError`（API 模型无本地权重可载，语义正确）；`predict` 由各 vendor 适配器实现（rg 核实 kling/luma/minimax/runway/sora/veo 均自带 `predict`）。可行性依据：`BaseSynthesis` 虽为 ABC 但无任何 `@abstractmethod`，继承不引入实例化约束；全部 9 个子类（含 ApiVideoSynthesis/OpenAiVideoSynthesis/DashScope 中间层）均单继承，ABCMeta 元类传播无冲突。
- 验证：`py_compile` 通过；python3 运行类层次检查——`issubclass` 断言 `CredentialedSynthesis`/`ApiVideoSynthesis`/`OpenAiVideoSynthesis` 及 KlingApiSynthesis、LumaRay2Synthesis、Hailuo2p3Synthesis、RunwayGen4p5Synthesis、Sora2Synthesis、Veo3Synthesis、Wan2p5Synthesis、WorldLabsSynthesis、DashScopeVideoSynthesis 共 12 类全部进入 `BaseSynthesis` 谱系；`ApiVideoSynthesis.api_init(endpoint=..., api_key=..., bogus_option=1)` 工厂 + 未知参数过滤行为保持、返回实例通过 `isinstance(x, BaseSynthesis)`；消费方 `worldfoundry.pipelines.kling.pipeline_kling_api`、`...wan.pipeline_wan_2p5` 真实 import 通过（只读验证，未改 pipelines）。全仓 `rg` 确认当前无 `isinstance(..., BaseSynthesis)` 调用点，即本改动无既有行为可破坏、纯粹使未来类型分派成立。
- 风险：`api_init` 的 classmethod-工厂 vs 基类实例方法的签名语义差异是**既有设计**（改成实例方法需同步改 10 个 pipeline 调用点，超出本轮 scope），已在两处 docstring 显式记录；若未来有代码按基类契约 `instance.api_init(api_key, endpoint)` 位置传参，会得到参数错位的新实例——与修复前行为相同，非本次引入。

## Deferred（评估后不在本轮处理）

- **[VI-18] 残余**：`allegro/allegro_runtime/allegro/utils/adaptor.py:23` 也有 `nn.LayerNorm.forward` 覆写（函数内、非 import 时执行），报告未点名 framepack 以外条目，且风险等级不同（需调用才生效），留给后续专项。
- **[VI-23] 残余**：报告点名的 splatt3r/unik3d/unidepth/cut3r/wan resident 等 8+ 处在 `base_models/`、`diffusion_model/` 树内，超出本 session 的目录权限（另一并发 agent 负责其它模块），未动。
- **[VI-21] 第 2 部分**（`ActionModelSynthesis` 反向依赖 evaluation 层的分层倒置）：修复需下沉 `RuntimeProfileSynthesis` 到 synthesis/core 并改 evaluation 侧注册，涉及 `worldfoundry/evaluation/`（禁改目录），按任务分工明确不在本单元范围。
- **framepack 运行时端到端回归**：环境无 GPU/diffusers/accelerate/transformers，无法跑 framepack 官方推理确认数值路径逐字节一致；已用"补丁体零改动 + 入口处显式激活 + AST 顶层零赋值"三重静态证据覆盖，建议有 GPU 环境后跑一次 framepack smoke 视频生成做最终确认。
