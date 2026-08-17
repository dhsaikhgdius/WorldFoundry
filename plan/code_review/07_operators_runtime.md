# operators 与 runtime 评审

> 状态：已完成。评审人：infra 专项 agent（operators/ + runtime/）。

## 评审范围与方法

- **范围**：`worldfoundry/operators/`（85 个 `<model>_operator.py` + `base_operator.py` + `_media.py` + `__init__.py`，约 12.9k 行）；`worldfoundry/runtime/`（14 个顶层文件 + `platforms/` 5 个文件，约 5.6k 行）。
- **方法**：
  - 精读：`base_operator.py`、`_media.py`、`operators/__init__.py`，以及 runtime/ 全部 19 个文件（含 platforms/ 子包）。
  - 精读/实读代表性 operator 15 个：API 型（kling_api、sora2、veo3、worldlabs、wan_2p5 + 与 wan_2p6/wan_2p7 逐行 diff）、本地 GPU 型（wan_2p2、matrix_game_2、flash_world）、3D 型（vggt、部分 neoverse/worldfm）、VLA 型（openvla、gr00t、embodied_action）、通用型（runtime_video、world_model_runtime）。
  - 其余 operator 用 `rg` 扫描共性反模式（BGR 启发式、三件套克隆、异常处理、print、设备/路径硬编码、跨层 import、网络/subprocess/env var 使用）并对命中处抽查确认；所有计数（"11 份""5 处"等）均来自 rg 全目录统计。
  - runtime 关键 API（`execute_in_tree`、`AsyncCommandJobStore`、`CudaDeviceLeasePool`、`stage_checkpoint_for_realtime`、`run_bounded_command`）用 rg 统计了全库调用方，以评估影响面。
- **对照文档**：`docs/fumadocs/content/docs/maintainers/architecture/runtime-assembly.mdx`、`model-runtime.mdx`。文档声称 `BaseOperator` 只负责 "input and interaction shaping"，operator 负责 "validation, image/video loading, camera/action parsing, interaction shaping, perception preprocessing"，且 "Provider credentials belong in environment variables only"。

## 发现（按主题分组）

### 主题 A：base_operator 契约设计

#### [OR-01] P1 BaseOperator 契约几乎为空，核心状态字段是"只写不读"的仪式代码

- **位置**：`worldfoundry/operators/base_operator.py:4-61`
- **证据**：基类全部内容只有 62 行，核心是 4 个 `pass` 方法和 3 个 list 字段：

```26:49:worldfoundry/operators/base_operator.py
    def get_interaction(self, interaction):
        """
        utilize this function to update the interaction list
        """
        pass

    def check_interaction(self, interaction):
        ...
        pass
```

  全库检索确认：`operation_types` 在 operators/ 包外使用次数为 **0**，`interaction_template` 包外使用次数为 **0**，`interaction_history` 仅 2 处（rg 全库统计）。而 85 个 operator 的 `__init__` 都在忠实地设置这三个字段。
- **问题**：
  1. 基类没有定义任何真正被下游消费的抽象：没有 `process_prompt`/`process_perception` 的签名约定（`process_perception` 在基类里甚至不接受参数，而所有子类都带参数重定义，签名互不兼容：kling 是 `images=None, image_field=None`，veo3 是 `prompt, *, images=None, ...`，vggt 是 `input_signal`，matrix_game_2 是 `input_image, num_output_frames, resize_H, ...`）。pipelines 里 198 处调用 `process_*` 全靠"记得每个 operator 自己的参数拼写"。
  2. `operation_types`/`interaction_template` 是纯装饰：设置它们没有任何运行时效果，`interaction_template_init()` 只检查"是不是 list"（`base_operator.py:21-24`）。
  3. 没有生命周期定义：无 `load/unload/close`，无资源语义（对纯输入整形层可接受，但文档没有明说 operator 必须无状态，实际上 `current_interaction` 就是跨调用可变状态）。
- **影响**：类型检查器/IDE 无法对 operator 调用做任何检查；新增模型时作者只能拷贝一个旧 operator 改名（见 OR-05 的克隆漂移证据）；"契约"退化为命名习惯。
- **建议**：把 `process_prompt/process_perception/process_interaction` 定义为带 `**kwargs` 的显式抽象接口并规定返回 dict 的最小键集合（如 `images/video/extra_inputs`）；删除或真正启用 `operation_types`；为交互式模型单独定义 `InteractiveOperatorMixin`（`get_interaction/check_interaction/delete_last_interaction` 只对交互式模型有意义）。可参考 `EmbodiedActionOperator`（`embodied_action_operator.py:63-156`）——VLA 家族已经自发长出了一个像样的中间基类，说明基类缺位。

#### [OR-02] P2 `delete_last_interaction` 语义在基类与子类间相互矛盾

- **位置**：`worldfoundry/operators/base_operator.py:59-61` vs `flash_world_operator.py:526-531`、`vggt_operator.py:192-197` 等 10 个文件
- **证据**：基类版本静默容忍空列表：

```59:61:worldfoundry/operators/base_operator.py
    def delete_last_interaction(self):
        """Remove the last recorded interaction from the current list."""
        self.current_interaction = self.current_interaction[:-1]
```

  而 10 个子类重写为空列表时抛 `ValueError`（rg `def delete_last_interaction` 计 10 处），例如：

```526:531:worldfoundry/operators/flash_world_operator.py
    def delete_last_interaction(self):
        """Delete the last interaction from current_interaction list."""
        if len(self.current_interaction) > 0:
            self.current_interaction = self.current_interaction[:-1]
        else:
            raise ValueError("No interaction to delete.")
```

- **问题**：同名方法在不同 operator 上行为不同（静默 no-op vs 抛异常），调用方（如 `pipelines/vmem/pipeline_vmem.py:79`）无法写出统一的回退逻辑。
- **影响**：多轮交互回退在部分模型上崩溃、部分模型上静默，行为不可预测。
- **建议**：在基类中确定一种语义（建议抛异常，快速失败），删除全部子类重写。

### 主题 B：85 个 operator 的复制粘贴漂移

#### [OR-03] P0 BGR/RGB 通道均值猜测启发式会静默破坏红色主导图像的数据，且复制了 5 份

- **位置**：`worldfoundry/operators/flash_world_operator.py:500-503`、`vggt_operator.py:88-91`、`cut3r_operator.py`、`depth_anything_operator.py`、`depth_anything_v3_operator.py`（rg `mean\(\) > .*mean\(\)` 命中 5 个文件）
- **证据**：

```500:503:worldfoundry/operators/flash_world_operator.py
            # Convert BGR to RGB if needed
            if len(input_signal.shape) == 3 and input_signal.shape[2] == 3:
                if input_signal[..., 0].mean() > input_signal[..., 2].mean():
                    input_signal = input_signal[..., ::-1]
```

```88:91:worldfoundry/operators/vggt_operator.py
        elif isinstance(input_signal, np.ndarray):
            image_rgb = input_signal / 255.0 if input_signal.max() > 1.0 else input_signal
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                if image_rgb[..., 0].mean() > image_rgb[..., 2].mean():
                    image_rgb = image_rgb[..., ::-1]
```

- **问题**：用"红通道均值 > 蓝通道均值"猜测输入是 BGR 并翻转通道。任何正常的暖色调 RGB 图（日落、人脸特写、红色物体）都会被静默翻转成蓝色调；反之冷色调的 BGR 图不会被纠正。这是不可判定问题上的猜测式修复，且作为评测框架的输入路径，会直接改变被评测模型看到的数据。
- **影响**：评测输入被静默破坏 → 分数不可信且难以排查（没有任何日志）；5 份拷贝意味着修一处漏四处。
- **建议**：删除启发式。ndarray 输入统一约定为 RGB（文档化），或要求调用方显式传 `channel_order="bgr"`。至少要在翻转时打 warning。

#### [OR-04] P2 `input_signal.max()>1.0` 归一化猜测：对全黑/低亮度 uint8 图判断错误

- **位置**：`worldfoundry/operators/vggt_operator.py:84-88`（tensor 与 ndarray 两个分支）、flash_world/cut3r/depth_anything 同款
- **证据**：

```84:86:worldfoundry/operators/vggt_operator.py
            if image_rgb.max() > 1.0:
                image_rgb = image_rgb / 255.0
            return image_rgb
```

- **问题**：一张全部像素 ≤1 的 uint8 图（接近全黑的真实图像，合法输入）会被判为"已归一化"而跳过 /255，输出值域错误。与 OR-03 同属"用数据内容猜数据格式"。
- **影响**：暗图输入时深度/重建结果错误且无告警。
- **建议**：按 dtype 判断（`np.issubdtype(arr.dtype, np.integer)` → /255），不要按值域猜。

#### [OR-05] P1 API 型 operator 是 11 份逐字克隆，而共享基类其实已经存在却未被使用

- **位置**：`sora2_operator.py:53-72`、`wan_2p5_operator.py:46-65`、`kling_api_operator.py:35-54`、`veo3_operator.py:126-145`、`hailuo_2p3_operator.py`、`luma_ray2_operator.py`、`runway_gen4p5_operator.py` 等
- **证据**：`rg '"processed_prompt": now_interaction'` 命中 11 个文件；`rg 'Interaction must be a string, got'` 命中 11 个文件。`diff wan_2p5_operator.py sora2_operator.py` 除类名/docstring 外主体一致。三件套的典型形态：

```53:72:worldfoundry/operators/sora2_operator.py
    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction
        }
```

  且已出现漂移：`worldlabs_operator.py:49-56` 的同名方法把"空交互抛异常"改成了"返回空串"，`check_interaction` 接受 None——语义分叉没有任何文档说明。克隆文件全名单（rg 命中）：sora2、veo3、kling_api、wan_2p5、wan_2p6、wan_2p7、hailuo_2p3、luma_ray2、runway_gen4p5、worldlabs、wow。
  关键是：**承载这个三件套的共享基类已经存在**——`runtime_video_operator.py:10-37` 的 `RuntimeVideoOperator`（docstring 自称 "Shared prompt/image operator for individually wrapped local video models"）实现了完全一样的逻辑，但 11 个 API operator 没有一个复用它。
- **问题**：这正是基类/中间基类该提供的默认实现。11 份拷贝已经开始各自演化，而现成的共享实现闲置。
- **影响**：行为漂移不可审计；修 bug（例如给 `process_interaction` 加历史长度上限）要改 11 个文件。
- **建议**：将 11 个 API operator 改为继承 `RuntimeVideoOperator`（或提取 `TextPromptOperator`），只保留 payload 构造差异。同理 `world_model_runtime_operator.py:31-42` 的 `_actions` 与 `embodied_action_operator.py:22-30` 的 `_as_action_list` 也是一对重复实现，应合并。

#### [OR-06] P2 图像/视频加载逻辑在 ≥13 个 operator 各写一份，而 core 已有现成实现

- **位置**：`rg 'Image\.open'` 命中 13 个 operator 文件；视频帧加载重复 5 份：`depth_anything_v3_operator.py:221`、`neoverse_operator.py:138`、`recammaster_operator.py:79`、`infinite_world_operator.py:188`、`pi3_operator.py:91`
- **证据**：`wow_operator.py:9-11` 表明 core 层已有统一实现且部分 operator 已在用：

```9:11:worldfoundry/operators/wow_operator.py
from worldfoundry.core.io.media import VIDEO_EXTENSIONS
from worldfoundry.core.io.video import load_frames_from_video
from worldfoundry.core.utils import load_pil_image
```

  但 `lyra1_operator.py:8` 却从 pipelines 层拿同名函数：`from ..pipelines.lyra.lyra_utils import load_pil_image`；`flash_world_operator.py:444-524` 又手写了一遍 80 行的"路径/base64/bytes/ndarray/tensor → PIL"多态加载；`neoverse_operator.py:94-121` 手写 `_to_pil_image`；`wan_2p2_operator.py:12-21` 手写 `_load_input_image`。
- **问题**：同一能力有 4+ 个实现源（core.utils、pipelines.lyra.lyra_utils、各 operator 手写），行为细节不一致（是否支持 base64、是否取视频首帧、BGR 处理，见 OR-03）。
- **影响**：输入兼容性因模型而异；`_media.py` 名义上是"Shared media helpers"（架构文档原话）但只有 4 个 PNG 编码函数，真正被复制的加载逻辑没有下沉。
- **建议**：把"任意输入 → PIL/ndarray 帧序列"下沉到 `_media.py`（或直接复用 `core.io`），operator 只保留模型特有的 resize/crop/归一化。

#### [OR-07] P2 operators 反向依赖 pipelines/synthesis，违反架构文档声明的装配方向

- **位置**：`lyra1_operator.py:8`、`lyra_operator.py:12`（← pipelines）；`yume_operator.py:13`、`hunyuan_worldplay_operator.py:9-14`（← synthesis，模块级）；`neoverse_operator.py:147`（← synthesis，函数级）
- **证据**：

```13:13:worldfoundry/operators/yume_operator.py
from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume import YUME_SIZE_CONFIGS
```

```8:8:worldfoundry/operators/lyra1_operator.py
from ..pipelines.lyra.lyra_utils import load_pil_image
```

  架构文档（runtime-assembly.mdx:12-20）声明链路是 `PipelineABC -> BaseOperator -> BaseSynthesis`，即 pipeline 依赖 operator、operator 不应回头依赖 pipeline/synthesis。
- **问题**：operator ← pipelines 是循环依赖的温床（pipelines 本来就 import operators）；operator ← synthesis 使"轻量输入整形层"在 import 时拉起模型运行时栈（yume 是模块级 import）。
- **影响**：懒加载 `__getattr__` 的收益被抵消——touch 一个 yume operator 就要 import synthesis 子树；分层文档与实际不符。
- **建议**：把 `YUME_SIZE_CONFIGS`、`load_pil_image`、pose_utils 这类被两层共享的常量/工具移到 `core` 或 `data/models` 配置，恢复单向依赖。

#### [OR-08] P2 `embodied_action_operator.py` 尾部维护了第二份 operator 注册表，与 `__init__.py` 重复

- **位置**：`worldfoundry/operators/embodied_action_operator.py:159-205` vs `worldfoundry/operators/__init__.py:9-102`
- **证据**：

```159:166:worldfoundry/operators/embodied_action_operator.py
_OPERATOR_EXPORTS = {
    "ACTOperator": ".act_operator",
    "BeingH05Operator": ".being_h05_operator",
    "CogACTOperator": ".vla_native_operator",
    "DiffusionPolicyOperator": ".diffusion_policy_operator",
    "DreamZeroOperator": ".dreamzero_operator",
    "DBCogACTOperator": ".vla_native_operator",
    "GigaBrain0Operator": ".giga_brain_0_operator",
```

  同样的 22 条 类名→模块 映射在 `operators/__init__.py` 里也各有一份（"Lazy compatibility export for older imports"）。
- **问题**：新增/改名一个 VLA operator 要同步改两处映射；注释自认是兼容垫片但没有废弃计划。
- **影响**：两份注册表漂移后，`from worldfoundry.operators import X` 与 `from worldfoundry.operators.embodied_action_operator import X` 会解析出不同结果或一边 AttributeError。
- **建议**：兼容垫片直接 `from . import <name>` 代理到包级 `__getattr__`，或加 DeprecationWarning 并排期删除。

#### [OR-09] P2 官方基准动作生成使用全局随机源且无种子，评测不可复现

- **位置**：`worldfoundry/operators/matrix_game_2_operator.py:9-40`
- **证据**：

```19:24:worldfoundry/operators/matrix_game_2_operator.py
    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]["keyboard_condition"]
        if mouse:
            m = data[rd]["mouse_condition"]
```

  函数名为 `_combine_official_action_data`，被 `process_official_bench_actions`（`matrix_game_2_operator.py:337-345`）调用，用于官方基准动作序列。无 seed 参数，用进程级 `random` 全局状态。另外 `selections = [12]`（`matrix_game_2_operator.py:17`）使第一处 `random.randint(0,0)` 恒为 12——死随机性。
- **问题**：同一模型两次跑官方 bench 得到不同动作条件序列；结果差异无法归因。`repeat_time = rd_frame // 4` 依赖 `num_samples_per_action=4` 的隐式假设（`k.repeat(3,1)` 恰好 12 行），改采样数即静默错位。
- **影响**：基准分数抖动；跨模型对比失真。
- **建议**：接受显式 `seed` 或 `random.Random` 实例；`selections` 死代码删除；用断言固定 `rd_frame % num_samples_per_action == 0`。

#### [OR-10] P3 `flash_world_operator.process_interaction` 的 `text_prompt` 恒为空串

- **位置**：`worldfoundry/operators/flash_world_operator.py:228-247`
- **证据**：

```228:233:worldfoundry/operators/flash_world_operator.py
        text_prompt = ""
        # Preserve list order; every non-text entry is a motion segment (forward, left, camera_l, ...)
        camera_actions = [
            a for a in self.current_interaction
            if a != "text_prompt"
        ]
```

- **问题**：`interaction_template` 里声明了 `"text_prompt"` 交互类型，但 `process_interaction` 永远返回 `text_prompt=""`——用户通过交互通道注入的文本被丢弃（真实 prompt 只能从 pipeline 侧另行传入）。返回一个恒空字段徒增困惑。
- **影响**：低——但 API 撒谎（模板声明支持、实现丢弃）。
- **建议**：要么真正透传文本交互，要么从模板和返回值中删掉 `text_prompt`。

#### [OR-11] P2 operator 层残留调试 print 与设备硬编码

- **位置**：print：`astra_operator.py:277-373`（6 处）、`lingbot_world_operator.py:213,309`、`recammaster_operator.py:86`；设备硬编码：`wan_2p2_operator.py:91-97`
- **证据**：

```91:97:worldfoundry/operators/wan_2p2_operator.py
            elif prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    mode=mode,
                    is_vl=images is not None,
                    device=0,
                )
```

```86:86:worldfoundry/operators/recammaster_operator.py
            print(f"WARNING: Your video clip is too short. The length of video clip need longer than {self.max_num_frames}")
```

- **问题**：
  1. `device=0` 把 prompt 扩写用的 Qwen 模型钉死在 cuda:0，绕过 `runtime/device_pool` 的设备分配；在被分到其它 GPU 的评测作业里会与别的作业争抢 0 号卡。此外，在名义上"不创建模型、不做推理"（该文件 docstring `wan_2p2_operator.py:30`）的 operator 里加载 LLM 本身就与自我声明矛盾。
  2. print 直写 stdout，绕过统一日志，批量评测时无法定位来源。
- **影响**：多卡评测互相干扰；日志卫生。
- **建议**：device 由调用方注入；print 改 `logging`。

### 主题 C：operators/__init__.py 导入策略

#### [OR-12] P3 包级懒加载做得好，但兼容注册表一分为二（见 OR-08）

- **位置**：`worldfoundry/operators/__init__.py:107-115`
- **证据**：

```107:115:worldfoundry/operators/__init__.py
def __getattr__(name):
    """Getattr   implementation."""
    if name not in _OPERATOR_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{_OPERATOR_MODULES[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
```

- **评价**：正面发现。85 个 operator 通过 PEP 562 `__getattr__` 按需加载并缓存到 `globals()`，`import worldfoundry.operators` 本身不拉起任何 torch/cv2 依赖——这是正确的设计。**但**懒加载收益被两处削弱：(a) OR-07 中 yume/hunyuan_worldplay 等模块级反向 import synthesis；(b) 23 个 operator 文件顶层 `import torch`/`import cv2`（rg 统计），即使只做纯字符串交互也要付出 torch 导入成本。注册表映射需要手工维护，缺一个静态一致性测试（断言 `_OPERATOR_MODULES` 每项可导入且类名存在）。
- **建议**：加一个参数化冒烟测试遍历 `_OPERATOR_MODULES`；把只在个别方法里用到的 torch/cv2 改函数级导入。

### 主题 D：runtime/ 环境装配与作业生命周期

#### [OR-13] P1 同一包内三种"仓库根"推导，`probes.py` 的 `parents[4]` 与 `conda.py` 的 `parents[5]` 都指向仓库外

- **位置**：`worldfoundry/runtime/probes.py:22`、`worldfoundry/runtime/conda.py:37-43`、`worldfoundry/runtime/performance.py:517`
- **证据**：

```22:22:worldfoundry/runtime/probes.py
REPO_ROOT = Path(__file__).resolve().parents[4]
```

```37:43:worldfoundry/runtime/conda.py
def project_root() -> Path:
    """Resolve the project root by searching for ``pyproject.toml`` upwards."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[5]
```

  实测（本仓库布局 `<...>/WorldFoundry/worldfoundry/runtime/probes.py`）：`parents[2]` = 仓库根，`parents[4]` = `/mnt/cpfsB/yangboxue/visual_generation`（仓库外两级）。`performance.py:517` 用的是正确的 `parents[2]`。`evaluation/utils.py:153` 则有第四种：`worldfoundry_repository_root()`。
- **问题**：
  1. `probes.py` 的 `REPO_ROOT` 被用作探针子进程的 `PYTHONPATH` 注入（`probes.py:324,329,367` 等），指向错误目录后该注入完全失效——探针在需要 `vbench`/`worldscore` 源码路径时会误报 import 失败或依赖外部环境巧合。
  2. `conda.py` 的 `project_root()` 在包内零调用（rg 全库仅此一处定义，studio 用的是 `core.io.paths.project_root`）——死代码；且兜底 `parents[5]` 更加错误。
- **影响**：环境体检结果不可信；死代码误导后来者复制错误算法。
- **建议**：统一使用 `worldfoundry.core.io.paths.project_root` / `evaluation.utils.REPO_ROOT`；删除 `conda.project_root`；修正 `probes.REPO_ROOT` 为 `parents[2]` 并加一条 `assert (REPO_ROOT / "pyproject.toml").exists()` 类的自检。

#### [OR-14] P1 `execute_in_tree` 对模型 CLI 子进程不设超时，10+ 个 synthesis 运行时受影响

- **位置**：`worldfoundry/runtime/in_tree_cli.py:99-107`
- **证据**：

```99:107:worldfoundry/runtime/in_tree_cli.py
    completed = subprocess.run(
        rendered,
        cwd=workdir,
        env=process_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
```

  调用方包括 hydra、uni3c、video_x_fun、versecrafter、moverse、liveworld、minwm、magic_world、spatia、hunyuan_world 等 10+ 个 synthesis 运行时（rg 统计）。对照：同包 `jobs.py:54-68` 的 `run_bounded_command` 专门为"官方基准子进程会在 native 代码里卡死"设计了硬超时 + 进程组击杀，docstring 说得明明白白。
- **问题**：模型推理 CLI 恰恰是最容易 hang 的一类子进程（CUDA 死锁、NCCL 等待、数据下载卡住）。`execute_in_tree` 没有 timeout 参数，也没用 `start_new_session`，卡死后评测 worker 永久阻塞且无法连子进程一起清理。
- **影响**：单模型 hang 拖死整批评测；无人值守跑批过夜后发现全部排队。
- **建议**：`execute_in_tree` 增加必填/带默认的 `timeout` 参数并复用 `run_bounded_command`（或至少 `start_new_session=True` + `communicate(timeout=)` + 进程组击杀）。

#### [OR-15] P1 `AsyncCommandJobStore`：异常路径遗留孤儿进程 + readline 64KiB 上限炸雷 + 作业字典无限增长

- **位置**：`worldfoundry/runtime/jobs.py:358-365`（异常路径）、`jobs.py:330-341`（spawn/读流）、`jobs.py:215-217`（无淘汰）
- **证据**：异常处理不终止已 spawn 的子进程：

```358:365:worldfoundry/runtime/jobs.py
        except Exception as exc:  # noqa: BLE001 - surfaced through UI/MCP status.
            job.status = "failed"
            job.error = str(exc)
            self._append_job_log(job, "stderr", f"{type(exc).__name__}: {exc}\n")
            self._write_lifecycle_event(job, "ERROR", "job.failed", "WorldFoundry job failed", exception=exc)
        finally:
            if job.completed_at is None:
                job.completed_at = _utc_now_iso()
```

  读流用 `stream.readline()`（`jobs.py:377`），而 `create_subprocess_exec`（`jobs.py:330-337`）未调大 `limit`——asyncio 默认 64KiB。一旦子进程输出单行超过 64KiB（模型 CLI 打印大 JSON result 是常见场景，且 `_extract_json_from_logs` 的设计恰恰鼓励子进程往 stdout 打 JSON），`readline` 抛 `ValueError` → 落入上面的 `except Exception` → 作业标记 failed，**但子进程既没被 kill 也没被 wait**（`start_new_session=True` 的孤儿进程组继续占着 GPU）。此外 `self._jobs` 只增不减（`jobs.py:217` 无任何 prune/evict API），每个 job 最多驻留 4000 行日志，长驻 MCP/UI 进程内存单调增长；`submit()` 也没有 per-job 超时。该 store 被 `mcp/tools/context.py` 使用。
- **影响**：GPU 僵尸占用 + 状态误报（failed 但其实还在跑）+ 长驻进程内存泄漏。
- **建议**：(a) `except Exception` 分支加 `await self._terminate_process(job)`；(b) spawn 时传 `limit=2**20` 或改用 `stream.read()` 分块读；(c) 提供 `prune(max_age/max_jobs)`；(d) submit 接受可选 timeout，超时走 cancel 流程。

#### [OR-16] P1 `local_checkpoint_cache` 跨进程发布竞争：并发进程可把对方刚发布的检查点整树删掉

- **位置**：`worldfoundry/runtime/local_checkpoint_cache.py:137-164`
- **证据**：

```158:163:worldfoundry/runtime/local_checkpoint_cache.py
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
```

  staging 的并发防护只覆盖同一 `torch.distributed` 组内（rank0 独占 I/O + broadcast，`local_checkpoint_cache.py:198-221`）。但两个**互不相干的进程**（例如两个单卡评测作业共享同一 `WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE`）同时 stage 同一 checkpoint 时：A、B 都判 `_is_ready` 为假 → 各自复制到 `.tmp-{pid}`（这一步没问题）→ A `rename(target)` 成功并开始从 target 加载模型 → B 执行 `if target.exists(): shutil.rmtree(target)` **把 A 正在读的目录删除**，A 的 `safetensors`/mmap 读取中途 ENOENT。
- **影响**：并发首跑随机崩溃且难复现（第二次跑就好了）；是"缓存目录约定缺进程间锁"的典型案例。
- **建议**：发布前用 `fcntl.flock` 文件锁（或 `os.rename` 到位后先重查 `_is_ready` 再决定是否删除重建；更简单的做法：`rename` 失败（目标已存在）即认为他人已发布，转为复用并清理自己的 tmp）。同文件 `local_checkpoint_cache.py:140` 的 `print(...)` 应改 logging。

#### [OR-17] P2 资产"就绪"判定只看 `path.exists()`，无大小/校验和验证

- **位置**：`worldfoundry/runtime/assets.py:87-105`
- **证据**：

```87:90:worldfoundry/runtime/assets.py
        path = expand_worldfoundry_path(raw_path, env) if raw_path else None
        canonical_path = expand_worldfoundry_path(raw_canonical_path, env) if raw_canonical_path else None
        ready = bool(path and path.exists())
        status = "available" if ready else "missing"
```

- **问题**：runtime 层的资产可用性契约是纯存在性检查。中断的下载/拷贝留下的半截文件、空目录都会被判 `ready=True` 并进入 `resolve_benchmark_repo_root`（`benchmark_repos.py:104-106`）等下游决策。manifest 的 metadata 允许携带任意字段，但没有任何 checksum/size 字段被消费（对照：`in_tree_cli.py:140` 输出侧倒是算了 `artifact_sha256`）。
- **影响**：坏资产被静默采用，报错发生在深处的模型加载而非资产解析处，排障成本高。
- **建议**：manifest 项支持可选 `sha256`/`size_bytes`/`min_entries`（目录），`LocalAsset.from_manifest_item` 消费之；至少对 file 类资产加 `st_size > 0` 检查。

#### [OR-18] P2 `env.py` 诊断命令超时未捕获：`nvidia-smi` 卡死会让 preflight 整体崩溃

- **位置**：`worldfoundry/runtime/env.py:481-498`
- **证据**：

```488:496:worldfoundry/runtime/env.py
    if shutil.which(argv[0]) is None:
        return None
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
```

- **问题**：`subprocess.run(timeout=5)` 超时会抛 `subprocess.TimeoutExpired`，此处不捕获，沿 `capture_runtime_environment`（`env.py:549-553`）直接炸掉调用方。驱动挂掉时 `nvidia-smi` 卡住是常见故障态——恰恰是 preflight 最该稳住的场景。对照：`cuda_tiers.py:72-82` 的同类探测就正确地 `except Exception: return None`。
- **影响**：诊断工具在最需要诊断的机器上先崩。
- **建议**：包一层 `try/except (OSError, subprocess.SubprocessError): return None`。

#### [OR-19] P2 `probes.py` 把某一部署的 conda 环境名/目录布局硬编码进库代码

- **位置**：`worldfoundry/runtime/probes.py:268-274`、`probes.py:319-412`
- **证据**：

```268:274:worldfoundry/runtime/probes.py
    candidates = {
        "benchmark_cu113": conda_envs_root / "worldfoundry-zeroscope-cu113" / "bin" / "python",
        "benchmark_worldplay": conda_root / "worldplay" / "bin" / "python",
        "benchmark_worldscore": conda_root / "worldscore" / "bin" / "python",
        "benchmark_cu113_animatediff": conda_envs_root / "worldfoundry-animatediff-official-cu113" / "bin" / "python",
        "benchmark_cu113_zeroscope": conda_envs_root / "worldfoundry-zeroscope-cu113" / "bin" / "python",
    }
```

  `build_environment_report`（`probes.py:319-412`）同样硬编码 worldplay/worldscore 两套环境的 30+ 模块清单和 `model_root / "VBench"` 布局。
- **问题**：环境清单本应来自 `conda.py` 的 manifest（`data/models/runtime/environments`），这里却平行维护了一份写死的快照；conda 环境更名/新增 tier 后报告静默失真（全部 `python_not_found`）。`benchmark_cu113` 与 `benchmark_cu113_zeroscope` 两个键还指向同一路径，探测重复。
- **影响**：GPU/环境体检覆盖面停留在写死那天；与 unified-env 路由（conda.py）演进脱节。
- **建议**：候选环境从 `load_runtime_conda_env_specs_with_overrides()` 生成；模块清单挂到各环境 spec 的 `validation_imports`（字段已存在，`conda.py:138`）。

#### [OR-20] P2 `newest_media` 的 mtime 启发式：并发作业互捡产物 + 排序期 stat 竞态

- **位置**：`worldfoundry/runtime/in_tree_cli.py:40-68`、`in_tree_cli.py:124-135`
- **证据**：

```55:63:worldfoundry/runtime/in_tree_cli.py
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            try:
                if path.stat().st_mtime >= since - 1.0:
                    candidates.append(path)
            except OSError:
                continue
    candidates.sort(key=lambda item: (item.stat().st_mtime, item.stat().st_size), reverse=True)
```

- **问题**：
  1. "命令跑完后从输出根目录里找最新媒体文件"本身是弱契约：两个作业若共享 `search_roots`（同一模型的官方 output 目录很常见），会把对方刚写的视频当成自己的产物归档——评测结果串号。
  2. 第 63 行排序 key 里再次 `item.stat()`，此时文件可能已被清理，`OSError` 未捕获直接炸（第一处 stat 有 try/except，第二处没有）。
  3. 每个文件 stat 两次，量大时慢。
- **影响**：产物错配是评测正确性问题；竞态崩溃偶发难查。
- **建议**：收集时缓存 `(mtime, size)` 一次性排序；对 search_roots 引入 run 级隔离目录（run_id 子目录），或要求模型 CLI 显式输出 `--output` 并放弃全局扫描。

#### [OR-21] P2 设备池仅进程内有效，跨进程/跨 worker 无协调；`wan_2p2` 的 `device=0` 直接绕开它

- **位置**：`worldfoundry/runtime/device_pool.py:52-114`
- **证据**：

```52:61:worldfoundry/runtime/device_pool.py
class CudaDeviceLeasePool:
    """Thread-safe allocator for non-overlapping CUDA worker assignments."""

    def __init__(self, devices: Sequence[str]) -> None:
        normalized = normalize_cuda_device_groups(tuple(str(device) for device in devices))
        self._devices = tuple(normalized)
        self._available = list(self._devices)
        self._leased: set[str] = set()
        self._waiting = 0
        self._condition = threading.Condition()
```

- **问题**：池本身实现干净（Condition + 可取消等待 + 幂等 release），但它是纯进程内对象（仅 `studio/conda_dispatch.py` 使用）。同一台机器上并行启动两个评测进程时各自 new 一个池，GPU 双重分配无任何防护；而库内已经存在绕过案例（OR-11 的 `device=0`）。另有小问题：多卡请求（count=2）在单卡请求持续到达时可能饿死（无 FIFO 公平性）；`acquire` 的等待即使无 `cancel_requested` 也按 0.1s 轮询空转。
- **影响**：多进程并发评测时显存冲突 OOM；问题归因困难。
- **建议**：文档明确"仅进程内"边界；跨进程场景用文件锁（`/dev/shm` flock per device token）或交给上层调度器；operator/synthesis 层禁止字面量 device。

#### [OR-22] P2 `suggested_sourceable_env_value`：runtime 层硬编码 60+ 行具体基准的 env var 命名启发式

- **位置**：`worldfoundry/runtime/env.py:412-478`
- **证据**：

```424:433:worldfoundry/runtime/env.py
    if name == "DATA_PATH":
        return "${WORLDFOUNDRY_DATA_DIR}"
    if name == "COPPELIASIM_ROOT":
        return "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/coppeliasim"
    if name == "WORLDFOUNDRY_HF_CACHE_DIR":
        return "${WORLDFOUNDRY_CACHE_DIR}/huggingface/hub"
    if name == "WORLDFOUNDRY_CHRONOMAGIC_CHSCORE_CKPT":
        return "${WORLDFOUNDRY_CKPT_DIR}/hfd/BestWishYsh--ChronoMagic-Bench/CHScore/cotracker2.pth"
    if name == "WORLDFOUNDRY_CHRONOMAGIC_MTSCORE_CKPT":
        return "${WORLDFOUNDRY_CKPT_DIR}/hfd/BestWishYsh--ChronoMagic-Bench/MTScore/InternVideo2-stage2_1b-224p-f4.pt"
```

- **问题**：ChronoMagic/iWorld/IPV/CoppeliaSim 等具体基准的路径知识内嵌在通用 runtime 模块里，靠 `_RESULTS_PATH`/`_ROOT` 等 20 条后缀规则兜底。每接入一个新基准都可能要改 runtime/env.py——依赖方向反了（runtime 不该知道基准细节）。同时反映了 env var 泛滥：仅此函数就隐含 20+ 种基准 env var 命名模式。
- **影响**：runtime 层成为基准接入的隐性改动点；后缀规则误命中会给出错误建议值。
- **建议**：把建议值声明移到各基准的 manifest（`data/benchmarks/**`，与 `worldfoundry-benchmark` skill 的 catalog 思路一致），runtime 只保留通用后缀规则。

### 主题 E：runtime/ 缓存与全局状态

#### [OR-23] P2 `compile_cache` 编译失败静默降级为未编译，无任何告警

- **位置**：`worldfoundry/runtime/compile_cache.py:343-353`（module 版），`compile_cache.py:398-408`（callable 版）
- **证据**：

```343:353:worldfoundry/runtime/compile_cache.py
        try:
            compiled = _compile_target(
                compile_fn,
                module,
                policy=selected,
                options=options,
            )
        except Exception:
            if strict:
                raise
            return module
```

  同样地，`compile_module_cached` 里 `configure_persistent_compile_cache` + `import torch` 失败也走 `except Exception: return module`（`compile_cache.py:320-326`）。
- **问题**：非 strict 路径下 torch.compile 失败无声回退 eager，性能腰斩但用户毫无感知——对一个专门做性能基线（performance.py/realtime_regression.py）的框架，这会让回归数据"莫名变慢"且无迹可循。
- **影响**：性能回归误归因；调参浪费时间。
- **建议**：fallback 时 `warnings.warn` 或 logging.warning 一次（带 module 名与异常类型）；`OptimizationSnapshot.fallbacks` 字段已经为此设计（`performance.py:187`），应把该事件写进去。另：`configure_persistent_compile_cache(namespace=...)` 参数直接 `del namespace`（`compile_cache.py:231`）——死参数应删除。

#### [OR-24] P3 `detect_nvidia_driver_cuda` 的 `lru_cache(1)` 忽略 env 覆盖变化，与同包 env-key 缓存策略不一致

- **位置**：`worldfoundry/runtime/cuda_tiers.py:63-84`
- **证据**：

```63:68:worldfoundry/runtime/cuda_tiers.py
@lru_cache(maxsize=1)
def detect_nvidia_driver_cuda() -> str | None:
    """Detect the NVIDIA driver CUDA version via ``nvidia-smi`` or env override."""
    override = os.environ.get("WORLDFOUNDRY_DETECTED_DRIVER_CUDA")
    if override:
        return override
```

- **问题**：函数体读 `WORLDFOUNDRY_DETECTED_DRIVER_CUDA` 但整体被 `lru_cache(1)` 冻结——测试或长驻进程内改 env 后结果不更新。对照 `conda.py:326-348` 专门构造了 env 快照 cache key 来处理同类问题，两套策略并存。另外 `best_cuda_tier_for_driver`（`cuda_tiers.py:87-97`）在驱动版本探测不到时返回最新 tier `cu128` 且 `cap_tier_to_driver` 对 `(0,0)` 直接放行（`cuda_tiers.py:116-117`）——未知驱动默认选最激进档，安装后才发现不兼容。
- **建议**：cache key 纳入 env 值（或提供 `cache_clear` 的公开入口并在测试用）；未知驱动时降级到最保守 tier 并打 warning。

#### [OR-25] P3 runtime 层其余卫生问题（汇总条）

- **位置/证据**：
  1. `conda.py:378`：`pip_extra_index_url="" if pip_extra_index is None else str(pip_extra_index)`——dataclass 字段声明 `str | None = None`（`conda.py:135`），加载路径却把 None 规整成 `""`，下游 `is None` 判断永假。类型契约与实现不一致。
  2. `probes.py:42-59`：探针代码以字符串拼 Python 源码（`code = "import importlib,...;" f"mods={json.dumps(...)}"`），虽然用了 `json.dumps` 转义是安全的，但混合 `;` 与 `\n` 的写法极难维护——建议放进独立的 `_probe_script.py` 用 `-m` 跑。
  3. `assets.py:190`：默认 manifest 候选包含 `repo_root / "tmp" / "benchmark_zoo" / ...`——仓库内 `tmp/` 目录作为一级配置源被固化（`benchmark_repos.py:71` 同款 legacy 路径），legacy 迁移路径没有废弃时间表。
  4. `jobs.py:453-476` `_extract_json_from_logs`/`_json_candidates`：从 stdout 反向逐行试 `json.loads` 抓结果——子进程 stdout 中任何一行合法 JSON（比如进度日志 `{"step": 1}`）都会被当成最终 result。契约脆弱，建议改为显式 sentinel（如 `WORLDFOUNDRY_RESULT_JSON: {...}` 前缀）或结果文件。
  5. `env.py:63-73` `SOURCEABLE_ENV_BASE_LINES` 以 shell 行硬编码默认目录布局，与 `core.io.paths` 的 Python 端布局是两份真相，漂移无检测。
- **影响**：各为局部一致性/可维护性问题。
- **建议**：如上逐条；其中 4 值得在 MCP 面向用户前修。

### 主题 F：正面观察（简要）

以下 runtime 设计是**好的实践**，评审确认无需改动，列出以防"只报忧"失真：

- `performance.py`：严格 JSON 校验（拒绝 NaN/Inf，`performance.py:30-48`）、`write_json` 的 mkstemp+fsync+`os.replace` 原子写（`performance.py:384-408`）、best-effort 指纹采集全部 `_safe_call` 包裹。`realtime_regression.py` 同样水准。
- `local_checkpoint_cache.py` 的 rank0 staging + `broadcast_object_list` 错误传播（`local_checkpoint_cache.py:198-221`）在**单 distributed 组内**是正确的（跨进程见 OR-16）。
- `compile_cache.py` 的 torch/硬件指纹分区（`compile_cache.py:130-190`）与写权限探测回退（`compile_cache.py:193-212`）考虑周到。
- `conda.py` 的 unified-env 阻断策略（`unified_env_blocker`，`conda.py:263-309`）对 ABI pin、上界约束的处理有注释有道理。
- `runtime/__init__.py` 刻意零依赖（`runtime/__init__.py:1-7`），`platforms/` 的 Protocol + ABC 双契约、CPU 兜底探测（`platforms/detect.py:62-66`）干净利落。
- `env.py` 的秘密值 redaction（值→presence 布尔，`env.py:376-382`）是正确的 manifest 卫生。
- operators 层没有任何网络调用/API key/subprocess（rg 验证 `import requests|httpx|urllib`、`subprocess`、`os.environ` 全部 0 命中）——API 调用与凭证确实如文档所说隔离在 synthesis/pipelines 层。评审标准第 4 条（API key、重试、费用保护）在 operators 层无发现，责任在别的模块范围。

## 汇总

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0 | 1 | OR-03 |
| P1 | 6 | OR-01, OR-05, OR-13, OR-14, OR-15, OR-16 |
| P2 | 14 | OR-02, OR-04, OR-06, OR-07, OR-08, OR-09, OR-11, OR-17, OR-18, OR-19, OR-20, OR-21, OR-22, OR-23 |
| P3 | 4 | OR-10, OR-12, OR-24, OR-25（5 条子问题合并计 1） |

共 25 个编号发现（OR-25 为 5 合 1 汇总条）。operators 层 12 条（OR-01～OR-12），runtime 层 13 条（OR-13～OR-25）。

### Top 5 问题

1. **OR-03（P0）**：BGR/RGB 通道均值猜测启发式静默翻转暖色调图像通道，复制 5 份——评测输入被破坏，分数不可信。
2. **OR-16（P1）**：checkpoint staging 跨进程发布竞争，`rmtree` 可删除并发进程正在读取的已发布目录——随机崩溃。
3. **OR-14（P1）**：`execute_in_tree` 模型 CLI 子进程无超时，10+ synthesis 运行时受影响——单模型 hang 拖死整批评测。
4. **OR-15（P1）**：MCP 作业执行器异常路径不清理子进程 + asyncio readline 64KiB 上限 + 作业表无限增长——GPU 僵尸与内存泄漏。
5. **OR-01+OR-05（P1）**：BaseOperator 契约空心化导致 85 个 operator 各自为政，11 份逐字克隆已开始漂移，现成共享基类闲置——每修一个共性 bug 都要 touch 两位数文件。

### 修复优先级建议

- **立即**：OR-03（删启发式，一次 rg 可定位全部 5 处）；OR-18（两行 try/except）；OR-13 的 `probes.REPO_ROOT`（一行）。
- **本迭代**：OR-14/15/16（子进程与缓存生命周期，都是局部改动）；OR-02（统一 delete 语义）。
- **规划重构**：OR-01/05/06（operator 契约与三件套下沉，建议随下一次新模型接入一并做，先加冒烟测试 OR-12 防回归）；OR-19/22（probes/env 的部署快照外移到 manifest）。

