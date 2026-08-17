# pipelines 层评审

> 评审人：infra 代码评审 agent；日期：2026-08-14
> 状态：已完成

## 评审范围与方法

- 范围：`worldfoundry/pipelines/` 全部（164 个 py 文件，约 90 个模型目录 + 5 个根级共享模块）。
- 方法：
  1. **共享模块全部精读**：`pipeline_utils.py`、`native_diffusion.py`、`native_diffusion_video.py`、`component_pipelines.py`、`api_runtime.py`，及家族级基类 `videocrafter/base.py`、`world_model/pipeline_runtime_manifest.py`、`three_d_four_d/pipeline_runtime.py`、`video_official/pipeline_official_video.py`。
  2. **代表 pipeline 精读 14 个**（覆盖四种组织模式）：`wan/pipeline_wan_2p1_t2v.py`（声明式 native）、`cosmos/pipeline_cosmos_predict2.py`（手写 native）、`wan/pipeline_wan_2p5.py` 与 `kling/pipeline_kling_api.py`（hosted API）、`vggt/pipeline_vggt.py` + `vggt/official_runtime.py`、`cut3r/official_runtime.py`（官方 runtime 包装）、`hunyuan_world/pipeline_hunyuan_mirror.py`、`matrix_game/pipeline_matrix_game_2.py`（realtime 交互）、`depth_anything/pipeline_depth_anything_v1.py`/`_v2.py`（感知批处理）、`bernini/pipeline_bernini.py`、`lyra/lyra_utils.py`、`echo_memory/pipeline_echo_memory.py`（部分）。
  3. **全量 rg 扫描 + 抽查**：bare except / `torch.load` / 硬编码路径 / 越层 import / `no_grad` / `print` / `sys.path` / `os.environ` / `time.sleep` 轮询 / `subprocess` / `DiffusionRequest` 复制 / `api_init`、`get_operator` 重定义计数；抽查 `flash_world`、`solaris`、`kling/pipeline_astra.py`、`hunyuan_world/pipeline_hunyuan_worldplay.py`、`pi3/pipeline_loger.py` 及 8 个子包 `__init__.py`。
  4. 对照架构文档 `docs/fumadocs/content/docs/maintainers/architecture/`（runtime-assembly / model-runtime / native-diffusion）核对契约与分层规则。
- 结构速览：
  - **根级共享模块（5 个）**：`pipeline_utils.py`（`PipelineABC` 非严格基类）、`component_pipelines.py`（声明式工厂，惰性 target 生成 ~50 个 VLA/视频 pipeline 类）、`native_diffusion.py`（`NativeVisualDiffusionPipeline`，本地扩散统一适配器）、`native_diffusion_video.py`（兼容别名）、`api_runtime.py`（API key 解析）。
  - **约 90 个模型目录**，四种组织模式：
    1. **声明式 native diffusion 子类**（wan 2.1、cogvideox、hunyuan_video、sana 等）：仅覆写 ClassVar，最薄最规范；
    2. **手写 hosted-API 包装**（wan 2.5/2.6/2.7、kling、sora、veo、luma、minimax、runway、worldlabs 共 10 个 `api_init` 文件）：复制同一模板；
    3. **手写本地重型 pipeline**（vggt、cut3r、pi3、hunyuan_world 系、matrix_game 系、lyra 等）：pipeline 内直接做模型执行/几何后处理；
    4. **官方 runtime 包装**（`official_runtime.py`：vggt、vggt_omega、cut3r）与**清单驱动运行时**（`world_model/pipeline_runtime_manifest.py`、`three_d_four_d/pipeline_runtime.py`）。
  - 根包与约 55 个子目录无 `__init__.py`（PEP 420 隐式命名空间包），另 ~35 个子目录有 `__init__.py`——混用。
  - pyproject `[tool.setuptools]` 排除 `worldfoundry.pipelines.*.runtime*`，当前树中无该类目录（规则为历史遗留防御）。

## 发现（按主题分组）

### 主题 A：抽象设计与复制粘贴漂移

#### [PL-01] P1 `stream()` 语义三态分裂：生成器 / 直接返回值 / 生成器内 return 丢值
- 位置：`worldfoundry/pipelines/pipeline_utils.py:195-211`、`worldfoundry/pipelines/native_diffusion.py:393-403`、`worldfoundry/pipelines/vggt/pipeline_vggt.py:920-940`
- 证据：

```195:203:worldfoundry/pipelines/pipeline_utils.py
    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Yield pipeline outputs using the same call semantics as ``__call__``."""
        if self._uses_component_contract() and getattr(self, "synthesis_model", None) is not None:
            if kwargs.get("images") is None and kwargs.get("video") is None:
                previous = self.memory_module.select() if self.memory_module is not None else None
                if isinstance(previous, dict):
                    kwargs["images"] = previous.get("artifact_path")
            return self(*args, **kwargs)
        return self._stream_fallback(*args, **kwargs)
```

- 问题：基类 `stream()` 在组件契约路径**直接返回结果**（非生成器），fallback 路径返回**生成器**；`NativeVisualDiffusionPipeline.stream()` 又返回单值。调用方无法统一 `for chunk in pipe.stream(...)`。架构文档称 stream 用于 Studio 增量输出，但三种实现语义互斥。
- 影响：Studio/CLI 消费方必须按 pipeline 类型分支处理；`isinstance(result, Iterable)` 的兜底在 str/tensor 上行为不同，易出隐性 bug。
- 建议：契约上固定 `stream()` 必须返回迭代器（哪怕单元素）；基类统一包装。

#### [PL-02] P1 10 个 hosted-API 包装复制同一模板，含相同缺陷的 N 份拷贝
- 位置：`wan/pipeline_wan_2p5.py`、`wan/pipeline_wan_2p6.py`、`wan/pipeline_wan_2p7.py`、`kling/pipeline_kling_api.py`、`sora/pipeline_sora2.py`、`veo/pipeline_veo3.py`、`luma/pipeline_luma_ray2.py`、`minimax/pipeline_hailuo_2p3.py`、`runway/pipeline_runway_gen4p5.py`、`worldlabs/pipeline_worldlabs.py`
- 证据（wan_2p5 与 kling 同构，均为手写 `__init__/api_init/process/__call__/get_operator/get_synthesis_model`）：

```22:31:worldfoundry/pipelines/kling/pipeline_kling_api.py
    def __init__(
        self,
        operator: Optional[KlingApiOperator] = None,
        synthesis_model: Optional[KlingApiSynthesis] = None,
        endpoint: str = "https://api.klingapi.com",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Kling")
        self.endpoint = endpoint
```

- 问题：任务轮询（`_poll_task_status` + `time.sleep`，8 个文件各一份：runway:200、worldlabs:256、sora:281、kling:169、wan_2p6:130、wan_2p7:131、luma:131、minimax:128）、状态/URL 提取、下载逻辑每家复制一份且细节漂移（kling 的 `_extract_video_url` 查 9 个候选键，wan_2p5 只查 `output.video_url`）；`get_operator/get_synthesis_model` 在 26 个 pipeline 文件中重定义（基类已提供同名方法）。修 bug（如轮询超时、重试、代理）需改 10 处。
- 影响：维护成本线性放大；行为不一致（有的失败抛异常、有的静默返回 None）。
- 建议：抽 `HostedApiPipeline` 基类（poll/extract/download 模板方法），子类只声明 endpoint、env 键与 payload 组装。

#### [PL-03] P2 API 包装 `__init__` 不调 `super().__init__()`，基类状态缺失
- 位置：`worldfoundry/pipelines/wan/pipeline_wan_2p5.py:23-43`、`worldfoundry/pipelines/kling/pipeline_kling_api.py:22-34`、`worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py:27-39`、`worldfoundry/pipelines/vggt/pipeline_vggt.py:97-108` 等
- 证据：

```39:43:worldfoundry/pipelines/wan/pipeline_wan_2p5.py
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.5")
        self.endpoint = endpoint
        self.api_key = api_key
        self.operator = operator
        self.synthesis_model = synthesis_model
```

- 问题：跳过 `PipelineABC.__init__`，实例缺 `model_id/memory_module/options/operators/device` 属性。基类 `stream()`/`_call_component_pipeline` 等方法若被触发会 `AttributeError`（当前靠 `_uses_component_contract()` 返回 False 侥幸绕开）。
- 影响：基类新增依赖实例属性的功能时，这批子类全部隐性破坏。
- 建议：强制 `super().__init__(...)`；或基类提供 `__init_subclass__` 校验。

#### [PL-04] P2 `PipelineABC._call_component_pipeline` 硬编码 `processed["actions"]` 键契约
- 位置：`worldfoundry/pipelines/pipeline_utils.py:278-286`
- 证据：

```278:286:worldfoundry/pipelines/pipeline_utils.py
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            video=processed["video"],
            interactions=processed["actions"],
            output_path=output_path,
            fps=fps,
            **synthesis_kwargs,
        )
```

- 问题：`process()` 的返回由 operator 三个方法的 dict 合并而成，`prompt/images/video/actions` 四键是隐式契约，operator 少返回一个键即 `KeyError`，错误信息不含指导性；契约没有 TypedDict/Protocol 约束，全靠运行时约定成立。
- 影响：新写 operator 时踩坑概率高，报错点远离根因。
- 建议：定义 `ProcessedInputs` TypedDict/dataclass 或在合并处做显式校验并给出修复提示。

#### [PL-09] P1 8 个家族绕过 `NativeVisualDiffusionPipeline` 适配器，手搓同一套 native 流程
- 位置：`cosmos/pipeline_cosmos_predict2.py`、`cosmos/pipeline_cosmos_predict2p5.py`、`cosmos/pipeline_cosmos_transfer2p5.py`、`cosmos/pipeline_cosmos3.py`、`gen3c/pipeline_gen3c.py`、`hunyuan_video/pipeline_hunyuan_video.py`、`vchitect/pipeline_vchitect_2_t2v.py`、`t2v_turbo/pipeline_t2v_turbo_t2v.py`
- 证据：`native_diffusion.py:27-31` 明言该适配器存在的目的是"so image and video families do not grow parallel pipeline implementations"，但 cosmos_predict2 完整复刻了 `_options`/dtype/offload 策略组装/`DiffusionRequest` 构造/产物保存：

```144:157:worldfoundry/pipelines/cosmos/pipeline_cosmos_predict2.py
        native = NativeDiffusionPipeline.from_pretrained(
            resolved_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="Cosmos Predict2",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
```

- 量化：`DiffusionRequest(` 手工构造出现在 9 个文件；负种子习语 `secrets.randbits(63) if int(seed) < 0 else int(seed)` 复制 8 份（`native_diffusion.py:365`、`gen3c:303`、`hunyuan_video:176`、`cosmos_predict2:237`、`vchitect:133`、`t2v_turbo:146`、`cosmos_transfer2p5:185`、`cosmos_predict2p5:170`）。
- 问题：与 wan2.1 那种"只覆写 ClassVar"的声明式子类形成两套并行写法；改动 request 组装（如新增 alias、修 seed 语义）要同步 9 处。
- 影响：这正是适配器要消灭的复制粘贴漂移，且已经在漂（各家 `_checkpoint_overrides` 的角色名、探测逻辑均不同步）。
- 建议：cosmos/gen3c/hunyuan_video 等改为继承 `NativeVisualDiffusionPipeline`，把 image/video 条件输入的差异做成 ClassVar 或钩子（`ACCEPTS_IMAGES`/`REQUEST_INPUT_DEFAULTS` 已支持大半）。

#### [PL-10] P1 `VGGTPipeline.stream()` 是生成器函数却在 CLI 分支 `return` 值——返回值必然丢失
- 位置：`worldfoundry/pipelines/vggt/pipeline_vggt.py:920-940`
- 证据：

```932:940:worldfoundry/pipelines/vggt/pipeline_vggt.py
        data = images if images is not None else image_path
        if data is None:
            raise ValueError("Provide image_path or images.")
        if task_type == "vggt_two_stage_3dgs_stream_cli":
            return self.run_two_stage_3dgs_stream_cli(image_path=data, **kwargs)

        result = self.process(input_=data, interaction=interactions, **kwargs)
        for img in result.images:
            yield torch.from_numpy(np.array(img))
```

- 问题：函数体含 `yield`（940 行），整个函数是生成器函数；936 行的 `return <value>` 只会把视频路径塞进 `StopIteration.value`。调用 `stream(task_type="vggt_two_stage_3dgs_stream_cli")` 得到的是一个生成器：不迭代则什么都不执行；迭代则先跑完整个交互式 CLI，然后一个元素都不产出、路径被丢弃。docstring 宣称"-> output_video_path"的行为不可能发生。
- 影响：该分支功能损坏；调用者要么拿到空生成器，要么被阻塞在 `input()`。
- 建议：把 CLI 分支拆成独立方法（本就有 `run_two_stage_3dgs_stream_cli` 公开方法），`stream()` 内不做 return-with-value；或改为 `yield self.run_...` 单元素生成。

#### [PL-11] P2 `VGGTPipeline` 1012 行巨类：相机数学、PLY IO、视频导出、交互式 REPL 全部内嵌
- 位置：`worldfoundry/pipelines/vggt/pipeline_vggt.py:866-890`（`input()` REPL）、`541-602`（交互 token→相机增量映射）、`231-271`（视频导出）
- 证据：

```866:870:worldfoundry/pipelines/vggt/pipeline_vggt.py
        while True:
            interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()
            if interaction_input in ["n", "q"]:
                print("Stopping interaction loop...")
                break
```

- 问题：架构文档要求 pipeline 是"thin public wrapper"，交互解析归 operators、几何归 representations、CLI 归 cli 层。在库代码里调 `input()` 会挂死任何非 TTY 调用方（Studio/评测 worker）。`_apply_interaction_to_camera` 对未知 token 静默不动（fall through 无 else），非 CLI 路径不校验 token，拼错交互名会生成一段静止视频且无任何警告。
- 影响：不可测试、不可复用；静默 no-op 使评测结果失真。
- 建议：REPL 移到 CLI 层；token 映射表移入 `vggt_operator`；未知 token 抛 `ValueError`。

### 主题 B：分层违规

#### [PL-05] P1 pipelines 向上 import evaluation（2 处）
- 位置：`worldfoundry/pipelines/video_official/pipeline_official_video.py:8`、`worldfoundry/pipelines/bernini/pipeline_bernini.py:10`
- 证据：

```8:8:worldfoundry/pipelines/video_official/pipeline_official_video.py
from worldfoundry.evaluation.models.pipelines.invocation import PipelineInvocation
```

- 问题：架构文档（model-runtime.mdx"Bridge to evaluation"）规定 evaluation 经 bindings 解析后调用 pipelines，方向是 evaluation→pipelines；此处反向依赖使 `PipelineInvocation` 类型跨层泄漏，`import worldfoundry.pipelines.video_official` 会拉起 evaluation 包。
- 影响：层间循环依赖风险；打包/裁剪 evaluation 时 pipelines broken。
- 建议：把 `PipelineInvocation` 下沉到 core（或 pipelines 层定义协议），evaluation 适配。

#### [PL-06] P1 pipelines 向上 import studio（1 处）
- 位置：`worldfoundry/pipelines/cut3r/official_runtime.py:25`
- 证据：

```25:25:worldfoundry/pipelines/cut3r/official_runtime.py
from worldfoundry.studio.visualization.core.geometry import depth_to_world_points
```

- 问题：pipelines 依赖 studio 的可视化几何函数，studio 是最上层产品面；`depth_to_world_points` 是纯几何工具，应位于 core 或 representations。
- 影响：无 studio 依赖的部署（纯评测环境）import cut3r 即失败。
- 建议：函数迁至 `core` 几何模块，studio 与 pipelines 共同引用。

#### [PL-12] P1 `lyra_utils` 成为事实共享库：synthesis 反向 import pipelines、跨家族引用、与 core 重复实现
- 位置：`worldfoundry/pipelines/lyra/lyra_utils.py`（539 行）；引用方：`synthesis/visual_generation/lyra_2/runtime.py:317,450,647`、`pipelines/video_official/pipeline_official_video.py:10`、`pipelines/helios/pipeline_helios.py:7`
- 证据（下层 synthesis 向上 import pipelines）：

```317:317:worldfoundry/synthesis/visual_generation/lyra_2/runtime.py
        from worldfoundry.pipelines.lyra.lyra_utils import working_directory
```

同时 `core/utils/image_utils.py:49,88` 已有规范的 `load_pil_image`/`materialize_image_input`（`bernini/pipeline_bernini.py:9` 正确使用），而 `lyra_utils.py:482-539` 维护着一份行为有差异的副本（lyra 版要求恰好一张图，core 版有 `first_sequence_item` 参数）。
- 问题：三重违规——synthesis→pipelines 反向依赖；helios/video_official 跨家族 import lyra 目录；与 core 重复实现且已漂移。
- 影响：依赖图成环的温床；两份 `load_pil_image` 行为差异会造成同输入不同结果。
- 建议：`working_directory`/`load_pil_image`/`materialize_image_input`/`build_subprocess_env` 全部收敛到 core；`lyra_utils` 只留 lyra 专属的权重布局探测。

#### [PL-13] P2 `world_model/__init__.py` 跨家族 re-export dreamx_world
- 位置：`worldfoundry/pipelines/world_model/__init__.py:4`
- 证据：

```4:4:worldfoundry/pipelines/world_model/__init__.py
from ..dreamx_world import DreamXWorld5BARPipeline, DreamXWorld5BCamPipeline
```

- 问题：`world_model` 包的 `__init__` 顺带导入兄弟家族，import `world_model` 就会加载 dreamx_world 及其依赖链；包边界被模糊。
- 影响：加载耦合、循环 import 风险。
- 建议：调用方直接从 `dreamx_world` import；或在 catalog 绑定层做别名。

### 主题 C：模型加载、路径与全局状态

#### [PL-14] P2 `lyra_utils.project_root()` 假设源码检出，安装为 wheel 后失效
- 位置：`worldfoundry/pipelines/lyra/lyra_utils.py:30-36`、`341`、`512-526`
- 证据：

```30:36:worldfoundry/pipelines/lyra/lyra_utils.py
def project_root() -> Path:
    """Project root helper function."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[4]
```

- 问题：向上找 `pyproject.toml`，wheel 安装环境找不到时兜底 `parents[4]`（落在 site-packages 附近的任意目录）；`prepare_lyra1_checkpoint_root` 会往 `project_root()/cache/runtime` 写权重软链，`build_subprocess_env` 把 `root/"src"` 塞进 PYTHONPATH——全部基于这个脆弱假设。
- 影响：非源码部署下 lyra1/lyra2、以及依赖 `build_subprocess_env` 的子进程运行时路径全错。
- 建议：用 `worldfoundry.core.io.paths` 的 cache/checkpoint root API（已有 `checkpoint_root_path()`）统一解析，禁止从 `__file__` 反推仓库根。

#### [PL-15] P2 `HunyuanMirrorPipeline` 把 skyseg.onnx 下载到进程 CWD
- 位置：`worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py:242-247`
- 证据：

```242:247:worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )
```

- 问题：模型权重下载到相对路径 `"skyseg.onnx"`（当前工作目录），不同 CWD 下重复下载；并发进程写同一文件会损坏；也绕过了框架统一的权重缓存根。
- 影响：污染用户工作目录；并发评测竞态。
- 建议：走 `checkpoint_root_path()`/hf 缓存目录，下载用带锁的原子重命名。

#### [PL-16] P2 CWD 相对默认输出路径散布在多个 pipeline
- 位置：`vggt/pipeline_vggt.py:62-63,412,679`（`./vggt_output`）、`hunyuan_world/pipeline_hunyuan_mirror.py:30`（`./output/hunyuan_mirror`，构造函数即 mkdir）、`wan/pipeline_wan_2p5.py:146`（`./output/wan25`）、`video_official/pipeline_official_video.py:97`（`tmp/pipeline_eval/...`）、`bernini/pipeline_bernini.py:207`（同前）
- 证据：

```97:97:worldfoundry/pipelines/video_official/pipeline_official_video.py
        target = Path(output_path or f"tmp/pipeline_eval/{self.model_id}.mp4")
```

- 问题：产物落点取决于调用者 CWD；`HunyuanMirrorPipeline.__init__` 甚至在构造时就创建目录。库代码不应隐式写 CWD。
- 影响：Studio/评测多任务并发时产物互相覆盖或散落。
- 建议：默认输出统一走 core 的 run-dir/artifact-root API；无 output_path 时报错或用系统临时目录。

#### [PL-17] P2 进程级全局状态被 pipeline 调用路径修改（chdir/sys.path/env/logging）
- 位置：`lyra/lyra_utils.py:471-479`（`os.chdir` 上下文）、`:360-365`（`sys.path.insert(0,...)`）、`:368-372`（`NVTE_FUSED_ATTN` 环境变量）、`matrix_game/pipeline_matrix_game_2.py:145-148`（全局压低 torch 日志级别）
- 证据：

```145:148:worldfoundry/pipelines/matrix_game/pipeline_matrix_game_2.py
        if not visualize_warning:
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)
```

- 问题：`__call__` 一次即永久改变全进程 torch 日志级别；`working_directory` 的 chdir 对多线程不安全（其他线程的相对路径解析全被影响）；sys.path 注入与 env 写入影响同进程所有模型。
- 影响：多 pipeline 共存（Studio 常态）下互相踩踏、诊断信息丢失。
- 建议：日志级别调整移到 CLI/应用入口；chdir 改为绝对路径传参；env/sys.path 注入仅限子进程 env dict。

#### [PL-18] P2 实例状态突变使 pipeline 不可重入
- 位置：`hunyuan_world/pipeline_hunyuan_mirror.py:596-601`（`__call__` 改写 `self.output_path`）、`videocrafter/base.py:154-161`（`__call__` 把本次调用参数 `setattr` 进共享 generator）
- 证据：

```154:161:worldfoundry/pipelines/videocrafter/base.py
        if runtime_overrides:
            runtime_kwargs = getattr(self.synthesis_model, "runtime_kwargs", None)
            if isinstance(runtime_kwargs, dict):
                runtime_kwargs.update(runtime_overrides)
            generator = getattr(self.synthesis_model, "generator", None)
            if generator is not None:
                for key, value in runtime_overrides.items():
                    setattr(generator, key, value)
```

- 问题：单次调用的 `num_frames/ddim_steps` 被永久写进 generator 对象，下一次不带该参数的调用继承上次的值（跨调用泄漏）；`self.output_path` 同理。对比 `world_model/pipeline_runtime_manifest.py:41-48` 用 try/finally 恢复 options 的做法（正确但仍非线程安全）。
- 影响：同一 pipeline 实例服务多请求时结果不可复现。
- 建议：per-call 覆盖通过参数透传到 predict，不落在长生命周期对象上；必须落时用 try/finally 恢复。

### 主题 D：错误处理与静默降级

#### [PL-07] P2 hosted-API 失败路径静默降级为 `video_url=None` + print
- 位置：`worldfoundry/pipelines/wan/pipeline_wan_2p5.py:219-225`
- 证据：

```219:225:worldfoundry/pipelines/wan/pipeline_wan_2p5.py
        else:
            if response and hasattr(response, 'status_code'):
                print(f"API调用失败，状态码: {response.status_code}")
                if hasattr(response, 'message'):
                    print(f"错误信息: {response.message}")
            result['video_url'] = None
            result['task_id'] = None
```

- 问题：API 失败不抛异常、不走 logging，仅 print 后返回带 None 的 dict；批量评测中失败样本会被当作"成功但无产物"继续流转。
- 影响：评测静默丢样本、产物路径 None 的下游崩溃点远离根因。
- 建议：失败抛 `RuntimeError`（或统一的 `PipelineExecutionError`），用 logger 记录。

#### [PL-08] P2 depth_anything 批处理循环零信息吞样本（`except Exception: continue`）
- 位置：`worldfoundry/pipelines/depth_anything/pipeline_depth_anything_v1.py:238-241`（v2 同构 `pipeline_depth_anything_v2.py:167-170`）；图像分支 print+continue（v1:216-218、v2:151-153）
- 证据：

```238:241:worldfoundry/pipelines/depth_anything/pipeline_depth_anything_v1.py
            try:
                raw_frames, metadata = read_video(filename)
            except Exception:
                continue
```

- 问题：视频批处理中读取失败的样本被**无任何输出**地跳过（连 print 都没有）；图像分支虽有 print 但也吞掉异常继续。深度估计常作为 benchmark 的前置步骤，样本悄悄缺失会让指标失真。
- 影响：评测结果样本数不一致且无从发现；坏文件/编解码问题被掩盖。
- 建议：logger.warning 记录被跳过的文件与原因，结果对象中带 `skipped` 列表；严格模式下抛异常。

### 主题 E：导入卫生与包结构

#### [PL-19] P3 顶层 import 卫生良好（无根 `__init__.py`），但子包三种风格并存
- 位置：`worldfoundry/pipelines/`（无 `__init__.py`）；`wan/__init__.py`（空文件）；`ati/__init__.py`（re-export）；约 55 个目录无 `__init__.py`
- 证据：`ls worldfoundry/pipelines/__init__.py` → No such file；`wan/__init__.py` 0 字节；`ati/__init__.py` 做 `from .pipeline_ati import ATIPipeline`。
- 问题：**优点**：`import worldfoundry.pipelines` 不会拉起任何重依赖（评审关注点 5 通过：没有顶层聚合 import，`component_pipelines.py` 也用惰性 target 字符串延迟加载 synthesis/operator）。**缺点**：无 init / 空 init / re-export init 三种风格随机分布，无规则可循；PEP 420 隐式命名空间包对 mypy/某些打包工具不友好；`pyproject.toml:510` 的 `worldfoundry.pipelines.*.runtime*` 排除规则在当前树中已无匹配目录（历史遗留）。
- 影响：新增家族时无所适从；IDE/静态工具解析不稳定。
- 建议：统一为"每个家族一个显式 `__init__.py` + re-export 公开类"；清理 pyproject 失效排除项或注明保留原因。

### 主题 F：配置与参数管理

#### [PL-20] P2 参数管理四种风格并存 + 多处"声明了但被静默忽略"的死参数
- 位置与证据：
  1. 声明式 ClassVar + 严格校验（最佳）：`native_diffusion.py:339-340` 对未识别 kwargs 抛 `TypeError`；
  2. 宽松 dict merge：`cosmos/videocrafter/echo_memory` 的 `_options()`；
  3. 签名过滤静默丢弃：

```164:171:worldfoundry/pipelines/pipeline_utils.py
        supported_kwargs = {
            key: value
            for key, value in init_kwargs.items()
            if key in parameters
            and parameters[key].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return cls(**supported_kwargs)
```

  4. dataclass 风格（仅个别家族）：`kling/pipeline_astra.py:17-45` 的 `AstraConfig` dataclass——本是好做法，但全目录只有零星几家采用，与其余 kwargs 风格并存；
  5. 死参数：`wan/pipeline_wan_2p5.py:145-146` 的 `save_content`/`output_dir` 声明后从未使用；`vggt/pipeline_vggt.py:687,756` 的 `camera_trajectory` 两处声明零引用；`hunyuan_world/pipeline_hunyuan_mirror.py:368` 的 `save_colmap=True` 只创建空 `sparse/0` 目录（:417-419），COLMAP 输出从未实现。
- 问题：用户传 `output_dir` 给 wan2.5、传 `camera_trajectory` 给 vggt、指望 mirror 出 COLMAP——都会静默无效；`from_pretrained` 的签名过滤会把拼错的参数名无声吞掉。
- 影响：配置错误不可见，调参实验结果不可信。
- 建议：以 `native_diffusion.py` 的严格校验为标准；死参数删除或实现；签名过滤路径至少 warning 被丢弃的键。

### 主题 G：显存与生命周期

#### [PL-21] P3 无统一卸载/关闭 API；显存策略在 native 路径已集中，手写路径自担
- 位置：`pipeline_utils.py`（`PipelineABC` 无 `unload/close/__enter__`）；对照 `native_diffusion.py:194-198`（offload 统一走 core `RuntimePolicy`）；`matrix_game/pipeline_matrix_game_2.py:322-328` 与 `hunyuan_world/pipeline_hunyuan_worldplay.py:384-388`（realtime session 各自实现 `reset_realtime`，但只清缓存不卸权重）
- 证据：

```194:198:worldfoundry/pipelines/native_diffusion.py
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner=cls.OWNER,
                ),
```

- 问题：多 pipeline 共存（Studio 切换模型）只能靠 GC 释放显存，没有显式 `unload()`；realtime 系列各自发明 `reset_realtime/prepare_realtime/configure_realtime/stream_realtime` 四件套（matrix_game_2、matrix_game_3、hunyuan_worldplay、helios 均有副本），命名一致但无协议类约束。
- 影响：显存回收时机不可控；realtime 协议漂移风险。
- 建议：`PipelineABC` 增加 `unload()` 默认实现（drop refs + `empty_cache`）；realtime 四件套抽 Protocol。

### 主题 H：性能

#### [PL-22] P3 推理保护与拷贝模式总体尚可，个别点可加固
- 位置与证据：
  - 好的一面：`vggt/official_runtime.py:150-152`（`no_grad` + 按算力选 bf16/fp16 autocast）、`hunyuan_world/pipeline_hunyuan_mirror.py:228-231`（`no_grad`+autocast）；native 家族的采样循环在 `base_models/diffusion_model` 内部统一管理。
  - 可加固：`matrix_game/pipeline_matrix_game_2.py:106-111` 在 pipeline 层直接调 `vae.encode`/`clip.encode_video`，无 `no_grad` 包裹（权重已 `requires_grad_(False)`，实际不建图，但缺 `inference_mode` 的显式保护，且输入若带 grad 会意外建图）；`hunyuan_world/pipeline_hunyuan_mirror.py:442-445` 逐帧 `save_depth_png/npy` 每帧单独 GPU→CPU。
- 建议：pipeline 层直接触碰模型组件的调用统一套 `torch.inference_mode()`；批量转移后再落盘。

### 主题 I：死代码、命名与日志卫生

#### [PL-23] P2 `vggt` 顶层 import 两个未使用的重型 vendored 符号；`matrix_game_2` 有一个必崩的死函数
- 位置：`worldfoundry/pipelines/vggt/pipeline_vggt.py:22-28`、`worldfoundry/pipelines/matrix_game/pipeline_matrix_game_2.py:22-26`
- 证据：

```22:28:worldfoundry/pipelines/vggt/pipeline_vggt.py
from ...base_models.three_dimensions.point_clouds.gaussian_splatting.scene.dataset_readers import (
    storePly,
    fetchPly,
)
from ...base_models.three_dimensions.point_clouds.flash_world.render import (
    gaussian_render,
)
```

`fetchPly`、`gaussian_render` 全文件零引用（仅 `storePly` 在 :423 使用），却在 import 时加载 gaussian_splatting 训练仓的 `dataset_readers` 与 flash_world 渲染栈。`matrix_game_2:22-26` 的 `tensor_to_pil` 对 `torch.Tensor` 调 `.astype(np.uint8)`（Tensor 无此方法），全仓零调用——死且必崩。
- 问题：无谓的重依赖导入放大 import 失败面与耗时；死函数误导后来者。
- 影响：裁剪部署中缺 flash_world 依赖时 vggt 无法 import。
- 建议：删除未用 import 与死函数；重型依赖改为方法内惰性 import（`run_official_scene_export` 已示范此模式）。

#### [PL-24] P2 print/emoji 代替 logging 遍布手写 pipeline（约 60 处）
- 位置（计数）：`vggt/pipeline_vggt.py`（20）、`hunyuan_world/pipeline_hunyuan_mirror.py`（14）、`kling/pipeline_astra.py`（6）、`wan/pipeline_wan_2p5.py`（3）、`hunyuan_world/pipeline_hunyuan_world_voyager.py`（3）等 20 个文件
- 证据：

```218:221:worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py
        print(f"📸 Loaded {S} images with shape {imgs.shape}")
        
        # Inference
        print("\n🚀 Starting inference pipeline...")
```

- 问题：库层 print 无法被日志系统收集/分级/关闭；emoji 输出进评测日志；`hunyuan_world_voyager.py:373` 手工检查 `LOCAL_RANK` 决定是否 print——分布式日志去重应由 logging 框架处理。
- 影响：批量评测日志噪声大、无法定位；与 repo 其余部分的 logging 规范割裂。
- 建议：统一 `logging.getLogger(__name__)`；进度类输出仅在 CLI 层。

#### [PL-25] P3 命名与文档卫生：错位 docstring、双语混杂、易混淆文件名、空 stub
- 位置与证据：
  - `hunyuan_world/pipeline_hunyuan_mirror.py:3-7`：import 之后跟游离字符串字面量（本应是模块 docstring，现在是无效表达式）：

```3:7:worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py
from ..pipeline_utils import PipelineABC
"""
input image and output 3D reconstruction (depth, normal, point cloud, gaussians, colmap)
load operators and WorldMirror representation model
"""
```

  - `wan/pipeline_wan_2p5.py:3` 过时注释"这里面包含了sora2， veo3 和wan2.5"（sora2/veo3 各有独立文件）；
  - `pi3/pipeline_loger.py`：LoGeR 模型文件名拼作 `loger`，与 "logger" 一字之差；
  - `hunyuan_world/pipeline_hunyuan_worldplay.py:390-394`：`save_pretrained` 空 stub（`pass` + 中文 TODO）；`pipeline_hunyuan_mirror.py:652-668` 的 `save_pretrained` 用 `torch.save` 序列化纯 dict 配置（应为 JSON）；
  - 中英文 docstring 混杂（wan_2p5/mirror 全中文，其余英文），与仓库其他层的英文规范不一致。
- 建议：修正 docstring 位置；统一语言；`pipeline_loger.py` 在模块 docstring 里注明是 LoGeR 模型（避免被当成 logger 工具误改）；删除空 stub。

## 汇总

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0（损坏/危险） | 0 | — |
| P1（严重设计缺陷） | 7 | PL-01, PL-02, PL-05, PL-06, PL-09, PL-10, PL-12 |
| P2（应修复） | 14 | PL-03, PL-04, PL-07, PL-08, PL-11, PL-13, PL-14, PL-15, PL-16, PL-17, PL-18, PL-20, PL-23, PL-24 |
| P3（改进建议） | 4 | PL-19, PL-21, PL-22, PL-25 |
| 合计 | 25 | |

### Top 5 问题

1. **[PL-10] `VGGTPipeline.stream()` CLI 分支功能损坏**：生成器函数内 `return 值`，视频路径永远丢失，调用方拿到的生成器一迭代就阻塞在交互式 `input()`。这是范围内唯一"按文档用法必然失败"的路径。
2. **[PL-09] 8 个家族绕过 `NativeVisualDiffusionPipeline` 适配器**：cosmos×4、gen3c、hunyuan_video、vchitect、t2v_turbo 手搓同一套 policy/DiffusionRequest/保存流程（9 处 `DiffusionRequest(`、8 份负种子习语），适配器"防止并行实现"的设计目标正在失守。
3. **[PL-12] `lyra_utils` 造成三重分层违规**：synthesis 反向 import pipelines、helios/video_official 跨家族引用、与 `core.utils.image_utils` 重复实现且行为已漂移；外加 `project_root()` 源码检出假设（PL-14）。
4. **[PL-02+PL-07] 10 个 hosted-API 包装复制同一模板且失败静默**：8 份 `time.sleep` 轮询循环、各自的状态提取器，失败路径 print 后返回 `video_url=None`，批量评测会把失败样本当成功流转。
5. **[PL-05+PL-06] pipelines 向上 import evaluation/studio**：`video_official`、`bernini` 依赖 `evaluation.models.pipelines.invocation`；`cut3r` 依赖 `studio.visualization`——层级方向反转，裁剪部署即 broken。

### 总体评价

pipelines 层呈**双轨状态**：框架侧（`pipeline_utils`/`component_pipelines`/`native_diffusion`/`world_model` manifest/`three_d_four_d`）是高质量的声明式设计——惰性加载、严格参数校验、集中 offload 策略，`import worldfoundry.pipelines` 零重依赖（评审点 5 无问题）；但手写侧（vggt/hunyuan_world/API 包装/matrix_game 等约 30 个文件）大量保留"研究脚本"形态：print/emoji 日志、CWD 相对路径、实例与进程级状态突变、复制粘贴轮询循环、死参数。无 P0：未发现 `torch.load` 反序列化风险（范围内零 `torch.load` 调用）、无硬编码绝对路径、无凭据泄漏（API key 统一走 `resolve_api_key` 环境变量）。修复优先级应放在：vggt stream 修复（1 处）、native 适配器归拢（8 个家族）、`lyra_utils`/`PipelineInvocation`/`depth_to_world_points` 三个错位模块的搬家、hosted-API 基类抽取（10 个文件）。
