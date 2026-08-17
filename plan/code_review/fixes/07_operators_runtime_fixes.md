# operators 与 runtime 修复日志

> 对应评审报告：`plan/code_review/07_operators_runtime.md`。
> 约束：只改 `worldfoundry/operators/`、`worldfoundry/runtime/`，测试只新建 `test/test_operators_*.py`。
> 环境说明：本机缺少部分第三方依赖（如 `imageio`），相关模块导入冒烟以 `py_compile` + AST 检查代替，见各条记录。

## 已修复

### OR-03 (P0) 删除 BGR/RGB 通道均值猜测启发式（5 处）

- **改动**：
  - `operators/flash_world_operator.py`：`process_perception` ndarray 分支删除 `[...,0].mean() > [...,2].mean()` 翻转。
  - `operators/vggt_operator.py`：同上；docstring 本就声明 ndarray 输入为 RGB。
  - `operators/depth_anything_operator.py`：同上，docstring "RGB or BGR" 改为 "RGB"。
  - `operators/cut3r_operator.py`：同上，docstring 同步修正。
  - `operators/depth_anything_v3_operator.py`：`_to_uint8_rgb` 删除末尾的均值猜测翻转。
- **约定**：ndarray/tensor 输入统一按 RGB 处理（cv2.imread 路径分支仍显式 `COLOR_BGR2RGB`，不受影响）。
- **验证**：`py_compile` 5 文件通过；`rg 'mean\(\) > .*mean\(\)' worldfoundry/operators/` 0 命中。`representations/`、`base_models/` 下同款启发式超出本次修复边界，未动（已在 deferred 备注）。
- **风险**：向 operator 喂 BGR ndarray 的调用方（如直接把 `cv2.imread` 结果传入）不再被"纠正"——这正是评审要求的显式契约；暖色调 RGB 图不再被破坏。

### OR-04 (P2) 归一化/反归一化按 dtype 判断，不再纯值域猜测

- **改动**（与 OR-03 同批文件）：
  - `vggt/depth_anything/cut3r`：`/255` 归一化改为 `np.issubdtype(dtype, np.integer) or max() > 1.0`——整型必除 255（修复全黑 uint8 图误判），浮点保留值域启发式兜底（浮点 0-255 输入仍可用）。
  - `cut3r` PIL 分支：在 astype(float32) 前记录 `is_integer_input`，整型输入必除 255。
  - `flash_world`：float→uint8 方向同理——整型直接 `astype(uint8)` 不再乘 255（修复全黑 uint8 图被 ×255 破坏），浮点保留 `max() <= 1.0` 启发式。
- **验证**：`py_compile` 通过；操作语义仅对"整型且 max≤1"（近全黑图）分支改变，属评审确认的行为 bug 修复。
- **风险**：低。整型输入的正常图（max>1）行为完全不变。

### OR-02 (P2) 统一 `delete_last_interaction` 语义为"空列表抛 ValueError"

- **改动**：
  - `operators/base_operator.py`：基类改为空列表抛 `ValueError("No interaction to delete.")`（fail-fast）。
  - 删除语义与新基类一致的重写：`flash_world`、`cut3r`、`depth_anything`、`vggt`、`pi3`、`infinite_vggt`（6 处，raise 版）；`astra`、`hunyuan_mirror`（2 处，旧基类静默版的逐字拷贝）。
  - **保留** `lingbot_map_operator.py` 的重写：其语义是"清空整个列表"而非"删最后一条"，且被 `pipelines/lingbot_map/pipeline_lingbot_map.py:315` 在 finally 中调用，合并会改变多元素时的行为。已在 deferred 段落备注建议 owner 澄清。
- **调用方核实**：rg 全仓 `delete_last_interaction(` ——全部 pipeline 调用都是 `get_interaction → try process → finally delete` 模式，正常路径删除时列表非空；astra/hunyuan_mirror/pi3/infinite_vggt 的 pipeline 均无调用。
- **验证**：`py_compile` 9 文件通过；功能冒烟：非空删除成功、空列表抛 ValueError。
- **风险**：无 override 的 operator 在"空列表上调用 delete"从静默 no-op 变为抛错。全仓调用方审计显示此路径仅出现在 process_interaction 已先失败的异常分支，异常链（`__context__`）保留原始错误。

### OR-09 (P2) matrix_game_2 官方基准动作支持显式 seed；死代码清理

- **改动**（`operators/matrix_game_2_operator.py`）：
  - `_combine_official_action_data(..., rng=None)`：接受 `random.Random`；缺省仍用模块级 `random`（保留旧行为）。
  - `official_bench_actions_{universal,gta_drive,templerun}(..., seed=None)`、`MatrixGame2Operator.process_official_bench_actions(num_frames, seed=None)`：seed 给定时用独立 `random.Random(seed)`，不污染全局随机态。
  - 删除死代码 `selections = [12]` + `random.randint(0,0)` 恒取 12 的假随机，提为常量 `_OFFICIAL_SEGMENT_FRAMES = 12`。
  - `repeat_time = rd_frame // 4` 的隐式 `num_samples_per_action=4` 假设改为 `rd_frame // k.shape[0]` 并加断言 `rd_frame % samples_per_action == 0`（改采样数时 fail-fast 而非静默错位）。
- **调用方核实**：`scripts/model_zoo/matrix_game2_demo_common.py:108`、`pipelines/matrix_game/pipeline_matrix_game_2.py:120` 均仅用 `num_frames=` 关键字，追加可选参数向后兼容；既有测试 `test/test_matrix_game_2_seed_repro.py` 只断言形状与不等式，不依赖具体随机序列。
- **验证**：功能冒烟通过——同 seed 两次结果 tensor 相等、不同 seed 不等、无 seed 旧路径可用、三种 mode 形状正确（(57,4)/(57,2)/(57,7)）。
- **风险**：无 seed 时全局随机消耗序列有变化（少了一次恒 0 的 randint），本就无复现契约，不构成行为回归。

### OR-11 (P2) operator 层 print→logging；wan_2p2 设备硬编码可注入

- **改动**：
  - `astra_operator.py`：6 处 print → `logger.info`（新增模块级 `logger = logging.getLogger(__name__)`）。
  - `lingbot_world_operator.py`：unknown action → `logger.warning`，轨迹生成 → `logger.info`。
  - `recammaster_operator.py`：短视频告警 → `logger.warning`。
  - `wan_2p2_operator.py`：`QwenPromptExpander(device=0)` → 新增关键字参数 `prompt_extend_device: Union[int, str] = 0` 由调用方注入，默认 0 保持行为不变。
- **验证**：`py_compile` 4 文件通过；`wan_2p2`/`lingbot_world` 导入冒烟通过并断言新参数默认值；`astra`/`recammaster` 因本机缺 `imageio`（预先存在的依赖缺口，两文件改动前顶层就 `import imageio`）无法导入，用 AST 解析确认 `logger` 名字已定义、无语法/名称错误。
- **风险**：日志走 logging 后默认 WARNING 级别下 info 消息不再输出到 stdout——这正是"统一日志"意图；warning 级别照常可见。

### OR-13 (P1) probes.REPO_ROOT 修正；conda.project_root 死代码删除

- **改动**：
  - `runtime/probes.py:22`：`parents[4]` → `parents[2]`（与同包 `performance.py:517` 一致），附路径推导注释。
  - `runtime/conda.py`：删除 `project_root()`（含错误的 `parents[5]` 兜底）。
- **删除符号核实**：rg 全仓 `from worldfoundry.runtime.conda import`、`runtime\.conda`、`conda.project_root`——所有导入方（studio/evaluation/synthesis/test 共 13 处）均不导入 `project_root`；`evaluation/models/runtime/profiles.py:39` 是另一份本地定义，不受影响。
- **验证**：`py_compile` 通过；运行时断言 `REPO_ROOT/pyproject.toml` 存在、`conda` 模块无 `project_root` 属性，均通过。
- **风险**：极低。`REPO_ROOT` 仅用于探针子进程 PYTHONPATH 注入（probes.py 5 处），修正后注入才真正生效。

### OR-14 (P1) `execute_in_tree` 支持超时并击杀进程组

- **改动**（`runtime/in_tree_cli.py`）：
  - 新增可选参数 `timeout: float | None = None`；未传时读环境变量 `WORLDFOUNDRY_IN_TREE_CLI_TIMEOUT_SECONDS`（opt-in 全局兜底）；两者都缺省则保持旧行为（无超时、`subprocess.run`，路径逐字节不变）。
  - 有超时时改用 `Popen(start_new_session=True)` + `communicate(timeout)`，超时后复用 `jobs._kill_process_group` 击杀整个进程组，再收尸；返回结构化 `{"status": "failed", ..., "metadata": {"timed_out": True, ...}}`（与该函数既有的"非零退出返回 failed dict"契约一致，不抛异常）。
  - stdout/stderr 日志落盘逻辑不变。
- **验证**：功能冒烟 5 例通过——超时击杀（含孙进程确认死亡）、无超时成功路径、有超时不触发的成功路径、env var 兜底、日志保留子进程输出。
- **风险**：默认行为不变（None=无超时）；10+ synthesis 调用方无需改动即可通过参数/env var 获得超时保护。

### OR-15 (P1) `AsyncCommandJobStore` 生命周期修复

- **改动**（`runtime/jobs.py`）：
  - (a) `_run` 的 `except Exception` 分支先 `await self._terminate_process(job)`（cleanup 自身异常被抑制，不掩盖原错误）——运行器失败不再遗留占 GPU 的孤儿进程组。
  - (b) `create_subprocess_exec(..., limit=2**20)`（默认 64KiB → 1MiB）；`_read_stream` 捕获 readline 的 `ValueError`（超限行被 StreamReader 丢弃后），记录 marker 行继续读流，作业不再因超长行误标 failed。
  - (c) 新增 `prune(max_age_seconds=None, max_jobs=None)`：只淘汰 terminal 作业；构造器新增可选 `max_jobs`（默认 None 不改行为），设置后 submit 时机会式 prune。
  - (d) `submit(..., timeout=None)`：per-job 超时，经 `asyncio.wait_for` 包裹泵流+等待，超时走 `_terminate_process` 并标记 `failed` + `job.timeout` 生命周期事件；`timeout<=0` 抛 ValueError。
  - 附带：`_terminate_process` 的 `except TimeoutError` 改为 `except (TimeoutError, asyncio.TimeoutError)`——py3.10 上 `asyncio.wait_for` 的超时不是内建 TimeoutError，原代码在 3.10 上永远不会升级到 SIGKILL（仓库 requires-python >=3.10；本机 3.12 行为不变）。
- **验证**：功能冒烟通过——200KB 单行作业 completed 且 JSON result 正确提取；3MB 超限行作业 completed 且含 marker；timeout=2 作业 failed 且孙进程确认被杀；prune 淘汰 terminal 保留 running；timeout=0 拒绝。既有测试 `tests/runtime/test_jobs_logging.py` 通过（pytest -p no:cacheprovider，1 passed）。
- **风险**：limit 提升与异常路径清理为纯防御性；prune/timeout 均为可选新 API，默认不启用。

### OR-16 (P1) `local_checkpoint_cache` 跨进程发布竞争

- **改动**（`runtime/local_checkpoint_cache.py`）：
  - 复制阶段仍可并发（各自 `.tmp-{pid}`）；发布步骤（`rmtree(target)` + `rename`）改为在 `cache_root/.{target}.lock` 上 `fcntl.flock(LOCK_EX)` 内执行，并在锁内**重查 `_is_ready`**：他人已发布有效树 → 丢弃自己的 tmp、直接复用，不再删除并发进程正在读取的目录。
  - 非 POSIX 平台（无 fcntl）降级为无锁（保持旧行为）。
  - `print(...)` → `logger.info(...)`。
- **验证**：跨进程竞争窗口重现测试通过——进程 B 的复制被人为拖到进程 A 发布之后，B 返回同一 target 且 A 的树 inode 不变（未被 rmtree+rename 替换）、无 tmp 残留；正常路径冒烟通过（staging/复用/禁用三态）。
- **风险**：flock 在共享文件系统（NFS/CPFS）上语义视挂载选项而定，最坏退化为"与旧行为相同"，锁内 `_is_ready` 重查在大多数场景仍能消除竞争；同一 `_is_ready` 的读侧无锁（只影响重复拷贝功，不影响正确性）。

### OR-20 (P2) `newest_media` stat 竞态与双重 stat

- **改动**（`runtime/in_tree_cli.py`）：收集阶段一次性缓存 `(mtime, size, path)`，排序用缓存值；文件型 root 的 stat 也补了 try/except（文件在 exists 与 stat 间消失不再炸）。
- **验证**：冒烟通过（新旧文件排序、preferred_names、缺失 root）。
- **未修部分**：并发作业共享 search_roots 时互捡产物属 run 级隔离设计问题，需改 synthesis 调用方（超边界），已列入 deferred。

### OR-17 (P2) assets 文件资产就绪检查补 `st_size > 0`

- **改动**（`runtime/assets.py`）：`LocalAsset.from_manifest_item` 的就绪判定抽为 `_path_is_ready`：目录仍按 `exists()`；**文件**改为 `is_file() and stat().st_size > 0`（stat 失败按未就绪处理）——空文件/下载中断的 0 字节残留不再被标记 `available`。
- **验证**：`py_compile` + 导入冒烟通过；功能冒烟 4 例通过（非空文件 available、空文件 missing、目录 available、缺失 missing）。
- **风险**：合法的 0 字节占位文件会从 available 变 missing——制品清单里的权重/资源不存在合法空文件场景，属评审确认方向。

### OR-18 (P2) env.py 诊断命令异常兜底

- **改动**（`runtime/env.py`）：`_run_command` 包 `try/except (OSError, subprocess.SubprocessError)` 返回 `None`——覆盖 `TimeoutExpired`（驱动挂死时 nvidia-smi 卡住）、`FileNotFoundError`/`PermissionError` 等，preflight 诊断报告不再因诊断命令本身失败而崩溃。
- **验证**：`py_compile` + 导入冒烟通过；功能冒烟：不存在的命令返回 None、`sleep 10`（timeout=5）返回 None 且 ~5s 内返回、正常命令输出不变。
- **风险**：无。诊断字段本就是 `str | None` 契约。

### OR-23 (P2) compile_cache 静默降级加一次性 warning

- **改动**（`runtime/compile_cache.py`）：新增 `_warn_compile_fallback`——`torch.compile` 失败回退 eager 时记 `logger.warning`（含目标名与异常摘要），按 `(目标, 异常类型)` 去重只告警一次（`_FALLBACK_WARNED` set + 锁），避免每次调用刷屏。`strict=True` 路径照旧抛异常不变。
- **验证**：`py_compile` + 导入冒烟通过；功能冒烟：首次降级出 warning、同目标同异常第二次静默、不同目标再次告警。
- **风险**：无行为变化，仅新增日志。

### OR-24 (P3) cuda_tiers env override 移出 lru_cache

- **改动**（`runtime/cuda_tiers.py`）：`detect_nvidia_driver_cuda` 拆为无缓存外壳（每次读 `WORLDFOUNDRY_DETECTED_DRIVER_CUDA` override）+ `@lru_cache` 的 `_detect_nvidia_driver_cuda_from_smi`（昂贵的 nvidia-smi 探测仍只跑一次）。修复"先探测后设 override 不生效 / 测试间 env var 状态泄漏"问题。
- **验证**：`py_compile` 通过；功能冒烟：探测后设 override 立即生效、清除后回落到缓存的探测值、smi 探测只执行一次。
- **风险**：零。override 语义只会更正确；缓存粒度不变。

### OR-25.1 (P3) conda.py `pip_extra_index_url` 缺省规整为 None

- **改动**（`runtime/conda.py`）：manifest 无 `pip_extra_index` 时 `pip_extra_index_url` 存 `None`（原来存 `""`），与字段类型注解 `str | None` 对齐。全仓消费方均为真值判断（`if spec.pip_extra_index_url:`），`None`/`""` 等价。
- **验证**：`py_compile` + 导入冒烟；断言全部 spec 的该字段为 `None` 或非空 str。
- **归因说明**：修后 `test/eval_core/test_official_model_conda_envs.py` 出现 3 处失败，经 HEAD 版 `conda.py`+`cuda_tiers.py` 对照验证：全部 spec 的 `(env_name, driver_compatible, cuda_profile)` 输出与 HEAD **逐字节一致**（ROUTING_IDENTICAL），失败与本次改动无关——该测试文件本身有他人未提交修改（import 路径改动），且对照运行期间 `evaluation/models/runtime/profiles.py` 一度出现 `NameError: BaseSynthesis`（其他 agent 正在共享工作树上做迁移重构）。三处失败分别断言 manifest 数据（driver_compatible 期望值）、shell 脚本 help 文本、openpi 路由期望，均由 `data/` 清单与脚本内容决定，不经过本次修改的代码路径。
- **风险**：零（真值判断等价）。

### OR-25.2 (P3) conda.py 其余项——未修

- 报告中 OR-25 其余小项（如 spec 字段命名、日志措辞）核实后无行为影响或已随 OR-13 的 `project_root` 删除一并处理，不再单独改动。

### OR-12 (P3) 新增 operator 注册表/导入防回归冒烟测试

- **新文件**：`test/test_operators_registry_smoke.py`（357 个参数化用例，纯 CPU、不加载权重、导入期强制 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`）。
- **覆盖**：
  - 静态（不依赖第三方包，必须全绿）：注册表 85 项每项模块文件存在；AST 顶层能定位类名（类定义/别名赋值/re-export）；`__all__` 与 `_OPERATOR_MODULES` 同步；`__getattr__` 对未知名抛 AttributeError；`BaseOperator` 契约方法存在。
  - 动态：每个注册模块可导入 + 注册类可解析且具备 4 个契约方法（`get_interaction`/`process_interaction`/`process_perception`/`delete_last_interaction`）。缺第三方依赖 `pytest.skip`；`worldfoundry` 内部 ImportError/NameError/SyntaxError 一律 fail（靠 `ModuleNotFoundError.name` 根包名区分）。
- **运行结果**：`pytest -p no:cacheprovider` → **347 passed, 10 skipped, 8.5s**。跳过全部为本机缺依赖：imageio×4、plyfile×2、ftfy×2、easydict×2（各为"模块导入+类契约"两条用例）。
- **风险**：无（纯新增测试文件）。

## Deferred 方案

评审明确标注"规划重构"的项不在本轮执行；以下为可执行方案（文件清单/步骤/风险/验证）。防回归安全网已就位：`test/test_operators_registry_smoke.py`（OR-12，348 用例）。

### OR-01 (P1) BaseOperator 契约空心化 —— deferred（规划重构）

- **文件清单**：`operators/base_operator.py`（新增 `InteractiveOperatorMixin`、契约签名）；85 个 operator 文件（分批迁移）；`pipelines/**` 198 处 `process_*` 调用点（只读核对，不必改）。
- **步骤**：
  1. 基类定义 `process_perception(self, input_signal, **kwargs)`、`process_interaction(self, **kwargs) -> dict` 显式签名，规定返回 dict 最小键集合（`images/video/extra_inputs` 任选其一非空）；`get_interaction/check_interaction/delete_last_interaction` 移入 `InteractiveOperatorMixin`。
  2. `operation_types/interaction_template` 二选一：真正启用（`check_interaction` 校验 type ∈ operation_types）或删除。全仓 rg 已证实包外 0 使用，删除风险集中在 operator 自身 `__init__`。
  3. 分批迁移：先 VLA 家族（已有 `EmbodiedActionOperator` 中间基类，改动最小），再 API 家族（与 OR-05 合并做），最后感知类。
- **风险**：签名收紧会让现有关键字调用暴露不匹配（这正是目的，但需要一次全量 pipeline 回归）；建议随下一次新模型接入一并做（评审原话）。
- **验证**：OR-12 冒烟测试的契约用例扩展为断言签名（`inspect.signature` 含 `**kwargs`）；`pytest test/` 全量。

### OR-05 (P1) 11 份 API operator 三件套克隆 —— deferred（规划重构）

- **文件清单**：sora2、veo3、kling_api、wan_2p5、wan_2p6、wan_2p7、hailuo_2p3、luma_ray2、runway_gen4p5、worldlabs、wow 共 11 个 operator；基类候选 `runtime_video_operator.py`（现成 `RuntimeVideoOperator`）。
- **步骤**：
  1. 先固化行为快照：对 11 个类各写一条"`get_interaction→process_interaction` 返回 `{"processed_prompt": ...}`"的参数化测试。
  2. 逐个改继承 `RuntimeVideoOperator`，删除三件套方法，只保留 payload/参数差异；`worldlabs_operator` 的分叉语义（空交互返回空串、check 接受 None）需 owner 决策：保留则显式重写并加注释，否则并入统一语义。
  3. 合并 `world_model_runtime_operator._actions` 与 `embodied_action_operator._as_action_list`。
- **风险**：worldlabs 分叉语义是否有调用方依赖需先审计；其余 10 个 diff 确认逐字一致，风险低。
- **验证**：步骤 1 的快照测试全绿；OR-12 冒烟不回归。

### OR-06 (P2) 图像/视频加载 13+ 份重复 —— deferred（规划重构）

- **文件清单**：下沉目标 `operators/_media.py`（或直接 `core/io`、`core/utils.load_pil_image`）；改造点 `flash_world_operator.py`（80 行手写多态加载）、`neoverse_operator._to_pil_image`、`wan_2p2_operator._load_input_image`、`lyra1_operator.py:8`（从 pipelines 拿 `load_pil_image`，与 OR-07 合并处理）、视频帧加载 5 处（depth_anything_v3/neoverse/recammaster/infinite_world/pi3）。
- **步骤**：以 `core.utils.load_pil_image` + `core.io.video.load_frames_from_video` 为唯一实现源（wow_operator 已示范），逐文件替换手写加载，operator 内只留模型特有 resize/crop/归一化；行为差异点（base64 支持、视频首帧）先写对照表再定统一语义。
- **风险**：各手写实现的边角行为不一致（是否收 base64/bytes），统一后个别调用路径行为变化；需按 operator 逐个核对 pipeline 输入类型。
- **验证**：对照表驱动的单测 + OR-12 冒烟。

### OR-07 (P2) operators 反向依赖 pipelines/synthesis —— deferred（超边界）

- **文件清单**：`yume_operator.py:13`、`hunyuan_worldplay_operator.py:9-14`（模块级 ← synthesis）；`lyra1_operator.py:8`、`lyra_operator.py:12`（← pipelines）；`neoverse_operator.py:147`（函数级 ← synthesis）。迁出目标在 `core/` 或 `data/models` —— 均超出本轮边界（不许改 core/pipelines/synthesis）。
- **步骤**：`YUME_SIZE_CONFIGS` 等常量移 `core` 或模型 manifest；`load_pil_image` 统一走 `core.utils`（与 OR-06 同批）；恢复 `pipeline -> operator -> synthesis` 单向依赖后，在 CI 加 import-linter 契约。
- **风险**：常量迁移需同步更新 synthesis 侧引用（跨包改动）；低风险但触碰面广。
- **验证**：import-linter 规则 + OR-12 冒烟（yume/hunyuan_worldplay 目前因缺依赖 skip，迁移后应转绿）。

### OR-08 (P2) 第二份 operator 注册表 —— 部分缓解，重构 deferred

- **已做**（零风险防漂移）：`test/test_operators_registry_smoke.py::test_embodied_action_compat_registry_consistent_with_package` 断言 `embodied_action_operator._OPERATOR_EXPORTS` 每项与包级 `_OPERATOR_MODULES` 一致（当前 22 项全一致，测试通过）。
- **重构方案**：`embodied_action_operator.__getattr__` 改为代理 `getattr(worldfoundry.operators, name)`（单一事实源），或加 `DeprecationWarning` 并排期删除垫片。改动一处文件，但会改变"从该模块 import 未注册名字"的报错形态，且需确认无循环导入 —— 收益低于风险，随 OR-01 批次一起做。

### OR-10 (P3) flash_world `text_prompt` 恒空 —— deferred（无零风险修法）

- 两种修法都是行为变更：真正透传文本会改变模型输入；从模板/返回值删除 `text_prompt` 键可能破坏读取该键的 pipeline 调用方。P3 且不满足"零行为风险"门槛，留待 owner 决定语义。

### OR-19 (P2) probes.py 硬编码部署快照 —— deferred（规划重构，评审指定）

- **文件清单**：`runtime/probes.py:268-274`（候选环境表）、`probes.py:319-412`（`build_environment_report` 的 30+ 模块清单与 `model_root / "VBench"` 布局）；数据源 `runtime/conda.py`（`load_runtime_conda_env_specs_with_overrides`）、manifest `data/models/runtime/environments/**`（超边界）。
- **步骤**：
  1. 候选环境改由 `load_runtime_conda_env_specs_with_overrides()` 生成（env_name → `envs_root/<env_name>/bin/python`），删除写死的 5 键 dict（其中 `benchmark_cu113` 与 `benchmark_cu113_zeroscope` 本就重复）。
  2. 每环境的模块探测清单挂到 spec 的 `validation_imports` 字段（`conda.py` 已有该字段），manifest 里补数据。
  3. `model_root / "VBench"` 等布局假设移到对应基准 manifest。
- **风险**：探测报告的键名会变（消费方若按键名解析需同步）；manifest 数据补齐属跨目录改动。
- **验证**：对新旧报告做一次字段对照；`pytest test/` 中 probes 相关用例。

### OR-21 (P2) device_pool 仅进程内有效 —— deferred（需跨层设计）

- **文件清单**：`runtime/device_pool.py`（加文档与可选文件锁）；使用方 `studio/conda_dispatch.py`（超边界）。
- **步骤**：docstring 明确"仅进程内互斥"；跨进程方案二选一：per-device token 文件 `fcntl.flock`（复用 OR-16 的锁模式），或交上层调度器；FIFO 公平性与 0.1s 轮询空转顺带修（Condition 通知代替轮询）。`wan_2p2` 绕过池的 `device=0` 已在 OR-11 改为可注入，调用方接线在 synthesis/studio 层（超边界）。
- **风险**：文件锁在共享 FS 上的语义限制同 OR-16；公平性改动需并发测试护航。
- **验证**：多进程抢占冒烟（两进程各 acquire，token 文件互斥）；现有 device_pool 单测。

### OR-22 (P2) env.py 60+ 行基准 env var 启发式 —— deferred（规划重构，评审指定）

- **文件清单**：`runtime/env.py:412-478`（`suggested_sourceable_env_value`）；迁出目标 `data/benchmarks/**` manifest（超边界）。
- **步骤**：各基准 manifest 增加 `env_hints: {VAR: template}` 字段；`suggested_sourceable_env_value` 改为先查 manifest 聚合表、仅保留通用后缀规则（`_RESULTS_PATH`/`_ROOT` 等）作兜底；ChronoMagic/iWorld/IPV/CoppeliaSim 等 20+ 专有条目迁走。
- **风险**：manifest 加载引入 env.py → data 的读依赖（需保持惰性，避免 preflight 变慢）；建议随 worldfoundry-benchmark skill 的 catalog 演进一起做。
- **验证**：迁移前后对全部已知 env var 名跑 `suggested_sourceable_env_value` 输出 diff，应逐字节一致。

### 其余备注

- **OR-03 残留**：`representations/`、`base_models/` 下存在同款 BGR 均值猜测启发式，超出本轮边界（只许改 operators/runtime），需另开批次同法删除。
- **OR-20 残留**：并发作业共享 `search_roots` 互捡产物需 run 级隔离目录（run_id 子目录）或模型 CLI 显式 `--output`，改动在 synthesis 调用方（超边界）。
- **OR-23 残留**：(a) `configure_persistent_compile_cache(namespace=...)` 死参数**不删**——rg 确认 core/kernels、core/attention、studio、synthesis 共 12+ 处边界外调用点显式传参，删除即破坏；启用它会改缓存目录布局（行为风险）。(b) fallback 事件写入 `OptimizationSnapshot.fallbacks`（performance.py:187）属跨模块特性接线，warning 日志已解决核心可见性诉求，接线随性能基线批次做。
- **OR-02 残留**：`lingbot_map_operator.delete_last_interaction` 语义是"清空整个列表"（被 pipeline finally 调用），与基类"删最后一条"不同，保留重写并建议 owner 澄清命名。

## 验证汇总

- 全部改动文件（23 个）+ 新测试文件一次性 `PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile` 通过。
- 终轮导入冒烟：18 个改动模块（runtime 9 + operators 9）`PYTHONPATH=. python -c "import ..."` 全部通过；`flash_world_operator` 因顶层依赖本机缺失的 `plyfile`（预先存在缺口，OR-12 跳过名单可证）以 py_compile + 注册表静态用例覆盖。本机缺 `imageio/plyfile/ftfy/easydict/transformers/diffusers/decord`。
- 删除符号（`conda.project_root`、各 operator 的 `delete_last_interaction` 重写、BGR 启发式）均 rg 全仓确认无残留引用。
- 新增测试：`test/test_operators_registry_smoke.py` —— **348 passed, 10 skipped（全部为缺第三方依赖）**（`pytest -p no:cacheprovider`）。
- 既有测试终轮复跑：`tests/runtime/test_jobs_logging.py` + `test/eval_core/test_cuda_tiers.py` —— **12 passed**。`test/eval_core/test_official_model_conda_envs.py` 3 处失败经 HEAD 对照证实与本批改动无关（OR-25.1 条目"归因说明"，ROUTING_IDENTICAL）。
- 功能级冒烟（不依赖 GPU/网络/权重）：OR-02/09/13/14/15/16/17/18/20/23/24/25.1 各自条目附证据。
