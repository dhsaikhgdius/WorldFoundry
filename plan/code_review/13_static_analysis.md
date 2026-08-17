# 静态分析报告（ruff/compileall/导入图/死代码）

> 状态：已完成（2026-08-14）。结论速览：P0×0 / P1×3 / P2×8 / P3×5，见文末汇总。

## 工具与方法

- **ruff 0.16.2**（`python -m pip install ruff --user`，aliyun 镜像；默认镜像 mirrors.cloud.aliyuncs.com SSL 握手失败）
- **Python 3.12.3**（系统解释器；仓库 pyproject 声明 target py310，ruff 按 `target-version = "py310"` 检查）
- 基线：`ruff check worldfoundry --statistics --no-cache`，遵守仓库 `pyproject.toml` 中 `[tool.ruff]` 配置（select E4/E7/E9/F/I，ignore E501，exclude thirdparty、base_models、synthesis、representations、studio/native/world_explorer 等 vendored 目录）
- 扩展扫描：仅对自研层（core/evaluation/cli/mcp/pipelines/operators/runtime/training/studio），命令行覆盖 `--select B,C4,SIM,PLE,PLW,RUF,ARG,PTH --ignore E501`，不改仓库配置文件，全部 `--no-cache`
- 语法检查：`PYTHONPYCACHEPREFIX=/tmp/pycache python -m compileall -q -j 8 worldfoundry`（pyc 写到 /tmp，不污染仓库）
- 导入图/死代码/TODO：/tmp 下的 ast 分析脚本（不写仓库）

## 基线（仓库自身 ruff 配置）结果

**基线不干净：`ruff check worldfoundry --statistics --no-cache` 共 3456 个错误**（仓库自身 select E4/E7/E9/F/I）。其中 2262 个可 `--fix` 自动修复。

Top 规则分布：

| 规则 | 数量 | 含义 |
|---|---|---|
| F401 | 1083 | 未使用 import |
| I001 | 961 | import 未排序 |
| E402 | 274 | import 不在文件顶部 |
| F841 | 240 | 未使用局部变量 |
| F541 | 222 | 无占位符的 f-string |
| F722 | 155 | 前向注解语法错误（见下：jaxtyping 误报） |
| F405/F403 | 131/24 | import * 相关 |
| E722 | 78 | 裸 except |
| F821 | 67 | **未定义名字（潜在 NameError）** |
| F811 | 57 | 重复定义未使用符号 |
| 其余 | ~230 | E7 系列、F601/F402 等 |

按目录：`evaluation` 2997（占 87%）、`studio` 263、`pipelines` 118、`operators` 49、`core` 21、`data` 7。

**关键结论 1（配置缺口）**：按路径特征（`/runtime/`、`/vendor/`、`/official/`、`_official_impl`）拆分，**2528 个（73%）错误位于 evaluation 内嵌的 vendored benchmark 运行时**（如 `evaluation/tasks/execution/runners/*/runtime/`、`tasks/metrics/fvmd/vendor/`）。pyproject 的 `extend-exclude` 只排除了顶层 `thirdparty`、`base_models`、`synthesis`、`representations`，没有覆盖这些 evaluation 内的 vendored 目录 → **仓库自身的 lint 门禁对全包是红的，形同虚设**（任何人跑 `ruff check worldfoundry` 都会淹没在 vendored 噪声里）。

**关键结论 2（自研层仍有 928 个）**：主要是 I001(353)/F401(147)/E402(64)/F841(63) 卫生问题 + 155 个 F722 jaxtyping 误报（`Float[Tensor, "#channel"]` 字符串标注，`studio/visualization/plugins/scene3d/pixelsplat_full/` 与 `studio/visualization/providers/evaluation.py`，应对这些文件 ignore F722 而不是无视）。其余为下述已确认的真实 bug。

### [SA-1] P1 robotics.py 插件模块导入必炸：类体内前向引用注解且无 `from __future__ import annotations`
- 位置：`worldfoundry/studio/visualization/plugins/robotics/robotics.py:1018`
- 证据：`def from_dict(cls, data: dict) -> ExoskeletonCalibration:` 位于 `class ExoskeletonCalibration` 类体内（类定义始于 993 行），文件无 `from __future__ import annotations`（rg 确认）。py310-312 下注解在 `def` 执行时求值，此时类名尚未绑定。
- 问题：模块一旦被 import（且重依赖 dlimp/octo/jax 都装齐后）必然 `NameError: name 'ExoskeletonCalibration' is not defined`。该插件当前不可用。
- 建议：文件头加 `from __future__ import annotations`，或注解改字符串 `-> "ExoskeletonCalibration"`。

### [SA-2] P1 studio catalog 字典重复键 `"cut3r"`，第一个条目（含 aliases）被静默覆盖
- 位置：`worldfoundry/studio/catalog.py:4956` 与 `:7990`（同一个 dict 字面量，F601）
- 证据：4956 行条目含 `aliases: ("cut3r-512", "cut3r_512_dpt_4_64")`、`default_call_kwargs: {"output_type": "all"}`、`default_interactions: ("point_cloud",)`；7990 行条目是另一套完整配置（`default_task_type`、`call_params`…）但**无 aliases、无 notes**。Python dict 后键覆盖前键。
- 问题：第一个条目整体是死配置；`cut3r-512`/`cut3r_512_dpt_4_64` 别名解析静默丢失，且两处配置漂移无人察觉。
- 建议：删除/合并为一个条目，把 aliases/notes 并入保留条目；给 catalog 构建处加重复键检测。

### [SA-3] P2 human_pose.py `draw_mask` 调用不存在的 `alphaMerge` + 变量名拼写错误使 resize 结果被丢弃
- 位置：`worldfoundry/studio/visualization/plugins/perception/human_pose.py:1029-1031`
- 证据：`backgournd = cv2.resize(background, (w, h))`（拼写错误，赋给死变量，F841），下一行 `return alphaMerge(img_rgba, background, 0, 0, ...)` —— `alphaMerge` 全文件唯一出现（rg 确认无定义、无导入，F821），且传入的是未 resize 的 `background`。
- 问题：`draw_mask` 一被调用即 NameError；即便补上函数，resize 结果也没被使用。双重死函数。
- 建议：补实现或删除该函数；修正拼写。

### [SA-4] P3 `evaluation/utils.py` 多模块拼接体，重复 import（F811 x8 / E402）
- 位置：`worldfoundry/evaluation/utils.py:68/69/138/149/216/222`
- 证据：文件按 `# io.py`、`# manifest.py` 等注释分段，各段重复 `from pathlib import Path`、`from typing import Any...`（Read 确认 68-69 行与 8-10 行重复）。
- 问题：可读性差、易产生遮蔽；是历史合并脚本的遗迹。
- 建议：合并头部 import，一次导入。

### [SA-5] P3 `workspace_app.py` 循环变量 `field` 遮蔽 `dataclasses.field`（F402 x5）
- 位置：`worldfoundry/studio/workspace_app.py:843/1128/1612/1623/1709`
- 证据：15 行 `from dataclasses import ... field ...`；843 行 `for field in fields:`（函数内局部遮蔽，Read 确认）。模块级 `field(default_factory=...)` 用在 226 行类体，先于这些函数执行，无运行时故障。
- 问题：纯卫生问题，但未来在这些函数内新增 `field(...)` 调用会踩坑。
- 建议：循环变量改名 `fld`/`input_field`。

### [SA-16] P2 ruff exclude 未覆盖 evaluation 内嵌 vendored 目录，lint 门禁全红形同虚设
- 位置：`pyproject.toml` `[tool.ruff] extend-exclude`（474-494 行）
- 证据：`ruff check worldfoundry` 3456 错，73%（2528 个）位于 `evaluation/tasks/execution/runners/*/runtime/`、`runners/*/official/`、`tasks/metrics/*/vendor/` 等 vendored 子树；exclude 列表只有顶层 thirdparty/base_models/synthesis/representations/studio native。
- 问题：仓库自身 lint 永远不通过，开发者无法用 ruff 守住自研层质量（新增违规淹没在 3400+ 存量噪声中）。
- 建议：extend-exclude 增加 `worldfoundry/evaluation/tasks/execution/runners/*/runtime`、`worldfoundry/evaluation/tasks/execution/runners/*/official`、`worldfoundry/evaluation/tasks/metrics/*/vendor` 等模式（或加 per-file-ignores）；同时对 jaxtyping 文件 ignore F722；然后把自研层清零并在 CI 强制。

## 扩展规则扫描结果（按规则类别，含确认过的真实违例样本）

命令：`ruff check worldfoundry/{core,evaluation,cli,mcp,pipelines,operators,runtime,training,studio} --select B,C4,SIM,PLE,PLW,RUF,ARG,PTH --ignore E501 --no-cache`

**共 8501 个**；按 vendored 路径特征拆分（vendored = 顶层包内出现 `runtime/vendor/official/third_party/vendored` 子目录或 `_official_impl` 文件，不含顶层 `worldfoundry/runtime` 自研包）：**自研层 3029 / vendored 内嵌 5472**。以下只讨论自研层。

自研层规则分布（Top）：PTH 系列合计 ~700（os.path 旧式用法）、B905 zip 无 strict 193、RUF002/001 全角标点 230、RUF100 失效 noqa 172、SIM108 139、ARG001/002 247、RUF012 可变类属性 93、PLW0603 global 85、PLW2901 76、SIM105 60、B006 13、B008 21、B023 12、B904 14、PLE 级 3。（顶层 `worldfoundry/runtime` 自研包仅 22 个，全部为 SIM/RUF/B010 卫生级。）

高价值规则逐类核查结论：

| 规则 | own 数 | 抽查结论 |
|---|---|---|
| PLE0604 | 2 | **误报**：`evaluation/__init__.py:50`、`operators/embodied_action_operator.py:184` 的 `__all__` 用 `*dict` 解包，运行时全是字符串，ruff 静态无法证明 |
| PLE0704 | 1 | 真问题（见 SA-6） |
| B006 | 13 | 真问题但低危：`core/attention/extension_context_parallel.py:21-23,140-142` `num_token_list=[]` 等 6 处；rg 确认函数体内**无 append/变异**，当前只读 → 潜在坑（P3），vendored metrics 里另有零散 |
| B008 | 21 | 半误报：全部是 `torch.device("cuda")` 作默认值（`core/attention/rope.py` x4、`core/vram/initialization.py` x2、clean_fid…），`torch.device()` 是轻量对象构造、不初始化 CUDA，惯用法可接受（P3） |
| B023 | 12 | 已核查 2 处（见 SA-9）：闭包在同一迭代内被同步消费，当前行为正确但极脆弱 |
| PLW0603 | 85 | 集中于 `core/distributed`（57 处，megatron 风格全局进程组状态，属设计选择）；`studio/execution.py:187` `global torch` 懒加载惯用法。真正值得改的是 studio/conda_dispatch 的全局工作线程/GPU 池状态（P3） |
| RUF012 | 93 | 抽查 `operators/vla_native_operator.py:63` 等 6 处 `DEFAULT_INPUT_SCHEMA`：基类 `embodied_action_operator.py:86` 用 `{**self.DEFAULT_INPUT_SCHEMA, ...}` 拷贝后使用，**无共享变异 bug**，属 ClassVar 注解卫生（P3） |
| B904 | 14 | 真问题（P3）：`core/io/config_utils.py:166/219`、`core/structures/tree_utils.py:35/45` 等 except 内 raise 无 from，破坏调试回溯链 |
| SIM105 | 60 | 真问题（P3）：try/except/pass 应改 `contextlib.suppress`，`cli/tui_app.py`、`core/attention/varlen.py` 等 |
| SIM108 | 139 | 风格建议（P3） |
| ARG001/002 | 137/110 | 大多是协议/回调签名保留参数（P3，可用 `_` 前缀显式化）；`core` 57 个 ARG001 值得过一遍确认无逻辑遗漏 |
| PLW1510 | 6 | 半误报：抽查 `evaluation/tasks/embodied/docker_runner.py:24/29/33`，`subprocess.run(...).returncode` 都有显式检查，仅建议加 `check=False` 表意 |
| RUF100 | 172 | 值得清理：大量 `# noqa` 已失效（规则未启用或已修复），说明历史上换过 lint 工具/配置 |

### [SA-6] P2 `human_pose.py` 裸 `raise` 不在 except 内，触发即 RuntimeError
- 位置：`worldfoundry/studio/visualization/plugins/perception/human_pose.py:865`（PLE0704）
- 证据：ruff PLE 级判定 `Bare raise statement is not inside an exception handler`。
- 问题：该分支一旦执行，抛出的不是业务异常而是 `RuntimeError: No active exception to re-raise`。与 SA-3 同文件，该可视化插件质量整体堪忧。
- 建议：改为 `raise ValueError(...)` 等具体异常。

### [SA-7] P2 fire-and-forget `asyncio.create_task` 未保存引用，清理任务可能被 GC 吞掉（RUF006 x2）
- 位置：`worldfoundry/cli/tui_app.py:839`（`asyncio.create_task(self._stop_process())`）；`worldfoundry/studio/visualization/backends/world_realtime.py:1855`（WebRTC channel `on_close` 里 `asyncio.create_task(self.close_active())`）
- 证据：Read 确认两处均未保存返回的 Task；同文件 1857-1859 行的其它 task 均保存在 `active.*` 上，唯独 close 路径没存。
- 问题：事件循环仅弱引用 task，未引用的 task 可能在执行中途被垃圾回收——停进程/关会话的清理逻辑会间歇性丢失，出难查的资源泄漏。
- 建议：保存引用并在 done 回调里释放（官方文档推荐模式），或用 TaskGroup。

### [SA-8] P2 `subprocess.Popen(preexec_fn=os.setsid)` 在多线程 FastAPI 服务内使用
- 位置：`worldfoundry/studio/workspace_app.py:712-720`（PLW1509）
- 证据：Read 确认该 Popen 在 HTTP handler 路径中执行（同函数内 `raise HTTPException`），服务是多线程 web 服务。
- 问题：CPython 文档明确 `preexec_fn` 在多线程程序中不安全（fork 后子进程内执行 Python 代码，可能死锁）。
- 建议：改用 `start_new_session=True`（等效 setsid 且安全）。

### [SA-9] P3 suite 编排里闭包捕获循环变量（B023 x12，当前正确但脆弱）
- 位置：`worldfoundry/evaluation/tasks/execution/orchestration/model_benchmark_suite.py:1321-1330`（`acquire_runner` 捕获循环变量 `plan`/`runner_state`）；`worldfoundry/studio/execution.py:3678-3686`（`persisted_preview` 捕获 `payload`/`recovered_previews`）
- 证据：Read 确认两处闭包都在同一迭代内被同步调用（前者 1332 行传入 `_run_cell` 立即消费；后者 3697-3699 行立即调用），当前无错误。
- 问题：一旦有人把 `run_cell`/`RunRecord` 构造改成延迟/并行执行，全部 cell 会静默使用最后一次迭代的 `plan`——这是 B023 的经典事故模式，且这里是基准测试编排核心。
- 建议：闭包改用默认参数绑定 `def acquire_runner(plan=plan, runner_state=runner_state)` 或改为显式传参。

## 语法编译检查

`PYTHONPYCACHEPREFIX=/tmp/pycache python -m compileall -q -j 8 worldfoundry` → **exit 0，全包无 SyntaxError**（含全部 vendored 目录，未发现 py2 残留）。pyc 全部写入 /tmp，未污染仓库。

产生 **84 条 SyntaxWarning（invalid escape sequence），涉及 18 个文件**，按目录：synthesis 62、evaluation（runners runtime）11、base_models 5、studio（native/world_explorer vendored）5、**core 1**。

- 自研层唯一一处：`worldfoundry/core/distributed/model_parallel_groups.py:89` docstring 里 `\i` 非法转义（P3，加 r-string 前缀即可）。
- 其余全部在 vendored 代码（allegro/irasim/magi/open_sora 等的同一段坏正则 `"\)"+"\("...` 复制传播了 6 份）。
- 注意：CPython 计划将 invalid escape 从 SyntaxWarning 升级为 SyntaxError，vendored 代码届时会批量炸；但属上游问题，跟踪即可。

## 分层违规与循环导入

方法：/tmp/import_graph.py 用 ast 解析自研 10 个顶层目录共 **2769 个模块**的 import（仅模块级，已排除 `if TYPE_CHECKING:` 块与函数内延迟导入），按可疑方向检测 + 互引对检测。

模块级分层违规共 **14 个（模块, 目标层）对**，归为 4 类：

| 方向 | 数量 | 涉及模块 |
|---|---|---|
| runtime → evaluation | 3 | `runtime/{assets,benchmark_repos,conda}.py` import `evaluation.utils` 的 `REPO_ROOT/DATA_ROOT/BENCHMARKS_DATA_ROOT/load_manifest` |
| core → pipelines | 1 | `core/inference.py:34` import `pipelines.gen3c.constants` 的默认 prompt 常量 |
| core → runtime | 6 | `core/{acceleration,attention,kernels,io}` 6 个模块 import `runtime.compile_cache` |
| pipelines → evaluation / studio | 3 | `pipeline_bernini`、`pipeline_official_video` import `evaluation.models.pipelines.invocation.PipelineInvocation`；`pipelines/cut3r/official_runtime` import `studio.visualization.core.geometry` |

另有 `evaluation/__main__.py` → `cli.main`（入口委托，文档声明 cli 包装 evaluation 流程，属刻意设计，不计违规）。

### [SA-10] P1 runtime → evaluation 反向依赖：路径常量放错层
- 位置：`worldfoundry/runtime/assets.py:23`、`runtime/benchmark_repos.py:18`、`runtime/conda.py:21`
- 证据：rg 确认三处模块级 `from worldfoundry.evaluation.utils import REPO_ROOT, DATA_ROOT, BENCHMARKS_DATA_ROOT, load_manifest...`；同文件同时已经 import 了 `worldfoundry.core.io.paths`。
- 问题：runtime（环境/资产管理层）依赖 evaluation（上层业务）才能拿到仓库根路径常量——任何 runtime 使用者被迫连带加载 evaluation 包；且 `evaluation/utils.py` 本身是拼接遗迹（SA-4），根基不稳。
- 建议：把 `REPO_ROOT/DATA_ROOT/BENCHMARKS_DATA_ROOT/load_manifest` 下沉至 `core.io.paths`（大部分能力已在那里），evaluation 层转为 re-export。

### [SA-11] P2 core → pipelines：core 引用具体 pipeline 的业务常量
- 位置：`worldfoundry/core/inference.py:34-37`
- 证据：Read 确认模块级 `from worldfoundry.pipelines.gen3c.constants import DEFAULT_GEN3C_NEGATIVE_PROMPT, DEFAULT_GEN3C_PROMPT`。
- 问题：core（最底层公共设施）耦合了 gen3c 这个具体 pipeline 的默认 prompt，方向颠倒；import core.inference 就会连带加载 pipelines 包。
- 建议：常量移入 core 或通过参数注入；core 不应知道任何具体 pipeline。

### [SA-12] P2 core → runtime：编译缓存被 6 个 core 模块横向引用
- 位置：`core/acceleration/triton_nvfp4.py`、`core/attention/{rope_kernel,triton_piecewise_attention}.py`、`core/io/video_tiling.py`、`core/kernels/{triton_diffusion,triton_group_norm_silu}.py`
- 证据：均为模块级 `from worldfoundry.runtime.compile_cache import ...`。
- 问题：core 的 triton kernel 依赖 runtime 层的 `compile_cache`，说明 `compile_cache` 实际是 core 级基础设施但被放在 runtime；core 与 runtime 形成事实上的双向耦合面（runtime 亦大量 import core）。
- 建议：`runtime/compile_cache.py` 下沉到 `core`（如 `core/compile_cache.py`），runtime 保留 re-export。

### [SA-13] P3 pipelines → evaluation/studio 的零散上向引用
- 位置：`pipelines/bernini/pipeline_bernini.py`、`pipelines/video_official/pipeline_official_video.py`（import `evaluation.models.pipelines.invocation.PipelineInvocation`）；`pipelines/cut3r/official_runtime.py`（import `studio.visualization.core.geometry.depth_to_world_points`）
- 问题：`PipelineInvocation` 是 pipeline 调用契约，被定义在 evaluation 下却被 pipelines 消费——契约类应在更低层；`depth_to_world_points` 是纯几何函数，放在 studio 可视化目录里被 pipeline 反向引用。
- 建议：契约类与纯几何工具分别下沉到 core（或 pipelines 内部）。

**循环导入候选**：互引对 4 组，全部是"包 `__init__` 急切 re-export 子模块 × 子模块经包引用兄弟模块"模式：

- `core.distributed/__init__`（第 5 行起 `from .context_parallel import ...`）↔ `context_parallel.py:33`（`from worldfoundry.core.distributed import torch_process_group`）
- 同一包内 `inference_runtime.py:23`、`pipeline_parallel.py:21`（`from . import model_parallel_groups`）↔ `__init__` 32/61 行反向 re-export
- `core.distributed.sequence_parallel/__init__` ↔ `parallel_state.py:42`（`from . import envs`）

### [SA-14] P3 `core.distributed` 包级循环导入（当前可用但对 import 顺序敏感）
- 位置：见上 4 组
- 证据：实测 `import worldfoundry.core.distributed` 成功（依赖 Python 的子模块绑定回退机制）；但 `__init__` 的 re-export 顺序一旦调整、或子模块改成 `from worldfoundry.core.distributed import <attr>`（属性而非子模块），会立刻 ImportError。
- 建议：子模块之间直接 `from worldfoundry.core.distributed.model_parallel_groups import ...`，不经过包 `__init__` 中转。

## TODO/死代码统计

### TODO/FIXME/HACK/XXX 密度（`#` 注释内，按目录）

| 目录 | 标记数 | 文件数 | kLOC | 密度/kLOC |
|---|---|---|---|---|
| studio（自研） | 27 | 191 | 77.9 | 0.35 |
| evaluation（vendored 内嵌） | 24 | 827 | 173.4 | 0.14 |
| core | 11 | 258 | 68.0 | 0.16 |
| evaluation（自研） | 9 | 615 | 134.3 | 0.07 |
| pipelines | 2 | 164 | 28.7 | 0.07 |
| cli / mcp / operators / runtime / training / data | 0 | 712 | 148.0 | 0.00 |

类型分布：TODO 63、FIXME 5、HACK 5、XXX 0。总体密度极低（自研层约 0.11/kLOC），无恶性 HACK 聚集。值得注意的两点：`training` 112 kLOC 零标记（结合下文死代码结论看，该目录更像"整体搬入后未再迭代"）；studio 密度最高且集中在 `visualization/plugins`（与 SA-1/SA-3/SA-6 同区域，进一步佐证该子树质量最弱）。

### 死代码候选（从未被静态 import 的自研模块）

方法：/tmp/dead_modules.py。候选 = 自研 10 目录内、非 `__init__`/`__main__`/cli/测试文件、且满足：① worldfoundry+test+tests+scripts 下任何 .py 都没有 ast 级 import 它（含函数内与相对导入）；② 其完整点号名不出现在 worldfoundry/configs/docker/docs/scripts/packages/Makefile/pyproject 的任何文本文件中（拦截 importlib/入口点/catalog 字符串引用）。

结果：2768 个自研模块中 **756 个候选（约 151 kLOC）**，但其中 672 个在 evaluation 的 vendored runtime 子树（这些是 runner 用 subprocess 按文件路径调起的脚本，"未被 import"是正常形态，不算死代码）。**剔除 vendored 后自研层候选 172 个**，分布：training 35、studio 27、core 10、operators 4、data 4、runtime 2、pipelines 1、mcp 1、evaluation（自研部分）若干。

已识别的系统性误报类别（诚实披露，均已从"确认死亡"里排除）：
- `training/post_training/{distillation,rl}/*/builder.py、session.py`：由包 `__init__` 的**字符串惰性导出表**（如 `"build_native_senseflow_training_stack": ".senseflow.builder"` + `__getattr__`/`import_module`）按名触达，ast 看不见。但注意：部分符号（如 senseflow builder）除再导出链外**找不到静态消费方**，是否真有人用需 owner 确认。
- `operators/*_operator.py`：经 `_OPERATOR_EXPORTS` 类名→模块字符串注册表动态解析。
- `pipelines/minimax/...`：经 catalog 别名/点号字符串绑定（test/test_hailuo_2p3.py 亦引用）。

以下 5 处为逐个 rg 全库精确复核后**确认无任何引用**的死模块：

### [SA-15] P2 确认死代码：core/runtime/studio 五处约 100 KB 无引用模块
- 位置与证据（每处均 rg 全库复核，无 import、无点号引用、无路径引用）：
  1. `worldfoundry/core/attention/avatar_context_parallel.py`（16.1 KB）、`extension_context_parallel.py`（9.8 KB）、`reference_context_parallel.py`（9.4 KB）——三个 context-parallel 变体全库零引用；其中 extension 版还携带 6 处 B006 可变默认参数。
  2. `worldfoundry/core/utils/cuda_graph.py`（22.8 KB）——全库仅有的"cuda_graph"匹配都是无关标识符（`use_cuda_graph` 参数、`cuda_graph_prewarm_steps` 函数）。
  3. `worldfoundry/runtime/probes.py`（16.1 KB）——仅自引；test 里的 `runtime_probes.py` 是另一个文件；`platforms/base.py` 的"probes"是 docstring 用词。
  4. `worldfoundry/runtime/benchmark_repos.py`（4.9 KB）——零引用，且它本身还是 runtime→evaluation 分层违规者之一（SA-10）：死代码+违规双重身份，删除即可同时消掉一处违规。
  5. `worldfoundry/studio/visualization/plugins/scene3d/pixelsplat_full/encoder_visualization/encoder_visualizer_epipolar.py`（19.7 KB）——`pixelsplat_full` 包按名字动态加载，但该子模块无任何导入方；同文件是基线 own 层错误数第二多的文件（24 个）。
- 问题：死代码持续吃 lint/评审/维护成本，且这些文件恰是静态错误重灾区。
- 建议：与 owner 确认后删除或移入 attic；training 的惰性导出表建议生成端到端引用清单再清理。

## 汇总

**严重度统计（16 项，均经抽查确认）**：P0 × 0，P1 × 3，P2 × 8，P3 × 5。

| 严重度 | 条目 |
|---|---|
| P1 | SA-1 robotics 插件 import 必炸（类体前向注解无 future import）；SA-2 catalog `"cut3r"` 重复键静默覆盖丢 aliases；SA-10 runtime→evaluation 反向依赖（路径常量放错层） |
| P2 | SA-3 `draw_mask` 调用不存在的 `alphaMerge`；SA-6 裸 raise 不在 except 内；SA-7 asyncio 清理任务无引用可被 GC；SA-8 多线程服务用 `preexec_fn`；SA-11 core→pipelines；SA-12 core→runtime compile_cache 放错层；SA-15 五处确认死代码约 100 KB；SA-16 ruff exclude 缺口致门禁全红 |
| P3 | SA-4 utils.py 拼接体重复 import；SA-5 `field` 遮蔽；SA-9 B023 闭包脆弱；SA-13 pipelines→evaluation/studio 零散上向引用；SA-14 core.distributed 包级循环 |

**总体画像**：
- 基线不干净（3456 错），根因是 lint 配置未把 evaluation 内嵌 vendored 划出去（SA-16）；自研层真实存量约 928 个，其中约 155 个是 jaxtyping F722 误报。
- 扩展规则下自研层 3029 个，绝大多数是 PTH/SIM/RUF 卫生级；真正的行为级风险集中在 asyncio 任务丢失（SA-7）、preexec_fn（SA-8）、B023 闭包（SA-9）。
- 语法层面全包干净（compileall 零错误，无 py2 残留）；自研层仅 1 条转义序列告警。
- 架构方向上有一条清晰的主线问题：**路径/缓存/契约类等基础设施被放在上层（evaluation.utils、runtime.compile_cache、evaluation.models.pipelines.invocation），导致 14 处模块级反向依赖**（SA-10/11/12/13），下沉到 core 可一揽子解决。
- 死代码：自研层 172 个未引用候选，其中 5 处（约 100 KB）已逐一确认可删（SA-15）；training 的字符串惰性导出层掩盖了真实可达性，建议出一份端到端引用清单。
- TODO 密度极低（自研 0.11/kLOC），无 HACK 聚集；studio/visualization/plugins 子树同时是 TODO 密度、基线错误、真 bug（SA-1/3/6）的三重重灾区，值得整体评审。

**质量弱点排序（按目录）**：studio/visualization/plugins ≫ evaluation/utils+内嵌 vendored 边界 > core/distributed（global 状态+包循环）> 其余自研层（总体健康）。

## 附录：工件清单

- 报告：本文件
- 临时脚本（/tmp，未入库）：`/tmp/import_graph.py`（导入图/分层/循环）、`/tmp/dead_modules.py`（死代码候选）
- 中间数据（/tmp）：`ruff_baseline.json`、`ruff_ext.json`、`compileall.log`
- 仓库内无任何代码修改。注意：ruff 的 `--no-cache` 实际语义是"禁用缓存**读取**"（`--help` 确认），仍会写 `.ruff_cache/<version>/`；本次运行产生的 `.ruff_cache/0.16.2/` 已删除（更早版本的缓存目录先于本次分析存在，未动）。一次包导入验证产生的 2 个 `core/distributed/__pycache__` pyc 也已删除。compileall 的 pyc 全部走 `PYTHONPYCACHEPREFIX=/tmp/pycache`。
