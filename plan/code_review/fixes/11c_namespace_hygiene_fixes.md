# base_models sys.path / 命名空间卫生修复日志（11c）

> 修复人：infra 修复 agent（base_models 命名空间批次）；日期：2026-08-14
> 对应评审报告：`plan/code_review/11_vendored_integration.md`（[VI-2]、[VI-4]、[VI-5]、[VI-14]）
> 约束：只改 `worldfoundry/base_models/`，且不得触碰 7 个由 sibling agent 并行修改的文件（dust3r/mast3r 的 model.py、splatt3r worldfoundry_runtime.py、unik3d.py、unidepthv2.py、cut3r model.py、wan resident.py）；不改 pyproject/MANIFEST/core/evaluation 等。编辑前逐一 `stat -c '%y'` 检查 mtime——本批次全部目标文件 mtime 为 2026-07-21，无并发修改，未触发 skip。
> 验证手段：`python3 -m py_compile` 全部改动文件；实际 import 修好的 wrapper 并打印路径；`types.ModuleType` 伪造冲突模块验证 fail-fast；重复 import/重复调用验证幂等；无 GPU/pypi，深层依赖缺失的 ImportError 按模块来源判定是否既有问题。

## 已修复

### [VI-5] 新增共享设施 `worldfoundry/base_models/_vendor_imports.py`（新文件）
- 文件：`worldfoundry/base_models/_vendor_imports.py`
- 改动：复刻评审点名的两个好模式为 base_models 内可复用的 stdlib-only 模块：
  - `prepend_import_path(path)`：幂等 `insert(0)`——vendored 路径优先于 site-packages，但**绝不 remove 已存在条目**（消灭 remove+insert(0) 抢占）；参考 `evaluation/tasks/metrics/_shared/imports.py` 的模式在本树内本地复刻（不跨层 import evaluation）。
  - `assert_top_level_not_shadowed(name, expected_root)`：若顶层名已被 import 且其 `__file__`/`__path__` 不在期望 vendored 根下，抛出说明冲突双方路径、建议进程隔离的 **ImportError**（open_flamingo `_assert_top_level_package_not_shadowed` 的泛化版）。
- 验证：`py_compile`；被下述全部 wrapper import 并实测行为（见各条）。
- 风险：无重依赖、无 import 副作用；`any(origin under root)` 语义与 open_flamingo 原实现一致（命名空间包多 origin 时任一命中即放行）。

### [VI-2] `general_3d/dust3r/__init__.py`：remove+insert(0) 抢占 → 冲突断言 + 幂等插入
- 文件：`worldfoundry/base_models/three_dimensions/general_3d/dust3r/__init__.py`
- 改动：`ensure_import_paths` 先 `assert_top_level_not_shadowed("dust3r", SOURCE_ROOT)` 与 `("croco", CROCO_ROOT)`，再幂等 prepend（保持 SOURCE_ROOT 在 CROCO_ROOT 之前的原有顺序）；**保留** `IMPORT_PATHS = ensure_import_paths()` 顶层执行与函数名/返回值契约（`tuple[str, str]`，171 处裸 `import dust3r` 与 studio/synthesis 两处显式调用方不受影响）。
- 验证：
  - `import worldfoundry.base_models.three_dimensions.general_3d.dust3r` 成功，`IMPORT_PATHS` 输出两条正确路径；
  - 预置 `sys.modules['dust3r'] = ModuleType`（伪 `__file__` 指向 site-packages）后 import wrapper → 抛出新 ImportError（消息含冲突来源与期望根）；无 `__file__` 的伪 `croco` 模块 → 同样报错（`<unknown origin>`）；
  - 伪造 `__file__` 位于 vendored 根下的 `dust3r` → 放行（不误伤已加载的自家副本）；
  - 重复调用 `ensure_import_paths()` ×2 + 删 `sys.modules` 后重 import → `sys.path` 长度与顺序完全不变、无重复条目。
- 风险：行为变化点是"已加载外来同名副本时从静默错版本变为显式 ImportError"——这正是修复目标（fail-fast 优于静默错版本）；另外已在 sys.path 中的条目不再被强制置顶，极端场景（有人先把 site-packages 版路径手工插到更前面）下解析结果与旧行为不同，但旧行为本身是顺序敏感的 bug 源。

### [VI-2] `general_3d/mast3r/__init__.py`：同型修复
- 文件：`worldfoundry/base_models/three_dimensions/general_3d/mast3r/__init__.py`
- 改动：`ensure_import_paths` 断言 `mast3r`→SOURCE_ROOT、`dust3r`→DUST3R_ROOT、`croco`→DUST3R_ROOT/croco（该函数插入的 DUST3R_ROOT 同时暴露 dust3r 与 croco 两个顶层名），幂等 prepend 保持原迭代顺序；`reexport_dust3r()` 逻辑与顶层 `dust3r = reexport_dust3r()` 执行时机不变（仍显式 reexport canonical wrapper 到 `...mast3r.dust3r`）。
- 验证：import wrapper 成功且 `m.dust3r.__name__` 仍指向 canonical dust3r wrapper；伪造外来 `mast3r` 模块 → ImportError；幂等性同上（重复调用/重 import 后 `sys.path` 稳定）。
- 风险：同上条；splatt3r 运行时对 `ensure_import_paths` 的两处调用（`splatt3r_runtime/model.py`、`worldfoundry_runtime.py`）契约未变。

### [VI-2] `general_3d/mast3r/mast3r/__init__.py` + `mast3r/utils/path_to_dust3r.py`：vendored 包内 bootstrap 去抢占
- 文件：`.../mast3r/mast3r/__init__.py`、`.../mast3r/mast3r/utils/path_to_dust3r.py`
- 改动：
  - 内层包 `ensure_dust3r_import_paths`：断言 `dust3r`→DUST3R_ROOT + 幂等 prepend（原为 remove+insert(0)）；该文件本就 import worldfoundry 模块（reexport），引入 `_vendor_imports` 不新增依赖方向；保留上游版权头，docstring 标注 "Modified by WorldFoundry"。
  - `path_to_dust3r.py`（被 mast3r 的 model/sparse_ga/fast_nn 等 import 时执行）：remove+insert(0) → `not in sys.path` 才 insert(0)，加 "Modified by WorldFoundry" 注释；保持 stdlib-only（不引 helper，该文件需在最小依赖下可执行）。
- 验证：`import worldfoundry...mast3r.mast3r` 全链成功——顶层 `dust3r` 实际解析到 canonical 副本（`general_3d/dust3r/dust3r/__init__.py`），`sys.path` 无重复条目；`py_compile` 通过。
- 风险：内层包新增对 `worldfoundry.base_models._vendor_imports` 的顶层 import——worldfoundry 不可 import 的"独立 clone"场景本就无法工作（reexport 已硬依赖 worldfoundry），无新增破坏面。

### [VI-2] `point_clouds/pi3/__init__.py`：同型修复
- 文件：`worldfoundry/base_models/three_dimensions/point_clouds/pi3/__init__.py`
- 改动：断言 `pi3`→SOURCE_ROOT + 幂等 prepend（原为 remove+insert(0)）；函数签名/返回值（`tuple[Path, ...]`）不变，调用方 `warp_as_history/camera_warp.py` 不受影响。
- 验证：import + 首次调用注册路径后重复调用 ×3，`sys.path` 稳定；伪造外来 `pi3` → ImportError。
- 风险：同 dust3r 条。

### [VI-2][VI-5] `llm_mllm_core/mllm/open_flamingo/__init__.py`：好模式归一到共享 helper，去掉残留抢占
- 文件：`worldfoundry/base_models/llm_mllm_core/mllm/open_flamingo/__init__.py`
- 改动：原有 shadow 断言改为委托 `assert_top_level_not_shadowed("open_flamingo", PACKAGE_ROOT)`（保留 `_assert_top_level_package_not_shadowed` 函数名）；`ensure_import_paths` 的 remove+insert(0) → 幂等 prepend；删除仅内部使用的 `_is_under`（rg 全仓确认无外部引用）。
- 验证：import + 幂等测试通过；伪造外来 `open_flamingo` → ImportError；调用方 `roboflamingo/imports.py` 契约（`PACKAGE_ROOT`、`ensure_import_paths`）未变。
- 风险：**冲突时异常类型由 RuntimeError 变为 ImportError**——rg 确认无人捕获该 RuntimeError，且报告/任务均要求 ImportError；语义更准确。

### [VI-2][VI-14] `path_to_croco.py` ×3：import 时无条件 insert(0) → 幂等 insert
- 文件：`general_3d/dust3r/dust3r/utils/path_to_croco.py`、`general_3d/monst3r/dust3r/utils/path_to_croco.py`、`general_3d/stable_virtual_camera/.../third_party/dust3r/dust3r/utils/path_to_croco.py`
- 改动：三份副本的模块顶层 `sys.path.insert(0, CROCO_REPO_PATH)` 加 `not in sys.path` guard + "Modified by WorldFoundry" 注释。canonical 那份的直接收益：wrapper `__init__` 已注册 CROCO_ROOT 后，裸 `import dust3r` 触发本文件不再产生重复置顶条目（重复条目本身就是一次隐性抢占）。
- 验证：`py_compile` ×3；canonical 份经 inner-mast3r 全链 import 实测执行且 `sys.path` 无重复。
- 风险：仅去重，不改变各自副本"插自己 croco"的语义；三副本收敛是 [VI-8] 范畴（见 Deferred）。

### [VI-4] `slam/mega_sam_runtime/camera_tracking_scripts/test_demo.py`：CWD 相对路径锚定到文件位置
- 文件：`worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/camera_tracking_scripts/test_demo.py`
- 改动：`sys.path.append("base/droid_slam")` → `sys.path.append(str(Path(__file__).resolve().parents[1] / "base" / "droid_slam"))`，保留 append 优先级语义，加 "Modified by WorldFoundry" 标记注释说明上游的 CWD 假设。
- 验证：锚定表达式独立执行确认目标目录存在且含 droid_slam 模块；实际 exec_module 该脚本：在任意 CWD 下 droid_slam 路径均已正确入 `sys.path`，import 终止于 `ModuleNotFoundError: lietorch`——lietorch 是 mega-sam 的 CUDA 扩展依赖，本环境未安装，属**既有环境缺口**（失败点在第三方依赖模块，与本次改动无关）。
- 风险：脚本模式下行为从"仅 CWD=mega_sam_runtime 可用"变为任意 CWD 可用，纯改善。

## Deferred（超出本批次范围或判定风险大于收益）

1. **`worldfoundry/evaluation/.../stevo_bench/.../generate_lingbot_poses.py` 的占位符死路径（[VI-4]）**：evaluation 树，owned by evaluation module agents，本批次未触碰。
2. **helper 提升到 `worldfoundry.core` 级公共设施 + CI lint 禁裸写 `sys.path.insert`（[VI-5] 报告建议）**：本轮按约束只在 base_models 内落地 `_vendor_imports.py`；core 级晋升 + evaluation/synthesis 两树接入 + CI 守护列为第二轮工作。
3. **monst3r / stable_virtual_camera 的 dust3r/croco 副本收敛（[VI-8]/[VI-14]）**：两副本的 `path_to_croco.py` 本轮只做了幂等 guard，**未加 shadow 断言**——它们与 canonical 竞争同一批顶层名，加断言会把当前"靠 import 顺序碰巧工作"的流程变成硬失败，正确解法是副本收敛（variant 注入或顶层名改名），属 [VI-8] 专项。
4. **CROCO_ROOT 入 path 暴露的通用顶层名 `models`/`utils`（[VI-14]）**:对这两个名字加断言会大面积误报（多树合法暴露同名），需按报告建议改写 vendored import（`croco.models.*`）后才能收紧，本轮不动。
5. **函数内/脚本级 sys.path hack（[VI-2] 的 75 处函数内命中，base_models 子集）**：modelscope_swift 各 model 文件的 `local_repo_path` append、yolo_world/opens2v_nexus 的 `paths.py`、dreamsim/hpsv3/dvlt/lagernvs/grounding_dino slconfig、svc seva preprocessor 等均为函数内执行（非 import 时），grit ×2 / mmyolo tools ×3 / monst3r raft.py / mvdiffusion pano_video_generation.py 为上游脚本的锚定式 append（非抢占、非 CWD 相对或仅脚本模式执行）——按任务约定 out-of-batch，仅记录。
6. **七个并行修改中的文件**：`dust3r/dust3r/model.py`、`mast3r/mast3r/model.py`、`splatt3r/worldfoundry_runtime.py`、`unik3d.py`、`unidepthv2.py`、`cut3r/model.py`、`wan/resident.py` 由 sibling agent 处理（git status 已见其改动），本批次未触碰。
