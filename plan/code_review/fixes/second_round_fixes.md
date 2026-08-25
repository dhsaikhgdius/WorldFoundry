# 第二轮修复日志（orchestrator 直接执行）

记录模块修复 agent 未覆盖、但静态分析/横切/vendored 报告点名的定点修复。每条含证据与验证。

## studio 静态分析定点 bug（模块 agent 完成后补修）

### SA-1 (P1) robotics.py 插件 import 即 NameError — 已修
- 文件：`worldfoundry/studio/visualization/plugins/robotics/robotics.py`
- 问题：`from_dict` 方法（第 1018 行）返回注解 `-> ExoskeletonCalibration` 引用尚未绑定的所属类，且文件无 `from __future__ import annotations`，注解在类体执行时被 eager 求值 → import 抛 NameError，插件不可用。
- 修复：文件首行加 `from __future__ import annotations`（PEP 563，注解变惰性字符串）。
- 验证：`py_compile` 通过；此前若安装了 flax/gym/jax 等重依赖，import 到该类定义即崩，现不再。零行为变化（仅修复崩溃路径）。

### SA-2 (P1) catalog.py 重复 dict 键 "cut3r"，首个条目（含 aliases）被静默覆盖 — 已修
- 文件：`worldfoundry/studio/catalog.py`
- 问题：同一 dict 字面量内 `"cut3r"` 键定义两次（原 4956 与 7990 行）。Python 保留最后者，故首个条目（含 `aliases: ("cut3r-512", "cut3r_512_dpt_4_64")`）完全失效；两条目 `default_model_ref` 同为 `_cut3r_default_ref`（同一模型）。
- 独立核实：`cut3r-512`/`cut3r_512_dpt_4_64` 在 catalog.py 内仅出现于 (a) 死条目的 aliases，(b) `_cut3r_default_ref()` 里的 checkpoint model-ref 字符串（第 2236 行，非 catalog 键/别名）—— 无键/别名冲突，合并安全。
- 修复：删除失效的首个 `"cut3r"` 条目（零运行时变化，其本已被覆盖）；将 `aliases` 合并进存活的第二条目（恢复作者意图，别名解析到同一 CUT3R 模型）。
- 验证：`py_compile` 通过；`"cut3r"` 顶层键计数 2→1；F601 重复键违例 0。
- 备注（行为增量）：`cut3r-512`/`cut3r_512_dpt_4_64` 现能在 studio catalog 解析到 CUT3R（此前解析失败）——因是同一模型的别名，属修复而非语义改变。

## 跨 agent 覆盖缺口（第二轮定点修）

### TR-12/13/14 (P1+P2) Ray 生命周期泄漏 — `training/distributed/ray_runtime.py` — TR-12/13 已修（infra 任务 6）
- 来源：training 配方 review（09/TR-12）发现，但该文件属 training 引擎 agent（08）的目录范围，而 08 的发现（TE-*）未覆盖此 ray 泄漏 → 两个 agent 之间漏掉。
- 问题：`RayDevicePool.setup()` 中 placement group 创建先于 `self._ray` 赋值，`ray.get(ready)` 失败时 `shutdown()` 因 `self._ray is None` 早退，placement group 与 ray 会话在失败恢复场景永久泄漏。
- 修复（infra 任务 6，分支 `cursor/infra-code-review-fixes-2f62`）：
  - `setup()` 全程用局部变量持有 ray 句柄与已建 placement group；`ready()` 等待失败时逐个 `remove_placement_group()` 回滚，若 ray 会话由本次 `setup()` 启动则 `ray.shutdown()`，仅在完全成功后才赋值 `self._ray`/`self._placement_groups`（TR-12）。
  - 新增 `RayDevicePoolConfig.placement_timeout_seconds`（可选、正数校验）作为 `ray.get(ready, timeout=...)` 上界；超时/失败时尝试读取 `placement_group_table()` 诊断并入日志再抛出（TR-13）。
- 验证：`tests/training/test_training_fix_ray_device_pool_lifecycle.py`（8 条，mock ray：成功路径、`ray.get` 失败回滚 PG、自启动会话 shutdown、外部会话不误关、timeout 透传、配置校验）全部通过；`py_compile` 通过。
- TR-14（shutdown 吞错 + 无重启策略）仍 deferred：重启策略需要 owner 决策（重试次数/退避），非最小修复。

### TE-13 (training 层问题，修复落点在 core/) — 已修（infra 任务 6）
- training 引擎 agent 标注 deferred（超出其 training/ 范围）→ 由 infra 任务 6 落地。
- 修复：`core/distributed/fsdp2_sharding.py` 与 `core/distributed/block_fsdp.py` 增加模块级与 `shard_model` docstring，标注 inference/vendored-only，训练一律用 `worldfoundry.training.distributed.apply_fsdp2`（其 fallback 分支静默退化单卡，用于训练是灾难）。
- 验证：`py_compile` 通过；纯 docstring，无行为变化。

### SA-10 (P1) runtime → evaluation.utils 反向依赖 — 已修（infra 任务 6）
- 来源：`13_static_analysis.md` SA-10；`README.md` 交叉核对清单点名 `runtime/{assets,conda,benchmark_repos}.py` 仍 `from worldfoundry.evaluation.utils import ...`。
- 修复：
  - 新建 `worldfoundry/core/io/manifests.py`：`load_manifest` / `manifest_paths` / `load_manifest_collection` / `MANIFEST_SUFFIXES` 的单一权威实现（YAML 解析失败带文件路径重抛）。
  - `worldfoundry/evaluation/utils.py` 改为 re-export（公共契约不变，`from worldfoundry.evaluation.utils import load_manifest` 依旧可用）。
  - `runtime/assets.py`、`runtime/conda.py`、`runtime/benchmark_repos.py` 改 import `core.io.manifests`；`REPO_ROOT`/`DATA_ROOT`/`BENCHMARKS_DATA_ROOT` 改由 `core.io.paths.project_root()`/`package_data_root()` 派生。
- 验证：`tests/runtime/test_runtime_no_evaluation_import.py`（4 条：runtime 三模块 import 不再引入 `worldfoundry.evaluation`；evaluation.utils re-export 与 core 实现同一对象；路径常量两侧一致）+ `tests/core/test_core_io_manifests.py`（11 条）全部通过。

### 待第二轮清理（本文件两处的 F811）
- `robotics.py` 有 4 处 F811（`require_package` 等重复定义/导入）——pre-existing 卫生问题，非本次引入；留待第二轮批量卫生清理时处理（需先确认是重复 import 还是真覆盖）。
