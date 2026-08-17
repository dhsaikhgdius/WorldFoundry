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

### TR-12/13/14 (P1+P2) Ray 生命周期泄漏 — `training/distributed/ray_runtime.py`
- 来源：training 配方 review（09/TR-12）发现，但该文件属 training 引擎 agent（08）的目录范围，而 08 的发现（TE-*）未覆盖此 ray 泄漏 → 两个 agent 之间漏掉。
- 问题：`RayDevicePool.setup()` 中 placement group 创建先于 `self._ray` 赋值，`ray.get(ready)` 失败时 `shutdown()` 因 `self._ray is None` 早退，placement group 与 ray 会话在失败恢复场景永久泄漏。
- 处置：training 引擎 agent 已完成，`training/distributed/` 现无 agent 占用 → 第二轮由 orchestrator 定点修复 + 单测（mock ray 让 `ray.get` 抛异常，断言 placement group 被移除）。

### TE-13 (training 层问题，修复落点在 core/)
- training 引擎 agent 标注 deferred（超出其 training/ 范围）→ 并入 core 第二轮。

### 待第二轮清理（本文件两处的 F811）
- `robotics.py` 有 4 处 F811（`require_package` 等重复定义/导入）——pre-existing 卫生问题，非本次引入；留待第二轮批量卫生清理时处理（需先确认是重复 import 还是真覆盖）。
