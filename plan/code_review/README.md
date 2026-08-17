# WorldFoundry Infra 代码评审（进行中）

评审目标：`worldfoundry/` 包作为一个 infra 框架的合理性审查——分层与依赖方向、API 设计、导入卫生、全局状态、错误处理、资源管理、配置一致性、安全性、可测试性、重复代码等。

评审方式：14 个并行专项 agent，每个 agent 负责一个模块范围，持续写入各自的报告文件。全部完成后在 `SUMMARY.md` 汇总。

修复方式：某模块 review 完成后即启动对应修复 agent（文件范围与 review 范围一致、互不重叠）；修复日志在 `fixes/` 子目录。跨模块共享文件（pyproject.toml、worldfoundry/__init__.py 等）的修改留到最后统一处理。工作区在修复开始前已有 424 个既有未提交改动（快照存于 /tmp/wf_pre_fix_git_status.txt），修复只新增改动，不碰既有改动。

## 报告索引

| 报告文件 | 范围 | 评审 | 修复 |
| --- | --- | --- | --- |
| `01_core_foundation.md` | core/：configuration, registry, model_loading, checkpoint, io, structures, utils, logging | ✅ P0×1 P1×9 P2×26 P3×12 | ✅ 已修复 40 项（P0 原子写自愈 + 9×P1 全部 + 26×P2 + 12×P3）32 测试；eval_core collect 1512/exit0；15 项 deferred → `fixes/01_core_foundation_fixes.md` |
| `02_core_compute.md` | core/：attention, kernels, nn, acceleration, memory, vram, distributed, inference, realtime | ✅ P0×1 P1×10 P2×17 P3×10 | ✅ 已修复 23 项（P0 dist_init + attention 链 + patch 恢复 + TF32/mask 数值修正）47 CPU 测试通过；13 项 deferred → `fixes/02_core_compute_fixes.md` |
| `03_evaluation_framework.md` | evaluation/：framework, runner, public, api/, models/, reporting/ | ✅ P1×4 P2×17 P3×17 | ✅ 已修复 26 项（EF-30 profiles torch 669→0 / EF-17/08/01 + runner 445→208ms）collect 1516/0，22 测试；移交 EF-34→core → `fixes/03_evaluation_framework_fixes.md` |
| `04_evaluation_tasks.md` | evaluation/tasks/ 的 schema、注册表、runner 接线层 + data/ | ✅ P1×5 P2×12 P3×7 | ✅ 已修复 18 项（5×P1：failed scorecard/mtime/fail-fast/normalizer/registry）collect 1516/0err，36 测试；6 deferred → `fixes/04_evaluation_tasks_fixes.md` |
| `05_cli_mcp.md` | cli/, mcp/, __main__, 包入口 | ✅ P0×1 P1×4 P2×14 P3×14 | ✅ 已修复 19 项（CM-01 --help 3.24s→0.27s 去 torch / CM-07 json 错误契约 / CM-25 MCP 封套 / CM-26 async）13 测试；14 deferred → `fixes/05_cli_mcp_fixes.md` |
| `06_pipelines.md` | pipelines/ | ✅ P1×7 P2×14 P3×4 | ✅ 已修复 13 项（PL-10 stream bug + 死码/路径/状态泄漏 + 分层惰性化）；17 项 deferred（跨包搬家/大重构/需GPU）→ `fixes/06_pipelines_fixes.md` |
| `07_operators_runtime.md` | operators/, runtime/ | ✅ P0×1 P1×6 P2×14 P3×4 | ✅ 已修复 16 项（P0 BGR + 4×P1）+ 冒烟测试 348pass；OR-01/05/06/07/19/21/22 deferred（重构/跨边界）→ `fixes/07_operators_runtime_fixes.md` |
| `08_training_engine.md` | training/：engine, distributed, checkpoint, data, optimizers | ✅ P1×2 P2×5 P3×9 | ✅ 已修复 7 项（TE-01 FSDP2 一致判定 + TE-11 lr scheduler 接线 + TE-02/05/08）25 测试通过；TE-06/07/12/13 deferred → `fixes/08_training_engine_fixes.md` |
| `09_training_recipes.md` | training/：api, models, objectives, post_training, recipes, tuning, safety | ✅ P1×2 P2×5 P3×16 | ✅ 已修复 10 项（TR-10 校验链声明化 + TR-6/15/16/18/20）152 测试通过；TR-12/13/14 ray 泄漏移交第二轮 → `fixes/09_training_recipes_fixes.md` |
| `10_studio.md` | studio/ | ✅ P1×1 P2×8 P3×2 | ✅ 已修复 9/11（ST-2 deferred=pyproject；ST-9 验证后不删=测试在用）→ `fixes/10_studio_fixes.md`；SA-1/SA-2 已补修 → `fixes/second_round_fixes.md` |
| `11_vendored_integration.md` | base_models/, synthesis/, representations/ 的集成层卫生 | ✅ P0×1 P1×10 P2×10 P3×2 | 修复中（第二会话，4 个并行 agent，只碰三棵 vendored 树内文件）→ `fixes/11a_vendored_security_fixes.md` `fixes/11b_synthesis_hygiene_fixes.md` `fixes/11c_namespace_hygiene_fixes.md` `fixes/11d_provenance_fixes.md`；pyproject/MANIFEST/去重删除/批量 docstring 清理仍留第二轮 |
| `12_cross_cutting.md` | 全库横切关注点：全局状态、安全、子进程、env var、路径处理 | ✅ P1×3 P2×14 P3×8 | 第二轮（多数点与模块报告重叠，待模块修复后统一处理不重叠项） |
| `13_static_analysis.md` | ruff / 语法 / 导入静态检查结果 | ✅ P1×3 P2×8 P3×5（agent 完成版） | 交叉核对：见下方"静态分析交叉核对"清单 |
| `14_tests_ci.md` | test/ 布局、覆盖缺口、CI | ✅ P0×3 P1×3 P2×4 P3×2 | ✅ 修复完成（TC-01 CI 加 pytest / TC-02 collect 18err→0 / TC-03 conftest 隔离 44 脚本 / pytest.ini）+ 9 项 owner 移交清单；最终 eval_core 实跑数字并入 orchestrator 终检 → `fixes/14_tests_ci_fixes.md` |
| `SUMMARY.md` | 汇总 + 优先级排序的行动清单 | 待汇总 | — |

## 静态分析交叉核对（确保被修复，不遗漏）

静态分析（13号）发现的具体 bug 中，部分**不一定在对应模块的 review 报告里**，需在模块修复 agent 完成后逐一核验是否已修，未修则第二轮定点修：

- **SA-1 (P1)** `studio/visualization/plugins/robotics/robotics.py:1018` 类体内前向引用注解且缺 `from __future__ import annotations` → import 即 NameError（studio agent 范围，但报告 10 未必含此插件）
- **SA-2 (P1)** `studio/catalog.py` 重复 dict 键 `"cut3r"`（4956/7990 行），前者含 aliases 被静默覆盖（studio agent 范围）
- **SA-3 (P2)** `studio/.../human_pose.py` `draw_mask` 调用不存在的 `alphaMerge` + resize 结果被拼写错误丢弃（studio agent 范围）
- **SA-6 (P2)** `human_pose.py` 裸 `raise` 不在 except 内（studio agent 范围）
- **SA-8 (P2)** `subprocess.Popen(preexec_fn=os.setsid)` 多线程 FastAPI 内使用 ≈ 报告 10 的 ST-8（studio agent 应已覆盖）
- **SA-10 (P1)** `runtime/{assets,conda,benchmark_repos}.py` → `evaluation.utils` 反向依赖（operators/runtime agent 范围，报告 07 未必含）
- **SA-11/12 (P2)** `core → pipelines`（业务常量）/ `core → runtime`（compile_cache 被 6 个 core 模块引用）≈ 报告 01 的 CF-1/CF-2（core 基础层 agent 应已覆盖）
- **SA-13 (P3)** `pipelines → evaluation/studio` 上向引用 ≈ 报告 06 的 PL-12（pipelines agent 应已覆盖）
- **SA-14 (P3)** `core.distributed` 4 组包级循环导入（core 计算层 agent 范围）
- **SA-15 (P2)** 5 处约 100KB 死模块（context-parallel 变体、`runtime/probes.py` 等）→ 需 rg 全仓确认无引用后删除（第二轮）
- **SA-16 (P2)** ruff exclude 未覆盖 evaluation 内嵌 vendored 目录 → lint 门禁全红（pyproject，第二轮）

## 严重度定义

- **P0**：损坏的/危险的行为（bug、数据损坏风险、安全漏洞、崩溃路径）
- **P1**：严重设计缺陷，影响框架的可维护性/可扩展性/正确性
- **P2**：应当修复，但影响面可控（一致性、健壮性、性能）
- **P3**：改进建议（风格、文档、命名）
