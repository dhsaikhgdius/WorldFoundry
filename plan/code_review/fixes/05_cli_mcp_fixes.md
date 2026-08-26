# CLI/MCP 层修复日志

> 修复人：infra 修复 agent；日期：2026-08-14
> 对应评审报告：`plan/code_review/05_cli_mcp.md`（33 条发现 CM-01～CM-33）
> 约束：只改 `worldfoundry/cli/`、`worldfoundry/mcp/`、`worldfoundry/__main__.py`；不动 `worldfoundry/__init__.py`、`worldfoundry/core/`、其它子包、pyproject、既有测试、docs；pypi 不可用（不装新依赖）；不 git commit。
> 验证手段：`py_compile`；只读 CLI 命令实测（`--help`、`zoo benchmarks --json`、`zoo benchmark-show`、`run --help`）；`python -X importtime | rg torch` 查询路径核查；假 FastMCP 对象注册冒烟；`test/eval_core` 修复前后对比（基线含 18 个收集错误，属评测侧另一 agent 的修复范围，本次改动不新增）；新增回归测试 `test/test_cli_mcp_fix_contracts.py`（13 项全过）。
> 环境注记：本机未安装 `mcp`/`fastmcp`/`textual` 包——`import worldfoundry.mcp` 冒烟通过（惰性导出），`create_mcp_server()` 在缺包时按既有设计抛带指引的 RuntimeError；MCP 侧验证经由 payload 函数直调 + 假 mcp 对象完成。

## 启动时间前后对比（实测）

| 命令 | 修复前 | 修复后（当日首测） | 备注 |
| --- | --- | --- | --- |
| `python -m worldfoundry.cli --help` | 3.24s | 0.27s | torch 不再加载 |
| `python -m worldfoundry.cli zoo benchmarks --json` | 4.13s | 1.07s | torch 不再加载；imports 仅 ~236ms，剩余 ~0.8s 是评测层 manifest 全量解析（见"移交"） |

`python -X importtime -m worldfoundry.cli --help 2>&1 | rg torch` 与 `... zoo benchmarks --json ... | rg torch` 均为空（查询路径零 torch，修复后多次复测均成立）。

**复测注记（当日晚间，环境已变化）**：--help 复测 1.9–2.2s、zoo --json 复测 8.5–10.5s。经 importtime 归因，膨胀全部来自 cli/mcp 边界之外、且与本次修复无关：
- 评测层 agent 的未提交改动使 `worldfoundry.evaluation.utils` 从报告时的 ~5ms 涨到 ~610ms（新增 `core.io.serialization`/`evaluation.api`/`yaml`/`importlib.metadata` 链）——正是 CM-06 预警的脆弱点成真，佐证"utils 保持纯 stdlib"约定需要评测层落实；
- 同机 4 个并发 `pytest test/eval_core`（其他 agent）打满共享网络盘，裸 `python -c "pass"` 都要 0.307s（site+numba redirector ~290ms）；
- zoo --json 的 handler 段（manifest 全量解析+新出现的 status 归一化警告）来自评测层当前在改的代码。
- 本次修复负责的 cli/mcp 段（parser 构建 + 分发）经 importtime 核对无回归：`cli.main` 自身（扣除 evaluation.utils）~185ms，torch-hits=0。

剩余瓶颈与移交事项：

- `core/distributed/__init__` 的急切 re-export 链（torch）未动——凡真正配置日志的命令仍要付出该成本，属 core 计算层修复范围（评审已注明）。
- `zoo benchmarks` 剩余耗时在 `worldfoundry/evaluation` 的 manifest/registry 全量加载，超出本次 cli/mcp 边界，移交评测框架层。
- `evaluation.utils` 顶层变重（见复测注记）移交评测框架层核实其修复是否保留轻量顶层。

## 已修复

### [CM-01] P0 `--help`/纯查询命令 4 秒（configure_logging 无条件拉起 torch）
- 文件：`worldfoundry/cli/main.py`
- 改动：`main()` 不再无条件 `_configure_cli_logging`。改为三类触发：(1) 显式 `--log-level/--log-file/--log-json` 或 `WORLDFOUNDRY_LOG_*` 环境变量（保持今日急切行为）；(2) `_EAGER_LOGGING_COMMANDS`（tui/mcp/train* 等长驻或自有工作负载的命令）；(3) 解析后 namespace 带非空 `output_dir` 且非 dry-run（`plan_only`/`print_config` 除外）。`--tui` 快捷路径保持急切配置（交互长会话）。命令结束/失败的 `get_logger().event()` 一律加 `logging_setup.is_configured()` 守卫，未配置路径零日志副作用。
- 验证：见上表实测；`rg torch` 空；`worldfoundry train --help`、`worldfoundry mcp --help` 等仍正常。

### [CM-02] P1 `evaluation_intents.py` 顶层导入 orchestration.service（~0.7s）
- 文件：`worldfoundry/cli/evaluation_intents.py`
- 改动：`GenerateAndScoreIntent`/`ScoreArtifactsIntent`/`prepare_evaluation` 等重导入从模块顶层下沉到 `_handle_score`/`_handle_reproduce`/`_handle_generate_score`/`_finish` 内部，与同目录其它 register 模块惯例一致；注册函数只留常量。
- 验证：新回归测试断言 `_build_parser()` 之后 `sys.modules` 无 `orchestration.service` 与 torch。

### [CM-01 关联] `training_commands/register.py` 顶层导入全部 handler
- 文件：`worldfoundry/cli/training_commands/register.py`
- 改动：6 个 handler 模块的导入移入 `register_training_subparser()` 体内（注册期才导入，构建根 parser 的其它命令不再付出该成本）。

### [CM-04] P2 `--task-type/--benchmark-name/--data-path` 死路提示误导
- 文件：`worldfoundry/cli/main.py`
- 改动：`run` 缺参错误从"三旗标 required"改为指向现行入口：`run --benchmark <id> --model <id>` 或 `run <model-id>`，并明说 legacy 三旗标流程已裁撤；三旗标 help 文本标注 retired。旗标本身保留（向后兼容约束：不删名）。
- 验证：`run`（无参）实测输出新提示，exit code 不变。

### [CM-07] P1 `--json` 模式错误契约中断；运行期失败与用法错误共用 exit 2
- 文件：`worldfoundry/cli/main.py`
- 改动：兜底异常处理重写——`--json` 时 stdout 输出 `{"status":"error","command":...,"error":{"type","message"},"exit_code":1}`；人类模式 stderr 一行 `error: <Type>: <msg>`；traceback 仅 `-v/--verbose`（新增根旗标，argv 预扫描解析）；运行期失败 exit 1，argparse 用法错误维持 exit 2，`KeyboardInterrupt` 维持 130。既有 JSON 成功输出字段未动（只增错误形态）。
- 验证：`zoo benchmark-show --benchmark-id does-not-exist-xyz --json` → exit 1 + 可解析 JSON（stdout）+ 简洁 stderr 无 Traceback；`definitely-not-a-command` → exit 2 + usage。两者由新回归测试固化。

### [CM-09] P2 dry-run 命令也建 `logs/` 并改写进程环境
- 文件：`worldfoundry/cli/main.py`（`_prepare_cli_run_observability`）
- 改动：`plan_only`/`print_config` 的 namespace 直接跳过 observability（不建目录、不 `configure_logging(force=True)`、不写 `WORLDFOUNDRY_LOG_FILE/RUN_ID` 环境变量），只保留日志上下文绑定。
- 验证：`run --plan-only ...` 不再生成 `benchmark_results/logs/`。
- 剩余（deferred）：把 observability 进一步下沉到真正执行工作负载的 handler、环境变量只在 spawn 子进程时注入——影响面大，见 Deferred。

### [CM-11] P2 `cli/embodied.py` 整模块死代码（163 行，与 main.py 内联实现漂移双拷贝）
- 改动：删除 `worldfoundry/cli/embodied.py`。
- 验证：`rg register_embodied_subparser` 全仓仅定义处；`test_evaluate_embodied_unification.py` 中的 "cli-embodied" 是 run-id 字符串字面量非 import。`worldfoundry embodied --help` 走 main.py 内联版，行为不变。

### [CM-12] P3 `cli/models.py` 带越界 bug 的死常量
- 改动：删除 `REPO_ROOT = Path(__file__).resolve().parents[3]`（指向仓库父目录）与 `SRC_ROOT`；本文件零使用。

### [CM-13] P3 `_task_roots_from_args` 两处逐字重复
- 改动：合并为 `worldfoundry/cli/utils.py::task_roots_from_args`；`tasks.py`、`plan_metric.py` 改 import 别名，本地重复体删除。
- 备注：报告同条提到的 `_add_task_selector` 同名不同义（main.py 必填 vs tasks.py 可选）未合并——两处语义确实不同，强行统一会改变某侧命令契约；deferred（改名建议见报告）。

### [CM-14] P3 `zoo.py` `_dispatch_zoo_handler` 字符串间接层
- 改动：全部 `set_defaults(func=_dispatch_zoo_handler("..."))` 改直接函数引用；删除 helper。静态可达性恢复。

### [CM-19] P3 `train-audit-rollout-prompts` 被根 help 隐藏
- 改动：加入 `main.py::_PUBLIC_ROOT_COMMANDS`；根 `--help` metavar 现列出该命令。

### [CM-21] P2 `worldfoundry tui --write-suite-plan` 未注册（三处镜像漂移实例）
- 改动：main.py 的 tui 子命令 parser 补 `--write-suite-plan`，`_handle_tui` 补转发。`worldfoundry tui --write-suite-plan ...` 与 `worldfoundry-tui` 入口行为对齐。
- 备注：三处镜像的结构性合并（parent parser / parse_known_args）见 Deferred。

### [CM-24] P3 小工具函数跨模块逐字重复
- 改动：新增 `cli/utils.py::append_optional_arg`；`zoo.py` 的 `_add_path_arg`/`_add_value_arg` 与 `tui_discovery.py` 的 `_append_optional`/`_append_optional_path`（4 份同体）改为别名绑定，调用点零改动。
- 备注：报告同条点名的 `_dedupe_text`（model_run.py vs tui_discovery.py）**未合并**——两者语义不同（model_run 版按 `_normalise` 归一化去重、tui_discovery 版精确文本去重），合并任一方向都会改变去重行为；记录为报告勘误性备注。

### [CM-25] P1 `responses.py` 错误封套死代码；27 个工具裸抛异常
- 文件：`worldfoundry/mcp/tools/responses.py`、`worldfoundry/mcp/tools/registration.py`
- 改动：(1) `responses.py` 新增 `_normalized_error`，`invoke_tool`/`invoke_tool_async` 从白名单异常改为捕获全部 `Exception`（白名单外的 OSError/TypeError/YAML 错误等映射为 `error_type="internal"`，不再裸抛给 MCP 客户端）；(2) `registration.py` 全部 27 个 `@mcp.tool()` 接线 `invoke_tool`/`invoke_tool_async`。成功响应统一带 `ok: true`（字段只增不删），失败响应 `{"ok": false, "error": ..., "error_type": "error"|"runtime"|"internal"}`。
- 验证：假 FastMCP 注册后 `list_runs(limit=0)`、`evaluate(wait="bogus")` 返回结构化封套而非抛异常（新回归测试固化）。

### [CM-26] P1 Studio 等待类工具同步 `time.sleep` 轮询 + 默认无限超时
- 文件：`worldfoundry/mcp/tools/studio.py`、`worldfoundry/mcp/tools/registration.py`
- 改动：(1) 新增 `DEFAULT_STUDIO_WAIT_TIMEOUT_S = 600.0`，`submit_studio_inference_payload`/`wait_for_studio_job_payload` 及新 async 变体的等待默认值从 0（无限）改为 600s，0/负值仍表示无限（允许覆盖）；(2) 新增 `submit_studio_inference_payload_async`/`wait_for_studio_job_payload_async`：HTTP 调用经 `asyncio.to_thread`、轮询经 `asyncio.sleep`（对照组 `evaluate` 的实现模式）；(3) registration 中 `submit_studio_inference`/`wait_for_studio_job` 改 `async def` 调 async 变体——事件循环在整个等待期间保持响应。同步 payload 保留给一次性脚本（docstring 注明服务器代码必须用 async 变体）。
- 验证：事件循环响应性冒烟（等待期间并发 asyncio 任务持续推进）；签名默认值由新回归测试固化。

### [CM-29] P2 MCP 默认输出/数据路径相对 CWD
- 文件：`worldfoundry/mcp/tools/context.py`、`worldfoundry/mcp/tools/readiness.py`
- 改动：(1) `DEFAULT_MCP_OUTPUT_ROOT` 默认锚定 `REPO_ROOT/runs/mcp`（绝对路径）；`WORLDFOUNDRY_MCP_RUN_ROOT` 覆盖值 import 时一次性 resolve，服务器生命周期内稳定；(2) `check_benchmark_datasets_payload` 的默认 `data_root` 从 CWD 相对 `"datasets"` 改为 `core.io.paths.local_data_root_path()`（尊重 `WORLDFOUNDRY_DATA_DIR`/`WORLDFOUNDRY_BENCHMARK_DATA_ROOT`，与评测栈同源）。`server_info` 的 `output_root` 现回显解析后的绝对路径。
- 实测：`output_root=<repo-root>/runs/mcp`、`dataset_root=/root/.cache/worldfoundry/data`，均绝对。
- 备注：`DEFAULT_CONTEXT` 全局单例与 server 显式 context 并存的问题未动（删除会破 27 个 payload 函数的默认参数契约），见 Deferred。

### [CM-30] P3 discovery 每次调用全量重载 manifest + conda 探测
- 文件：`worldfoundry/mcp/tools/discovery.py`
- 改动：加 30s TTL 进程内缓存 `_load_catalog(ctx)`（键为两个 manifest 目录，线程锁保护），4 个 discovery payload 全部走缓存；manifest 磁盘编辑 30s 内自动生效，无需重启。
- 实测：本机目录首載 19.10s（含 conda 探测），缓存命中 0.0003s；两次调用 total 一致。

### [CM-31] P3 工具面手工同步；`preview_run` 与 `evaluate` 参数面不一致
- 文件：`worldfoundry/mcp/tools/registration.py`
- 改动：`preview_run` 参数面对齐 `evaluate`：新增 `tasks`/`benchmarks`/`resume`/`generation_cache_dir`/`generation_cache_mode`，`benchmark` 转可选（MCP 按名传参，纯增量，旧调用不受影响）；`tasks`→`benchmarks` 的别名语义与 `evaluate` 相同。现可预览 `evaluate` 实际会提交的多基准命令。
- `MCP_TOOL_NAMES` 保留静态清单（是 `server_info` payload 契约的一部分），漂移风险改由新回归测试守护：假 FastMCP 注册集合与清单严格相等；另一测试断言 `evaluate` 参数集合 ⊆ `preview_run` ∪ {wait, wait_timeout_s}。

### [CM-32] P3 `MCPClient` 每次调用重新拉起服务器子进程
- 文件：`worldfoundry/mcp/client.py`
- 改动：`MCPClient` 支持 `async with`（`__aenter__`/`__aexit__`）持久会话——上下文内 `list_tools`/`call_tool` 复用同一服务器子进程，退出时关闭；不进上下文时保持原每调一进程行为（一次性脚本零改动）。docstring 注明两种用法与成本。
- 验证：假会话单测——默认模式 2 次调用开 2 个会话且都关闭；持久模式 3 次调用共 1 个会话、退出即关（新回归测试固化）。

### [CM-33] P3 `worldfoundry` 入口的 usage/help 品牌固定为 `worldfoundry-eval`
- 文件：`worldfoundry/cli/main.py`
- 改动：`prog=_cli_prog_name()`——`sys.argv[0]` 名为 `worldfoundry`/`worldfoundry-eval` 时如实显示，其它（`python -m worldfoundry.cli`、测试）回退历史品牌 `worldfoundry-eval`。
- 验证：5 个 console 入口中本 parser 只服务 `worldfoundry`/`worldfoundry-eval` 两个（`worldfoundry-mcp`/`worldfoundry-studio`/`worldfoundry-tui` 各有独立 parser，未动）；`python -m worldfoundry.cli` usage 输出与修复前一致（回退分支）。

## Deferred（评估过、暂不动，附方案）

- **[CM-03] P2 argparse 急切注册全部子命令**：短期已把最重的两条链（CM-01/CM-02）切断，并以回归测试固化"parser 构建不得拉 torch/orchestration.service"约定（`run_mode`/`runtime_preflight` 两个轻量 parser-data 叶子模块保留急切）。长期方案：argv[0] 预分派只注册命中的子 parser——需要重构 `_build_parser` 的注册协议，收益 <100ms，不值本轮风险。
- **[CM-05] P2 `run` 四形态旗标组合路由**：结构性。方案：拆 `run plan/model/suite` 显式子命令 + `_handle_run` 组合矩阵单测；本轮仅以 CM-04 修正了死路提示。**第二轮更新**：显式子命令拆分仍 deferred（契约面过大）；组合路由经逐条审计未发现错误分派，唯一真实 exit 缺陷（`run --print-config` 缺位置模型时走 ValueError→exit 1）已随 CM-08 修正为 exit 2 并补测试（见下"第二轮修复"）。
- **[CM-06] P3 CLI 顶层依赖 `evaluation.utils` 常量**：现状轻量（~5ms）。新回归测试对 parser 构建路径的重导入有守护作用；"utils 保持纯 stdlib"约定留给评测层文档。
- **[CM-08] P3 handler 错误风格不统一（`return 2` vs `raise ValueError`）**：~~下一轮做~~ **已在第二轮落地**（`CliUsageError` + main 兜底分流 exit 1/2，见下"第二轮修复"）。
- **[CM-10] P3 手写 argv 预扫描日志旗标**：行为保留（`-v/--verbose` 沿用同机制以保持一致）。方案：parent parser（`add_help=False` + `parents=[...]`）注入各子命令；涉及全部子 parser 注册点，本轮不动。
- **[CM-15] P2 `model_run.py` 1221 行契约推断长在 CLI**：结构性，且目标位置（`worldfoundry/evaluation/models/`）超出本次修改边界。方案：`_field_kind`/`_call_key`/fallback schema 合成迁到 evaluation 层，CLI 留 argparse 装配；迁移时 MCP/TUI 同步改 import。
- **[CM-16] P2 `zoo.py` readiness/videoscore/成功判定内嵌**：同上，目标位置在 `evaluation.tasks.catalog`（超边界）。方案：readiness 谓词与 `_benchmark_run_cli_success` 移入 catalog/orchestration result；videoscore 命令写回 manifest 数据文件。
- **[CM-17] P3 `models.py` visualize/assets 执行逻辑**：目标位置 `worldfoundry/studio/visualization/`（超边界）。方案同报告。
- **[CM-18] P2 跨子命令旗标名/默认值不一致**：修复约束明确"旗标名不得改名/删除、默认值属行为契约"。方案：下一个版本周期加 alias（`--metrics`↔`--metric`、`--limit`↔`--num-samples`）并统一新默认，旧名保留一周期。
- **[CM-20] P3 训练命令无条件打印 JSON、无 `--json` 旗标**：现行为是既有输出契约（脚本方可能已依赖恒 JSON），加 `--json` 并把默认改人类摘要会破坏它。方案：加 `--json`（默认沿用现行 JSON 以保兼容）+ 文档声明，另行评审。
- **[CM-22] P2 TUI 两处同步阻塞事件循环**：本机未安装 `textual`，任何改动无法运行验证——违背"修一条验一条"。方案（已核对 Textual API）：`action_refresh`/`_show_artifacts` 改 `self.run_worker(..., thread=True)` + `call_from_thread` 回填；`__init__` 的首次 `load_tui_catalog` 移到 `on_mount` worker。待有 textual 环境的机器执行。**第二轮复核**：环境仍无 `textual`（`import textual` → ModuleNotFoundError），按约不装包硬改，维持 deferred。
- **[CM-23] P2 TUI 500 行手工变体注册表**：结构性；目标位置是 model zoo manifest 数据文件（超边界）。方案：manifest 增加变体分组/标签结构化字段，`tui_discovery` 消费后删表。
- **[CM-27] P2 MCP 反向依赖 `cli.tui_discovery`**：需要新建库层包（如 `worldfoundry/catalog/`，超边界）。本轮以 TTL 缓存（CM-30）缓解了该依赖的性能面；方向性迁移与 CM-15/16/23 一揽子做。
- **[CM-28] P2 作业状态仅存内存**：~~超边界~~ **已在第二轮落地**（`runtime/jobs.py` 落盘 JSON 索引 + pid 对账 + 上限，MCP context 接线，见下"第二轮修复"与 `07_operators_runtime_fixes.md`）。
- **[CM-29 余项] `DEFAULT_CONTEXT` 双 store 并存**：~~下一轮做~~ **已在第二轮落地**（`set_default_context` 模块级 setter + server 写回，见下"第二轮修复"）。

## 验证汇总

- `py_compile`：全部改动文件通过（cli：main/evaluation_intents/training_commands/register/models/tasks/plan_metric/utils/zoo/tui_discovery；mcp：client/tools/{context,readiness,discovery,registration,responses,studio}）。
- 只读 CLI 实测：`--help`（root/run/train/tui/mcp）、`zoo benchmarks --json`、`zoo models --json`、`zoo benchmark-show`（成功与失败两态）、`run`（缺参提示）——输出契约与修复前一致，仅新增错误 JSON 形态与 usage 品牌修正。
- MCP 冒烟：`PYTHONPATH=. python -c "import worldfoundry.mcp"` 通过；`mcp`/`fastmcp` 包缺失时 `create_mcp_server` 报既有指引性 RuntimeError（记录在案）；27 工具经假 FastMCP 注册并逐类抽测。
- 新增回归测试：`test/test_cli_mcp_fix_contracts.py`——13 项覆盖 CM-01/07/25/26/29/31/32 契约，全过（12.3s）。
- `test/eval_core` 对比：全量套件在本机（4 个并发 pytest + 评测层大范围在改）单轮 >50 分钟且中途被裁剪，改用**受控 A/B 对照**严格归因：把本次修改的 16 个 cli/mcp 文件临时还原到 HEAD（`git show HEAD:` 写回，不碰 index；评测层/训练侧他人改动原样保留），对 12 个 CLI/MCP 相关测试文件跑同一子集——
  - HEAD 版：36 failed / 77 passed / 16 skipped；
  - 修复版：35 failed / 78 passed / 16 skipped；
  - 失败清单逐行 diff：修复版**零新增失败**；唯一差异是 `test_cli_ux.py::test_help_prints_command_areas_without_heavy_runtime_imports` 在 HEAD 失败、修复后通过（该测试正是 CM-01 的既有守护，佐证修复有效）。
  - 其余 35 个失败两侧完全一致，经抽样归因均来自评测层进行中的未提交改动或既有缺口（示例：`zoo env-check`/`validate` 系列——`git grep env-check HEAD -- worldfoundry/` 为空，功能在 HEAD 即不存在而测试存在；`vbench` 掉出 `--integration-status planned` 过滤——评测层新的 status 归一化把未识别状态降级为 unknown，stderr 有其警告）。对照后已从备份恢复全部修复文件并复核逐字节一致、`embodied.py` 保持删除。

---

## 第二轮修复（2026-08-25，CLI/MCP job-store 修复 agent，分支 `cursor/cli-mcp-jobstore-fixes-2f62`）

> 范围：首轮 Deferred 中正确性清晰的三项（CM-08 / CM-28 / CM-29 余项）+ CM-05 路由审计点修。约束沿用首轮：不改旗标名/默认值（CM-18 禁区）、不回退 CM-01 守护、跳过 CM-03、CM-22 无 textual 不动。

### [CM-08] handler 用法错误统一为 `CliUsageError`，main 兜底按类型分流 exit 1（运行期）/ 2（用法）

- 文件：`worldfoundry/cli/utils.py`（新增异常类型）、`worldfoundry/cli/main.py`、`worldfoundry/cli/evaluation_intents.py`、`worldfoundry/cli/tasks.py`、`worldfoundry/cli/reporting.py`。
- 改动：
  - `cli/utils.py` 新增 `CliUsageError(Exception)`——用户可通过改命令行修复的错误（旗标互斥/缺参）。
  - `main()` 新增独立 `except CliUsageError` 分支（先于通用 `except Exception`）：stderr 一行 `error: <msg>`（无堆栈、日志事件不带 exc_info）；`--json` 时 stdout 输出 `{"status":"error","error":{"type":"usage","message":...},"exit_code":2}` 封套；**`return 2` 而非 `parser.exit`**——进程内调用 `main()` 的测试/嵌入方仍拿返回值（与既有 `return 2` handler 契约逐字节兼容）。运行期异常分支维持 CM-07 的 exit 1 + `error.type=异常类名` 不变。
  - 两种旧风格全部收敛：`print(error, file=sys.stderr); return 2` 共 17 处（main.py 11、reporting.py 4、tasks.py 2）与用法类 `raise ValueError` 共 5 处（evaluation_intents.py 2、tasks.py 1、main.py 2）统一改 `raise CliUsageError`。消息文本逐字保留（main 渲染时统一加 `error: ` 前缀；tasks.py 两处原本无前缀的 stderr 文本获得前缀，全仓无测试断言该文本）。
  - 附带修正：`_handle_score` 的旗标校验前移到 orchestration 重导入**之前**——用法错误现在毫秒级失败，不再先付 ~0.7s 导入成本。
- 不变式核对：`zoo benchmark-show --benchmark-id 不存在 --json` 仍 exit 1 + CM-07 封套（未知 id 是运行期查找失败，非用法错误）；argparse 自身用法错误仍 exit 2；`KeyboardInterrupt` 仍 130。
- 验证：新增 6 项契约测试（exit 2 + 简洁 stderr、`--json` usage 封套、score 快速失败且不导入 orchestration.service、进程内 `main()` 返回值语义、runtime/usage 分流、`run --print-config` 组合）全过。

### [CM-05 审计] `run` 组合路由逐条核对，点修一处真实 exit 缺陷

- 审计四形态（`--plan` 重放 / 位置模型直推 / unified facade / 已裁撤三旗标路径）的分派谓词（`_uses_direct_model_run`、`_run_uses_unified_framework`、`_has_complete_task_args`）：未发现错误分派。
- 唯一真实缺陷：`run --print-config` 缺位置模型时 `_model_run_plan` 抛 `ValueError` → exit 1（运行期语义），实为用法错误；已随 CM-08 改 `CliUsageError` → exit 2，测试 `test_run_print_config_without_model_is_usage_error` 固化。
- 显式子命令拆分（`run plan/model/suite`）维持 deferred：契约面过大，不属本轮"点修"范围。

### [CM-28] `AsyncCommandJobStore` 落盘 JSON 索引 + 启动 pid 对账 + 保留上限（实现在 runtime 层，详见 `07_operators_runtime_fixes.md` OR-15 续）

- 文件：`worldfoundry/runtime/jobs.py`、`worldfoundry/mcp/tools/context.py`。
- 改动（jobs.py，纯增量 API，默认行为不变）：
  - `AsyncCommandJobStore(state_path=...)`：设置后在 submit/进程 spawn（记录 pid）/终态/cancel/prune 各状态转移点把全部作业的元数据索引（job_id/run_id/**pid**/status/output_dir/日志三路径/命令/metadata/时间戳/returncode/error）原子写入该 JSON 文件（tmp + `os.replace`；写失败静默——持久化是尽力而为，绝不拖垮提交路径）。内存日志尾不入索引（原始 stdout/stderr/events 本就逐作业落盘，评审备注"缺的只是索引"）。
  - 构造时恢复 + 对账：读索引重建 `CommandJob(restored=True)`；非终态作业查 pid 存活（`pid_alive`，signal 0）——活着保持 `running`（元数据与磁盘日志路径可查，无进程句柄），死了标 `failed` 并注明 pid，对账结果立即写回。索引损坏/缺失从空启动，不抛。
  - 恢复作业可取消：`_terminate_process` 对无句柄但有 pid 的 restored 作业直接对进程组走同一 SIGTERM→等待→SIGKILL 阶梯（子进程 `start_new_session=True`，pid==pgid）。
  - `CommandJob` 新增 `pid`/`restored` 字段并透出到 `to_summary`（字段只增不删）。
- MCP 接线（context.py，只消费新 API，27 个工具名零改动）：`MCPToolContext` 未显式传 `job_store` 时自动创建持久化 store——`state_path=output_root/jobs-index.json`、`max_jobs=DEFAULT_MCP_MAX_TRACKED_JOBS(256)`（终态作业超限按最旧淘汰，长驻 server 内存与索引不再无界涨）。显式传 store 的既有调用方（含全部既有测试）行为不变。
- 验证：新增 `tests/runtime/test_jobs_store_persistence.py` 7 项——索引跨重启往返、无 state_path 零落盘（legacy）、死 pid 对账为 failed 且写回、活 pid 保持 running 且 cancel 经 pid 阶梯真实击杀（真子进程验证 SIGTERM 收尸）、queued 无 pid 标 failed、损坏索引空启动、max_jobs 淘汰 + 索引同步。既有 `tests/runtime/test_jobs_logging.py` 与 `test/eval_core/test_mcp_web_interfaces.py` 全过。

### [CM-29 余项] `DEFAULT_CONTEXT` 双 store 收敛：模块级 setter + server 写回

- 文件：`worldfoundry/mcp/tools/context.py`、`worldfoundry/mcp/server.py`、payload 消费方 5 个文件（runs/discovery/readiness/server_info/registration）。
- 改动：
  - context.py 新增 `get_default_context()`/`set_default_context()`；`_default_mcp_output_root` 转正为公开 `resolve_mcp_output_root()`（语义不变，`DEFAULT_MCP_OUTPUT_ROOT` 仍是 import 时快照）。
  - 5 个消费模块的 `ctx = context or DEFAULT_CONTEXT`（import 时绑定，setter 对其无效的隐患）全部改 `ctx = context or get_default_context()`——27 个 payload 函数签名零改动。
  - `create_mcp_server()`：`context = set_default_context(MCPToolContext(output_root=resolve_mcp_output_root()))`——server 自建 context 写回为进程默认，payload 直调与注册工具共用同一 job store。**顺带修正一处首轮遗留 bug**：server.py:44 原来仍用 CWD 相对的 `Path(os.environ.get(..., "runs/mcp"))` 构建 context（CM-29 首轮只改了 context.py 侧，server 侧漏改），现统一走绝对路径解析。
- 验证：新增 3 项契约测试——MCP context 默认 store 持久化+有界、持久化索引经 payload 函数可见（restored 作业跨"重启"可 list/status）、假 FastMCP 下 `create_mcp_server` 写回默认 context 且注册工具与默认 store 共视同一作业（真实提交作业经 `list_runs` 工具封套回读）。测试带 fixture 守卫，退出时恢复原默认 context。

### 本轮验证汇总（实测数字）

- `py_compile`：16 个改动文件全过（cli：main/utils/evaluation_intents/tasks/reporting；mcp：server/tools 的 context/runs/discovery/readiness/server_info/registration/__init__；runtime：jobs；test 2 个）。
- `test/test_cli_mcp_fix_contracts.py`：13 → **23 项全过**（5.5s；新增 CM-08 x6、CM-05 x1、CM-28 接线 x2、CM-29 x2，首轮 13 项零回归）。
- `tests/runtime/test_jobs_store_persistence.py`（新增）：**7 项全过**（9.6s）；`tests/runtime/test_jobs_logging.py` + `test/eval_core/test_mcp_web_interfaces.py`：13 passed / 9 skipped（skip 均为缺 `mcp` 包的既有守卫）。
- CLI/MCP 相关面回归（12 个测试文件两批）：第一批 82 passed / 5 failed / 10 skipped，第二批 42 passed / 20 failed / 7 skipped——**25 个失败经 main worktree 复跑逐行 diff 完全一致（IDENTICAL_FAILURES），全部为 main 既有失败**（`test_cli_ux.py` 的 zoo readiness JSON 字段 5 项 + `test_eval_cli_contract.py`/`test_tui_cli.py` 的 zoo env-check/validate、vbench 等 20 项，属评测层既有缺口），本分支零新增失败。
- 启动守护复测：`python -m worldfoundry --help` 与 `python -m worldfoundry.cli zoo benchmarks --help` 经 `-X importtime` 核查 torch-hits=0（CM-01 守护未回退）。
- 环境注记：本机无 `mcp`/`fastmcp`/`textual` 包；MCP 侧验证经假 FastMCP + payload 直调完成，CM-22 维持 deferred。
