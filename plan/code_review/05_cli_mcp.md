# CLI 与 MCP 评审

> 评审对象：`worldfoundry/cli/`（34 文件）、`worldfoundry/mcp/`（14 文件）、`worldfoundry/__init__.py`、`worldfoundry/__main__.py`、`worldfoundry/data/__init__.py`、`pyproject.toml` 入口。
> 评审日期：2026-08-14。状态：已完成（33 条发现：P0×1、P1×4、P2×14、P3×14）。

## 评审范围与方法

- 逐文件精读上述范围内全部 Python 源码；`tui_app.py`（约 10.7 万字符）、`tui_discovery.py`（约 9.4 万字符）、`main.py`（约 7.6 万字符）、`zoo.py`（约 5.3 万字符）为重点大文件。
- 实际运行只读命令验证启动性能与行为：`python -X importtime -m worldfoundry.cli --help`、`worldfoundry zoo benchmarks --json` 等（仅只读，不下载、不写仓库文件）。
- 评审维度：启动性能 / 参数解析一致性 / 退出码与错误呈现 / JSON 输出契约 / TUI 线程模型 / MCP 工具质量 / 分层 / 全局状态与信号 / 错误处理 / 重复与死代码。
- 严重度：P0=损坏或危险；P1=严重设计缺陷；P2=应修复；P3=改进建议。

## 发现（按主题分组）

### 主题 A：启动性能与导入链

### [CM-01] P0 每次 CLI 启动都会加载 torch：`configure_logging` → `core.distributed.logging` → `core.distributed/__init__` 急切导入 `context_parallel`

- 位置：`worldfoundry/cli/main.py:1792-1796`、`worldfoundry/core/logging_setup.py:592-604`、`worldfoundry/core/distributed/__init__.py:5-15`
- 证据：`main()` 第一步即调用 `_configure_cli_logging`（main.py:1796），其内部 `from worldfoundry.core import configure_logging`；`configure_logging` 走到：

```592:604:worldfoundry/core/logging_setup.py
def _apply_distributed_logger(level: int) -> None:
    """Reparent the rank-aware ``distributed_logger`` onto the root pipeline.

    Lazily imported so this module never eagerly pulls ``torch`` / the
    distributed stack; if it is unavailable we simply skip reparenting.
    """
    try:
        from worldfoundry.core.distributed.logging import distributed_logger
    except Exception:
        return
```

  而 `worldfoundry/core/distributed/__init__.py` 顶层是急切 re-export：`from .context_parallel import (...)`，`context_parallel.py:23` 顶层 `from torch.distributed import (...)`。实测（`python -X importtime -m worldfoundry.cli --help`）：

```text
import time:      19 | 2519382 | worldfoundry.core.distributed.logging   ← 运行期由 configure_logging 触发
import time:   41092 | 1595172 |       torch
real    0m4.045s   （--help 全程 4.0 秒）
real    0m4.172s   （zoo benchmarks --json 全程 4.2 秒）
```

- 问题：docstring 明确写着 “Lazily imported so this module never eagerly pulls torch”，但惰性导入的目标模块位于急切 re-export 的包内，`import worldfoundry.core.distributed.logging` 必然执行 `distributed/__init__.py` → `context_parallel` → torch。torch 装了就成功导入（付出 2.5s），没装才走 `except` 跳过——防护条件恰好和设计意图相反。
- 影响：直接违反项目设计原则“CLI/目录查询路径必须 stdlib-first、轻量导入”。`--help`、`zoo benchmarks --json` 等纯查询命令固定付出约 4 秒（其中 torch ≈2.5s），MCP server 每次 stdio 启动同样中招（mcp 工具链也 import CLI 层）。在无 GPU 的轻量环境里还会因 torch 初始化产生额外告警/副作用。
- 建议：`_apply_distributed_logger` 改为 `sys.modules` 探测（仅当 `worldfoundry.core.distributed.logging` 已被导入时才 reparent），或把 `distributed_logger` 定义挪到不触发 torch 的独立模块；同时把 `core/distributed/__init__.py` 改成 PEP 562 惰性 re-export（仓库里 `evaluation/__init__.py` 已有现成模式）。

### [CM-02] P1 `cli/evaluation_intents.py` 顶层导入 orchestration.service，构建 parser 即付出约 0.7 秒

- 位置：`worldfoundry/cli/evaluation_intents.py:8-16`
- 证据：

```8:16:worldfoundry/cli/evaluation_intents.py
from worldfoundry.evaluation.tasks.execution.orchestration.service import (
    GenerateAndScoreIntent,
    ReproduceIntent,
    ScoreArtifactsIntent,
    ScoreResultsIntent,
    execute_prepared_evaluation,
    prepare_evaluation,
)
```

  importtime 实测：`worldfoundry.cli.evaluation_intents` 累计 727ms（其中 `orchestration.service` 722ms，连带拉起整个 `tasks.execution.orchestration`、`models.catalog`、`models.pipelines.bindings`、benchmark 契约注册表等）。
- 问题：`_build_parser`（main.py:1203-1205）无条件导入该模块以注册 `score`/`generate-score`/`reproduce` 三个子命令；同目录其它模块（`dataset.py`、`config.py`、`tasks.py` 等）都遵守“注册函数只建 parser、handler 内再函数级导入”的惯例，唯独此文件把重导入放在顶层。`register_*` 仅需要 `BENCHMARK_ZOO_DIR`/`TMP_ROOT` 两个常量，intent 类只在 `_handle_*` 里使用。
- 影响：即使修复 CM-01，`--help` 仍要付出约 1 秒的 parser 构建成本；这是启动热路径上第二大的开销。
- 建议：把 intent 类与 `prepare_evaluation`/`execute_prepared_evaluation` 的导入下沉到 `_handle_score`/`_handle_reproduce`/`_handle_generate_score` 内部，与本目录其它文件保持同一惯例。

### [CM-03] P2 `_build_parser` 每次调用都急切注册全部子命令，无法按需短路

- 位置：`worldfoundry/cli/main.py:1083-1612`（`_build_parser`）、`worldfoundry/cli/main.py:1186-1229`（十余个 `register_*` 导入）
- 证据：`_build_parser` 顺序导入 `.tasks`/`.dataset`/`.config`/`.plan_metric`/`.evaluation_intents`/`.reporting`/`.models`/`.preflight`/`.training`/`.zoo`/`.mcp` 并注册全部子命令；importtime 显示除 CM-02 外每个模块 4-12ms、`cli.models` 还拉 `worldfoundry.base_models.capabilities`（15ms）。`worldfoundry zoo benchmarks` 这类命令同样要为 `train`/`embodied`/`tui` 等无关子命令买单。
- 问题：argparse 天然不支持惰性子命令，但当前结构把“注册”与“实现导入”耦合在模块导入上，任何一个子模块顶层变重（如 CM-02）都直接拖慢全部命令。
- 影响：启动时间随子命令数量线性增长；目前 parser 构建约 1s（含 CM-02）。
- 建议：短期先修 CM-02 并为 `register_*` 模块加“顶层禁止导入 evaluation 执行栈”的约定/测试（可用 `sys.modules` 断言写一个冒烟测试）；长期可考虑第一段 argv 预分派（`argv[0]` → 仅注册对应子 parser）。

### 主题 B：入口与分发

### [CM-04] P2 `run`/`evaluate` 的 `--task-type/--benchmark-name/--data-path` 路径是保留旗标的死路：必然抛错

- 位置：`worldfoundry/cli/utils.py:101-107`、`worldfoundry/cli/main.py:434-445`、`worldfoundry/cli/main.py:995-1016`
- 证据：

```101:107:worldfoundry/cli/utils.py
def resolve_cli_benchmark_for_materialize(task_type: str, benchmark_name: str) -> Any:
    """Resolve a benchmark adapter for legacy task-type/materialize CLI flows."""
    raise ValueError(
        "Task-type/benchmark-name materialization is retired for benchmark-zoo entries. "
        "Use `worldfoundry-eval run --benchmark <id> --model <id>` or "
        "`worldfoundry-eval task materialize` with a filesystem task YAML."
    )
```

  `_handle_run_in_process`（main.py:1008-1016）与 `_handle_evaluate`（main.py:434-445）都以此为第一步；而 `run` 缺参数时的提示仍是 “--task-type, --benchmark-name, and --data-path are required unless --plan is provided”（main.py:995-1000），引导用户走向必败路径。
- 问题：已裁撤的功能只在运行期抛 `ValueError`（经 main 的兜底转成 exit 2），但旗标、help 文本、错误提示、以及 `_handle_run_in_process` 整个函数（约 70 行）仍原样保留；`validate` 子命令也仍要求同一组旗标。
- 影响：用户按 help/错误提示操作会得到二段式挫败（先被要求提供三旗标，提供后又被告知该路径已裁撤）；死代码增加维护面。
- 建议：要么在 argparse 层面直接拒绝（`deprecated` 帮助文本 + 立即报错并给出替代命令），要么删除三旗标与 `_handle_run_in_process`，把 main.py:995 的提示改为指向 `--benchmark/--model`。

### [CM-05] P2 `run` 子命令承载 60+ 旗标、四种执行形态，路由规则靠旗标组合推断

- 位置：`worldfoundry/cli/main.py:1432-1609`（parser）、`worldfoundry/cli/main.py:839-1002`（路由）
- 证据：`run` 同时支持 (a) `--plan` 重放、(b) 位置参数模型直推（`_uses_direct_model_run`）、(c) 统一 facade（`_run_uses_unified_framework`：`--all-benchmarks/--suite/--benchmark/--results-path` 任一命中）、(d) 遗留 task-type 路径（已死，见 CM-04）。`_uses_direct_model_run`（main.py:839-853）要靠 7 个旗标的否定组合来判定。
- 问题：单一子命令的行为由旗标组合隐式决定，用户与维护者都难以预测；例如给了位置模型又给 `--results-path` 会静默切换到 facade 路径，`--engine existing-results` 与位置模型组合则报错要求 in-process（main.py:974-976）。帮助文本无法解释这些交互。
- 影响：可用性与可测试性差；每加一个旗标都要重新推理 4 条路径的组合语义。
- 建议：中期把 4 形态拆成显式子命令（如 `run plan`、`run model`、`run suite`），或至少在 `--help` epilog 里写明路由决策表；短期为 `_handle_run` 补组合矩阵单测。

### [CM-06] P3 CLI 顶层 `from worldfoundry.evaluation.utils import ...` 是常量导入，轻量但形成硬依赖

- 位置：`worldfoundry/cli/main.py:17-22`
- 证据：`from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR, REPO_ROOT, TMP_ROOT`；importtime 显示 `worldfoundry.evaluation`（惰性 `__init__`）+`evaluation.utils` 合计约 5ms，可接受。
- 问题：`evaluation/__init__.py` 采用 PEP 562 惰性导出（见 `worldfoundry/evaluation/__init__.py:65-86`），此处只触发 utils 子模块，当前无性能问题；但 CLI 的 4 个路径常量绑定在 evaluation 包上，若 utils 将来变重会直接影响所有命令（与 CM-01 同型的隐患）。
- 影响：低；仅是导入拓扑上的脆弱点。
- 建议：保持 `evaluation.utils` 纯 stdlib 的约定并加注释/测试固化。

### 主题 C：退出码与错误呈现

### [CM-07] P1 `--json` 模式下错误不输出 JSON；运行期失败与用法错误共用 exit 2

- 位置：`worldfoundry/cli/main.py:1874-1892`
- 证据：

```1874:1892:worldfoundry/cli/main.py
    except Exception as exc:
        from worldfoundry.core import get_logger

        get_logger(__name__).event(
            "ERROR",
            "cli.command_failed",
            "CLI command failed",
            exc_info=True,
            command=getattr(args, "command", None),
        )
        ...
        parser.exit(2, f"error: {exc}\n")
```

- 实测（只读命令）：`worldfoundry zoo model-show --model-id does-not-exist-xyz --json` → 退出码 **2**、stdout **0 字节**、stderr 先输出**完整 traceback**（logger `exc_info=True` 落到 stderr sink）再跟一行 `error: "unknown model-zoo entry: 'does-not-exist-xyz'"`。
- 问题：四点。(1) 任何 handler 抛出的异常统一变成 exit 2——与 argparse 用法错误（也是 2）无法区分；框架层多数 handler 通过 `result.exit_code` 返回 1 表示运行失败，兜底路径却用 2。(2) `--json` 模式下 stdout 为空、契约中断：机器调用方拿不到结构化错误（`{"error": ...}`），必须解析 stderr 文本。(3) “未知 id”这类普通用户错误也把全量 Python traceback 打到 stderr（经日志管道），对终端用户就是裸 traceback 体验。(4) `parser.exit` 打印 `str(exc)`，对 `KeyError` 会输出裸 key，可读性差。
- 影响：自动化消费方（脚本/CI/agent）在失败时需要额外的 stderr 解析分支；退出码语义不稳定。
- 建议：在兜底处判断 `getattr(args, "json", False)`，为 JSON 模式输出 `{"status": "error", "error": {...}}` 到 stdout；运行期异常用 exit 1，保留 2 给用法错误；对常见异常类型格式化消息。

### [CM-08] P3 处理器内部错误返回码不统一：参数校验有的 `return 2`、有的 `raise ValueError`

- 位置：`worldfoundry/cli/main.py:362-380`（`return 2` 风格）、`worldfoundry/cli/evaluation_intents.py:49-65`（`raise ValueError` 风格）
- 证据：`_handle_evaluate` 对旗标互斥直接 `print(..., file=sys.stderr); return 2`；`_handle_score` 对同类校验 `raise ValueError("score --benchmark requires --artifacts")`，靠 main 兜底转 exit 2。两者最终退出码一致，但前者信息是定制的 `error: --samples-path requires --embodied-spec`，后者会先被 logger 记一条带堆栈的 ERROR 事件再打印。
- 问题：同类用户错误走两条呈现路径，日志噪声与输出格式不一致。
- 建议：统一用 `parser.error()`/自定义 `CliUsageError`，在 main 兜底里区分“用户用法错误（无堆栈日志）”与“内部错误（记堆栈）”。

### 主题 D：日志副作用与全局状态

### [CM-09] P2 只要给了 `--output-dir` 的命令就会在输出目录写 `logs/<run-id>/events.jsonl` 并改写进程环境变量

- 位置：`worldfoundry/cli/main.py:1708-1759`（`_prepare_cli_run_observability`）
- 证据：`main()` 对所有带 `output_dir` 属性的命令调用该函数：生成 run_id、`configure_logging(log_file=event_path, json=True, force=True)`、`os.environ["WORLDFOUNDRY_LOG_FILE"]=...`、`os.environ["WORLDFOUNDRY_RUN_ID"]=...`、`os.environ.update(log_context_environment())`，并立即写入 `run.started` 事件行。
- 问题：(1) 这对 `score --plan-only`、`run --plan-only`、`run --print-config` 这类“干跑/只读”命令同样生效——干跑也会在磁盘上创建 `logs/` 目录并写事件文件；(2) 环境变量污染会传给之后 fork 的一切子进程，包括与本次运行无关的；(3) `_handle_worldfoundry_run`（main.py:781-791）还会按 `--performance-profile` `os.environ.setdefault` 六个 WORLDFOUNDRY_* 变量，进一步扩大隐式全局状态。
- 影响：违反最小副作用预期；plan-only 输出目录被提前创建可能干扰后续“输出目录必须不存在”的原子占用逻辑（代码里用 `_requires_exclusive_output_dir` 特判侧目录，正说明这个副作用已经咬过一次）。
- 建议：把 observability 初始化下沉到真正执行工作负载的 handler；plan-only/print-config 路径跳过；环境导出改为只在启动子进程时注入。

### [CM-10] P3 全局日志旗标靠手写 argv 预扫描，绕过 argparse

- 位置：`worldfoundry/cli/main.py:1615-1652`（`_extract_logging_flags`）、`worldfoundry/cli/main.py:1098-1121`
- 证据：`--log-level/--log-file/--log-json` 在 root parser 注册“仅为出现在 --help 里”（注释原话 “argparse never enforces them”），实际解析由 `_extract_logging_flags` 手工完成，支持任意位置出现。
- 问题：手写扫描与 argparse 语义有微妙差异：`--log-level` 后跟以 `--` 开头的值会被跳过（留给 argparse 报错，尚可）；但 `--log-file --weird` 这类合法文件名 `--weird` 无法表达；子命令 help 里看不到这些全局旗标。
- 影响：低；边缘行为不一致。
- 建议：改用 parent parser（`argparse.ArgumentParser(add_help=False)` + `parents=[...]`）在每个子命令继承全局旗标，或至少在文档中说明取值限制。

### 主题 E：死代码与重复

### [CM-11] P2 `cli/embodied.py` 整个模块（163 行）是死代码，且与 main.py 内联实现形成漂移的双份拷贝

- 位置：`worldfoundry/cli/embodied.py:119-162`、`worldfoundry/cli/main.py:502-590`、`worldfoundry/cli/main.py:1231-1270`
- 证据：全仓 grep `register_embodied_subparser` 仅有定义处与 `__all__`，无任何调用方；main.py 内联注册了自己的 `embodied` 子命令与四个同名 handler。两份实现已经漂移：
  - `embodied.py:67-69` 对 `--shard-id`/`--num-shards` 有互斥校验且错误打到 **stdout**（`print("error: ...")` 无 `file=sys.stderr`）；main.py 版本无此校验。
  - `embodied.py` 的 run 结果附加 `eval_id`/`raw_result_count` 字段，merge 要求 `--output-dir` 必填；main.py 版本均不同（merge 的 `--output-dir` 可选、payload 无附加字段）。
  - plan 的 payload 结构不同（`benchmark_count/episodes_per_task/max_tasks/params` vs `total_requests/sample_ids`）。
- 问题：无人引用的完整命令模块；其中部分行为（shard 校验、eval_id 透出）比 main.py 的活代码更完善，说明修复曾发生在死拷贝上。
- 影响：维护者极易改错文件；两个版本的 JSON 契约不一致。
- 建议：删除 `cli/embodied.py`，或反向操作——把 main.py 的内联 embodied 块换成 `register_embodied_subparser`（合并两版差异后保留一份）。

### [CM-12] P3 `cli/models.py` 顶部 `REPO_ROOT` 是带越界 bug 的死代码

- 位置：`worldfoundry/cli/models.py:15-16`
- 证据：

```15:16:worldfoundry/cli/models.py
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT
```

  实测 `parents[3]` = `/mnt/.../juanxi`（仓库的**父目录**），真实仓库根是 `/mnt/.../juanxi/WorldFoundry`；且两个常量在本文件内无任何使用（全文件 grep 仅此两行）。
- 问题：差一级的路径常量 + 死代码；将来若有人直接引用会得到错误路径。仓库已有权威 `REPO_ROOT`（`worldfoundry/evaluation/utils.py:153`）。
- 建议：删除这两行。

### [CM-13] P3 `_task_roots_from_args` 在 tasks.py 与 plan_metric.py 逐行重复

- 位置：`worldfoundry/cli/tasks.py:216-227`、`worldfoundry/cli/plan_metric.py:27-49`
- 证据：两份函数体一致（含 `WORLDFOUNDRY_TASK_ROOTS`/`WORLDFOUNDRY_BENCHMARK_INCLUDE_PATH` 环境变量合并、`dict.fromkeys` 去重）；另有 `_add_task_selector` 在 main.py:187-191（benchmark-name 必填）与 tasks.py:20-24（benchmark-name 可选）同名不同义。
- 问题：环境变量解析规则改一处漏一处；同名函数行为不同增加误用风险。
- 建议：提到 `cli/utils.py` 共享；`_add_task_selector` 二选一并重命名。

### [CM-14] P3 `zoo.py` 的 `_dispatch_zoo_handler` 字符串间接层无实际收益

- 位置：`worldfoundry/cli/zoo.py:967-976`
- 证据：`set_defaults(func=_dispatch_zoo_handler("_handle_zoo_models_list"))` 通过 `globals()[name]` 查找同模块函数；注册时函数已定义，无惰性收益，反而丢失静态可达性（IDE/vulture/重构工具都看不到引用）。
- 建议：直接 `set_defaults(func=_handle_zoo_models_list)`。

### 主题 F：分层与业务逻辑位置

### [CM-15] P2 `cli/model_run.py`（1221 行）把模型契约推断的业务逻辑长在 CLI 层

- 位置：`worldfoundry/cli/model_run.py:141-210`（`_field_kind` 名称启发式）、`model_run.py:495-755`（fallback schema 合成）、`model_run.py:758-1043`（`load_model_run_schema`）
- 证据：`_field_kind` 用 55 行字符串启发式（`"scale"/"guidance"/"threshold"` → number 等）猜测参数类型；`_call_key` 维护 `frames→num_frames/video_length/frame_num` 等跨模型别名表；`_fallback_input_fields`/`_fallback_generation_fields` 按 task 文本关键词（`"geometry"、"robot"、"vla"`）合成输入契约。这些是模型目录/运行时契约域的知识，与参数解析无关。
- 问题：违反“cli 只做参数解析+调度”的分层原则；TUI（`tui_discovery.py`）与 MCP 若需要同样的契约推断只能 import CLI 模块（事实上 `mcp/tools` 已经在 import `worldfoundry.cli.tui_discovery`，见 CM-27）；启发式无法单独测试演进。
- 影响：CLI 成为契约推断的事实核心，evaluation/models 层反而不完整；heuristic 改动的影响面难以评估。
- 建议：把 schema 推断迁到 `worldfoundry/evaluation/models/`（或 core.inference 旁），CLI 保留 argparse 装配（`register_model_run_arguments`）。

### [CM-16] P2 `zoo.py` 内嵌 readiness 语义与 videoscore 特例，成功判定读 scorecard 内部结构

- 位置：`worldfoundry/cli/zoo.py:463-468`（videoscore 硬编码）、`zoo.py:903-925`（`_benchmark_run_cli_success`）、`zoo.py:363-512`（约 150 行 readiness 推导）
- 证据：

```463:468:worldfoundry/cli/zoo.py
    if entry.benchmark_id == "videoscore" and "normalizer_run" not in commands:
        commands["normalizer_run"] = (
            "worldfoundry-eval zoo benchmark-run --benchmark-id videoscore "
            "--mode official-validation --official-results-path '<eval_*_videoscore.json>' "
            ...
```

  `_benchmark_run_cli_success` 在 CLI 里读 scorecard JSON 并识别 `contract_only`/`normalizer_only`/`normalization_ok` 字段来决定退出码。
- 问题：(1) 单一 benchmark 的知识硬编码在 CLI（应写进 manifest 的 `validation_command`）；(2) “这次 run 算不算成功”是评测框架的领域判断，CLI 却复刻了 scorecard 的内部 schema——scorecard 字段一变 CLI 判定就悄悄失效；(3) readiness 推导（`_benchmark_official_runner_ready` 等 8 个谓词 + 17 个布尔旗标的 `_benchmark_command_readiness`）是目录域逻辑，却以 `zoo.py` 私有函数形式存在，TUI/MCP 需要同一信息时只能重复实现或跨层 import（MCP 现状见 CM-27）。
- 建议：readiness/success 判定移入 `evaluation.tasks.catalog`（或 orchestration 的 result 对象直接暴露 `cli_success` 语义）；videoscore 命令写回 manifest。

### [CM-17] P3 `models.py` 的 `visualize` 与 `assets` 处理器承载渲染/下载执行逻辑

- 位置：`worldfoundry/cli/models.py:157-316`（`_npz_value`/`_media_frames`/`_write_visualization`/`_handle_models_visualize`）、`models.py:78-100`（`_execute_download_commands`）
- 证据：npz 键探测、逐帧渲染、libx264 写视频、子进程下载执行与输出截断（`stdout[-4000:]`）全部实现于 CLI。
- 问题：与 CM-15 同型但规模较小；`studio.visualization` 已有渲染函数，此处的 npz/媒体装载器应一并下沉，让 Studio/MCP 可复用。
- 建议：把装载与写出移入 `worldfoundry/studio/visualization/`，CLI 只留 kind → 函数映射。

### 主题 G：参数与输出一致性（跨子命令）

### [CM-18] P2 同一概念的旗标名/默认值跨子命令不一致

- 位置：`worldfoundry/cli/zoo.py:1120-1124`、`worldfoundry/cli/main.py:1571-1580`、`worldfoundry/cli/evaluation_intents.py:162-166`、`worldfoundry/cli/tasks.py:389-429`
- 证据（对照）：
  - 指标：`run/evaluate/score` 用 `--metric`（append），`zoo benchmark-run` 用 `--metrics`（append+逗号分隔，zoo.py:1120）。
  - 清单目录：根命令用 `--model-manifest-dir`/`--benchmark-manifest-dir`，zoo 子命令统一叫 `--manifest-dir`（同一个词在 models/benchmarks 下含义不同）。
  - 生成缓存默认：`run`/`evaluate` 默认 `off`（main.py:204），`generate-score` 默认 `read-write`（evaluation_intents.py:165）。
  - 输出目录默认：`run` 默认字符串 `"./benchmark_results"`（main.py:1562），`evaluate`/`score`/`reproduce` 默认 `TMP_ROOT/...`（Path），`zoo benchmark-run` 则 `--output-dir` 必填。
  - 数据集限量：全 CLI 用 `--num-samples`，`zoo benchmark-run` 用 `--limit`（zoo.py:1125）。
  - 根命令 `tasks`（内建注册表）与 `task`（文件系统 YAML）单复数并存，`tasks list` 与 `task list` 行为完全不同（tasks.py:389 与 429）。
- 问题：用户在子命令间迁移时必须重查 help；脚本封装容易踩错默认值（尤其 cache 默认 off/read-write 的差异会改变行为）。
- 影响：可用性。`--json` 旗标本身覆盖良好（几乎全部查询/执行命令都支持），这点值得肯定。
- 建议：定跨命令旗标词表（metric/limit/manifest-dir/output-dir/cache-mode），旧名以 alias 保留一个版本周期。

### [CM-19] P3 帮助面板与实际注册的命令不一致：`train-audit-rollout-prompts` 被隐藏

- 位置：`worldfoundry/cli/main.py:47-75`（`_PUBLIC_ROOT_COMMANDS`）、`worldfoundry/cli/training_commands/register.py:20-25`
- 证据：`register_training_subparser` 注册了 6 个命令（含 `train-audit-rollout-prompts`），但 `_PUBLIC_ROOT_COMMANDS` 只列了 `train-audit-prompts` 等 5 个；`_curate_root_subparser_help` 用该列表覆盖 metavar，导致根 help 看不到 rollout 版本。`embodied`/`validate`/`evaluate` 在列表里但 `eval` 别名不显示（合理），可 rollout 命令被漏列更像疏忽。
- 建议：从 subparsers 实际注册项生成 metavar，或在列表测试中断言两者一致。

### [CM-20] P3 训练类命令输出契约与其它命令不同：无 `--json` 旗标、无条件打印 JSON

- 位置：`worldfoundry/cli/training_commands/handlers/train.py:128-129`、`handlers/cache.py:75-95、178`、`handlers/audit.py:54-67`
- 证据：`train`/`post-train`/`train-cache`/`train-audit-*` 直接 `print(json.dumps(...))`，不提供 `--json`/人类可读双态；而 CLI 其余命令都是 `--json` 切换。
- 问题：一致性；脚本方无法依赖“加了 --json 才是 JSON”的全局约定。
- 建议：统一加 `--json`（默认人类摘要），或在文档声明训练命令恒为 JSON 输出。

### 主题 H：TUI

### [CM-21] P2 TUI 旗标在三处手工同步（tui.py 解析器、main.py tui 子命令、`_handle_tui` 转发），已出现漂移：`--write-suite-plan` 只有一处有

- 位置：`worldfoundry/cli/tui.py:33-128`、`worldfoundry/cli/main.py:1124-1185`、`worldfoundry/cli/main.py:252-340`
- 证据：同一批约 50 个旗标定义了两遍（`worldfoundry-tui` 入口的 `build_parser` 与 `worldfoundry tui` 子命令），`_handle_tui` 再用约 90 行手工把 Namespace 序列化回 argv 调 `tui_main`。实测漂移：`tui.py:123-127` 定义了 `--write-suite-plan`，`rg "write.suite.plan" worldfoundry/cli/main.py` 无结果——`worldfoundry tui --write-suite-plan` 报“unrecognized arguments”，而 `worldfoundry-tui --write-suite-plan` 可用。
- 问题：三份镜像必然继续漂移；`_handle_tui` 的手工转发对 `store_true+default=None` 旗标依赖 `getattr(..., False)` 的微妙约定。
- 建议：main.py 的 tui 子命令改成 `add_parser("tui", add_help=False, parents=[tui.build_parser_parent()])` 或干脆 `parse_known_args` 后把剩余 argv 原样交给 `tui_main`，删除 90 行转发。

### [CM-22] P2 TUI 内两处同步阻塞事件循环：目录刷新与产物列举

- 位置：`worldfoundry/cli/tui_app.py:841-852`（`action_refresh`）、`tui_app.py:905-922`（`_show_artifacts`）
- 证据：

```841:852:worldfoundry/cli/tui_app.py
    def action_refresh(self) -> None:
        """Reload the catalog from disk and re-sync all UI state."""
        try:
            self.catalog = load_tui_catalog(
                model_manifest_dir=self.model_manifest_dir,
                ...
```

  `load_tui_catalog` 同步读全部模型/基准 manifest + runtime profile + conda 环境探测（`tui_discovery.py:1908-1962`），在按键 action 里直接调用；`_show_artifacts` 对输出目录 `rglob("*")` 并逐文件 `stat()`（tui_app.py:915），视频输出目录常有数千文件。整个文件无任何 `run_worker`/`@work`/thread 用法（grep 证实）。
- 问题：Textual 单事件循环，这两个 action 期间 UI 完全冻结（网络盘上的 manifest 目录可达秒级）。作为对照，命令执行（`_run_command_task`，tui_app.py:1287-1324）与 GPU 查询用 asyncio 子进程实现得很好，停止流程有 terminate→5s→kill 兜底（tui_app.py:1326-1341），退出时 `on_unmount` 也会清理子进程（tui_app.py:775-789）——阻塞问题仅存在于这两处纯 Python 同步调用。
- 建议：两处改用 `self.run_worker(..., thread=True)` + `call_from_thread` 回填 UI；`__init__` 中的首次 `load_tui_catalog`（tui_app.py:175-180）也可移到 `on_mount` worker 中以缩短黑屏时间。

### [CM-23] P2 TUI 自带约 500 行手工模型/变体注册表，与 model zoo 清单平行维护

- 位置：`worldfoundry/cli/tui_discovery.py:51-540`（`INFER_MODEL_VARIANTS`/`INFER_VARIANT_LABELS`/`INFER_VARIANT_TO_MODEL`/`INFER_VARIANT_GROUPS`/`INFER_MODEL_DEFAULT_VARIANTS`/`SCRIPT_INFER_FAMILY_TO_TUI_MODEL_ID`/`INFER_GROUP_TASKS`/`INFER_CONTROL_ORDER`）、`tui_discovery.py:781-999`（每变体控件文案覆盖）
- 证据：如 `"depth-anything-v3": ("da3-small", "da3-base", "da3-large", "da3-large-1.1", ...)` 等按模型逐一硬编码；`load_tui_catalog` 再把这些“虚拟行”与 zoo registry 合并（tui_discovery.py:1940-1951）。
- 问题：模型目录数据固化在 TUI 代码里，新增/改名模型要同时改 zoo manifest 和这批表；与 CM-15 的 CLI 契约推断共同构成“目录知识散落三处”（zoo manifest、model_run.py、tui_discovery.py）。
- 建议：把变体分组/标签/控件顺序迁入 model zoo manifest 或 studio catalog 的结构化字段，TUI 只消费。

### [CM-24] P3 小型工具函数跨模块逐字重复

- 位置：`worldfoundry/cli/tui_discovery.py:1503-1515`（`_dedupe_text`）与 `worldfoundry/cli/model_run.py:316-326`（同名同义）；`tui_discovery.py:2347-2356`（`_append_optional` 与 `_append_optional_path` 函数体逐字相同）；`worldfoundry/cli/zoo.py:535-544`（`_add_path_arg`/`_add_value_arg` 亦同体）
- 问题：小函数三处四份，纯维护噪声。
- 建议：合并进 `cli/utils.py`。

### 主题 I：MCP 服务器

总体评价：MCP 侧是本次评审中工程质量较好的部分——工具通过子进程调用 `python -m worldfoundry.cli run`（`runs.py:18,105`）复用 CLI 而非复制业务逻辑；`get_run_samples` 有路径穿越防护（`runs.py:496-514`、`533-537`）；作业存储有进程组终止（SIGTERM→SIGKILL）与内存日志上限（`runtime/jobs.py:382-383`）；每个工具函数有完整 docstring 供 FastMCP 生成 schema。以下为发现的问题。

### [CM-25] P1 精心设计的错误封套（responses.py）从未被接线，整个模块是死代码；实际错误契约回落到 FastMCP 默认行为

- 位置：`worldfoundry/mcp/tools/responses.py:25-55`、`worldfoundry/mcp/tools/registration.py:1-567`
- 证据：`responses.py` 定义了 `invoke_tool`/`invoke_tool_async`/`success_payload`/`error_payload`（`{"ok": false, "error": ..., "error_type": ...}` 结构化封套）。全仓 `rg "invoke_tool|success_payload|error_payload"` 仅命中 `responses.py` 自身——`registration.py` 的 27 个 `@mcp.tool()` 全部直接调用 payload 函数，异常（如 `discovery.py:82` 的 `ValueError("model not found: ...")`）裸抛给 FastMCP。
- 问题：1) 成功响应无 `ok` 字段、失败走 FastMCP 的 `isError=True` + 异常字符串，与 `responses.py` 声明的契约完全不一致；2) 未被 `invoke_tool` 白名单（ValueError/KeyError/FileNotFoundError/LookupError/RuntimeError）覆盖的异常（YAML 解析错、OSError、TypeError）以裸异常形式暴露给 MCP 客户端；3) 58 行死代码误导维护者以为错误已被规范化。
- 影响：Agent 消费方无法依赖稳定的错误结构；schema 文档与实际响应不符。
- 建议：要么在 `register_tools` 中统一用 `invoke_tool` 包裹所有工具，要么删除 `responses.py`。二选一，不要保持现状。

### [CM-26] P1 Studio 等待类工具是同步 `time.sleep` 轮询且默认无限超时，可能挂死整个 MCP 服务器

- 位置：`worldfoundry/mcp/tools/studio.py:247-254`（`wait_for_studio_job_payload`）、`studio.py:118-120`（`wait_timeout_s: float = 0` 即无限）、`registration.py:487-510`/`400-459`（注册为同步 `def`）
- 证据：

```247:254:worldfoundry/mcp/tools/studio.py
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while True:
        payload = get_studio_job_payload(job_id, base_url=base_url, timeout_s=max(1.0, min(poll_interval_s, 30.0)))
        if str(payload.get("status") or "").lower() in {"completed", "failed", "cancelled", "canceled"}:
            return payload
        if deadline is not None and time.monotonic() >= deadline:
            return payload
        time.sleep(max(0.25, poll_interval_s))
```

  `wait_for_studio_job` 与 `submit_studio_inference(wait=True)` 都注册为**同步**工具；依赖为官方 `mcp>=1.28.1`（pyproject:269）的 FastMCP。官方 python-sdk 的 FastMCP 历来在事件循环线程内联调用同步工具（`call_fn_with_arg_validation` 对非协程直接 `fn(**args)`；本环境未安装 mcp 包，1.28 具体行为建议复核）。
- 问题：若同步工具内联执行，一次 `wait_for_studio_job(timeout_s=0)`（默认值！）+ 推理作业跑数小时 = `time.sleep` 轮询占死事件循环——stdio 传输下服务器不再响应任何请求（包括 list_tools、其它工具、甚至协议 ping），客户端只能杀进程。即便 SDK 已改为线程池执行，默认无限等待也会长期占用工作线程且无法取消。对照组：`evaluate` 用 `async` + `asyncio.sleep` 实现了正确的等待（`runs.py:575-596`），且 `wait_timeout_s` 默认 90 秒有限值。
- 建议：`wait_for_studio_job`/`submit_studio_inference` 改 async + `asyncio.sleep`（或 `anyio.to_thread`），并把无限等待默认值改为有限（如 600s）；`time.sleep` 轮询循环不应存在于任何服务器代码中。

### [CM-27] P2 MCP 反向依赖 CLI 表现层：目录加载与命令构造从 `cli.tui_discovery` 导入

- 位置：`worldfoundry/mcp/tools/discovery.py:13`、`worldfoundry/mcp/tools/runs.py:18`
- 证据：

```13:13:worldfoundry/mcp/tools/discovery.py
from worldfoundry.cli.tui_discovery import load_tui_catalog
```

```18:18:worldfoundry/mcp/tools/runs.py
from worldfoundry.cli.tui_discovery import build_model_benchmark_command, build_suite_command
```

- 问题：复用本身是对的（避免了复制粘贴），但被复用的"统一目录视图 + run 命令构造"逻辑长在 TUI 的 discovery 模块里（连同 CM-23 的 500 行手工变体表）。MCP 返回的模型列表因此包含 TUI 专属的"虚拟行"合并语义；`worldfoundry.mcp → worldfoundry.cli` 的依赖方向违反"cli 只做解析+调度"的分层设计。
- 建议：把 `load_tui_catalog`/`build_*_command` 下沉到 `worldfoundry/evaluation/`（或新的 `worldfoundry/catalog/`）库层，cli 与 mcp 同时消费。

### [CM-28] P2 作业状态仅存内存：服务器重启后 run_id 全部失效，而子进程作为孤儿继续运行

- 位置：`worldfoundry/runtime/jobs.py:217`（`self._jobs: dict[str, CommandJob] = {}`）、`jobs.py:336`（`start_new_session=True`）、`worldfoundry/mcp/tools/context.py:36`
- 证据：作业注册表是进程内 dict，无落盘；子进程以新会话启动（有意与服务器解耦信号）。MCP stdio 服务器由客户端拉起/杀掉是常态——重启后 `list_runs` 返回空，而上一个实例提交的评测子进程仍在跑，只能靠 `ps` 人工找回。`_jobs` 也没有淘汰机制（每条含最多 `max_log_lines` 行日志，长驻进程缓慢增长）。
- 建议：把作业元数据（run_id、pid、output_dir、状态）写入 `output_root` 下的 JSON 索引，启动时恢复/对账（pid 存活性检查）；`_jobs` 加上限或按完成时间清理。
- 备注：每个 run 的 `events.jsonl`/stdout/stderr 已经落盘（`jobs.py:319-328`），缺的只是索引。

### [CM-29] P2 MCP 默认输出/数据路径是相对路径，取决于服务器进程 CWD

- 位置：`worldfoundry/mcp/tools/context.py:16`、`worldfoundry/mcp/tools/readiness.py:26`
- 证据：

```16:16:worldfoundry/mcp/tools/context.py
DEFAULT_MCP_OUTPUT_ROOT = Path(os.environ.get("WORLDFOUNDRY_MCP_RUN_ROOT", "runs/mcp"))
```

  `readiness.py:26` 同样：`cache_dir = Path(data_root or "datasets")`。
- 问题：MCP 服务器由 Claude Desktop/Cursor 等客户端以任意 CWD（常为 `/` 或客户端安装目录）拉起。`evaluate` 会在未知位置创建 `runs/mcp/...` 目录（或因权限失败）；`check_benchmark_datasets` 在错误位置找 `datasets/` 恒报 not-ready。另有小陷阱：`context.py:39` 的 `DEFAULT_CONTEXT = MCPToolContext()` 是 import 时创建的全局单例（自带独立 job_store），与 `server.py:44` 创建的 context 并存——payload 函数默认参数指向前者，直接调用时作业会落进与服务器不同的 store。
- 建议：默认锚定到明确的绝对根（如 `~/.worldfoundry/runs/mcp` 或 `WORLDFOUNDRY_REPO_ROOT`），启动时在 `server_info` 里回显解析后的绝对路径；删除 `DEFAULT_CONTEXT` 或让 server 显式注入。

### [CM-30] P3 每次目录查询全量重载 manifest + conda 探测，长驻服务器无缓存

- 位置：`worldfoundry/mcp/tools/discovery.py:40-43`、`75-78`、`108-111`、`140-143`
- 证据：四个 discovery 工具每次调用都执行 `load_tui_catalog(...)` → 读全部模型/基准 YAML + `load_runtime_profile_rows`（`tui_discovery.py:1879-1905`，无 `lru_cache`）→ conda 环境目录探测。Agent 会话中 list/get 是高频调用。
- 建议：在 `MCPToolContext` 挂一个带 TTL（如 30s）的 catalog 缓存。

### [CM-31] P3 工具面手工同步与暴露不全：`MCP_TOOL_NAMES` 27 项人工清单；`preview_run` 与 `evaluate` 参数面不一致

- 位置：`worldfoundry/mcp/tools/server_info.py:13-41`、`registration.py:194-227` vs `registration.py:229-283`
- 证据：`MCP_TOOL_NAMES` 是与 27 个 `@mcp.tool()` 装饰器平行维护的字符串清单（当前一致，但无机制保证）；`preview_run` 只接受单 `benchmark`、不暴露 `metrics/model_variant/requests_path/task_name/suite_ids`，而 `evaluate` 支持多 benchmark——用户无法预览 `evaluate` 实际会执行的多基准命令；底层 `run_evaluation_payload` 的 `model_variant/requests_path/task_name/metrics/suite_ids` 参数（`runs.py:131-136`）也没有从 `evaluate` 工具暴露。
- 建议：`server_info` 从 `mcp.list_tools()` 动态取名单；`preview_run` 与 `evaluate` 共享同一参数签名。

### [CM-32] P3 `MCPClient` 每次调用都重新拉起一个 MCP 服务器子进程

- 位置：`worldfoundry/mcp/client.py:39-45`、`57-70`、`108-144`
- 证据：`list_tools`/`call_tool` 各自 `await self._client()` → 新 `stdio_client(params)` + `ClientSession.initialize()`，用完即弃。结合 CM-1（服务器 import 链 ~4s），每次工具调用都要付出完整启动成本。
- 建议：提供持久会话用法（`async with MCPClient() as session` 缓存 exit stack），或至少在 docstring 标明该类只适合一次性脚本。

### 主题 J：交叉验证与正面观察

### [CM-33] P3 主入口 `worldfoundry` 的所有 help/usage/错误文案品牌为 `worldfoundry-eval`

- 位置：`worldfoundry/cli/main.py:1086`（`prog="worldfoundry-eval"`）、`pyproject.toml:[project.scripts]`
- 证据：`worldfoundry` 与 `worldfoundry-eval` 指向同一个 `worldfoundry.cli:main`，但 parser 硬编码 `prog="worldfoundry-eval"`。实测 `python -m worldfoundry.cli zoo …` 出错时 usage 打印 `usage: worldfoundry-eval zoo [-h] …`。用主命令名 `worldfoundry` 的用户会被文案引导去敲另一个命令。
- 建议：`prog=Path(sys.argv[0]).name`（回退默认），或统一文档只宣传一个入口名。

**交叉验证结论与正面观察**（有证据支撑，避免报告只见问题不见基线）：

- **入口一致性**：`[project.scripts]` 5 个入口（`worldfoundry`/`worldfoundry-eval` → `cli:main`、`worldfoundry-mcp` → `mcp:main`、`worldfoundry-studio` → `studio.cli:main`、`worldfoundry-tui` → `cli.tui:main`）经静态解析全部可解析到实际 `main` 函数，无断链。
- **包级入口文件干净**：`worldfoundry/__init__.py` 仅 docstring、明确声明不做重导出（`import worldfoundry` 本身轻量）；`worldfoundry/__main__.py` 与 `cli/__main__.py` 均为 3 行委托 + `SystemExit`；`worldfoundry/data/__init__.py` 为空文件，无隐藏代码。
- **JSON stdout 纯净性**：实测 `zoo benchmarks --json` stdout 可被 `json.load` 直接解析，日志（如 `INFO CLI command finished`）正确落 stderr——成功路径的 stdout/stderr 分离是干净的（失败路径见 CM-07）。
- **无裸 `except:`**：`cli/` 与 `mcp/` 全部 48 个文件中没有 bare except；`except Exception` 计 18 处，绝大多数带日志或有意降级（个别静默处已在 CM-15/CM-16 提及）。`mcp/client.py:137` 的 `except BaseException` 用于异步上下文栈清理后重抛，正当。
- **Ctrl-C 行为**：CLI 层 `KeyboardInterrupt` → exit 130（`main.py:1856`）；MCP 服务器把 Ctrl-C 视为正常停机返回 0（`server.py:61-65`）；TUI 退出时 `on_unmount` 主动终止遗留子进程（`tui_app.py:775-789`）。三个入口的信号语义都经过设计。
- **子进程执行**：TUI 与 MCP 的作业执行统一走 `runtime/jobs.py` 的 `AsyncCommandJobStore`（asyncio 子进程、进程组 SIGTERM→SIGKILL、内存日志截断、原始流落盘），是值得保持的共享基座。

## 汇总

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0 | 1 | CM-01 |
| P1 | 4 | CM-02, CM-07, CM-25, CM-26 |
| P2 | 14 | CM-03, CM-04, CM-05, CM-09, CM-11, CM-15, CM-16, CM-18, CM-21, CM-22, CM-23, CM-27, CM-28, CM-29 |
| P3 | 14 | CM-06, CM-08, CM-10, CM-12, CM-13, CM-14, CM-17, CM-19, CM-20, CM-24, CM-30, CM-31, CM-32, CM-33 |
| **合计** | **33** | |

按主题分布：启动性能 3（含 P0）、入口与分发 2、退出码与错误呈现 2、日志副作用 2、死代码与重复 4、分层 3、参数与输出一致性 3、TUI 4、MCP 8、交叉与其它 2。

### Top 5 问题

1. **[CM-01] P0 — 每次 CLI 启动都加载 torch，`--help` 固定 4 秒。** `configure_logging` 的“惰性”导入穿过急切 re-export 的 `core.distributed/__init__` 拉起 torch（2.5s），直接违反 stdlib-first 设计原则，波及全部 5 个入口（含 MCP 服务器启动）。修法明确：`sys.modules` 探测 + PEP 562 惰性 re-export。
2. **[CM-26] P1 — MCP 的 `wait_for_studio_job`/`submit_studio_inference(wait=True)` 是同步 `time.sleep` 无限轮询。** 注册为同步工具且默认 `timeout_s=0`（无限），一次调用即可让 stdio MCP 服务器停止响应一切请求；对照 `evaluate` 的 async 实现，属于同一代码库内的已知正解未被套用。
3. **[CM-07] P1 — `--json` 模式下错误契约中断且退出码语义混叠。** 实测失败时 stdout 0 字节、stderr 输出完整 traceback + `error:` 行、exit 2 与用法错误共用；自动化消费方（CI/agent）没有可依赖的失败结构。
4. **[CM-25] P1 — MCP 错误封套模块整体死代码。** `responses.py` 的 `{"ok": false, ...}` 契约从未接线，27 个工具全部裸抛异常给 FastMCP 默认处理，声明的错误契约与实际行为不符。
5. **[CM-04]+[CM-11] P2 — 死路旗标与死模块两处“遗留双轨”。** `--task-type/--benchmark-name/--data-path` 保留了旗标、help、必填校验但运行必抛"已裁撤"；`cli/embodied.py` 整模块无人调用且与 main.py 内联版漂移（修复曾打在死拷贝上）。两者共同说明裁撤/迁移未清尾。

### 一句话总评

CLI/TUI/MCP 三层的**执行基座**（`AsyncCommandJobStore`、asyncio 子进程、信号处理、JSON stdout 纯净性）质量扎实，但**启动路径违反自家 stdlib-first 原则**（P0）、**错误契约在 CLI 与 MCP 两侧都未兑现**（2×P1），且目录/契约知识散落在 `zoo.py`、`model_run.py`、`tui_discovery.py` 三处并被 MCP 反向依赖——建议优先修 CM-01/02（一天内可完成、全入口收益），随后统一错误契约（CM-07/25），再做目录逻辑下沉（CM-15/16/23/27 一揽子）。
