# evaluation tasks 层评审（schema/registry/runners 接线/data 清单）

> 状态：已完成。最后更新：2026-08-14

## 评审范围与方法

**范围**（`worldfoundry/evaluation/tasks/` 全部自研层 + `worldfoundry/data/` 清单数据）：

| 模块 | 内容 | 规模 |
| --- | --- | --- |
| catalog/ | schema.py(1292) registry.py(447) yaml.py(702) zoo_registry.py(483) integrity.py(251) benchmark_catalog.py(341) specs.py(495) __init__.py(767) | ~4.8k 行 |
| contracts/ | external.py(1804) registry.py(219) __init__.py(175) | ~2.2k 行 |
| execution/framework/ | 17 个文件：official_runner.py(887) in_tree_evaluator.py(1133) video_contract_evaluator.py(984) result_normalizer.py(1051) 等 | ~7.3k 行 |
| execution/orchestration/ | 16 个文件：benchmark_runner.py(2134) model_benchmark_suite.py(1726) evaluate.py(1301) contract.py(1240) 等 | ~12k 行 |
| execution/runners/ | 52 个 runner 目录（每个含自研 wrapper .py + 可选 vendored runtime/），wrapper 合计约 60k 行 | 抽查 |
| metrics/ | registry.py(886) + ~55 个指标目录 | 抽查 |
| datasets/ embodied/ | manager/manifest；embodied 仅查跨界问题 | 抽查 |
| data/ | benchmarks/{catalog,tasks/external,suites,runtime_profiles,eval_configs,...} models/{catalog,runtime_profiles,bindings,acquisition_targets} | 一致性脚本验证 |

**方法**：
1. 先读 `.cursor/skills/worldfoundry-evaluation-guide/`（SKILL + benchmarks/runners references）与 `.claude/skills/worldfoundry-benchmark/SKILL.md`，以其定义的"合格接入"标准审查现有 runner。
2. 框架代码（catalog/contracts/framework/orchestration/metrics registry）逐文件精读。
3. runner wrapper 精读 ≥10 个代表（大小/模式覆盖 vendored-runtime、importer、external-checkout、judge 型），其余 `rg` 扫共性反模式（sys.path、shell=True、bare except、subprocess 超时、硬编码路径）并抽查。
4. data 清单一致性用临时脚本（/tmp，不入仓库）交叉验证 catalog id ↔ task YAML ↔ runtime profile ↔ docs ↔ runner 注册表。
5. vendored `runtime/` 子目录只查集成卫生（sys.path hack、动态 import、subprocess 方式），不评上游代码风格。

严重度：P0=损坏/危险；P1=严重设计缺陷；P2=应修复；P3=改进建议。

**正面评价（先说结论）**：框架核心（orchestration 的 evaluate/cache/existing_results、catalog 的 YAML 继承加载、runtime_preflight、datasets/manifest 校验）设计与实现质量高：逐样本指标隔离、生成缓存的确定性判定与工件存在性校验、preflight 的超时+脱敏都做得规范。主要问题集中在 **runner 接线层**（多张平行注册表、subprocess 生命周期各写各的、失败不写 scorecard）与**指标归一化的静默启发式**。

## 发现（按主题分组）

### 主题 1：runner 接线层一致性（多张平行注册表 + CLI 契约漂移）

#### [ET-01] P1 同一 benchmark 的接线信息散落在 3 张手工维护的平行注册表，且相互缺漂
- 位置：
  - `worldfoundry/evaluation/tasks/execution/framework/runner_registry.py:32-239`（`VIDEO_RUNNER_REGISTRY`，~50 项：script + results_flag）
  - `worldfoundry/evaluation/tasks/execution/framework/integration.py:70-304`（`BENCHMARK_INTEGRATION_REGISTRY`，~45 项：tier + hf_dataset_id + judge_model_id）
  - `worldfoundry/evaluation/tasks/execution/runners/workspace_registry.py:58-489`（`CLI_RUNNERS`，~48 项：module + 每个 runner 的 CLI flag 名映射）
- 证据（/tmp 脚本交叉比对注册表键集）：`BENCHMARK_INTEGRATION_REGISTRY` 缺少 **5** 个 id：`4dworldbench`、`larybench`、`sana-wm-bench`、`worldolympiad`、`worldreasonbench`（均在 `VIDEO_RUNNER_REGISTRY`（49 项）与 `CLI_RUNNERS`（47 项）注册；integration 表只有 44 项）。而 integration.py 通过 `worldfoundry/evaluation/public.py:36-41` 公开导出为 `benchmark_integration_spec()` 公共 API，对这 5 个已接入 benchmark 返回 `None`。此外 vbench 家族没有进 `CLI_RUNNERS`，而是在 dispatcher 里以第四种形态特判：`workspace_registry.py:593` `return key in {"vbench", "vbench-2.0", "vbench-plus-plus"} or key in CLI_RUNNERS`。
- 问题：同一 benchmark 的"如何运行"被拆成 script 路径、tier/数据集、CLI flag 三份独立数据（外加 vbench 特判集合），新增 benchmark 需手工同步 3 处 + catalog/task/runtime profile/docs（skill 文档承认 ~10 处触点）。`validate_workspace_registry()`（workspace_registry.py:1144-1168）只交叉校验其中 2 张表，integration.py 不在校验范围内，缺漂已实际发生。
- 影响：公共 API 对部分已接入 benchmark 静默返回 None；新增 benchmark 极易漏一处（skill 里也说"the half that silently breaks is the wiring"）。
- 建议：将三表合一为单一 `RunnerSpec`（或由 catalog YAML 生成），`validate_workspace_registry` 扩展为覆盖 integration registry 的 wiring test；短期先补 5 个缺失项并加一致性测试。

#### [ET-02] P2 runner CLI 契约不统一：6 个 runner 不接受 `--benchmark-id`，flag 名逐 runner 漂移
- 位置：`worldfoundry/evaluation/tasks/execution/framework/runner_registry.py:56,81,118,123,180,185`（`pass_benchmark_id=False`：evalcrafter、iworld-bench、phygenbench、phyground、videophy、videophy2）；`workspace_registry.py:127-137`（fetv 用 `--fetv-metrics/--fetv-limit/--fetv-model-name`，其余 runner 用 `--metrics/--limit/--model-name`）
- 证据：

```
    "evalcrafter": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/evalcrafter/run_evalcrafter_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
```

- 问题：`.claude/skills/worldfoundry-benchmark/SKILL.md` 明确规定 "the zoo dispatcher always passes `--benchmark-id`, `--output-dir`, `--json` … so accept all of them"。6 个 runner 未达标，反而在 dispatcher 侧加特判 flag 兼容；fetv 等自定义前缀 flag 使 `CLI_RUNNERS` 需要 14 个可选字段来描述"每个 runner 的方言"。
- 影响：每接一个新 benchmark，dispatcher 表就可能多一个特判；批量调用（suite 模式）无法统一构造命令行。
- 建议：把 `--benchmark-id/--metrics/--limit/--model` 收敛为共享 parser（`official_runner.build_common_parser` 已存在，6 个漂移 runner 应迁移），删除 `pass_benchmark_id` 与自定义前缀 flag。

#### [ET-03] P3 `VideoRunnerSpec.results_flag` 50 项全部为同一值
- 位置：`runner_registry.py:32-239`，全部 50 项 `results_flag="--official-results-path"`
- 问题：该字段没有任何变体，逐条重复 50 次；`specialized_result_normalizer_scripts()` 还把它复制成 `(script, flag)` 元组传播到 benchmark_runner.py。
- 建议：删除字段用常量，或等真的出现变体时再引入。

### 主题 2：subprocess / conda 生命周期

#### [ET-04] P1 共享 runner CLI 框架：超时与意外异常不写 scorecard.json，违反自家"失败也要留 scorecard"契约；且不用仓库已有的进程组安全工具
- 位置：`worldfoundry/evaluation/tasks/execution/framework/official_runner.py:711-724`（subprocess.run）与 `official_runner.py:860-871`（异常捕获白名单）
- 证据：

```712:720:worldfoundry/evaluation/tasks/execution/framework/official_runner.py
            completed = subprocess.run(
                command,
                cwd=command_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
```

```863:871:worldfoundry/evaluation/tasks/execution/framework/official_runner.py
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        logger.event(
            "ERROR",
            "official_runner.failed",
            "Official runner failed",
            exc_info=True,
        )
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

- 问题：
  1. `subprocess.TimeoutExpired` 不在 `run_main` 的捕获白名单里（它是 `SubprocessError`，不是 OSError）——官方命令超时会直接 traceback 崩出，**不写 scorecard.json**。而 skill 契约明确要求 "On any failure, still write a valid `scorecard.json` with `run.status: 'failed'` and return 1; downstream validation reads the file, not the exit code"。KeyError/TypeError 等同样穿透。
  2. `subprocess.run(capture_output=True, timeout=...)` 超时时孙子进程不被清理（无 `start_new_session`/进程组 kill），GPU judge/dataloader 子进程会变孤儿；且 stdout/stderr 在超时路径上完全丢失（`stdout_path.write_text` 只在正常返回后执行，official_runner.py:723-724）。
  3. 仓库已有专门解决这两个问题的 `worldfoundry/core/process.py:run_logged_subprocess`（流式写日志 + `terminate_process_group`），编排层 benchmark_runner.py 已在用（benchmark_runner.py:1204-1212 并正确捕获 TimeoutExpired→status="timeout"），但被 ~40 个 runner 共享的 official_runner.py 没有用。
- 影响：所有走 `run_main` 的 runner（绝大多数 `run_*_official_runner.py`）在超时/意外异常时违反 scorecard 契约，下游 `_run_cli_command`（workspace_registry.py:1103-1107）会因 scorecard 缺失抛 RuntimeError，只能靠 stderr tail 诊断；长时间 GPU 评测超时后留下孤儿进程占卡。
- 建议：`run_main` 改为 `except Exception`，统一在 except 分支写 `run.status="failed"` 的 scorecard 后返回 1；`run_official_pipeline` 换用 `run_logged_subprocess`。

#### [ET-05] P2 Workspace 调度层 subprocess 无超时
- 位置：`worldfoundry/evaluation/tasks/execution/runners/workspace_registry.py:1092-1099`
- 证据：

```1092:1099:worldfoundry/evaluation/tasks/execution/runners/workspace_registry.py
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
```

- 问题：`_run_cli_command` 是 Studio/Workspace 与 `official-run` 模式的实际执行入口，对子 runner 无 timeout；内层 runner 的 `--timeout` 只约束它自己的孙进程，runner 本身 hang（如 judge API 卡死、模型加载死锁）时 Workspace 永久等待。`capture_output=True` 同样在长运行时把全部输出缓存在内存。
- 建议：透传 `config["timeout"]` 到这里的 `subprocess.run`（或改用 `run_logged_subprocess`）。

#### [ET-06] P2 7 个 runner 运行时文件的 subprocess 调用完全无超时
- 位置（文件内无任何 `timeout` 字样，均为长运行 GPU/judge 子进程）：
  - `runners/worldarena/run_worldarena_official_runner.py`
  - `runners/videophy/videophy_runtime.py`、`runners/videophy2/videophy2_runtime.py`
  - `runners/phygenbench/phygenbench_runtime.py`
  - `runners/wbench/run_wbench_official_runner.py`
  - `runners/wrbench/wrbench_runtime.py`
  - `runners/_scorers/vqa_score/_backend.py`（ffmpeg 探测，风险低）
- 问题：judge 推理/训练型子进程 hang 时无任何兜底；与 ET-04/ET-05 叠加形成"三层都可能无限等待"。
- 建议：统一走 `run_logged_subprocess(timeout=...)`，默认取 `WORLDFOUNDRY_BENCHMARK_TIMEOUT`。

#### [ET-07] P2 subprocess 工具使用分裂：8 个文件用共享的 `run_logged_subprocess`，16 个文件裸用 `subprocess.run/Popen`
- 位置：rg 统计（排除 vendored runtime/）——使用共享工具：likephys、ewmbench、vbench、vbench_2_0、videoscore、mirabench、videobench、worldmodelbench；裸用：larybench/cli.py(3 处)、t2v_compbench(2 处)、wrbench、worldscore、worldarena、wbench、vmbench、videophy、videophy2、phygenbench、memobench、iworldbench、fetv、chronomagic_bench 等
- 问题：同一类"驱动 vendored runtime 子进程"的需求，一半 runner 自己复制 env 组装/日志重定向/超时逻辑，行为差异（是否留 stdout 文件、是否杀进程组、是否记录 lifecycle 事件）取决于 runner 作者当天的选择——典型复制粘贴漂移。
- 建议：在 `official_runner.py` 提供统一的 `run_vendored_runtime()` helper 并迁移。
- 备注（正面）：全部自研 wrapper 无 `shell=True`（rg 0 命中，仅 benchmark_runner.py:1211 在 manifest 命令为字符串时受控使用）；无硬编码 `/mnt|/home` 路径；`run_*_official_runner.py` 顶层无 torch import（导入卫生达标）。

### 主题 3：sys.path 操纵与动态 import 卫生

#### [ET-08] P2 videoscore 声称"临时"加 sys.path 实际永久污染，且以顶级通用名 import、并全局 monkeypatch transformers
- 位置：`worldfoundry/evaluation/tasks/execution/runners/videoscore/run_videoscore_official_runner.py:801-807, 810-831`
- 证据：

```801:807:worldfoundry/evaluation/tasks/execution/runners/videoscore/run_videoscore_official_runner.py
    # Temporarily add benchmark_dir and videoscore_root to sys.path for import.
    inserted = [str(benchmark_dir), str(videoscore_root)]
    for item in reversed(inserted):
        if item not in sys.path:
            sys.path.insert(0, item)
    # Import the module.
    return importlib.import_module("eval_videoscore")
```

- 问题：
  1. 注释写 "Temporarily" 但没有任何移除逻辑，`sys.path` 头部永久多出两个 vendored 目录；此后同进程内任何 import 都可能命中 vendored 目录里的顶级模块（如 `utils`、`benchmark`），与 in-process 评测（`in_tree_evaluator` 走同进程）叠加时有模块遮蔽风险。
  2. `import_module("eval_videoscore")` 用顶级通用名而非包限定名，靠 sys.path 顺序保证正确性，并发/多 benchmark 复用同进程时不可靠。
  3. `patch_transformers_dynamic_cache_api()` 直接改 `DynamicCache` 类属性，影响整个进程的 transformers 行为，其他 runner/metric 在同进程中拿到的是被 patch 过的类。
- 影响：单跑 videoscore 无碍；但该 wrapper 被 `in_tree` 或 workspace 同进程调用链复用时，路径污染与 monkeypatch 会泄漏给后续 benchmark。
- 建议：用 `try/finally` 恢复 `sys.path`（或 `importlib.util.spec_from_file_location` 定向加载）；monkeypatch 至少要能撤销，或改为在 vendored runtime 子进程内做。

#### [ET-09] P3 wrapper 顶部 sys.path shim 写法不一致（条件插入 vs 无条件插入）
- 位置：`runners/sana_wm_bench/run_sana_wm_bench_official_runner.py:8-9`（`if str(_REPO_ROOT) not in sys.path: sys.path.insert(0, ...)`）；`runners/devil_dynamics/run_devil_dynamics_official_runner.py:12-13`（无条件 `sys.path.insert(0, ...)`）
- 问题：同一用途（保证 `python path/to/runner.py` 直跑时能 import worldfoundry）各写各的；无条件插入在重复调用时会堆叠重复路径项。
- 建议：抽成 `framework/bootstrap.py` 的单行 helper，或统一改用 `python -m` 调用便可删除全部 shim（workspace_registry 已按 module 方式调用）。

### 主题 4：指标接线（提取/归一化/聚合）

#### [ET-10] P1 `normalize_unit_score` 对 (1,100] 区间的值一律 /100，量纲启发式会静默歪曲非百分制指标
- 位置：`worldfoundry/evaluation/tasks/execution/framework/io.py:156-174`
- 证据：

```156:174:worldfoundry/evaluation/tasks/execution/framework/io.py
def normalize_unit_score(value: Any) -> float | None:
    """Normalizes an arbitrary numeric score into the unit interval [0.0, 1.0].

    Values between 1.0 and 100.0 are assumed to be percentages and are divided by 100.
    """
    number = scalar_number(value)
    if number is None:
        return None
    normalized = number / 100.0 if 1.0 < number <= 100.0 else number
    return min(max(normalized, 0.0), 1.0)
```

- 问题：仅凭数值大小猜量纲：1–5 Likert 打分（videophy 语义一致性等 judge 类分数）、FVD/FID 之类无界距离一旦流经此函数，4.2 会变 0.042、350 会被截到 1.0，且完全静默。调用方遍布 runner 指标提取路径（`rg normalize_unit_score` 命中 framework 与多个 runner 的 metrics 模块）。
- 影响：scorecard 的 normalized 值可能与官方口径系统性偏离，而 raw 值/normalized 值哪个进 leaderboard 取决于下游选择，错误难以察觉。
- 建议：量纲声明放到 `BenchmarkMetricSpec`（catalog 已有 min/max/unit 字段位），normalize 只按声明换算；无声明时不猜、保留 raw 并标记 `normalized=None`。

#### [ET-11] P1 注册表能通过校验的指标 id，`BuiltinExistingResultsMetric` 计算时静默不产出（校验面与实现面脱节）
- 位置：`worldfoundry/evaluation/tasks/metrics/registry.py:242-273`（`__call__` 只实现 5 类内置 id）与 `registry.py:492-673, 285-316`（注册表同时注册 vqa_score/cmmd/fid 等 ~60 个"声明式"条目）
- 证据：`BuiltinExistingResultsMetric.__init__` 用 `validate_metric_ids(raise_on_error=True)` 校验——`fid`、`cmmd`、`vqa_score` 等都能通过（它们在 `BUILTIN_METRIC_REGISTRY_ENTRIES`/`_DISCOVERABLE_METRIC_PACKAGES` 里）；但 `__call__` 的分支只有 `artifact_count / required_artifacts_present / numeric / has_artifact: / numeric:` 五种，其余 id 落空，循环结束后只回默认的 `generation_success`。
- 问题：`run_evaluate(metrics=["fid"])` 或 CLI `--metrics cmmd` 不报错、不产出该指标，scorecard 里只有 generation_success——用户以为跑了 FID 实际什么都没算。`MetricRegistryEntry.spec` 还把所有条目的 `implementation` 都写成 `...:BuiltinExistingResultsMetric`（registry.py:433），对 ~55 个指标包是错误的自我描述。
- 影响：静默丢指标是评测框架最危险的失败模式之一；且注册表元数据误导调用方以为这些指标可离线计算。
- 建议：`create_existing_results_metric` 应拒绝（或显式路由）非五类内置 id：要么报 "metric X requires benchmark-zoo in-tree evaluator / metric package API"，要么接通对应 metric 包的 `compute()`。`implementation` 字段按实际包路径填写。

#### [ET-12] P2 共享框架的指标别名解析用双向子串匹配，易误配
- 位置：`worldfoundry/evaluation/tasks/execution/framework/official_runner.py:183-190`；同模式又在 `runners/iworldbench/iworldbench_metrics.py:121-128` 复制了一份（单向子串）
- 证据：

```183:190:worldfoundry/evaluation/tasks/execution/framework/official_runner.py
    if normalized in config.metric_aliases:
        return config.metric_aliases[normalized]
    if normalized in config.metric_specs:
        return normalized
    for alias, metric_id in config.metric_aliases.items():
        if alias in normalized or normalized in alias:
            return metric_id
```

- 问题：`alias in normalized or normalized in alias` 是双向包含：上游列名 `consistency` 会命中 `subject_consistency`/`background_consistency` 中先迭代到的那个（dict 顺序决定）；短别名（如 `iq`）几乎匹配一切含它的列名。匹配错了没有任何日志。
- 影响：上游结果表列名轻微变化时，值可能被挂到错误的 metric id 下，进入 scorecard 后无从发现。
- 建议：只允许精确匹配 + 显式别名表；子串匹配至少要求唯一命中且记录 `logger.event("WARN", ...)`。

#### [ET-13] P3 `devil_dynamics` 把通用指标名别名到 dynamics 专属指标
- 位置：`runners/devil_dynamics/run_devil_dynamics_official_runner.py:28`（`metric_aliases={..., 'subject_consistency': 'dynamics_range', 'background_consistency': 'dynamics_range', 'motion_smoothness': 'dynamics_controllability', ...}`）
- 问题：与 ET-12 的子串匹配叠加时，通用名别名扩大误配面；跨 benchmark 语义上 `subject_consistency` ≠ dynamics 指标。
- 建议：别名只收录上游真实出现过的列名。

### 主题 5：错误处理 / scorecard 失败契约

#### [ET-14] P1 不走共享框架的独立 runner（如 worldscore）失败路径同样不写 scorecard，异常白名单再次复制
- 位置：`worldfoundry/evaluation/tasks/execution/runners/worldscore/run_worldscore_official_runner.py:1189-1193`
- 证据：

```1189:1193:worldfoundry/evaluation/tasks/execution/runners/worldscore/run_worldscore_official_runner.py
    try:
        scorecard = run_official_worldscore(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

- 问题：与 ET-04 同构但又是**另一份**手写实现：白名单集合都不一样（这里多了 TimeoutExpired、少了 ImportError），失败时同样只打 stderr、不写 `scorecard.json`，违反 skill 契约 "On any failure, still write a valid scorecard.json with run.status: 'failed'"。它还自带第三种 subprocess 超时实现（`run_command_with_timeout`，worldscore:925-954：0.5s 忙等轮询 + `process.kill()` 不杀进程组）。
- 影响：失败契约在共享框架和独立 runner 两个家族都不成立——下游 `_run_cli_command` 读不到 scorecard 只能抛 RuntimeError；每个独立 runner 的失败行为取决于各自的白名单。
- 建议：把"任何异常 → 写 failed scorecard → return 1"做成框架级 `main_guard()` 装饰器，独立 runner 与 `run_main` 共用。

#### [ET-15] P2 官方命令退出码非零时只要发现了结果文件就照常打分，且发现逻辑取 mtime 最新——失败重跑会静默复用上一轮陈旧结果
- 位置：`worldfoundry/evaluation/tasks/execution/framework/official_runner.py:725-732`（returncode 非零仍继续）与 `official_runner.py:605-618`（`discover_by_globs` 取 `st_mtime` 最大者）
- 证据：

```729:732:worldfoundry/evaluation/tasks/execution/framework/official_runner.py
            if discovered is not None:
                results_path = discovered
            elif returncode != 0:
                blocked.append(f"official command failed with exit code {returncode}")
```

- 问题：第 N 次运行崩溃（returncode≠0）但目录里残留第 N-1 次的结果文件时，`discover_by_globs` 会把旧文件当"发现的官方结果"正常解析并产出 scorecard，`blocked_reasons` 为空、`official_benchmark_verified` 可能为 true——陈旧数据被盖上"已验证"戳。部分结果文件（官方进程写到一半被杀）同理。
- 影响：评测结果可信性问题：失败被伪装成成功，且分数来自旧一轮运行。
- 建议：returncode≠0 时即使 discovered 非空也应标记 `blocked_reasons`/`run.status="failed"`；discovery 限定为本次 run 开始时间之后 mtime 的文件，或在运行前清理 glob 目标。

#### [ET-16] P2 evalcrafter 自维护 metric_scripts 存在 bare except 静默吞错
- 位置：`worldfoundry/evaluation/tasks/execution/runners/evalcrafter/metric_scripts/dover_evaluate_a_set_of_videos.py:98`（`except:` 裸捕获）
- 说明：`metric_scripts/` 虽源自 EvalCrafter 上游，但 import 已改写为 `worldfoundry.base_models.*`（文件头即 `from worldfoundry.base_models.perception_core...`），属于在库内维护、随库演进的代码，不是隔离的 vendored runtime/。
- 问题：视频解码/模型推理失败被裸捕获后静默跳过或给默认分，单视频损坏不会体现在错误通道里，只会拉低平均分。且该 bare except 已经在掩盖一个真实 bug：`metric_scripts/dover_evaluate_a_set_of_videos.py:92-99` 的断点续跑分支写的是 `open(..., "rb")` + `pkl.dump(all_results, rf)`（应为 `pkl.load`）——必然抛异常被裸捕获，"Starting from ..." 永远不会发生，续跑功能静默失效。
- 建议：至少收窄为 `except Exception` + 记录 sample id 与异常；失败样本数进 scorecard 的 diagnostics；修复 `pkl.dump`→`pkl.load`。

### 主题 6：schema 校验 / catalog 层健壮性

#### [ET-17] P2 catalog `status` 是自由文本，normalizer 靠硬编码别名表追赶，近半数值静默降级为 "unknown"，且近似字符串归一化结果相反
- 位置：`worldfoundry/evaluation/tasks/catalog/schema.py:27-47`（`_OPEN_SOURCE_ALIASES` 枚举 11 个 `confirmed_official_code*` 变体）、`schema.py:237-265`（`_normalize_source_status` 兜底 `return "unknown"`）、`schema.py:268-286`（`_normalize_integration_status` 兜底 `return "planned"`）
- 证据（/tmp 脚本对全部 72 个 catalog 条目实测）：catalog 中实际存在 15 种 status 字符串，其中 8 种不在别名表内被静默归一为 `unknown`（覆盖约 10 个条目），包括 `in_tree_runtime_ready`（3 个条目）、`official_runtime_start_ready`、`bounded_official_generative_numeracy_verified` 等。对比性最强的一组：`confirmed_official_code_and_data_in_github` → `open_source`（在别名表），而 `confirmed_official_code_in_github` → `unknown`（不在表）——一字之差结果相反，没有任何告警。
- 问题：状态字段没有封闭词表约束，作者持续发明新状态字符串，normalizer 的别名表永远追不上；降级路径静默，拼写错误与新词汇都无法被发现。
- 影响：以 `source_status`/`integration_status` 做筛选的上层（readiness 汇总、docs 生成、TUI 过滤）会把"已就绪"的 benchmark 当成 unknown/planned 处理。
- 建议：catalog 的 `status` 收敛为封闭枚举（schema 校验直接报错未知值）；自由文本另用 `status_note` 字段承载；normalizer 移除模糊兜底或至少 `logger.warning`。

#### [ET-18] P2 `benchmark_id` vs `id` 两种键的兼容不对称：路径索引两者都认，id 全集只认 `id`
- 位置：`worldfoundry/evaluation/tasks/catalog/benchmark_catalog.py:100-103`（`entry.get("benchmark_id") or entry.get("id")`）vs `benchmark_catalog.py:138-149`（`benchmark_catalog_ids` 只取 `entry.get("id")`）；schema.py:1225 也是双键兼容（`entry.get("benchmark_id", entry.get("id", ""))`）
- 问题：一个只声明 `benchmark_id:` 的 shard 能被 `resolve_benchmark_manifest_path` 解析、能被 `BenchmarkZooEntry.from_dict` 加载，但不会出现在 `benchmark_catalog_ids()` 返回的全集里——一致性检查、docs 生成等依赖全集的调用方会漏掉它。当前 72 个 shard 恰好全部用 `id:`（实测），问题处于潜伏状态。
- 建议：`benchmark_catalog_ids` 与路径索引共用同一提取函数；或 schema 校验强制统一键名。

#### [ET-19] P2 catalog 路径索引静默跳过坏 shard（`except Exception: continue`）
- 位置：`worldfoundry/evaluation/tasks/catalog/benchmark_catalog.py:94-99`
- 证据：

```94:99:worldfoundry/evaluation/tasks/catalog/benchmark_catalog.py
    for candidate in iter_benchmark_catalog_manifest_paths(directory):
        try:
            payload = load_manifest(candidate)
            entries = iter_benchmark_zoo_payloads(payload)
        except Exception:  # noqa: BLE001 - skip malformed shards so callers can fall back gracefully.
            continue
```

- 问题：某个 shard YAML 语法错误时，它的 benchmark 会从索引里无声消失，调用方拿到的错误是下游的 "benchmark not found"，与真实原因（YAML 坏了）相距很远。`lru_cache` 还会把这个"少了一块"的索引缓存下来。
- 备注（同类缓存约定问题）：`zoo_registry.py:445-458` 的 `_load_benchmark_zoo_registry_cached` 用 `lru_cache` 缓存整个 `BenchmarkZooRegistry` 实例并共享给所有调用方——长驻进程（Studio/TUI）内 catalog YAML 落盘更新后不会被感知，且共享实例意味着任何调用方的原地修改会泄漏给他人。
- 建议：收集失败 shard 列表并在索引结果里携带 / 记日志；至少把异常类型收窄为 `(OSError, yaml.YAMLError, ValueError, TypeError)`；缓存键中纳入目录 mtime 或提供显式失效接口。

#### [ET-20] P3 `BenchmarkZooEntry` 的状态字段族过多，语义靠层层回退推断
- 位置：`schema.py:1220-1245`（`from_dict` 同时维护 `status`→source.status、`source_status`、`open_source_status`、`release_status`、`maturity`、`integration_status`、`verification_status` 及三个布尔证据位）
- 问题：`release_status` 默认取 `open_source_status`，`open_source_status` 又默认取 `release_status`（1220-1240 行互为回退）；`maturity` 回退取 `integration` 键。7 个状态字段 + 复杂回退链使"这个 benchmark 到底能不能跑"的判定分散且难审计（ET-17 的静默降级叠加其上）。
- 建议：收敛为 3 个正交维度（源码可得性 / 集成程度 / 验证程度），在 schema 文档写明各字段唯一来源键。

### 主题 7：data 清单一致性（/tmp 脚本交叉验证结果）

#### [ET-21] P2 catalog / runtime profile / docs 三者存在缺漂：2 个 benchmark 无 runtime profile，3 个未进 docs 状态表
- 验证方法：/tmp/wf_consistency_check.py 交叉比对 `data/benchmarks/catalog/{video,embodied}/*.yaml`（72 id）、`data/benchmarks/tasks/*.yaml + tasks/external/*.yaml`（72 id）、`data/benchmarks/runtime_profiles/official/*.yaml`（70 个）、三张 runner 注册表、`docs/fumadocs/lib/benchmark-catalog-status.json`（69 id）。
- 结果：
  - catalog ↔ task YAML：**1:1 完全对应**（正面）。
  - catalog 有而 runtime profile 缺：`ai2thor`、`larybench`。larybench 有 runner 目录、进了 VIDEO_RUNNER_REGISTRY 与 CLI_RUNNERS，但没有 runtime profile——按 skill 的接入检查单（"Create the runtime profile"是必做项）属于漏项。
  - catalog 有而 docs 状态 JSON 缺：`apple-pi`、`larybench`、`physical-ai-bench`（docs 生成物落后于 catalog）。
  - 三张注册表间的缺漂见 ET-01（integration 缺 5 项）。
- 影响：larybench 这类"接了一半"的 benchmark 恰好证明 ET-01 所述的多触点手工同步不可持续；runtime preflight 对无 profile 的 benchmark 无从校验环境。
- 建议：把"catalog id ↔ task ↔ profile ↔ 注册表 ↔ docs"一致性做成 CI 测试（脚本已可直接改造）；补齐 larybench/ai2thor 的 profile 与 docs 再生成。

#### [ET-22] P3 scorecard 证据字段声明与 runner 能力一致（正面确认），但 catalog 顶层三证据位全库为 false，信息量有限
- 验证：72 个条目中 `official_benchmark_verified: true` 0 个、`leaderboard_valid: true` 0 个；与 runner 实际能力（scorecard 里运行时计算的 `official_benchmark_verified`）不冲突——catalog 没有夸大声明（诚实，符合 skill "Set evidence flags to what you actually verified"）。细粒度的验证证据放在 `official_gpu_validation.*`（如 vbench 记录了 bounded 验证的 scorecard 路径 `tmp/local_open_eval/...`，该路径是运行产物、不随库分发，引用会失效）。
- 建议：`official_gpu_validation.scorecard` 改为指向入库的 evidence 文件或哈希；顶层证据位与 `official_gpu_validation` 的布尔命名重复（同名嵌套字段 `official_benchmark_verified` 内外两份），建议只保留一处。

### 主题 8：可复现性与导入卫生

#### [ET-23] P3 可复现性：上游版本 pin 与数据校验完备（正面为主），但 framework 层无 seed 管理
- 正面：
  - catalog 每个条目记录 `official_sources.github.head_sha`（87 处 pin，如 vbench.yaml 记录 `head_sha: 45e79ec...` 及 `verified_by: git ls-remote ...` 命令）；
  - `datasets/manifest.py:349-395` 对 samples 文件做 sha256 + sample_count + sample_ids_sha256 三重校验，manifest 不匹配时逐项报 issue；
  - 生成缓存把 `temperature>0 / do_sample / n>1` 判为不可缓存（cache.py:278-313），version_context 进 cache key，防陈旧命中设计到位。
- 不足：framework/orchestration 层 `rg seed` 无命中——生成侧 seed 完全依赖各 model runner 自觉，`GenerationRequest.generation_kwargs` 没有约定 seed 键；judge 型 benchmark（LLM API）的温度/seed 也无统一约定，同一输入两次打分可能不同且不会被缓存层拦住（cache 只看请求参数，不看 judge 配置）。
- 建议：在 `GenerationRequest`/runtime profile 里约定可选 `seed` 字段并写进 scorecard 的 reproduction 块。

#### [ET-24] P3 导入卫生整体达标（正面），唯一例外是 catalog 查询路径的 `yaml` 依赖
- 正面：`run_*_official_runner.py` 顶层无 `import torch`（rg 0 命中）；重型依赖（transformers、decord）都在函数内延迟导入；conda 调度集中在 `worldfoundry/studio/conda_dispatch.py` 一处，runner 层无重复实现（rg 仅 `_scorers/vqa_score/_backend.py` 一处涉 conda 字样）。catalog 层实测冷 import ~0.9s（含 yaml 解析），CLI 可接受。
- 备注：metrics registry 的惰性发现（registry.py:761-768 `_register_discoverable_entries` 仅在未命中时触发 import）是好设计。

## 汇总：严重度统计表 + Top 5

### 严重度统计

| 严重度 | 数量 | 条目 |
| --- | --- | --- |
| P0 | 0 | — |
| P1 | 5 | ET-01, ET-04, ET-10, ET-11, ET-14 |
| P2 | 12 | ET-02, ET-05, ET-06, ET-07, ET-08, ET-12, ET-15, ET-16, ET-17, ET-18, ET-19, ET-21 |
| P3 | 7 | ET-03, ET-09, ET-13, ET-20, ET-22, ET-23, ET-24（后三条以正面确认为主） |
| 合计 | 24 | |

### Top 5

1. **[ET-04/ET-14] 失败不写 scorecard 的契约破坏是系统性的**：共享 CLI 框架 `official_runner.run_main` 的异常白名单漏掉 `TimeoutExpired`/`KeyError` 等（official_runner.py:863-871），独立 runner（worldscore 等）又各自复制了一份不同的白名单（worldscore:1189-1193）——两个家族在超时/意外异常时都只打 stderr 退出，不产出 `run.status="failed"` 的 scorecard，下游只能拿到 RuntimeError；超时路径还不清理进程组、丢失子进程日志。仓库明明已有 `run_logged_subprocess` 解决全部这三件事。
2. **[ET-11] 指标静默 no-op**：`fid`/`cmmd`/`vqa_score` 等 ~60 个注册表可校验通过的指标 id，传入 `BuiltinExistingResultsMetric` 后在 `__call__` 里没有任何实现分支，静默产出空结果——用户以为算了 FID，scorecard 里只有 generation_success。
3. **[ET-10] `normalize_unit_score` 按数值大小猜量纲**：(1,100] 一律 /100、界外一律截断，1–5 分制 judge 分数与无界距离指标会被静默歪曲，normalized 值系统性偏离官方口径。
4. **[ET-01] 三张平行 runner 注册表已实际缺漂**：`BENCHMARK_INTEGRATION_REGISTRY` 比另两张表少 5 个已接入 benchmark（4dworldbench/larybench/sana-wm-bench/worldolympiad/worldreasonbench），公共 API `benchmark_integration_spec()` 对它们返回 None；vbench 家族又在 dispatcher 里第四种形态特判。
5. **[ET-15] 失败重跑会拿旧结果冒充新分数**：官方命令 returncode≠0 时只要 glob 发现了结果文件（取 mtime 最新）就照常解析打分，`blocked_reasons` 为空——上一轮的陈旧结果会被盖上本轮"已验证"的戳。

### 修复优先级建议

- **立即**（改动小、收益大）：ET-01 补 5 个注册表缺项；ET-04/ET-14 统一 `except Exception → 写 failed scorecard → return 1`；ET-15 在 returncode≠0 时强制 failed。
- **短期**：ET-10/ET-11 指标语义修复（量纲声明化、拒绝不可计算的 metric id）；ET-05/ET-06 超时兜底；ET-16 修 `pkl.dump`→`pkl.load`。
- **中期**：三表合一（ET-01/02/03/07 的根因）；catalog status 封闭词表（ET-17/20）；清单一致性 CI（ET-21，脚本可直接改造）。
