# 测试与 CI 评审

> 评审对象：WorldFoundry 仓库测试体系（`test/`、`tests/`）、CI（`.github/workflows/`）、pre-commit、`test/run_tests_docker.sh`、`CONTRIBUTING.md`、`docs/fumadocs/content/docs/reference/validation.mdx`。
> 评审日期：2026-08-14。状态：**已完成**。发现 12 项：P0×3、P1×3、P2×4、P3×2。

## 评审范围与方法

- **静态审查**：通读 `test/conftest.py`、`test/eval_core/conftest.py`、`test/eval_core/` 抽样精读、10 个顶层 `test_<model>.py` 代表精读；`Makefile`、`pyproject.toml`、`.github/workflows/ci.yml`、`.pre-commit-config.yaml`、`test/run_tests_docker.sh`、`CONTRIBUTING.md`、`docs/.../validation.mdx` 全文精读。
- **覆盖矩阵**：对指定的 13 个自研 infra 模块逐一用 `rg`/文件枚举匹配对应测试，评估覆盖深度。
- **动态验证**：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q -p no:cacheprovider` 验证收集；挑 1–2 个纯 CPU 测试实跑（时间盒 5 分钟）。
- **约束**：评审过程不修改仓库任何文件（本报告除外），pytest 使用 `-p no:cacheprovider` 避免产物。

### 基础数据（文件统计）

| 位置 | 文件数 | 说明 |
| --- | --- | --- |
| `test/*.py`（顶层） | 81 | 以 `test_<model>.py` 为主 + `conftest.py` + `__init__.py` |
| `test/eval_core/*.py` | 152 | 评测框架核心测试（含 `conftest.py`、`contract_fixture.py`） |
| `tests/**/*.py` | 322 | 第二套测试树：`base_models`(28)、`cli`(1)、`core`(15)、`evaluation`(31)、`pipelines`(3)、`runtime`(5)、`scripts`(1)、`studio`(11)、`studio_visualization`(11)、`synthesis`(35)、`training`(181) |

## 测试布局

### 双测试树并存，职责边界未文档化

仓库同时存在 `test/`（单数）与 `tests/`（复数）两套测试树：

- **`test/`**（约 233 文件）：`test/conftest.py` 只做 sys.path 注入；`test/eval_core/`（152 文件）是评测框架核心测试，质量整体较高；**顶层 79 个 `test_<model>.py` 中 44 个没有任何 `def test_` 函数**，是模块级直接执行的 demo/推理脚本（import 即触发模型下载 + CUDA 推理），剩余 35 个是真实的 pytest 测试（多为 mock + CPU）。
- **`tests/`**（322 文件）：按源码结构组织（`core/`、`cli/`、`evaluation/`、`runtime/`、`studio/`、`training/`(181 个) 等），命名与源码模块对应良好，**但无根 `conftest.py`、无任何 Makefile/CI/docs/CONTRIBUTING 引用**——`rg 'python -m pytest tests|pytest tests/'` 在自研层内零命中（仅 thirdparty 的 mmyolo 自带 CI 提到）。这棵树是文档化工作流之外的"孤儿树"。

### 命名与发现机制

- 文档（`docs/.../validation.mdx:30`）唯一认可的测试入口是 `python -m pytest test/eval_core -q`；顶层 `test/*.py` 与整个 `tests/` 都不在任何文档化命令中。
- 无仓库级 `conftest.py`、无 `pytest.ini`/`setup.cfg`/`tox.ini`、`pyproject.toml` 无 `[tool.pytest.ini_options]`：rootdir/testpaths/markers 全部未配置。`pytest`（不带参数）会同时收集 `test/`、`tests/`、`test_stream/` 及 thirdparty 内的 test 文件，行为不可控。
- `tests/` 子目录大多无 `__init__.py`（仅 `tests/training/post_training/rl/algorithms/*/` 有），依赖 pytest rootdir 插入路径 + `pip install -e .`；两树中同名 basename 文件（如两处 conftest 语义不同）在无 ini 的 rootdir 推断下有冲突风险。
- 正面评价：`test/eval_core/` 与 `tests/` 内部文件命名规范（`test_<主题>.py`，主题与被测模块对应），`tests/` 的目录映射（`tests/core/test_logging_setup.py` ↔ `worldfoundry/core/logging_setup.py` 等）清晰。

## 覆盖矩阵（模块 → 测试 → 评价）

检索方法：`rg -l "<module path>" test/ tests/`，并抽样打开命中文件确认是"直接测试"还是"字符串顺带提及"。规模数据来自 `wc -l`。

| 自研模块（规模） | test/ 命中 | tests/ 命中 | 覆盖评价 |
| --- | --- | --- | --- |
| `core/configuration`（1,148 行） | 0 | 0 | **完全无测试**。lazy_config/hydra/flags/model_config 均无 |
| `core/registry.py`（170 行） | 0 直接 | 0 直接 | **完全无测试**。字面命中均为 `studio.visualization.core.registry`（另一个模块）；核心注册表本体零测试 |
| `core/distributed`（8,492 行，26 文件） | 2（仅字符串提及） | 1 | **接近零**。仅 `tests/core/test_logging_setup.py` 测了 `distributed/logging`；FSDP2、context/sequence parallel、collectives、rank orchestration 等全部无测试（虽属 GPU 域，但纯逻辑部分如 mesh state、plan 计算可 CPU 测） |
| `evaluation/api`（9 文件） | 28 文件 | 6 | **良好**。registry/metrics/models/tasks/json_contract 有直接契约测试（`test_api_contracts.py`、`test_metric_registry.py` 等） |
| `evaluation/models`（catalog/runners/runtime/pipelines） | 45 文件 | 2 | **良好**。catalog schema、resolver、runner registry、runtime layering 均有专测 |
| `evaluation/tasks`（catalog/execution/metrics/datasets） | 59 文件 | 29 | **最佳**。官方 runner 逐个有 `test_<bench>_official_runner.py`；metric 公式、task YAML、normalizer 有专测 |
| `cli`（27 文件） | 24 文件 | 15 | **良好**。`test_cli_ux.py`、`test_eval_cli_contract.py`、各子命令（dataset/metric/task/plan/tui）有专测 |
| `mcp`（server/client/tools） | 3 文件 | 1 | **薄弱**。`test_mcp_web_interfaces.py`(434 行) 覆盖工具面；`tests/evaluation/test_mcp_server.py` 仅 60 行；client.py 无测试 |
| `runtime`（16 文件） | 29 文件 | 10 | **良好**。cuda_tiers、env、device_pool、jobs、local_checkpoint_cache、in_tree_cli 有直接测试 |
| `operators/base_operator.py`（61 行；operators 共 88 文件） | 2 文件 | 0 | **薄弱但基类小**。`test_operator_memory_hygiene.py` 做全 operator 内存卫生扫描（好实践）；88 个具体 operator 绝大多数无单测 |
| `training/engine`（18,182 行） | 0 | 36 文件 | **测试全在孤儿树**。`tests/training/` 有 181 文件、质量可（见抽查），但不被任何 CI/文档入口执行 |
| `training/checkpoint`（1,140 行） | 0 | 48 文件命中 | 同上，**只在孤儿树** |
| `studio`（app/serving/execution 等） | 25 文件 | 29 文件 | **中等偏好**。stream init/step、resident workers、viewport routing、workspace app 等有 mock 化 CPU 测试；分散在两棵树 |

**结论：完全无测试的关键 infra 模块**：`core/configuration`、`core/registry.py`、`core/distributed`（除 logging 外全部，含 FSDP/并行/collectives 共 8,400+ 行）、`mcp/client.py`。**测试存在但从不被执行**：`training/engine`、`training/checkpoint`、`training/post_training`（tests/ 树 181 个文件无 CI/文档入口）。

## 可运行性验证（实测）

环境：Python 3.12.3、pytest 8.1.1、torch 2.7.0（CUDA 可用）。所有命令带 `-p no:cacheprovider`。

1. `PYTHONPATH=. python -m pytest test/eval_core --collect-only -q`：**收集 1,229 个测试，18 个收集错误**（`Interrupted: 18 errors during collection`）。逐一核对错误原因，确认至少 6 处是**测试引用了源码中已不存在的模块/符号**（非环境缺依赖）：
   - `scripts.docs.generate_benchmark_subpages`（`scripts/docs/` 下无此文件）
   - `worldfoundry.evaluation.tasks.official`（`tasks/` 下无 `official`）
   - `worldfoundry.runtime.resolve_cache_dir`（`runtime/__init__.py` 无此导出）
   - `worldfoundry.runtime.AsyncCommandJobStore`（实际在 `runtime/jobs.py`，包级 `__init__` 未重导出）
   - `worldfoundry.pipelines.component_pipelines.StepVideoT2VPipeline`（源文件无此符号）
   - `animatediff_synthesis.DEFAULT_ANIMATEDIFF_REPO_ROOT`、`worldfoundry.base_models.diffusion_model.video`（均不存在）
2. 实跑纯 CPU 测试：`pytest test/eval_core/test_cuda_tiers.py test/eval_core/test_evaluate_runner.py -q` → **16 passed in 6.58s**，1 个 `PytestUnknownMarkWarning`（`fast_eval_core` 未注册）。核心 runner/CUDA tier 逻辑测试在无 GPU 依赖下可正常运行。
3. 附加检查 `pytest tests --collect-only -q`（孤儿树）：**收集 2,563 个测试，53 个收集错误**，错误归类为缺可选依赖且未用 `importorskip` 防护（transformers×17、ftfy×17、diffusers×4、fastapi×3 等）。
4. `make -n test-eval-core` → `No rule to make target 'test-eval-core'`，证实 `run_tests_docker.sh` 默认路径已损坏。

## 测试质量发现

### [TC-01] P0 CI 完全不运行任何 pytest 测试

- 位置：`.github/workflows/ci.yml:40-55`
- 证据：唯一的 CI job `public-surface` 步骤为 Build docs → `make lint` → `make compile-eval` → `make cli-check` → `make docs-check`。全文无 pytest；`Makefile` 也没有任何 test 目标（`.PHONY` 列表止于 `preflight`）。
- 问题：约 2,800 个自研 py 文件、两棵共 ~550 文件的测试树，PR 合并前零测试执行。`compileall` 只验证语法，`cli-check` 只覆盖一条 existing-results 命令。
- 影响：任何逻辑回归（评测计分、runner 契约、CLI 行为）都无法在 CI 拦截；TC-02 的测试腐化正是这一缺口的直接后果。
- 建议：在 ci.yml 增加一个 job：`pip install -e . pytest && python -m pytest test/eval_core -q`（当前收集错误修复后约 1,200 个 CPU 测试，本地实测单文件级秒级完成，全量预计几分钟）；后续再把 `tests/` 中纯 CPU 子集纳入。

### [TC-02] P0 文档化发布门禁 `pytest test/eval_core` 在 HEAD 上收集即失败（测试腐化）

- 位置：`test/eval_core/`（18 个文件）；门禁定义于 `docs/fumadocs/content/docs/reference/validation.mdx:30`
- 证据：见"可运行性验证"第 1 条。18 个收集错误中至少 6 处为源码符号已删除/迁移而测试未同步（`tasks.official`、`resolve_cache_dir`、`StepVideoT2VPipeline` 等，已逐一在源码中反查确认缺失）。
- 问题：validation.mdx 将 `python -m pytest test/eval_core -q` 列为 core 变更的 release gate，但该命令当前无法完成收集（pytest 收集错误默认中断运行）。说明这条门禁长期没人执行，测试随重构腐化。
- 影响：发布门禁形同虚设；1,229 个可收集测试也因收集中断而无法一键运行，开发者只能挑文件跑。
- 建议：修复/删除这 18 个漂移测试（多为改 import 路径即可，如 `AsyncCommandJobStore` 改从 `worldfoundry.runtime.jobs` 导入）；把该命令纳入 CI（TC-01）防止再次腐化。

### [TC-03] P0 顶层 44 个 `test_<model>.py` 是 import 即执行的推理脚本：收集即触发模型下载 + CUDA 推理 + 写仓库目录

- 位置：`test/` 顶层 79 个 `test_*.py` 中的 44 个（按是否含 `def test_` 统计）；另有 `test_stream/` 目录同类脚本（`test_gen3c_stream.py` 等，均无测试函数）
- 证据：
  - `test/test_wan_2p2.py:22` 模块级 `Wan2p2Pipeline.from_pretrained(model_path="Wan-AI/Wan2.2-TI2V-5B", ...)`，L37 直接推理，L53 默认写 `./wan_app_demo_output.mp4`（CWD＝仓库根）；
  - `test/test_vggt.py:37` 模块级 `VGGTPipeline.from_pretrained("facebook/VGGT-1B")`，输出默认 `./vggt_output`；
  - `test/test_hunyuan_worldplay.py:15-21` 模块级 `from_pretrained(..., device="cuda")`，写 `./outputs`；
  - `test/test_yume.py:10` 模块级加载 `stdstu123/Yume-I2V-540P`；
  - `test/test_sora2.py:25-35` 模块级调用 OpenAI 视频 API，**硬编码 `api_key="your api key"`**，写 `./output/sora2`；
  - 以上文件均无任何 `pytest.mark.skipif` / `importorskip` / GPU 检测（`test/` 全树仅 8 个文件出现 `cuda.is_available`，多用于选 device 而非 skip）。
- 问题：pytest 收集阶段就会 import 模块并执行这些副作用。任何人运行 `pytest test/`（该目录有 `conftest.py`、文件名全部符合默认收集规则，这是最自然的命令）都会触发数十 GB checkpoint 下载、CUDA 推理、真实 API 调用，并向仓库工作区写视频文件。无 pytest 配置（TC-06）又放大了误收集面。
- 影响：无 GPU 环境直接崩溃；有 GPU 环境会产生代价高昂的意外副作用；产物污染 git 工作区（`output/`、`outputs/` 恰好已存在于仓库根）。
- 建议：三选一并保持一致：(a) 迁到 `examples/<model>.py` 并去掉 `test_` 前缀（推荐，这些本质是 demo）；(b) 包进 `def test_()` + `pytest.mark.gpu` + `skipif(not cuda)` + 环境变量门控下载；(c) 至少加 `collect_ignore` / ini 级排除，禁止 pytest 触碰。`test_sora2.py` 的占位 key 应改为环境变量读取（同 `test_kling_api.py:9` 的做法——该文件用 `main()` 守卫 + env key，是 44 个中少数安全写法）。

### [TC-04] P1 `run_tests_docker.sh` 默认路径调用不存在的 `make test-eval-core`，开箱即坏

- 位置：`test/run_tests_docker.sh:11,118-120`（默认 `exec make test-eval-core`）、`:18`（usage 示例 `make:test-ux`）
- 证据：`make -n test-eval-core` → `make: *** No rule to make target 'test-eval-core'.  Stop.`；`Makefile` 的 `.PHONY` 与正文均无 `test-eval-core`/`test-ux`（`rg 'test-eval-core|test-ux' Makefile` 零命中）。
- 问题：官方推荐的 Docker 测试入口（帮助文本第一行示例 `test/run_tests_docker.sh` 不带参数）在容器内装完依赖后立即失败；显式传测试路径的分支可用，但默认与文档示例全坏。
- 影响：新贡献者按 usage 操作必然失败；也说明该脚本同样缺乏 CI/日常执行。
- 建议：在 Makefile 补 `test-eval-core: ; PYTHONPATH=. $(PYTHON) -m pytest test/eval_core -q` 与 `test-ux`（或改脚本默认直接执行 pytest 命令）。

### [TC-05] P1 322 文件的 `tests/` 树是"孤儿树"：无 conftest、无任何 CI/文档/Make 入口，training 层测试事实上从不运行

- 位置：`tests/`（11 个子目录、322 文件，其中 `tests/training/` 181 文件）
- 证据：`rg 'python -m pytest tests|pytest tests/'` 在 Makefile、CI、docs、CONTRIBUTING、scripts 中零命中（仅 thirdparty 的 mmyolo 自带配置提及自己的 tests）；树内无根 `conftest.py`；collect-only 实测 53 个收集错误（缺 transformers/ftfy/diffusers 等可选依赖且未 `importorskip`）。
- 问题：`training/engine`（18,182 行）、`training/checkpoint`、post_training RL 算法等关键模块的**全部**测试都在这棵树里（覆盖矩阵），但没有任何文档化命令、CI job 或 Make 目标会执行它们；且在精简环境下收集即报错，说明作者默认"全依赖环境"而无降级路径。
- 影响：2,563 个测试写了等于没写——回归不会被发现；两棵树并存还造成入口混乱（`test/` vs `tests/` 职责无任何文档说明）。
- 建议：给 `tests/` 建根 conftest + 对可选依赖统一 `importorskip`；将纯 CPU 子集并入 CI；长期应合并两棵树（建议以源码镜像结构的 `tests/` 布局为目标，把 `test/eval_core` 迁入）。

### [TC-06] P1 全仓库无 pytest 配置：无 testpaths/markers 注册/收集边界，裸 `pytest` 行为危险且警告刷屏

- 位置：仓库根（无 `pytest.ini`/`setup.cfg`/`tox.ini`/根 `conftest.py`；`pyproject.toml` 无 `[tool.pytest.ini_options]`）
- 证据：`ls pytest.ini setup.cfg tox.ini conftest.py` 全部不存在；实测出现 `PytestUnknownMarkWarning: Unknown pytest.mark.fast_eval_core`；`pytest.mark.unit`(8 处)、`pytest.mark.gpu`(1 处) 同样未注册；根目录下 `test/`、`tests/`、`test_stream/` 三个目录均符合默认收集模式。
- 问题：(a) 在根目录裸跑 `pytest` 会同时收集三个测试目录 + thirdparty 内自带测试，撞上 TC-03 的脚本雷区；(b) marker 未注册无法用 `-m "not gpu"` 做可靠分层，拼写错误也不会被 `--strict-markers` 拦截；(c) rootdir 推断不稳定，两棵树的同名模块有 import 冲突隐患。
- 影响：测试运行方式完全靠口口相传；无 GPU/有 GPU 环境无法用统一命令跑各自子集。
- 建议：在 `pyproject.toml` 加 `[tool.pytest.ini_options]`：`testpaths = ["test/eval_core", "tests"]`、`markers = ["gpu: ...", "unit: ...", "fast_eval_core: ..."]`、`addopts = "--strict-markers"`、`norecursedirs = ["thirdparty", "test_stream"]`（并显式排除 `test/` 顶层直至 TC-03 处理完）。

### [TC-07] P2 GPU/网络/可选依赖的 skip 约定不统一，三种风格并存

- 位置：全部测试树
- 证据：
  - 良好：`test/eval_core/test_core_primitives.py:287` 等 26 处 `pytest.importorskip("torch")`；`test_cosmos3_in_tree_integration.py:530` `pytest.skip("requires two CUDA devices ...")`；`test_official_model_category_runtime.py:435` 对未 staged 的官方仓库 skip。
  - 缺失：`tests/` 中 17+17+4 个文件直接 `import transformers/ftfy/diffusers` 导致收集错误（见可运行性第 3 条）；`test/` 顶层 44 个脚本完全无防护（TC-03）。
  - 分层：全仓库仅 1 处 `pytest.mark.gpu`、8 处 `pytest.mark.unit`，无任何文档说明按标记选跑。
- 问题：同一仓库内"无 GPU 能否跑"取决于碰到哪棵树哪个文件，没有统一契约。
- 影响：无法给贡献者一条"无 GPU 也稳过"的命令；CI 引入 pytest 时（TC-01）也需要先解决这一层。
- 建议：约定并注册 `gpu`/`network`/`slow` 标记；对可选依赖统一 `importorskip`；在 CONTRIBUTING 写明分层运行命令。

### [TC-08] P2 断言质量两极分化：eval_core 优秀，顶层脚本零断言

- 位置：`test/eval_core/`（优）vs `test/` 顶层 44 脚本（零断言）
- 证据：
  - 优：`test_evaluate_runner.py:61-100` 断言 schema 版本、11 个 artifact 文件逐一存在、`summary["leaderboard"]["quality"] == pytest.approx(0.6)` 等具体数值；`test_import_is_lightweight.py` 用子进程验证公共 import 不拉起 torch/cv2 等重依赖（很好的架构守护测试）；`test_hfd_download_script.py:36` 起一个本地 `ThreadingHTTPServer` 模拟 HF 元数据 500，验证缓存保留——网络故障注入不出真网。
  - 劣：44 个脚本无 `def test_`、无 assert（`test_video_models.py` 有模块级 assert，`lazy=True` 不下载权重，属可接受的轻量冒烟但仍在 import 期执行）；跑完只 `print` 输出路径，"跑通即成功"。
- 问题：脚本类文件既不产生断言信号，又占用 `test_` 命名空间，稀释了测试树的可信度。
- 影响：外部观察者（或覆盖率工具）会高估模型层的被测程度。
- 建议：同 TC-03——脚本迁出；确需保留的最小推理冒烟应断言输出 shape/时长/文件存在。

### [TC-09] P2 fixture/mock/隔离总体健康，但顶层脚本写仓库目录破坏隔离

- 位置：全树
- 证据：
  - 好：132/233 个 `test/` 文件用 `tmp_path`/`TemporaryDirectory`，`tests/` 138 个；mock 广泛（`test/` 47 文件、`tests/` 77 文件用 monkeypatch/MagicMock）；`test/eval_core/contract_fixture.py` 提供内存型 `ContractFixtureRunner`（`memory://` URI，不落盘），被 registry/runner/CLI 契约测试复用——fixture 设计的正面样板；`test/test_worldfoundry_studio_stream_init.py:35` 等顶层真测试也规范使用 TemporaryDirectory + Dummy pipeline。
  - 坏：TC-03 的脚本写 `./outputs`、`./output/sora2`、`./vggt_output`、`./wan_app_demo_output.mp4`、`./depth_anything_v2_output`（全在仓库工作区，且 `output/`、`outputs/` 目录确已存在于仓库根）；`test/test_sora2.py:2` 还有无意义的 `sys.path.append("..")`。
- 问题/影响/建议：见 TC-03；另建议在 `.gitignore`/lint 中把这些输出目录标记为禁止提交（`make lint` 的 open_source_path_hygiene 已有类似检查，可扩展）。

### [TC-10] P2 无覆盖率统计、无测试基础设施文档；CONTRIBUTING 全文未提 pytest

- 位置：`CONTRIBUTING.md`、`requirements/`、`pyproject.toml`
- 证据：`rg "pytest-cov|coverage" requirements/ pyproject.toml .github/` 零命中；`pyproject.toml` 的 `dev` extra 只有 `build/pre-commit/ruff`（**连 pytest 都不在任何依赖组**，只在 docker 脚本里临时 `uv pip install pytest`）；`CONTRIBUTING.md` 的 PR checklist 只要求 `make lint`/`make docs-check`/"public validation commands"，未提任何测试命令；测试入口只散落在 `validation.mdx:30`。
- 问题：无覆盖率基线，覆盖缺口（如 `core/configuration` 零测试）不可见；贡献者按 CONTRIBUTING 操作可以合法地不跑任何测试。
- 影响：测试文化依赖个人自觉；覆盖矩阵中发现的空洞不会被度量暴露。
- 建议：`dev` extra 加 `pytest`、`pytest-cov`；CI 出覆盖率并对 `worldfoundry/evaluation`、`worldfoundry/cli` 等成熟层设阈值；CONTRIBUTING 增加"运行测试"一节。

### [TC-11] P3 `run_tests_docker.sh` 工程细节：镜像未 pin digest、依赖容器内即时安装、挂载 `~/.netrc`

- 位置：`test/run_tests_docker.sh:43,87-89,94-116`
- 证据：镜像 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` 按 tag 引用（无 digest）；容器每次启动 `apt-get install`（约 20 个包）+ 现装 uv + `uv pip install -e .[extras] build pytest PyYAML`（pytest 无版本 pin）；`~/.netrc` 只读挂载进容器（测试代码可读取用户凭证）。
- 问题：非 hermetic——同一命令不同日期解析出不同依赖集；首次运行耗时长；凭证暴露面大于必要（只为 HF 下载的话可改用 token env）。
- 亮点：uv/HF/triton/worldfoundry 四个缓存 + benchmark 数据目录的挂载设计合理，`--gpus none` 支持 CPU-only，`--shm-size=16g`、ulimit 设置符合 CUDA 测试需要。
- 建议：提供预构建测试镜像（或至少 digest pin + requirements 锁定）；`.netrc` 挂载改为可选开关。

### [TC-12] P3 死标记与 lint 覆盖窄：`fast_eval_core` 无消费者；pre-commit ruff 只管 6 个子目录

- 位置：`test/eval_core/conftest.py:6-9`、`.pre-commit-config.yaml:27-36`、`Makefile:21-30`
- 证据：conftest 给每个 item 注入 `pytest.mark.fast_eval_core`，但 `rg fast_eval_core` 在仓库其他任何地方（CI、Makefile、docs）零命中——纯产生警告的死代码；pre-commit 的 ruff `files:` 正则与 Makefile `RUFF_SOURCES` 一致，仅覆盖 `worldfoundry/{cli,mcp,runtime}`、`evaluation` 的 4 个子路径和 `scripts/{benchmark_zoo,model_zoo}`；`core/`、`training/`、`studio/`、`operators/`、`pipelines/` 等自研层不受任何格式化/静态检查约束；`pre-commit-hooks` 用 2021 年的 v4.0.1。
- 问题：标记体系半途而废；lint 白名单让大部分自研代码游离在质量门外（与 ruff `extend-exclude` 排除 base_models/synthesis 的合理豁免不同，core/training 是纯自研层）。
- 建议：删除或真正使用 `fast_eval_core`（注册 + CI `-m fast_eval_core`）；分阶段把 `core/`、`training/` 纳入 RUFF_SOURCES；升级 pre-commit-hooks。

## CI 与工具链

| 项 | 现状 | 评价 |
| --- | --- | --- |
| CI workflow | `ci.yml`：conda + node 环境，docs build、`make lint`（ruff 白名单 + compileall + shell -n + 2 条 CLI JSON 检查 + runtime registry 校验）、`make compile-eval`、`make cli-check`（1 条 existing-results 端到端最小命令）、`make docs-check` | 结构清晰、轻量（好），但**无 pytest**（TC-01）；`cli-check` 是唯一的行为级 e2e，值得肯定但太薄 |
| 文档部署 | `deploy-docs.yml`：main 分支 docs 路径触发，GitHub Pages | 正常 |
| pre-commit | trailing-whitespace/eof/merge-conflict/symlink + ruff(import-sort/format，白名单) + 本地 `make lint`（pass_filenames: false，全量跑） | 每次提交全量 `make lint` 偏重但可接受；ruff 白名单窄（TC-12） |
| Docker 测试脚本 | `test/run_tests_docker.sh` | 默认入口坏（TC-04）；缓存设计好；非 hermetic（TC-11） |
| pytest 配置 | 无任何配置文件 | TC-06 |
| 覆盖率 | 无 | TC-10 |
| GPU 分层 | 无标记约定、无 GPU CI；docker 脚本支持 `--gpus none` 但默认 `all` | TC-07 |
| e2e 冒烟 | `make cli-check` + `make preflight`（未进 CI） | preflight 建议纳入 CI 的 nightly |

## 测试基础设施缺失项清单

1. **CI 测试 job**（TC-01）——最高优先级，其余问题多因缺这道闸而滋生。
2. **pytest 配置**：testpaths、markers 注册、`--strict-markers`、norecursedirs（TC-06）。
3. **覆盖率统计**：pytest-cov + 阈值 + PR 报告（TC-10）。
4. **GPU/network/slow 分层标记约定** + 无 GPU 环境的绿色基线命令（TC-07）。
5. **pytest 进入依赖声明**（`dev` extra 目前连 pytest 都没有，TC-10）。
6. **e2e 冒烟分层**：现有 `cli-check` 之外，建议 nightly 跑 `preflight` + 1 个最小 model-mode 评测（可用 `ContractFixtureRunner` 无 GPU 完成）。
7. **测试树合并/职责文档**：`test/` vs `tests/` vs `test_stream/` 三处并存且互不引用（TC-03/05）。

## 汇总

### 严重度统计

| 严重度 | 数量 | 编号 |
| --- | --- | --- |
| P0 | 3 | TC-01（CI 无测试）、TC-02（发布门禁收集即失败）、TC-03（test_ 脚本 import 即下载/推理/写仓库） |
| P1 | 3 | TC-04（docker 脚本默认路径坏）、TC-05（tests/ 322 文件孤儿树）、TC-06（无 pytest 配置） |
| P2 | 4 | TC-07（skip 约定不统一）、TC-08（断言两极分化）、TC-09（隔离被脚本破坏）、TC-10（无覆盖率/无文档） |
| P3 | 2 | TC-11（docker 非 hermetic + netrc）、TC-12（死标记/lint 白名单窄） |

### Top 5 问题

1. **TC-01 (P0)** CI 不跑任何 pytest：约 550 个测试文件全部游离在合并门禁之外，是所有腐化问题的根因。
2. **TC-02 (P0)** 唯一文档化的测试门禁 `pytest test/eval_core -q` 在 HEAD 收集即中断（18 个错误、6+ 处已验证的源码符号漂移），发布门禁形同虚设。
3. **TC-03 (P0)** `test/` 顶层 44 个 `test_<model>.py` 是模块级推理脚本：`pytest test/` 收集阶段即触发数十 GB checkpoint 下载、CUDA 推理、真实 API 调用并写仓库目录；`test_sora2.py` 还硬编码占位 API key。
4. **TC-05 (P1)** `tests/`（322 文件，含 training/engine 18k 行代码的全部测试）没有任何 CI/文档/Make 入口，写了等于没跑；精简环境下 53 个收集错误。
5. **TC-06 (P1)** 全仓库无 pytest 配置：三个测试目录 + thirdparty 无收集边界，markers 未注册，裸 `pytest` 直接撞上 TC-03 雷区。

### 正面亮点（应保持）

- `test/eval_core/` 的契约测试设计：内存型 `ContractFixtureRunner`、本地 HTTP 故障注入（`test_hfd_download_script.py`）、import 轻量性守护（`test_import_is_lightweight.py`）、具体数值断言。
- `tests/` 树的源码镜像目录结构与 `tmp_path`/mock 纪律。
- `make cli-check` 这类零依赖行为级冒烟、`test_operator_memory_hygiene.py` 全 operator 扫描、docker 脚本的缓存挂载设计。

