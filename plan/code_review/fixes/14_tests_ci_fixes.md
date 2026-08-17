# 测试与 CI 修复日志

> 对应评审报告：`plan/code_review/14_tests_ci.md`。
> 约束：只改 `test/`、`tests/` 既有文件、`.github/workflows/`、`Makefile`、`test/run_tests_docker.sh`、根 `pytest.ini`（新建）；不动 `worldfoundry/` 源码与 `pyproject.toml`；不动并发中他人新建的 `test/test_operators_*.py`、`tests/training/test_training_engine_fix_*.py`、`tests/training/test_training_recipes_fix_*.py`。
> 环境：Python 3.12.3、pytest 8.1.1、torch 2.7.0（CUDA）；本机缺 ftfy/imageio/fastapi/transformers/diffusers/hydra/gradio/loguru/colorspacious/omegaconf/easydict 等可选依赖，pypi 不可用。
> 所有 pytest 命令带 `-p no:cacheprovider`。

## 已修复

### TC-02 (P0) `pytest test/eval_core` 18 个收集错误 → 0

修复前：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q` → **1,229 收集 + 18 错误（Interrupted）**。
修复后：**1,495 收集 + 0 错误**（数量增加来自原先因错误被排除的文件恢复收集，含 2 个模块级 skip 文件）。

逐条处置（方法：rg 反查缺失符号在当前源码的去向）：

| # | 文件 | 根因 | 处置 |
| --- | --- | --- | --- |
| 1 | `test_benchmark_zoo_in_tree_evaluators.py` | `tasks.official.in_tree` → 移动到 `tasks.execution.framework.in_tree_evaluator`；`tasks.official.physics_video` 整体删除 | 更新 import；删除 `is_physics_video_benchmark` 引用与断言子句 |
| 2 | `test_benchmark_zoo_normalizers.py` | `tasks.official.normalizers` → `tasks.execution.framework.normalizers` | 更新 import |
| 3 | `test_official_results_normalizer.py` | `tasks.official.result_normalizer` → `tasks.execution.framework.result_normalizer` | 更新 import |
| 4 | `test_benchmark_zoo_scripts.py` | `resolve_cache_dir` 在 `runtime/env.py:273`，包级 `__init__` 有意不重导出（docstring 明示 dependency-free） | 改从 `worldfoundry.runtime.env` 导入 |
| 5 | `test_mcp_web_interfaces.py` | `AsyncCommandJobStore`/`python_module_command` 在 `runtime/jobs.py` | 改从 `worldfoundry.runtime.jobs` 导入 |
| 6 | `test_runtime_env.py` | 13 个符号分散在 `runtime.env` / `runtime.assets` / `runtime.jobs` | 拆分为三个子模块 import |
| 7 | `test_official_model_category_runtime.py` | ① `StepVideoT2VPipeline` 移到 `pipelines.step_video.pipeline_step_video_t2v`（catalog binding 佐证）；② 第二层错误：`synthesis.action_generation.roboflamingo.roboflamingo_runtime.inference` → `roboflamingo.runtime` | ① 参数化测试改为惰性导入 + `importorskip`（新模块链上有 ftfy）；catalog 期望串同步为新 target；② 更新 import |
| 8 | `test_official_model_conda_envs.py` | `DEFAULT_ANIMATEDIFF_REPO_ROOT` 移到 `animatediff/worldfoundry_runtime.py`（值语义不变，仍指向 animatediff 包目录） | 更新 import |
| 9 | `test_tui_cli.py` | `build_training_command` 从 `cli/tui_discovery.py` 删除；`tui_app` 的 `_parse_training_env_overrides`/`_split_env_assignments` 同步消失（TUI training 支持整体移除，`studio/workspace_job.py` 已无 train 子命令） | 删除 import 与 2 个死测试（`test_tui_training_command_uses_workspace_training_runtime`、`test_tui_training_env_parser_keeps_comma_values_when_available`） |
| 10 | `test_packaging_dependencies.py` | `tools/packaging`（build_sanitized_sdist / check_release_worktree / check_sdist_hygiene）整个工具链已从仓库删除；`scripts/dev/check_dev_tools.py` 亦不存在 | 拆分：保留 5 个仍有效的 pyproject/requirements/MANIFEST 契约测试；删除 sdist 工具测试与 `test_development_and_ci_install_sdist_hygiene_dependencies` |
| 11 | `test_runtime_profile_pipeline_integration.py` | 环境缺 fastapi（`studio.workspace_app` 模块级 import） | 模块级 `pytest.importorskip("fastapi")` |
| 12 | `test_vchitect_runner.py` | 环境缺 ftfy（vchitect pipeline 链） | 模块级 `pytest.importorskip("ftfy")` |
| 13 | `test_wan22_example_mapping.py` | 环境缺 ftfy（wan pipeline 链） | 模块级 `pytest.importorskip("ftfy")` |
| 14 | `test_worldfm_runtime_migration.py` | 环境缺 imageio（worldfm synthesis 链） | 模块级 `pytest.importorskip("imageio")` |
| 15 | `test_benchmark_subpage_generator.py` | `scripts/docs/generate_benchmark_subpages.py` 已删（commit 1b29e446），全仓无替代 | **删除测试文件**（52 行，功能消失） |
| 16 | `test_physics_video_evaluator.py` | `tasks.official.physics_video` + `physics-video` 基准从 catalog/源码整体移除（全仓 0 命中） | **删除测试文件**（247 行，功能消失） |
| 17 | `test_cosmos3_in_tree_integration.py` | `base_models.diffusion_model.video.cosmos3.artifacts`/`.worldfoundry_runtime`、`synthesis...cosmos.cosmos3_synthesis` 均已删除；runtime 移到 `synthesis...cosmos.cosmos3_runtime`（API 面不同）；artifacts 助手（`resolve_cosmos3_model_source` 等）无在树替代 | **模块级 skip + HANDOVER 注释**（1,394 行集成套件需 owner 按新 API 重写） |
| 18 | `test_visual_generation_training_integration.py` | `worldfoundry.training.visual_generation` 包整体删除；断言的 `base_models/diffusion_model/video/` 目录布局也已不存在 | **模块级 skip + HANDOVER 注释**（252 行，待确认继任训练面后重写） |

附带（为无 torch/numpy 的最小 CI 环境铺路）：

- `test_worldarena_model_integrations.py`：模块级 `import torch` 改为 `torch = pytest.importorskip("torch")`（eval_core 内唯一硬 torch 收集依赖；仓库已有 26 处同款约定）。
- `test_cosmos3_in_tree_integration.py` 的 skip 提前到 numpy/yaml 导入之前，保证 numpy-less 环境也能收集。

**验证**：`PYTHONPATH=. python -m pytest test/eval_core --collect-only -q -p no:cacheprovider` → `1495 tests collected in ~8s`，0 错误。实跑通过率见文末。

### TC-02 附加：收集修复后暴露的运行期失败（修复的部分）

收集恢复后实跑暴露一批存量运行期腐化（与本次改动无关，单文件跑与合跑结果一致、确定性失败）。其中机械可修的已修：

- `test_mcp_web_interfaces.py::test_mcp_discovery_and_preview_are_available_without_mcp_dependency`：MCP preview 命令从 `worldfoundry-eval run` 控制台脚本改为 `sys.executable -m worldfoundry.cli run`（`mcp/tools/runs.py` 现经 `python_module_command` 包装）。更新两处前缀断言。✅ 单测通过。
- `test_benchmark_zoo_in_tree_evaluators.py::test_target_benchmark_task_yaml_runtime_roots_are_env_names`：`world-in-world.yaml` 改用 bundled assets（`asset_override_env`/`prompt_manifest_env`），不再有 `root_env`。测试对无 `root_env` 的条目放行（仅允许 world-in-world）。✅
- `test_benchmark_zoo_in_tree_evaluators.py` / `test_official_results_normalizer.py`：`t2vphysbench` 已从外部基准 catalog 移除（contracts registry known 列表无此 id、`data/benchmarks` 无文件），从两处基准 id 清单删除。✅
- `test_tui_cli.py` 3 处 training 残留断言（`summary["training_targets"]`、payload 同名键、fallback 文本 "Training Targets" 段）随 TUI training 支持移除而删除。✅

**验证**：上述文件定点重跑 → `test_benchmark_zoo_in_tree_evaluators.py + test_official_results_normalizer.py + test_mcp_web_interfaces.py`：32 passed, 9 skipped；tui 两个已修测试通过。

### TC-03 (P0) 44 个脚本型 `test_<model>.py` 阻止 pytest 收集

- **验证清单**：自行枚举 `test/` 顶层无 `def test_` 的文件，得到与报告一致的 **44 个**（含 3 个 `_registry` 后缀文件——内容同为模块级 assert+print 脚本；含 `test_kling_api.py`——虽有 main() 守卫仍无测试函数）。
- **改动**：`test/conftest.py` 增加 `collect_ignore` 精确列出 44 个文件名（非 glob，不会波及并发新建的 `test_operators_*.py`），注释说明原因与手动运行方式（`PYTHONPATH=. python test/<file>.py`）。文件本身未删未改（保留为手动 demo）。
- **test_sora2.py**：硬编码 `api_key="your api key"` 改为 `os.getenv("OPENAI_API_KEY")`（空则 `SystemExit` 提示），endpoint 同步支持 `OPENAI_API_ENDPOINT` 覆盖（参照 `test_kling_api.py` 的 env 约定）。
- **验证**：`pytest test --collect-only --ignore=test/eval_core` 收集完成（5.4s，无下载/推理/写盘副作用），44 个脚本 0 触碰。

### TC-01 (P0) CI 增加 pytest job

- **改动**：`.github/workflows/ci.yml` 新增 `eval-core-tests` job（与现有 `public-surface` 同风格：ubuntu-latest + setup-miniconda + `bash -el {0}`）。安装 `pip install -e . pytest numpy` 后跑 `make test-eval-core`。
  - eval_core 为纯 CPU 契约测试；无 GPU runner 可行性依据：torch 依赖全部走 `importorskip("torch")`（含本次补的 `test_worldarena_model_integrations.py`），numpy 为 2 个测试文件的模块级硬依赖故进安装集，ftfy/fastapi/imageio 等缺失时对应 4 个文件模块级 skip。
  - 不重构既有 job。
- **验证**：`python -c "import yaml;yaml.safe_load(open('.github/workflows/ci.yml'))"` 通过。
- **注意**：该 job 首跑会**红**——eval_core 存在存量运行期失败（见"移交"节）。这正是门禁恢复后的真实信号；红名单已在下方逐项归因。

### TC-04 (P1) `run_tests_docker.sh` 默认调用不存在的 make 目标

- **改动**（选择"补 Makefile 目标"方向，与脚本默认行为和 validation.mdx 门禁语义一致）：
  - `Makefile` 新增 `test-eval-core`（`PYTHONPATH=. $(PYTHON) -m pytest test/eval_core -q -p no:cacheprovider`）与 `test-training`（跑 `tests/training`），加入 `.PHONY` 与 `make help` 文案。
  - `run_tests_docker.sh` usage 示例中不存在的 `make:test-ux` 改为 `make:test-training`。
- **验证**：`make -n test-eval-core`、`make -n test-training` 输出正确命令；`bash -n test/run_tests_docker.sh` 通过。

### TC-05 (P1) `tests/` 孤儿树：Make 入口 + 53 个收集错误清零

- **入口**：`make test-training`（见 TC-04）。未搬移任何文件。
- **收集修复**：`pytest tests --collect-only -q` 基线 **2,600 收集 + 53 错误** → 处置：
  - 52 个文件加模块级 `pytest.importorskip("<dep>")` 守护（只加守护不改测试逻辑）。依赖分布：transformers×17、ftfy×15、diffusers×5、hydra×3、fastapi×2、loguru×2、colorspacious×2、imageio×1、omegaconf×1、easydict×1（+第二轮暴露的 `tests/base_models/test_staged_preprocessing.py` ftfy×1，共 53 个文件）。
  - `tests/evaluation/test_wrbench_generation_bridge.py`：非缺依赖——`from .wrbench_prompts import ...` 相对导入指向不存在的兄弟模块；实际符号在 `worldfoundry/evaluation/tasks/execution/runners/wrbench/`，改为绝对导入。
- **验证（前后数字）**：53 errors → **0 errors**，`2610 tests collected`（前后差含并发新建文件）。
- **同性质外溢**：`test/` 顶层还有 13 个真实测试文件因缺可选依赖收集失败，同法处理：11 个加 importorskip（imageio×5、ftfy×2、gradio×2、fastapi×1、transformers×1）；`test_lingbot_runtime_paths.py` 是模块重命名（`lingbot` → `lingbot_world`），更新 import；`test_solaris_multiplayer_eval.py` 的 `tasks.official.solaris_multiplayer` 已整体删除且无继任 → 模块级 skip + HANDOVER。验证：`pytest test --collect-only --ignore=test/eval_core` → **0 错误、522 收集**（此前 13 错误、125 收集）。

### TC-06 (P1) 根 `pytest.ini`

- **改动**：新建根 `pytest.ini`：
  - `testpaths = test tests`（裸 `pytest` 不再触碰 `test_stream/`、thirdparty 自带测试）；
  - `norecursedirs = thirdparty test_stream .* build dist node_modules __pycache__`；
  - `addopts = --strict-markers`；
  - 注册 markers：`gpu`、`unit`、`slow`、`network`、`fast_eval_core`（先扫描确认全仓自定义 marker 仅 unit/gpu/fast_eval_core 三种在用，strict 不会炸）。
- **验证**：根目录裸 `python -m pytest --collect-only -q` → **4,746 tests collected, 0 errors**（~74s），无脚本副作用；`PytestUnknownMarkWarning`（fast_eval_core/gpu）随注册消失。
- 注：评审建议写进 `pyproject.toml [tool.pytest.ini_options]`，但 pyproject 属禁改清单，故用等效的根 `pytest.ini`（任务说明允许新建）。

### 其余 P1/P2 逐条

- **TC-07（skip 约定不统一）**：部分收敛——本次给 65 个文件（tests/ 53 + test/ 顶层 11 + eval_core 4 + worldarena torch）统一了 `importorskip` 守护写法；`gpu`/`network`/`slow` marker 已注册可用。全仓既有 `skipif` 风格改写超出"最小修复"，未动。
- **TC-08（断言两极分化）/TC-09（脚本破坏隔离）**：根因是 TC-03 的脚本，已通过 collect_ignore 隔离；脚本迁出 `examples/` 属目录结构调整，记 deferred（见下）。
- **TC-12（死标记）**：`fast_eval_core` 已在 pytest.ini 注册并可用（`-m fast_eval_core`），不再产生 UnknownMark 警告；pre-commit ruff 白名单扩展涉及 `.pre-commit-config.yaml`（不在允许改动清单），deferred。

## 移交源码/owner 问题（不改源码，证据齐备）

1. **cosmos3 集成套件重写**（`test/eval_core/test_cosmos3_in_tree_integration.py`，已模块级 skip）：旧 `base_models.diffusion_model.video.cosmos3.artifacts` 助手（`resolve_cosmos3_model_source`/`cosmos3_revision_for_repo_id`/`checkpoint_revision` 等）全仓无替代；`Cosmos3Runtime` 移居 `synthesis.visual_generation.cosmos.cosmos3_runtime` 且 API 面不同。1,394 行套件需按新面重写。
2. **visual_generation 训练面**（`test/eval_core/test_visual_generation_training_integration.py`，已模块级 skip）：`worldfoundry.training.visual_generation`（build_training_plan/resolve_target/stage 别名/assets/cli）整包删除；hy-action2v/wan-action2v 仅存推理侧（minwm/studio catalog）。需确认训练面继任者后重写或删除。
3. **solaris multiplayer 评测**（`test/test_solaris_multiplayer_eval.py`，已模块级 skip）：`tasks.official.solaris_multiplayer` 删除无继任，solaris pipeline/synthesis 仍在。
4. **benchmark_zoo 脚本层测试大面积腐化**（`test/eval_core/test_benchmark_zoo_scripts.py`：单文件 83 failed / 4 passed / 54 skipped）：`framework/script_paths.py`（`resolve_benchmark_script`）已删除、`run_benchmark_execution` 等脚本从 `scripts/benchmark_zoo/` 消失、`orchestration/benchmark_runner.py` 无 `build_parser`。该 5.6k 行套件测的是已被拆除的"脚本 zoo"面，需 owner 按 orchestration/runners 新面重写；不属于单点 import 修复。
5. **TUI 推理路由行为漂移**（`test/eval_core/test_tui_cli.py` 3 个失败）：
   - `test_tui_infer_catalog_uses_official_script_models_by_default`：depth-anything-v2 现路由到 `workspace_job infer`（studio runtime）而非 `scripts/inference/run_infer.sh`；
   - `test_tui_script_infer_variants_all_build_official_commands`：`infer_model_variant_ids` 现按 runner 状态过滤（open-magvit2 仅剩 xl 变体），不再镜像 `INFER_MODEL_VARIANTS` 全表；
   - `test_tui_keeps_studio_runtime_for_non_script_infer_models`：hy-worldplay/multiworld 的 `runner_status` 现为 `'verified'`（测试期望 `'ready'`）；同时 catalog schema 打 WARNING "integration_status 'verified' ... normalized to 'planned'"——**catalog 数据与 schema 词表疑似不一致，建议 owner 先澄清词表再定测试期望**。
6. **conda env 契约漂移**（`test_official_model_conda_envs.py` 3 个失败）：env spec 覆盖集合、openpi 路由（`worldfoundry-openpi-cu12` vs `worldfoundry-unified-cu128`）、install 脚本 usage 文案（'dry'）与测试期望不符，需 owner 确认现行 conda env 布局。
7. **packaging 契约漂移**（`test_packaging_dependencies.py` 2 个失败）：`requirements/worldfoundry-unified.txt` 已从 `-e .[extras]` 形态改为 pinned 清单 + `stable-worldmodel[train]`；`MANIFEST.in` 不再含 `recursive-include worldfoundry/data/models/runtime_profiles *.yaml` 行。测试断言的是旧发布契约，需按新契约重写（MANIFEST.in/pyproject 均不在本任务可改清单）。
8. **worldarena 状态词表**（`test_worldarena_model_integrations.py` 2 个失败）：runtime profile 状态出现 `'runtime_ported'`（期望集 `{'planned','verified_official_wan_14b_workspace'}` 外）、ctrl_world test cases 不在 data package 断言失败。与第 5 条同根（状态词表演进），一并移交。
9. **catalog schema 状态告警**（运行期普遍 WARNING）：`Model catalog contains unregistered status values (first: integration_status 'verified', normalized to 'planned')`——建议在 schema.py 注册有意的新状态值。

## Deferred（需要装包/改禁改文件，另行统一安排）

1. **pyproject 依赖组**：`dev` extra 加 `pytest`（当前连 pytest 都不在任何依赖组）、`pytest-cov`；`[tool.pytest.ini_options]` 若最终希望配置进 pyproject 可把根 pytest.ini 内容平移（TC-10/TC-06，pyproject 禁改）。
2. **覆盖率工具链**：pytest-cov 安装 + CI 覆盖率报告与阈值（TC-10，需装包）。
3. **Makefile `install-dev` 补 pytest**：`$(PIP) install build pre-commit PyYAML ruff` 未含 pytest——与 pyproject dev extra 一起定夺（本次未动以免与 deferred 的 pyproject 方案冲突）。
4. **44 个 demo 脚本迁移 `examples/`**（TC-03 建议 a / TC-08 / TC-09）：涉及目录结构与文档联动，本次仅隔离收集。
5. **pre-commit ruff 白名单扩展、pre-commit-hooks 升级**（TC-12，`.pre-commit-config.yaml` 不在允许清单）。
6. **CONTRIBUTING.md 增加"运行测试"一节**（TC-10，不在允许清单）。
7. **docker 脚本 hermetic 化**（TC-11，P3）：镜像 digest pin、预装依赖镜像、`.netrc` 挂载改开关。
8. **`tests/` 与 `test/` 两树合并**（TC-05 长期项）。

## eval_core 最终实跑结果

<!-- FINAL_NUMBERS -->
