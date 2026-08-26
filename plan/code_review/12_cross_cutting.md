# 全库横切关注点评审（安全/路径/env/全局状态/并发/异常）

> 评审日期：2026-08-14 · 评审人：infra 横切面评审（自动化扫描 + 人工确认）
> 状态：已完成（主题 1–11 全覆盖，XC-1～XC-25；结尾 Top 5 已补全）

## 评审范围与方法

- **仓库**：`<WORKSPACE_ROOT>`，被评审包 `worldfoundry/`。
- **分层口径**（按评审要求）：
  - **自研层**（约 2,697 个 py 文件 / 61.5 万行）：`core`(258/68.0k)、`evaluation`(1442/307.6k)、`cli`(34/13.8k)、`mcp`(14/2.7k)、`pipelines`(164/28.7k)、`operators`(88/12.9k)、`runtime`(19/5.6k)、`training`(551/111.9k)、`studio`(121/63.0k)、`data`(6/0.9k)。
  - **vendored 层**（约 5,649 个 py 文件 / 35.5 万行）：`base_models`(3240/112.7k)、`synthesis`(2369/233.7k)、`representations`(40/8.3k)。
  - 备注：`evaluation/tasks/execution/runners/*/`（26 MB）内嵌各基准的 official/runtime 实现，属"半 vendored"（自研入口 + 移植的官方评测代码）。统计归入自研层，叙述中单独标注，修复优先级介于两层之间。
- **方法**：`rg` 全库正则扫描 → 人工 `Read` 确认上下文 → 按调用路径判定严重度。每条发现附 `路径:行号` 与摘录。排除 `__pycache__`、`tests`、文档中的示例（除非默认值本身有害）。
- **严重度**：P0=损坏/危险；P1=严重设计缺陷；P2=应修复；P3=改进建议。

## 发现

### 主题 1：安全扫描

**全库统计**（`rg` 行级匹配，`!__pycache__`）：

| 模式 | 自研层 | vendored 层 | 说明 |
|---|---|---|---|
| `shell=True` | 3 | 1 | 自研 1 处在 studio、2 处在 wrbench runner runtime |
| `os.system(` | 0 | 13 | 自研层干净 |
| `pickle.load/loads(` | 16 | 23 | 自研含 core/distributed 集合通信 6 处（进程内可信）+ runners 若干 |
| `torch.load(` 总数 / 不带 `weights_only` | 106 / 82 | 431 / 336 | 82 为行级启发式，含少量经 kwargs 传入的误报；核心加载器已默认安全（见下） |
| `eval(`（真实动态执行） | ~12 处 | 多 | core 3 个文件为配置系统特性；runners 多处对数据/LLM 输出 eval |
| `exec(` | 2（lazy_config，设计如此） | - | detectron2 风格 LazyConfig |
| `yaml.load(` 非 safe | 4 | 5 | 全部在 metrics/jedi 与 runner runtime |
| `requests.*` 无 timeout | 真实命中约 23（多数在 runner runtime；自研核心 4 处） | 未细分 | languagebind 的 docstring 示例已排除 |
| `verify=False` | 1 | 0 | localhost 探活，可接受 |
| 硬编码 token/key/secret | 0 | 0 | 多种模式（字面量赋值、sk-/hf_/ghp_/AKIA 前缀、getenv 默认值）均未命中 |
| `hashlib.md5` | 3（均非安全场景） | 17 | 自研用于 state-dict 指纹/文件校验 |

#### [XC-1] P1 评测 reward 服务通过 HTTP 接收 pickle payload —— 反序列化即 RCE 面

- **位置**：`worldfoundry/evaluation/tasks/execution/runners/worldolympiad/runtime/worldolympiad/3d_metrics/serve_reward_3d.py:89-93`
- **证据**：

```python
content_length = int(self.headers.get("Content-Length", "0"))
payload = self.rfile.read(content_length)
try:
    data = pickle.loads(payload)   # do_POST 直接反序列化网络 body
```

  服务端为 `ThreadingHTTPServer((args.host, args.port), ...)`（L323），`--host` 默认 `127.0.0.1`（L25），但可被 `REWARD_3D_HOST` 环境变量改为 `0.0.0.0`。
- **问题**：pickle 反序列化任意网络 payload = 远程代码执行原语。默认仅绑本地回环缓解了外部暴露，但多租户 GPU 集群上"本机其它用户/容器逃逸的进程"即可打此端口；且无任何鉴权。
- **影响**：跑 worldolympiad 评测期间，宿主机上任何能访问该端口的进程可在评测进程权限下执行任意代码。
- **建议**：改用 JSON + 文件路径/共享内存传输张量（该文件已有 `/score_file` 路径式接口，可迁移）；若必须 pickle，加 HMAC 校验（共享密钥经 env 注入）并显式拒绝非回环绑定。

#### [XC-2] P2 studio HED 标注器：下载的第三方权重用完整 pickle 反序列化

- **位置**：`worldfoundry/studio/visualization/plugins/perception/hed_annotator.py:73-81`
- **证据**：

```python
remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
...
urlretrieve(remote_model_path, modelpath)
self.netNetwork.load_state_dict(torch.load(modelpath))   # 无 weights_only
```

- **问题**：从第三方 HF 仓库（非本组织控制）下载 `.pth` 后不带 `weights_only=True` 直接 `torch.load`，属"下载内容 + 完整 unpickle"组合；同类还有 `evaluation/tasks/metrics/fsim/vendor/piq/utils/common.py:182,192`（先 `download_target` 再裸 `torch.load`，但 piq 有 SHA256 校验缓解）。
- **影响**：HF 仓库被投毒/劫持时在 studio 进程内执行任意代码。
- **建议**：改用 `worldfoundry.core.checkpoint.safe_loading.load_tensor_state_dict`（仓库已有现成安全工具，见 XC-3）。

#### [XC-3] P3（正面确认）核心 checkpoint 加载路径已系统性默认 `weights_only=True`

- **位置/证据**：`core/checkpoint/safe_loading.py:22-28`（`weights_only: True` 且对旧版 torch fail-closed）、`core/io/serialization.py:594`（`kwargs.setdefault("weights_only", True)`，unsafe fallback 需显式 `allow_unsafe_pickle_fallback=True` opt-in）、`core/model_loading/file.py:169`（同前）。
- **说明**：行级统计中的 82 处"不带 weights_only"绝大多数位于 `evaluation/tasks/metrics/*/vendor`（fdd/fvmd/jedi/dreamsim/fsim 等移植的官方 metric 实现，~16 处）与 `runners/*/runtime`（63 处）——即半 vendored 评测代码；纯自研核心仅 XC-2 一处未走安全加载器。
- **建议**：新增 runner/metric 接入时在 code review checklist 强制走 `safe_loading`；对存量 vendored metric 逐步替换（加载对象均为"下载的第三方权重"，正是 `weights_only` 的目标场景）。

#### [XC-4] P2 配置系统三处动态执行：`eval`/`exec` 是特性但缺乏边界声明

- **位置/证据**：
  - `core/io/python_config.py:101,109`：`eval(value[5:-1], {}, {"d": root})` —— 配置值支持 `eval(...)` 与 `${...}` 插值；空 globals 不会阻断 `__builtins__` 注入，表达式可执行任意代码。
  - `core/configuration/lazy_config/config.py:69,128`：`exec(compile(target.read_text(...)))` —— detectron2 风格 LazyConfig，配置文件即代码（设计如此）。
  - `core/io/file_utils.py:673` 与 `core/io/print_utils.py:65`：`eval("f" + shlex.quote(fmt_str))` —— 用 `eval` 模拟 f-string；`shlex.quote` 是 shell 语义的引号，并非 Python 字符串转义，`{...}` 内表达式照常执行（当前调用点 `suffix_template` 均为代码内字面量，无外部输入路径）。
- **问题**：三处均为"配置=代码"的有意设计，但没有任何 docstring/文档声明"配置文件必须可信"；`eval` f-string 模拟属重复造轮子且在参数外泄时变注入点。
- **建议**：在 `python_config`/`lazy_config` 模块 docstring 明确信任边界；`fstring()` 助手改为 `str.format(**kwargs)`（当前模板只用到 `{i}`/`{i+1}`，可先规范模板语法）。

#### [XC-5] P2 自研下载路径 `requests.get` 无 timeout（3 个文件 4 处）

- **位置/证据**：
  - `core/io/artifacts.py:247`：`requests.get(url, stream=True, allow_redirects=True)` —— artifact 下载主路径；
  - `studio/visualization/plugins/perception/sky_segmentation.py:448`：`requests.get(url, stream=True)`；
  - `studio/visualization/plugins/scene3d/glb_export.py:491,496`。
  其余 ~19 处集中在 runner runtime（videoscore/t2v_compbench 等，半 vendored）。
- **问题**：无 timeout 的网络请求在对端挂起时永久阻塞；`artifacts.py` 位于模型下载关键路径，会卡死整个评测/训练任务且无任何日志（该函数还把失败吞成 `return None` + `print`，见 XC-19 证据）。
- **建议**：统一 `timeout=(connect, read)` 常量（如 `(10, 300)`）；`artifacts.py` 失败应 raise 而非返回 None。

#### [XC-6] P3 其余安全项：影响可控，逐项说明

- **`shell=True`（3 处自研）**：`studio/visualization/backends/frontends.py:420` —— 命令模板来自 `WORLDFOUNDRY_STUDIO_RERUN_COMMAND` env，参数已 `shlex.quote`，本地开发工具风险低；`runners/wrbench/runtime/.../geometry.py:303`、`pose.py:287` —— 拼接的是内部生成的路径，建议改列表参数但非紧急。
- **`yaml.load` 非 safe（4 处）**：`evaluation/tasks/metrics/jedi/V_JEPA.py:181,191`（`FullLoader`，风险低）、`runners/fetv/runtime/fetv_eval/auto_eval.py:163` 与 `runners/chronomagic_bench/runtime/.../configs/config.py:277`（`yaml.Loader`，可实例化任意对象——但输入是仓库内配置文件）。建议统一 `safe_load`。
- **`verify=False`（1 处）**：`studio/gradio_runtime.py:96` —— monkeypatch Gradio 的 localhost 探活，目标恒为 127.0.0.1，可接受；建议加注释说明仅限回环。
- **`pickle` 其余自研命中**：`core/distributed/*`（6 处）为 torch collective 的进程组内序列化，参与方同信任域，业界通用做法；`core/kernels/triton_autotune.py:64` 加载本地 autotune 缓存（本机写本机读）；`core/utils/misc_utils.py:257-262` 的 `encode_base64/decode_base64` 是通用 unpickle 工具且**当前无调用方**——属死代码暴露面，建议删除。
- **`md5`（3 处自研）**：`core/model_loading/file.py:259,399` 为 state-dict key 指纹、`core/io/file_utils.py:568` 为文件校验和，均非对抗场景，可保留（如需合规可换 `sha256` 或 `md5(usedforsecurity=False)`）。
- **`eval()` 于数据文件/LLM 输出（runners 内）**：`runners/worldscore/runtime/.../prompt_generator.py:74`（`eval(response)`，response 为 GPT 返回值！）、`runners/videoscore/runtime/benchmark/get_*.py`（eval 标注文件字段）、`evaluation/tasks/metrics/artscore/datasets.py:147`。LLM 输出 eval 属真实注入面但仅在手动跑基准脚本时触发——建议换 `ast.literal_eval`，标 P2 跟踪。

### 主题 2：路径硬编码

**统计**（`*.py`，字面量出现次数；`/data/` 已过滤 `xx/data/yy` 型相对路径子串）：

| 模式 | 自研层 | vendored 层 |
|---|---|---|
| `/mnt/` | 0 | 2（latent_action 的 hdfs→`/mnt/hdfs/` 映射） |
| `/home/` | 1（devil_dynamics 内嵌官方代码） | 2（1 注释 + spatia argparse 默认值） |
| 绝对 `/data/...` | 3（1 默认参数 + 2 文档字符串示例） | ~4 |
| `/cpfs`、`C:\` | 0 | 0 |
| 自研层 yaml/json/toml 配置内集群路径 | 0 | - |

自研层非常干净：无 `/mnt/`、`/cpfs`、当前集群用户路径泄漏；输出目录均走 config/env。

#### [XC-7] P3 少量默认参数指向环境特定绝对路径

- **位置/证据**：
  - `worldfoundry/evaluation/tasks/embodied/simulators/calvin/benchmark.py:197`：`dataset_path: str = "/data/calvin/dataset/validation"` —— 构造器默认值指向容器约定路径；同类 `simulators/robocerebra/benchmark.py:67`：`robocerebra_root: str = "/workspace/RoboCerebra_Bench"`。
  - `worldfoundry/evaluation/tasks/embodied/docker_runner.py:100,147`：`"/workspace/results"`、`"/workspace/WorldFoundry"` —— 容器内挂载点，设计如此（可接受）。
  - `worldfoundry/evaluation/tasks/execution/runners/devil_dynamics/runtime/official/metrics_utils/standard_video_dataset.py:180`：`video_folder = '/home/LiaoMingxiang/Workspace2/DinaBench/candidate_videos'` —— 移植官方代码残留的作者家目录死默认值（实际由 runner 覆盖）。
- **问题**：calvin/robocerebra 的默认值在容器外直接实例化时给出误导性报错（`FileNotFoundError: /data/calvin/...`）；devil_dynamics 残留路径污染代码可读性。
- **影响**：低——均可被调用方覆盖，且 docker_runner 场景下默认值正确。
- **建议**：calvin/robocerebra 默认值改 `None` + 显式报错提示"传入 dataset_path 或用 docker profile"；devil_dynamics 删除死默认。
- **vendored 层被调用路径上的问题**：`base_models/perception_core/action_recognition/latent_action/backbones.py:30,116` 把 `hdfs:///` 前缀重写为 `/mnt/hdfs/`（源集群约定），该模块被 `operators/dreamdojo_operator.py`、`evaluation/models/*` 引用——若 checkpoint 清单含 hdfs URI 会静默解析到不存在的本地路径，建议在自研包装层拦截并报错。

### 主题 3：环境变量清单

**统计**（`os.environ[...]` / `.get` / `getenv` / `setdefault` 读取点，自研层）：

- 唯一变量名 **574** 个，其中：
  - `WORLDFOUNDRY_` 前缀 **372** 个（65%）；`WF_` 前缀 **0** 个（无短前缀混用，好）；
  - 标准生态变量（`CUDA_VISIBLE_DEVICES` 47 处、`LOCAL_RANK` 25、`RANK` 18、`OPENAI_API_KEY` 23、`MASTER_ADDR/PORT` 等）约 55 个；
  - **裸名自定义变量 147 个**（无任何前缀），绝大多数在 runner runtime：`VBENCH_ROOT`、`VIPE_ROOT`、`DATASET_BASE_DIR`、`DATA_DIR`、`PAVRM_MODEL_PATH`、`VLM_API_KEY`、`REWARD_3D_HOST` 等；
  - 另有两个**历史前缀家族**：`TRAINER_*` 16 个（`core/distributed/sequence_parallel/envs.py` 与 training 引擎）、`WM_*` 4 个（`studio/conda_dispatch.py`）。
- **文档覆盖**：`docs/`+`scripts/`+README 共出现 453 个 `WORLDFOUNDRY_*` 名字，但与代码实际读取集合对比，**372 个使用中的变量有 253 个（68%）未在任何文档/脚本出现**（多为 runner 专属，如 `WORLDFOUNDRY_CAMERABENCH_MODE`、`WORLDFOUNDRY_APPLE_PI_JUDGE_BACKEND`）。
- **正面**：`runtime/env.py:35-75` 已是事实上的中心注册表——`CORE_ENV_KEYS`/`HF_ENV_KEYS`/`RUNTIME_ENV_KEYS` 分组、`_SENSITIVE_MARKERS` 脱敏、`SOURCEABLE_ENV_BASE_LINES` 给出 `WORLDFOUNDRY_HOME → MODEL_DIR/CKPT_DIR → HFD_ROOT` 的一致派生链，设计良好；但注册表只覆盖 core 组，未收编 runner 侧的 300+ 变量。

#### [XC-8] P2 环境变量命名三套并存 + 68% 无文档

- **位置/证据**：见上统计；具体家族——`TRAINER_TORCH_PROFILER_DIR` 等 16 个 `TRAINER_*`（`core/distributed/sequence_parallel/envs.py:全文件`）、`WM_AUTO_CUDA_VISIBLE_DEVICES` 等 4 个 `WM_*`（`studio/conda_dispatch.py`）与主流 `WORLDFOUNDRY_*` 并存。
- **问题**：三套前缀是不同来源代码合入的痕迹；`TRAINER_*`、`WM_*` 无命名空间保护，易与用户环境/其它框架撞名（`DATA_DIR`、`VERBOSE`、`PYTHON` 这类裸名更危险——147 个裸名里含 `DATASET_BASE_DIR`、`DATA_DIR`、`VERBOSE`、`VLM_MODEL` 等通用词）。
- **影响**：集群作业环境里预设的同名变量会被静默采纳，产生难排查的行为差异；无文档变量实际上是隐藏配置面。
- **建议**：(1) `TRAINER_*`/`WM_*` 迁移到 `WORLDFOUNDRY_` 并保留旧名 fallback + DeprecationWarning 一个版本；(2) runner 专属变量强制 `WORLDFOUNDRY_<BENCH>_` 前缀（现有多数已遵守，清理 147 个裸名中自研新增的部分）；(3) 以 `runtime/env.py` 为唯一注册表，CI 校验"代码中读取的 `WORLDFOUNDRY_*` 必须在注册表或 benchmark runtime-profile 文档中声明"。

#### [XC-9] P3 个别变量名歧义/异常

- **证据**：
  - `worldolympiad` 用**小写** `api_key` 作为环境变量名（`runners/worldolympiad/runtime/worldolympiad/model/openrouter.py:74,133,152`：`Bearer {os.getenv('api_key')}`）——不符合 env 命名惯例，且与常见 shell 变量冲突风险高，`reward_3d.py:159` 还据此做分支判断。
  - `WORLDFOUNDRY_KERNEL_AUTOTUNE`、`WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE`（布尔开关，`core/kernels/autotune_cache.py:45`）与 `WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_DIR`（路径，同文件:169）三个近似名并存，布尔开关未用 `_ENABLED` 后缀区分。
- **建议**：`api_key` 改 `WORLDFOUNDRY_OPENROUTER_API_KEY`（或复用 `OPENROUTER_API_KEY` 生态惯例）；autotune 开关改 `..._CACHE_ENABLED`（保留旧名 fallback）。

### 主题 4：全局可变状态

**统计**：`global` 关键字——自研层 **100** 处（core/distributed 约 45、studio 约 15、其余散布），vendored 层 **82** 处。模块级可变容器（registry/cache 类）自研核心约 **14** 个（`_CLASS_REGISTRY`、`POLICY_REGISTRY`、`BENCHMARK_INTEGRATION_REGISTRY`、`VIDEO_RUNNER_REGISTRY`、`_FAILED_ATTENTION_SIGNATURES`、`_HOST_COPY_STREAMS`、`_SEQUENCE_PARALLEL_GROUPS`、`_COLLECTIVE_SHAPE_CACHE` 等）。

#### [XC-10] P1 core/distributed 内三套互不相通的并行状态单例并存

- **位置/证据**：
  1. `core/distributed/context_parallel_util.py:8-16`：**小写**模块级全局 `dp_size/cp_size/dp_group/cp_group/...` 共 9 个，全文件 17 处 `global`；被 `core/attention/long_context_ulysses.py` 与多个 synthesis 运行时使用。
  2. `core/distributed/sequence_parallel_runtime.py:24-25`：`_SEQUENCE_PARALLEL_STATE = False` + `_SEQUENCE_PARALLEL_GROUPS: dict`；被 `pipelines/hunyuan_world/*` 使用。
  3. `core/distributed/sequence_parallel/parallel_state.py:694-771`：vLLM 风格 `_WORLD/_TP/_SP/_DP` GroupCoordinator 单例；被 training 引擎使用。
- **问题**：同一进程内三套"当前并行拓扑"真值来源。若一条推理流水线同时触发 ulysses attention（读 1）和 hunyuan sequence-parallel（读 2），两套 group 各自初始化、互不感知；`context_parallel_util` 还是小写全局名 + 无 reset/teardown + `print` 调试输出（L28,30,40），不符合库代码标准。
- **影响**：并行策略组合时的 rank 错配/死锁风险；单测之间状态泄漏（无 reset 接口）；三份近似代码的维护成本。
- **建议**：以 `parallel_state.py`（最完整，含 destroy/patch 机制）为唯一真值源，另两套改为其只读视图；`context_parallel_util` 全局改 `_CP_STATE` 单对象 + 提供 `reset_context_parallel()`；`print` 改 logger。

#### [XC-11] P2 注册表静默覆盖重名注册

- **位置/证据**：`core/io/config_utils.py:121-127`：

```python
def register_class(cls, alias=None):
    _CLASS_REGISTRY[cls.__name__] = cls   # 无重名检查，静默覆盖
```

  同模式：`core/utils/functional_utils.py:273-279` 的 `make_registry_metaclass`（`cls.registry[name] = new_cls`，且该工厂**当前无调用方**，属死代码）。对照组：`evaluation/tasks/execution/framework/runner_registry.py:32` 的 `VIDEO_RUNNER_REGISTRY` 为字面量字典（构造期固定，无运行时突变，好）。
- **问题**：两个不同模块定义同名类并 `register_class` 时，后 import 者静默赢——在 1400+ 文件的 evaluation 包里同名概率不低，错绑类会在 config 实例化时以极难排查的方式表现。
- **建议**：注册时 `if name in _CLASS_REGISTRY and _CLASS_REGISTRY[name] is not cls: raise`；删除无调用方的 `make_registry_metaclass`。

#### [XC-12] P2 studio 在 import 时 monkeypatch 第三方库（Gradio）

- **位置/证据**：`studio/gradio_runtime.py:80,112,149`——模块顶层直接执行 `_install_template_response_guard()`、`_install_proxy_safe_url_check()`、`_install_api_info_guard()`，分别替换 `gradio.templates.TemplateResponse`、`gradio.networking.url_ok`、API schema 处理函数。
- **问题**：`import worldfoundry.studio.gradio_runtime` 的副作用是全进程性篡改 Gradio 行为；任何间接 import（如文档生成、单测收集）都会触发。虽有幂等哨兵（`_worldfoundry_*` 属性），但 patch 无法卸载，且 Gradio 升级后 patch 目标漂移只能靠运行时报错发现。
- **影响**：中——目前该模块仅 studio 入口 import；但作为库的一部分导出，破坏"import 无副作用"约定。
- **建议**：改为显式 `install_gradio_patches()` 由 studio 启动入口调用；对 patch 目标加版本断言。

#### [XC-13] P3 模块级缓存/状态容器无并发保护、无失效策略（清单）

- **证据**（均自研核心）：`core/attention/dispatch.py:303,306`（`_FAILED_ATTENTION_SIGNATURES`/`_UNAVAILABLE_ATTENTION_BACKENDS`，多线程 add 无锁——set 操作有 GIL 兜底，可接受但应注明）、`core/acceleration/frame_prefetch.py:19`（`_HOST_COPY_STREAMS: dict[int, Any]`，key 为 device id，懒创建无锁，两线程同时首次访问会各建一个 stream）、`core/distributed/sequence_parallel_runtime.py:26`（`_COLLECTIVE_SHAPE_CACHE` OrderedDict 做 LRU，跨 rank 不同步——已有 env 开关 `WORLDFOUNDRY_CACHE_COLLECTIVE_SHAPES` 控制，尚可）、`core/utils/misc_utils.py:216`（`_GLOBAL_ONCE_SET`）。
- **建议**：懒创建路径统一 `threading.Lock` 或 `functools.lru_cache`；在容器旁注释线程模型假设。

### 主题 5：并发安全

**统计**（自研层，按文件数）：`threading` 29 个文件、`multiprocessing` 19、`asyncio` 18；threading+asyncio 同文件混用 2 处（`evaluation/tasks/embodied/adapters/websocket_adapter.py`——模式正确，见下；iworldbench runtime 1 处）。`os.fork()` 直接调用：0。

**正面确认**（抽查通过）：
- `evaluation/tasks/execution/orchestration/model_benchmark_suite.py:1517` 显式 `multiprocessing.get_context("spawn")`——评测子进程正确规避 fork-after-CUDA；
- `websocket_adapter.py:28-70` 专用事件循环线程 + `run_coroutine_threadsafe` + `call_soon_threadsafe`，教科书式桥接；
- `studio/conda_dispatch.py:167-172` 为全局 GPU 池/常驻 worker 配了 4 把独立锁；
- `mcp/client.py:206-220` `_run_sync` 检测运行中事件循环并显式关闭 coroutine 再报错，处理干净；
- `serve_reward_3d.py:101` 推理临界区有 `self.server.inference_lock`。

#### [XC-14] P2 checkpoint shard 并行下载用默认 fork 上下文的 ProcessPoolExecutor

- **位置**：`worldfoundry/core/checkpoint/load.py:296`
- **证据**：

```python
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    for shard_file, path in pool.map(_hf_hub_download_shard_task, work):
```

  Linux 默认 fork。该函数在模型加载路径上被调用，届时进程通常已 import torch、可能已被 studio 的多线程 dispatch（conda_dispatch 常驻 worker 线程、reaper 线程）包裹。
- **问题**：(1) 在多线程进程里 fork，子进程可能继承处于加锁状态的锁（logging、HF 内部锁）→ 偶发死锁；(2) 若届时 CUDA 已初始化，虽然子任务只做网络下载不触 CUDA，但继承的 CUDA 上下文在某些驱动组合下 fork 即警告/泄漏；(3) 下载是 IO-bound，进程池本身就是错配。
- **影响**：低概率但难复现的下载卡死；每 worker 一份 fork 的内存页副本（大模型进程 fork 代价高）。
- **建议**：改 `ThreadPoolExecutor`（hf_hub_download 线程安全、IO-bound）；若确需进程隔离则 `ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))`。

#### [XC-15] P2 mind runner：fork 语义的 `mp.Pool` + 子进程用 CUDA（半 vendored）

- **位置**：`worldfoundry/evaluation/tasks/execution/runners/mind/runtime/mind/src/process.py:282-285,326`
- **证据**：`available_gpus = torch.cuda.device_count()`（父进程）→ `with mp.Pool(processes=num_gpus) as pool:` → worker 参数含 `f'cuda:{i}'`。未设 spawn 上下文，依赖"父进程尚未真正初始化 CUDA 上下文"这一脆弱前提；一旦上游 import 链里有人先 `torch.cuda.current_device()` 之类，全部 worker 报 `Cannot re-initialize CUDA in forked subprocess`。
- **建议**：`mp.get_context("spawn").Pool(...)`，与 larybench（`classification_cli.py:63` 已 `mp.set_start_method("spawn")`）对齐。

#### [XC-16] P3 studio `_TORCHRUN_CONTROL_GROUP` 全局无锁

- **位置**：`worldfoundry/studio/execution.py:461,534`——`global _TORCHRUN_CONTROL_GROUP` 读写无锁，依赖"分布式初始化阶段单线程"的隐式假设。当前调用时序成立，建议加注释固化假设或包一把锁，防止后续 studio 线程化改造踩雷。

### 主题 6：临时文件

**统计**（自研层）：`tempfile.*` 调用 **67 处 / 57 文件**；手工 `"/tmp/..."` 字面量仅 **4 处**（其中 2 处为容器内约定路径/帮助文本，真实手拼 2 处）；`delete=False` **13 处**；`mkdtemp` **23 处**。vendored 层手工 `/tmp` 2 处。整体纪律良好——手工拼 `/tmp` 基本绝迹。

**正面确认**：`pipelines/cut3r/official_runtime.py:389-392` `finally: shutil.rmtree(temp_dir, ignore_errors=True)`；training 侧 `delete=False` 均为"写临时文件→原子 rename"模式（`training/data/sana_cache.py:387`、`video_cache.py:428`、`engine/ltx/lora.py:79` 的 `.incomplete-` 前缀目录）；fetv/t2v_compbench 的 `.stage-` 目录同为 rename 交付。

#### [XC-17] P2 多个 pipeline 在未指定 output_dir 时把生成视频写进不清理的 mkdtemp

- **位置/证据**：
  - `pipelines/lyra/pipeline_lyra1.py:172-173`：`generated_root = Path(tempfile.mkdtemp(prefix=f"lyra1_{mode}_generated_"))`（另 `lyra_utils.py:463` `lyra2_runtime_`）——生成的视频/重建产物落 `/tmp`，无任何 rmtree/atexit；
  - `pipelines/matrix_game/pipeline_matrix_game_3.py:140-143`：同模式 `matrix_game_3_` 前缀；
  - `evaluation/models/runtime/profiles.py:1166`：`run_dir = ... or tempfile.mkdtemp(prefix=...)`——评测 run 产物默认落 `/tmp`；
  - `core/io/video.py:770-775`：`local_video_path()` 每次调用 `mkdtemp` 物化视频帧并返回路径，调用方无从清理（当前暂无外部调用方，属新 API 的先天泄漏设计）。
- **问题**：视频类产物单个可达数百 MB；长驻评测服务/批量跑分场景下 `/tmp`（常为内存 tmpfs 或小分区）会被写满，殃及同机其它作业。返回路径给调用方的场景无法用 context manager，但至少应该注册进程退出清理或写到 `WORLDFOUNDRY_CACHE_DIR` 下带 TTL 的目录。
- **建议**：统一"未指定 output_dir → 写 `${WORLDFOUNDRY_CACHE_DIR}/scratch/<date>/`"并在 CLI 层提供 `worldfoundry cache prune`；短生命周期中间文件改 `TemporaryDirectory` context manager。

#### [XC-18] P3 runner 内临时 mp4 清理不走 finally

- **位置/证据**：`runners/wbench/runtime/wbench/src/metrics/vlm/vlm_evaluator.py:81-98,124-144`——`NamedTemporaryFile(suffix=".mp4", delete=False)` 后 `os.remove(tmp_path)` 在正常路径执行，无 try/finally；cv2 写帧抛异常即泄漏。同模式：`runners/memobench/runtime/.../llm-vqa.py:133`、`runners/videoscore/runtime/.../qwenVL_eval.py:53`（半 vendored）。
- **建议**：包 `try/finally` 或 `contextlib.ExitStack`；新 runner 接入 checklist 加此项。

### 主题 7：日志 vs print

**统计**（行首 `print(`）：

| 区域 | 数量 | 定性 |
|---|---|---|
| evaluation | 2,880（其中 runners 内 2,710） | runners 为移植官方脚本，可暂容忍；**非 runner 的 170 处应清理** |
| cli | 221 | CLI 面向终端输出，大部分有意 |
| studio | 88 | 混杂：`frontends.py` 19、`sky_segmentation.py` 16 等属库代码 print |
| core | 67 | `termcolor.py` 10 处为 print 工具本体（合理）；`metric_sync.py` 4、`model_loading/file.py` 5、`artifacts.py` 4 等为残留 |
| pipelines / operators / runtime / data | 62 / 9 / 4 / 18 | 少量残留 |
| **training / mcp** | **0 / 0** | 干净（training 全量走 logger，标杆） |
| vendored 层 | 3,473 | 不处理 |

**正面确认**：`core/logging_setup.py` 提供了完整的中心化日志基建——contextvars 上下文绑定（`bind_log_context`）、敏感字段脱敏（`_redact_text`）、JSONL 事件流（`write_jsonl_event`）、`WorldFoundryLoggerAdapter.event()` 结构化接口，质量高。

#### [XC-19] P2 中心日志基建采用率极低：9 个文件 vs 90 个文件裸用 `logging.getLogger`

- **证据**：`from worldfoundry.core.logging_setup import ...` 仅 **9** 个文件；同层 `logging.getLogger` 直接使用 **90** 个文件；另有 3,300+ `print`。核心路径上的 print 实例：`core/distributed/metric_sync.py`（分布式集合通信代码用 print，rank 混排不可读）、`core/model_loading/file.py`（加载进度 print）、`core/io/artifacts.py:253-256`（下载失败仅 print 后返回 None）、`core/distributed/context_parallel_util.py:28,30,40`。
- **问题**：脱敏、上下文、JSONL 事件等能力只覆盖 9 个文件；其余日志绕过 redaction（token 有泄漏进日志的通道）且无 rank/job 上下文。
- **建议**：`ruff` 启用 `T201`（print 检查）并对 cli/termcolor 白名单豁免；core/pipelines/studio 的 print 分批替换为 `logging_setup.get_logger`。

#### [XC-20] P1 库代码在 import/调用时篡改全局 logging 配置

- **位置/证据**：
  - `evaluation/tasks/metrics/jedi/V_JEPA.py:25-28`（**import 时**执行）：

```python
import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 把 root logger 提到 INFO，影响全进程
```

  - `core/io/wan_video_geometry.py:43`：`merge_video_audio()` 函数体内 `logging.basicConfig(level=logging.INFO)`——工具函数每次调用尝试配置 root；
  - `runners/memobench/runtime/memobench/evaluation/run_eval.py:11`（import 时）：`logging.getLogger().setLevel(logging.WARNING)`——反向压低 root，**静默全进程其它组件的 INFO 日志**；
  - `runners/larybench/classification_cli.py:36`：CLI 入口内 basicConfig，可接受。
- **问题**：jedi 是 metric 模块，任何评测流程 import 它就改掉宿主进程 root logger 级别与 handler；memobench 则把宿主日志级别压到 WARNING——两者叠加时行为取决于 import 顺序，属典型"日志幽灵"。
- **影响**：评测编排进程日志时有时无、级别漂移，难以在多基准 suite 中定位问题。
- **建议**：删除模块级 basicConfig/setLevel，改模块私有 logger（`logging.getLogger(__name__)` + `addHandler(NullHandler())`）；库代码禁止碰 root logger，CI 加 rg 守卫（`logging\.basicConfig|getLogger\(\)\.setLevel` 白名单仅 cli 入口）。

### 主题 8：异常处理审计

**统计**（`*.py`，`!__pycache__`）：

| 模式 | 自研层 | 其中 runners/vendor 外 | vendored 层 |
|---|---|---|---|
| 裸 `except:` | 78 | **8**（全部在 metrics 移植代码：`jedi/utils.py:12`、`facescore/FaceScore.py:109`、`artscore/{datasets,utils,models}.py` 6 处） | 228 |
| `except Exception/BaseException: pass` | 86（core 12、evaluation 53、studio 12、runtime 5、cli 3、training 1、mcp 0、pipelines 0） | 约 33 | 63 |
| `contextlib.suppress`（对照） | 12 | - | - |

**分类抽查结论**（读取核实）：
- core 的 12 处 except-pass 多为"尽力而为"型：可选依赖探测（`core/attention/varlen.py:126,131`——建议收窄为 `ImportError`）、best-effort setattr（`core/inference.py:4776`）。
- evaluation 非 runner 的 12 处多为降级链（`framework/benchmark_data.py:193,203`——opencv→imageio→GIF 编码回退，模式合理但失败应 debug 级记录）与模拟器清理路径（calvin/robomme 等 benchmark.py，属 teardown 容错）。
- **编排/报告层无"吞错继续打分"模式**（`orchestration`/`framework`/`api`/`reporting` 未检出 `except Exception:` + `continue/return None`），分数完整性通道干净。
- `KeyboardInterrupt` 吞没仅 3 处，均为 studio 前端进程 Ctrl-C 退出路径，合理。

#### [XC-21] P2 精度/确定性配置失败被静默忽略

- **位置**：`worldfoundry/core/inference.py:4824-4836`
- **证据**：

```python
try:
    torch.set_float32_matmul_precision(matmul_precision)
except Exception:
    pass
...
    setattr(backend, "allow_tf32", bool(enable_tf32))
except Exception:
    pass
```

- **问题**：用户显式请求的 matmul 精度 / TF32 开关设置失败时无任何痕迹，实际数值行为与配置声明背离——这类"配置未生效"比报错更伤（评测结果不可复现却无线索）。
- **建议**：失败至少 `logger.warning`（一次性去重可用现有 `_GLOBAL_ONCE_SET`）。

#### [XC-22] P2 metrics 移植代码 8 处裸 `except:`（连 `SystemExit`/`KeyboardInterrupt` 一并吞掉）

- **位置**：`evaluation/tasks/metrics/jedi/utils.py:12`、`facescore/facescore_pkg/FaceScore.py:109`、`artscore/models.py:11,45,144,202`、`artscore/{datasets.py:26,utils.py:63}`。
- **问题**：裸 `except:` 捕获 `BaseException`，Ctrl-C 中断评测时这些路径可能吞掉中断继续跑；且这批文件在自研 metric 调用路径上（非隔离子进程）。
- **建议**：全部改 `except Exception`；ruff 启用 `E722`（自研层已基本达标，指纹显示只剩这一批）。

### 主题 9：随机性与确定性

**统计**：自研层（不含 runners/vendor）seed 设置点 **55** 处，分散在 training/engine、pipelines、evaluation/simulators；中心工具 `core/utils/torch_utils.py` 的 `set_seed_everywhere()`（random+numpy+torch+cuda+可选 TF，含 rank 处理与 invalid-seed 策略，L107-120）与 `set_deterministic()`（CUBLAS_WORKSPACE_CONFIG + cudnn.deterministic + use_deterministic_algorithms，L84-104）设计完备。

#### [XC-23] P2 seed 设置各自为政：中心工具存在但采用不全，训练会话漏掉 numpy

- **位置/证据**：
  - `training/engine/sessions/single_device.py:363-368`：手写 `random.seed + torch.manual_seed + torch.cuda.manual_seed_all`——**未 seed numpy**，数据增广若用 `np.random` 则不可复现；也未复用 `set_seed_everywhere`。
  - `training/engine/sana/scm_ladd.py:93-96`：同样手写 random+torch，漏 numpy。
  - `set_deterministic()` 全库 **0 个调用方**（rg 全库仅定义处）——确定性开关是死代码，评测框架没有任何路径能启用 `torch.use_deterministic_algorithms`。
  - `evaluation/api/models.py:25` 有 `seed: int | None = None` 字段，但编排层 `model_benchmark_suite.py` 全文无 seed 处理——基准复跑的种子策略完全下放给各 runner，scorecard 不记录生效种子。
- **影响**：跨 runner 结果复现性口径不一致；训练侧 numpy 随机性逃逸。
- **建议**：训练会话与 sana 引擎改调 `set_seed_everywhere(seed, deterministic=...)`；评测编排在 run manifest 里固化并记录 seed；`set_deterministic` 要么接入 `WORLDFOUNDRY_DETERMINISTIC=1` 环境开关要么删除。

### 主题 10：时间与区域

**统计**（自研层，不含 runner runtime/vendor）：无时区 `datetime.now()/utcnow()` **14** 处；tz-aware 用法 **10** 处；`time.time()` 计时赋值仅 **4** 处，`time.monotonic()/perf_counter()` **105** 处——计时纪律整体良好。

#### [XC-24] P3 时间戳写入产物时无时区、格式不统一（清单）

- **证据**：
  - `runners/camerabench/camerabench_metrics.py:108,190,404`：`"evaluation_timestamp": datetime.now().isoformat()`——**写入评测结果 JSON** 的时间戳是 naive 本地时间，跨时区机器汇总时错序；
  - `studio/execution.py:667`、`studio/native/world_explorer/api/lyra_persistent.py:482`、`core/io/file_utils.py:650`（`"_%H-%M-%S_%m-%d-%y"`——时-分-秒在前、两位年在后的非常规格式）：文件名时间戳，5 种互异格式；
  - `core/checkpoint/sharded_safetensors.py:26-48`：用 `datetime.now()` 差值计时（应 `perf_counter`）；`core/distributed/metric_sync.py:178-201` torchvision 风格 ETA 用 `time.time()`（仅日志展示，影响小）。
- **建议**：产物内时间戳统一 `datetime.now(timezone.utc).isoformat()`；文件名戳统一 `%Y%m%d-%H%M%S`；计时新增代码一律 `perf_counter`。

### 主题 11：弃用 API（torch/transformers）

**统计**（仓库 pin：`torch>=2.7,<2.12`，pyproject.toml:225）：

| 模式 | 自研层 | vendored 层 | 状态 |
|---|---|---|---|
| `torch.cuda.amp.autocast`（旧签名，2.4 起 FutureWarning） | 9（core 1 + runners 8） | 84 | 每次调用刷警告 |
| `torch.cuda.amp.GradScaler` | 1（vmbench runner） | 3 | 同上 |
| `torch.symeig`（**1.13 已移除**） | 1（fsim vendor，见下） | 0 | 调用即 AttributeError |
| `torch._six`（**1.11 已移除**） | 0 | 3（lietorch/opensora，均带 try-fallback） | import 兜底 |
| `torch.meshgrid` 无 `indexing=` | 0（5 处命中均为误报/旧版兼容分支，已逐个核实） | ~167（行级启发式） | 警告 |
| torchvision `pretrained=True`（0.13 起弃用） | 18（全部在 metrics vendor / runner runtime） | 45 | 警告 |
| transformers 弃用族（AutoModelWithLMHead/AdamW/旧 BertTokenizer） | 0 | 未查 | 干净 |

#### [XC-25] P2 唯一 core 层旧 AMP 调用 + 已移除 API 的死代码

- **位置/证据**：
  - `core/nn/diffusion_utils.py:27`：`with torch.cuda.amp.autocast(enabled=True, ...)`——core 层唯一残留，位于 diffusion 数值工具，每次前向刷 FutureWarning，torch 3.0 计划移除；
  - `evaluation/tasks/metrics/fsim/vendor/piq/iw_ssim.py:393`：`torch.symeig(C_u, eigenvectors=True)`——该 API 在 torch 1.13 已删除，本仓库 pin torch≥2.7 下**必崩**；核实 `fsim/wrapper.py:25` 仅 `from piq.fsim import fsim`，iw_ssim 当前不可达（死代码），但任何人扩展 vendored piq 的使用面就会踩雷。
- **建议**：`diffusion_utils.py` 改 `torch.amp.autocast("cuda", ...)`（一行改动）；删除或修复 `iw_ssim.py`（改 `torch.linalg.eigh`）并在 vendored piq 目录加 README 标注"仅 fsim 入口经过验证"；runners 内 8 处旧 AMP 随各基准升级顺带处理。

## 汇总

### 严重度统计表

| 严重度 | 数量 | 条目 |
|---|---|---|
| P0 | 0 | - |
| P1 | 3 | XC-1（评测 reward 服务 pickle-over-HTTP）、XC-10（三套并行状态单例并存）、XC-20（库代码篡改全局 logging） |
| P2 | 13 | XC-2、XC-4、XC-5、XC-8、XC-11、XC-12、XC-14、XC-15、XC-17、XC-19、XC-21、XC-22、XC-23、XC-25 中除去正面项（XC-25 计入后为 14 项，其中 XC-2/4/5/8/11/12/14/15/17/19/21/22/23/25） |
| P3 | 8 | XC-3（正面确认）、XC-6、XC-7、XC-9、XC-13、XC-16、XC-18、XC-24 |

（修正：P2 共 **14** 项。）

### Top 5 问题

1. **[XC-1] P1** worldolympiad 3D reward 服务对 HTTP POST body 直接反序列化不可信输入（`pickle.loads`），默认绑回环缓解了外部暴露，但多租户机器上同机进程即可触达且无鉴权——应改 JSON/文件路径传输（该文件已有 `/score_file` 接口可迁移）。
2. **[XC-10] P1** `core/distributed` 三套互不相通的并行拓扑单例（`context_parallel_util` 小写全局 / `sequence_parallel_runtime` / `parallel_state` vLLM 风格）并存，组合使用时 rank 错配/死锁风险且单测间状态泄漏——应以 `parallel_state` 为唯一真值源，另两套改只读视图。
3. **[XC-20] P1** 库代码在 import/调用时篡改全局 `logging`（jedi `V_JEPA.py` import 即 `basicConfig`+抬 root 到 INFO；memobench 压 root 到 WARNING；`wan_video_geometry` 函数内 `basicConfig`）——评测 suite 日志级别随 import 顺序漂移，应改模块私有 logger、禁碰 root。
4. **[XC-8] P2** 环境变量三套前缀并存（`WORLDFOUNDRY_*` 372 / `TRAINER_*` 16 / `WM_*` 4）+ 147 个无前缀裸名（`DATA_DIR`/`VERBOSE` 等易撞名）+ 68% 使用中的变量无文档——应统一前缀并以 `runtime/env.py` 为唯一注册表加 CI 校验。
5. **[XC-19]+[XC-23] P2** 中心基建存在但采用率低：结构化日志（`logging_setup`）仅 9 文件用、90 文件裸 `getLogger` + 3300+ `print`（token 有绕过脱敏进日志的通道）；`set_seed_everywhere`/`set_deterministic` 齐备但训练会话手写 seed 漏 numpy、`set_deterministic` 全库零调用方（确定性开关是死代码）。

> 说明：本报告主体（主题 1–11、XC-1～XC-25、统计表）已完成；上方 Top 5 结尾曾因内容审查在收尾处中断，现已补全。多数自研层修复点与 core/pipelines/training 各模块报告重叠，将在模块修复完成后的"横切第二轮"统一处理不重叠项（runner runtime 的 pickle/root-logger、env 注册表、注册表静默覆盖等），避免与在途 agent 争抢同一文件。
