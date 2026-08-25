# studio 层修复日志

> 对应评审报告：`plan/code_review/10_studio.md`。
> 约束：只改 `worldfoundry/studio/`；不改 `pyproject.toml`（ST-2 另行安排）、其它子包、`test(s)/`、`docs/`；不执行 git 变更命令。
> 环境说明：本机缺 `fastapi`/`gradio`（第三方依赖，非本次改动引入），对应模块导入冒烟以 `py_compile` 通过 + ImportError 仅报缺依赖为准；`aiohttp`/`aiortc` 可用，`world_realtime` 走了真实导入与功能冒烟。

## 已修复

### ST-3 (P1) 非 loopback 绑定强制 token 认证；public URL 处加安全警告

- **共享模块**（新文件 `serving/auth.py`，各栈薄适配复用，无复制）：
  - `is_loopback_host(host)`：`localhost`/loopback IP（含 `[::1]`）返回 True；空串、`0.0.0.0`、`::`、未知主机名一律按非 loopback 处理（fail-closed）。
  - `require_auth_token_for_host(host, server_name=...)`：loopback 返回 `""`（保持历史无认证行为完全不变）；非 loopback 时读 `WORLDFOUNDRY_STUDIO_AUTH_TOKEN`，未设置则 `SystemExit` 拒绝启动，错误消息指导"设 token 或改 --host 127.0.0.1 + SSH 隧道"。
  - `request_token_valid(token, authorization_header=..., query_token=...)`：接受 `Authorization: Bearer <token>` 或 `?token=<token>`；比较用 `hmac.compare_digest`（`token_matches`）。
  - `serving/network.py` 新增 `bind_security_warning(host)`：非 loopback 绑定返回显著 SECURITY WARNING 文本（含 token 是否已启用），loopback 返回 None。`serving/__init__.py` 导出上述符号。
- **各服务器栈接线**：
  - `workspace_app.py`（FastAPI）：`create_app(auth_token="")` 注册 `@app.middleware("http")`（`workspace_app.py:2614-2631`），豁免 `{"/", "/favicon.ico", "/assets/openenvision-logo.png"}` 三个静态路径，其余（jobs/settings/file/visualizer 等全部 API）未带合法 token 返回 401；`main()`（`workspace_app.py:5500-5501`）非 loopback 时取 token 并打印警告。
  - `world_realtime.py`（aiohttp）：`serve_realtime_world_frontend` 注册 aiohttp middleware（`world_realtime.py:2860-2877`），豁免 `{"/", "/world.css", "/world.js", "/favicon.svg"}`，`/api/*`（含 offer/upload/file/ws）未带 token 抛 `HTTPUnauthorized`；启动打印处输出 `bind_security_warning`（`world_realtime.py:3115`）。
  - `frontends.py`（http.server，Spark/Media viewer）：`serve_spark_frontend`/`serve_media_frontend` 取 token（`frontends.py:326,361`）传入 handler；共享 `_handler_request_authorized`（`frontends.py:452-465`）在 `do_GET` 入口校验，未过发 401。Spark 的 `AUTH_EXEMPT_PATHS` 仅豁免 `/__worldfoundry__/*` vendored JS 模块（three/spark 静态库，非用户数据）；两份 viewer HTML 内注入小段 JS，把页面 URL 上的 `?token=` 透传到后续资产请求，保证带 token 打开页面后功能完整。
  - `app.py`（Gradio）：`main()` 取 token（`app.py:2815`）；有 token 时设置 `launch_kwargs["auth"] = lambda _u, p: token_matches(p, auth_token)` + `auth_message`（Gradio 内建认证门禁覆盖页面与 API 路由）。
  - `frontends.print_remote_access`（Gradio/native 前端共用的 URL 打印函数）在打印 public URL 时输出 `bind_security_warning`（`frontends.py:268`）。
- **验证**：
  - `python -m py_compile` 全部改动文件通过（见文末汇总）。
  - auth 模块功能冒烟全过（脚本断言）：loopback 判定 8 例；loopback 免 token；非 loopback 无 env `SystemExit` 且消息含 env 名；有 env 返回 token；Bearer/`?token=` 均可、错 token/缺 token 拒绝；空 token 视为关闭认证；警告文案含 "SECURITY WARNING"。输出 `AUTH_FUNCTIONAL_SMOKE_OK`。
  - 实机测试（本会话早前）：aiohttp 内嵌 app 冒烟——豁免路径 200、API 无 token 401、Bearer/query token 200；真实 `MediaViewerHandler`/`SparkViewerHandler` 起线程服务——无 token 401、`?token=` 与 Bearer 200、vendored JS 豁免路径 200。
- **风险**：默认（127.0.0.1）行为零变化。非 loopback + 未设 token 从"静默暴露"变为"拒绝启动"，属评审要求的行为变更。query-string token 可能进入访问日志，属已知折衷（WebSocket/浏览器资产请求无法自定义 header）；文案引导优先用 SSH 隧道。

### ST-11 (P2) Spark/Media viewer 移除 `Path.cwd()` 授权根

- **改动**（`frontends.py`）：
  - `_spark_allowed_roots`（`frontends.py:867-`）：移除 `Path.cwd().resolve()`，保留三个显式 env 目录（WORKSPACE/ARTIFACT/MODEL_DIR）+ 所选 asset 的父目录，并加注释说明 cwd 故意不作为授权根。
  - Media viewer：`allowed_roots=(asset_path.parent.resolve(),)`（`frontends.py:375`），同样不含 cwd。
- **验证**：实机 handler 测试确认——服务所选 asset 及其父目录内文件仍 200；父目录之外（原先 cwd 可达的仓库文件）404/403。`py_compile` + `import worldfoundry.studio.visualization.backends.frontends` 通过。
- **风险**：依赖"cwd 隐式可读"的非常规用法（如从仓库根启动后直接浏览仓库内任意 splat）不再工作，需显式传 `--asset` 或设置 env 目录；这正是收敛目标。

### ST-6 (P2) WebRTC 半开会话 setup 超时

- **改动**（`world_realtime.py`）：
  - `_ActivePeer` 增加 `setup_task: asyncio.Task | None`（`world_realtime.py:1232`）。
  - `create_answer` 在会话置为 active 后创建 `active.setup_task = asyncio.create_task(self._peer_setup_watchdog(active))`（`world_realtime.py:1907-1908`）。
  - `_peer_setup_watchdog`（`world_realtime.py:2496-2519`）：`await asyncio.sleep(timeout)` 后复查会话状态，`active.closed or active.channel is not None` 则静默退出，否则 `logger.warning` 并 `await self.close_active()`。timeout 复用 `WORLDFOUNDRY_REALTIME_SOCKET_SETUP_TIMEOUT_SECONDS`（默认 15s，与 socket 栈语义一致）。
  - `close_active` 的任务取消集合加入 `setup_task`（`world_realtime.py:1742`），沿用既有 `task is not current_task` 防护，watchdog 自触发 close 时不会自取消死锁。
- **设计说明（与任务描述的偏差）**：任务建议"DataChannel 打开时取消该 task"，实现改为"到期后复查状态"——`on_datachannel` 不做取消，避免"打开与超时同时发生"时取消/触发路径与 `close_active`/`_drain_done` 竞争；channel 已开时 watchdog 到期自然 no-op，语义等价且更简单。代价仅是任务多存活至多 15s（已写入方法 docstring）。并发闭环核对：并发 close 场景下 watchdog 若误入 `close_active`，`_lock` 内 `has_session=False` + `_draining=True` 走 `wait_for_drain` 等待后返回，无双重关闭。
- **验证**：`py_compile` + 真实导入 `world_realtime` 通过（aiohttp/aiortc 本机可用）；close_active/watchdog 并发语义按上述路径逐一走查。
- **风险**：合法但极慢的信令客户端（>15s 才开 DataChannel）会被断开，可用 env 调大；原先此类会话会永久占住单会话槽位。

### ST-8 (P2) workspace_app 关停时清理受管 viewer 子进程

- **改动**（`workspace_app.py`）：
  - 新增 `_stop_all_visualizers()`（`workspace_app.py:423-`）：锁内快照 `VISUALIZER_MANAGED` 键，逐个 `_stop_visualizer(mode)`，单个失败仅 `logger.warning` 不中断其余清理。
  - `create_app` 注册 `app.add_event_handler("shutdown", _stop_all_visualizers)`（`workspace_app.py:2612`，与现装 FastAPI 版本兼容的 on-event 写法）。
  - `atexit.register(_stop_all_visualizers)` 兜底（`workspace_app.py:436`），防 uvicorn 异常退出路径。
  - 幂等性核对：`_stop_visualizer` 在 `_VISUALIZER_LOCK` 内 `pop` 记录，记录不存在直接返回——shutdown 事件与 atexit 双触发时第二次为 no-op（`workspace_app.py:397-398` 注释）。
- **验证**：`py_compile` 通过；`import worldfoundry.studio.workspace_app` 仅因本机缺 `fastapi` 失败（预先存在的依赖缺口，非本次改动）；`add_event_handler`/`atexit` 接线用 rg 复核存在且各注册一次。
- **风险**：低。清理是 best-effort（terminate→kill 沿用 `_stop_visualizer` 既有逻辑）；SIGKILL 级别的宿主退出仍无法拦截，属 atexit 固有边界。

### ST-4 (P2) 上传扩展名白名单 + per-file 大小上限

- **改动**（`world_realtime.py` `serve_realtime_world_frontend` 内 `upload_input`）：
  - 白名单 `upload_allowed_exts = frozenset(IMAGE_EXTS) | frozenset(VIDEO_EXTS)`（`world_realtime.py:2857`），从 `execution.py` 下游解码逻辑使用的常量推断，不另造集合。
  - 上限 `upload_max_bytes = _env_int("WORLDFOUNDRY_UPLOAD_MAX_BYTES", 512MB)`（`world_realtime.py:2858`）。
  - 流式落盘时累计写入量，超限即停（`world_realtime.py:3002-3006`）；扩展名不在白名单、超限均返回 400（`HTTPBadRequest`，消息含允许集/上限与 env 覆盖方式）；任何失败路径 unlink 残留临时文件。
  - 未做内容级解码校验（按任务要求，避免新依赖）。
- **验证**：aiohttp 实机冒烟——合法小图 200 且落盘、非法扩展 400、超限 400 且无残留文件；`py_compile` + 导入通过。
- **风险**：低。原有合法上传（图像/视频常见格式、<512MB）行为不变；aiohttp `client_max_size`（2GB）仍是外层总请求兜底。

### ST-5 (P2) `serve_rerun_frontend` 去掉 `shell=True`

- **改动**（`frontends.py:432`）：`subprocess.Popen(command, shell=True)` → `subprocess.Popen(shlex.split(command))`。模板占位符值先经 `shlex.quote`（`frontends.py:447` 一带既有逻辑）再渲染，"先构造再 split"与原 shell 解析等价。
- **验证**：`py_compile` 通过；对默认 `command_template` 渲染结果做 `shlex.split` 冒烟，得到预期 argv（可执行路径 + `--web-viewer` 等参数，带空格路径保持单元素）。
- **风险**：依赖 shell 特性（管道、`&&`、env 前缀、glob）的自定义 `WORLDFOUNDRY_RERUN_COMMAND` 模板不再工作，需改写为单命令 argv 形式；换取消除注入面。已知折衷，属评审要求方向。

### ST-7 (P2) SETTINGS/VISUALIZER_MANAGED 写路径加锁；单用户假设写入 docstring

- **改动**（`workspace_app.py`）：
  - 模块级 `_SETTINGS_LOCK = threading.Lock()`（`workspace_app.py:119`）：包住 `_load_settings_from_disk`（`:331`）与 `update_settings` 端点的 read-modify-write（`:2652`）。
  - 模块级 `_VISUALIZER_LOCK = threading.RLock()`（`workspace_app.py:276`）：包住 `_cleanup_finished_visualizer`（`:388`）、`_stop_visualizer`（`:398`）、`_launch_visualizer`（核心逻辑收进 `_launch_visualizer_locked`，`:660`）及 `_stop_all_visualizers`。RLock 因 launch 路径内部会调 cleanup/stop。
  - `create_app` docstring 明确"single-user local tool"假设与全局状态共享语义（`workspace_app.py:2599-2608`）。FastAPI 同步端点确认在线程池并发执行（starlette run_in_threadpool），锁必要。
- **验证**：`py_compile` 通过；rg 核对 `SETTINGS.update`/`VISUALIZER_MANAGED` 的全部写点均在锁内。
- **风险**：低。锁粒度小（dict 操作与子进程启停），无跨锁嵌套（`_VISUALIZER_LOCK` 为可重入），不会引入死锁；多用户隔离仍未解决，见 deferred。

### ST-10 (P3) 两处静默吞错改为 logger.warning

- **改动**（仅报告点名的两处）：
  - `world_realtime.py` `ResidentWorldRuntime.reset`（`world_realtime.py:1195-1199`）：`except Exception: pass` → `logger.warning("Realtime runtime reset action failed; continuing session teardown.", exc_info=True)`，原有"继续 teardown"行为不变。
  - `workspace_app.py` `_load_settings_from_disk`（`workspace_app.py:326` 一带）：`OSError/json.JSONDecodeError` 静默回退默认值 → 先 `logger.warning(..., exc_info=True)` 再走原回退。
- **验证**：`py_compile` + 导入通过；rg 确认改动文件内不再有这两处裸 `pass` 吞错（`world_realtime.py:1433` 处的 timing-JSONL warning 为预存代码，未动）。
- **风险**：无行为变化，仅增加日志。

### ST-1 (P2) 路径白名单校验统一到 `serving.path_allowed`

- **语义差异审查（改动前）**：三处实现——`serving/http.py:path_allowed`（resolve 后 `is_relative_to` 逐根比对）、`workspace_app._safe_file_response`（registered-artifact 集合 + 手写 workspace 前缀判断）、`world_realtime.file_response`（手写 resolve+前缀判断）。三者均 resolve symlink 后比对，统一不放宽语义。
- **改动**：
  - `workspace_app._safe_file_response`（`workspace_app.py:2583-2585`）：registered-artifact 白名单（本质是"精确路径注册表"，与目录白名单不同）保留原样；非注册路径的 workspace 目录判断改调 `path_allowed(path, (workspace_root,))`。
  - `world_realtime.file_response`（`world_realtime.py:3022`）：手写判断改调 `path_allowed(path, file_roots)`，`file_roots` 为 allowed_roots+upload_root 去重 resolve 后的元组。
  - 各框架的响应构造、Range 处理与缓存头**保持不变**（FastAPI `FileResponse` 带 `Cache-Control: no-store`；aiohttp `web.FileResponse` 默认头 + 既有 ETag 逻辑；http.server 走 `send_file_response` 自带 Range/immutable 缓存）。缓存头差异记录于此，不强行统一。
  - `frontends.py` 的 spark/media handler 原本就走 `serving.send_file_response` + `path_allowed`，无需改动。
- **验证**：`py_compile` + `world_realtime` 真实导入通过；实机冒烟中 file 端点对白名单内 200 / 白名单外 404 行为符合预期。
- **风险**：低。`path_allowed` 与被替换的手写判断同为 resolve-then-prefix 语义，未放宽；registered-artifact 路径保持精确匹配不受影响。

## 第三轮定点修复（静态分析/横切报告点名的 studio 遗留项）

> 分支 `cursor/studio-remaining-fixes-2f62`。条目编号对应 `plan/code_review/13_static_analysis.md`（SA-*）与 `plan/code_review/12_cross_cutting.md`（XC-*）。每条附回归测试；测试文件位于 `tests/studio/` 与 `tests/studio_visualization/`。

### SA-3 (P2) human_pose.py `draw_mask` 调用不存在的 `alphaMerge` + resize 结果被拼写错误丢弃 — 已修

- **文件**：`worldfoundry/studio/visualization/plugins/perception/human_pose.py`
- **问题确认**（改前）：`backgournd = cv2.resize(...)` 赋给死变量（F841）；`return alphaMerge(...)` 调用全库不存在的符号（F821）——函数一被调用即 NameError；且 `return_rgba` 形参被硬编码 `True` 忽略。
- **修复**：仓库无通用"全图 alpha 合成"工具（`core/io/artifacts.py:overlay_rgba_icon` 是定位/缩放的 icon 贴图语义，不匹配），改为函数内直接实现标准 alpha 合成（numpy 4 行）：mask 按 8-bit alpha 约定（0=背景，255=前景；bool mask 也接受，HxWx1 自动压平），`background` int 路径保留原 `np.ones*255*background` 语义（0=黑/1=白），图像 background 修正拼写后真正被 resize 且参与合成；`return_rgba` 现按形参生效（True 时把 mask 拼为 alpha 通道）。
- **测试**：`tests/studio_visualization/test_human_pose_regressions.py::TestDrawMask` 7 例——黑/白背景合成、半透明线性混合、**背景尺寸不匹配时被 resize（原 bug 回归点）**、HxWx1/bool mask、return_rgba 形状与 alpha 值。

### SA-6 (P2) human_pose.py 裸 `raise` 不在 except 内 — 已修

- **文件**：同上，`draw_aapose_new` 的 `stickwidth_type` 分支（原 865 行）。
- **修复**：`raise` → `raise ValueError(f"Unsupported stickwidth_type {stickwidth_type!r}; expected 'v1' or 'v2'.")`；同函数顺带删除紧邻的重复 `H, W, C = img.shape`（完全相同的相邻两行）。`draw_handpose_new`（95 行起）同一模式缺 else 分支（非法值 → UnboundLocalError），补同样的 ValueError。
- **测试**：`tests/studio_visualization/test_human_pose_regressions.py::TestStickwidthTypeValidation` 6 例——两函数非法值抛 ValueError（match "stickwidth_type"）、v1/v2 正常路径不回归。

### SA-5 (P3) workspace_app.py 循环变量 `field` 遮蔽 `dataclasses.field` — 已修

- **文件**：`worldfoundry/studio/workspace_app.py`
- **修复**：全部 Python 级 `field` 绑定改名 `input_field`（choice 查表处一处改 `choice_field`），共 13 处 for/推导式/赋值绑定；模块级 `from dataclasses import ... field ...` 及 `field(default_factory=dict)` 用法不变；HTML 模板字符串内的 JavaScript `field` 标识符（非 Python 绑定）不动。
- **测试**：`tests/studio/test_workspace_app_field_shadowing.py`——AST 断言全文件无名为 `field` 的 Store 绑定/形参；导入后 `workspace_app.field is dataclasses.field`。

### SA-7 (P2) world_realtime.py fire-and-forget `asyncio.create_task` 未保存引用（RUF006）— 已修（studio 侧）

- **文件**：`worldfoundry/studio/visualization/backends/world_realtime.py`
- **问题确认**（改前）：WebRTC DataChannel `on_close` 回调内 `asyncio.create_task(self.close_active())` 未保存返回 Task——事件循环仅弱引用任务，会话清理可能被 GC 中途吞掉；同文件其余 create_task 均已挂在 `active.*`/局部变量上。
- **修复**：`RealtimePeerManager.__init__` 增加 `self._background_tasks: set[asyncio.Task]`；新增 `_spawn_background_task(coro, name=...)`（官方文档推荐模式：add + done 回调 discard）；`on_close` 改走该 helper。SA-7 点名的另一处 `cli/tui_app.py:839` 属 cli 目录，超出本次 studio 边界，未动。
- **测试**：`tests/studio_visualization/test_world_realtime_background_tasks.py` 3 例——任务在飞行中被强引用、完成后被 discard、异常任务同样被 discard；另有 AST 回归断言全文件无"裸 `asyncio.create_task(...)` 表达式语句"。

### SA-8 (P2) workspace_app 多线程服务内 `subprocess.Popen(preexec_fn=os.setsid)` — 已修

- **文件**：`worldfoundry/studio/workspace_app.py`（visualizer 启动 Popen，原 768 行）
- **背景**：第一轮 ST-8 修的是 shutdown 清理；SA-8 的 `preexec_fn` 线程安全问题（CPython 文档明确多线程程序不安全，此 Popen 跑在 FastAPI 线程池内）当时未覆盖，本轮补上。
- **修复**：`preexec_fn=os.setsid if hasattr(os, "setsid") else None` → `start_new_session=hasattr(os, "setsid")`（等效 setsid、fork 后不执行 Python 代码；子进程仍是新进程组组长，`_terminate_process_group` 的 killpg 语义不变；Windows 上 hasattr 为 False 与原行为一致）。
- **测试**：`tests/studio/test_workspace_app_subprocess_safety.py`——AST 断言全文件无 `preexec_fn` 关键字实参 + `start_new_session` 存在。

### SA-9 (P3, studio 半) execution.py `persisted_preview` 闭包捕获循环变量（B023）— 已修

- **文件**：`worldfoundry/studio/execution.py`（`StudioManager.list_recent_runs` 内闭包）
- **修复**：按报告建议改默认参数绑定：`payload`/`recovered_previews` 以 keyword-only 默认值在 def 时刻绑定，未来 RunRecord 构造若改为延迟/并行执行不会串成最后一次迭代的值。当前行为零变化。SA-9 的另一处（`evaluation/.../model_benchmark_suite.py`）在 evaluation 目录，超出本次边界，未动。
- **测试**：`tests/studio/test_execution_recent_runs_previews.py`——两个 run 各带独立 preview 的 manifest，断言 `list_recent_runs` 返回的 preview 按 run 隔离不串扰。

### XC-2 (P2) hed_annotator.py 裸 `torch.load` — 已在先前提交修复，本轮验证后跳过

- **证据**：`worldfoundry/studio/visualization/plugins/perception/hed_annotator.py:81-83` 现为 `from worldfoundry.core.checkpoint.safe_loading import load_tensor_state_dict` + `load_tensor_state_dict(modelpath, map_location="cpu")`（weights_only 安全路径）；`git log` 显示由 commit `b92eb890`（"refactor(eval): deduplicate runner helpers and harden integrations"）落地。rg 复核 `worldfoundry/studio/` 下（排除 vendored `native/world_explorer/`）无其它裸 `torch.load`。`native/world_explorer/api/lyra_persistent.py:153` 的 `weights_only=False` 属 vendored 子树（ruff exclude 同一口径），不在 XC-2 范围，未动。

### 本轮核查后不修（记录理由）

- **SA-15 第 5 项**（`pixelsplat_full/encoder_visualization/encoder_visualizer_epipolar.py` 死代码删除）：报告要求"与 owner 确认后删除"，且 ST-9 的教训（评审"零引用"结论可能过时）表明删除类改动需 owner 决策，跳过。
- **10_studio.md Deferred 复核**：ST-2（pyproject，按任务安排不动）、ST-9（world.py，`tests/studio_visualization/test_backends_frontends_viser_world.py` 仍实例化 `WorldSession` 等符号，不删）、ST-7 多用户隔离与 ST-1 缓存头统一（设计级，维持 deferred）。

### 第三轮验证汇总

- `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile` 通过：`human_pose.py`、`workspace_app.py`、`world_realtime.py`、`execution.py`。
- 新增回归测试 5 个文件共 20 例全过（本机 CPU，python3.12 + CPU torch）：`test_human_pose_regressions.py` 12、`test_world_realtime_background_tasks.py` 3、`test_workspace_app_field_shadowing.py` 2、`test_workspace_app_subprocess_safety.py` 2、`test_execution_recent_runs_previews.py` 1。
- 相关既有套件回归数字见 PR 描述（`tests/studio/` 与 `tests/studio_visualization/` 全量）。

## Deferred / 未修复

### ST-2 (P3) pyproject `ui` extra 与 studio 实际依赖不符

- 按任务安排跳过：`pyproject.toml` 不在本次修复边界内，已另行安排。

### ST-9 (P2) world.py "死代码"——验证发现真实引用，按规则不删（原样保留）

- **独立验证过程**：rg 全仓（含 `test/`、`tests/`、`docs/`、字符串引用，排除 `plan/`）搜索 `WorldFrontendHandler|WorldFrontendState|WorldSession|_run_model` 及候选专属 helper。
- **发现真实引用**（`tests/studio_visualization/test_backends_frontends_viser_world.py`，测试目录不在本次可改边界内）：
  - `WorldSession`：`:1198,1209,1215,1227,3047` 直接导入并**实例化**、`fields()` 校验字段名。
  - `WorldFrontendState`：`:1235,1249` 导入并 `fields()` 结构校验。
  - `WorldFrontendHandler`：`:1705-1735` 导入并断言 `server_version`、`do_GET/do_HEAD/do_POST`、`_handle_websocket/_read_ws_frame/_send_ws_frame/_send_ws_json/_read_ws_json/_handle_ws_request` 等属性存在。
  - （`test/test_worldfoundry_studio_interactive_controls.py:17` 的 `_uses_state_init` 来自 `worldfoundry.studio.app`，与 world.py 同名符号无关。）
- **结论**：评审报告"约 870 行零引用"结论对当前测试套件已过时；handler 保留则其专属 helper（`_model_import_error_message`、`_record_payload`、`_embed_frame_payload`、`_open_image` 等）均被 handler 内部调用，不存在可安全删除的子集。按任务规则"发现任何真实引用→记录理由，不强删"处理。
- **过程记录**：会话中曾执行带边界断言的删除（3466→2697 行，py_compile 通过），复核 rg 时发现上述测试引用，随即用 `git show HEAD:<path>` 原内容整体还原（还原后 `git diff` 为空、3466 行、`py_compile` 通过、模块可导入）。工作区最终对 world.py **无任何改动**。
- **建议**（供 owner 决策）：若确认该结构性测试文件可同步删除/改写，则死代码删除可另开一次带测试更新的修复；仅在那时一并处理 `_legacy_world_frontend_js` 等未被报告点名的疑似死代码。

### ST-7 附带 deferred：多用户会话隔离

- 全局 `SETTINGS/JOBS/VISUALIZER_MANAGED/MANAGER` 的按会话隔离属设计级重构，改动面大，超出最小修复边界；本次仅消除线程竞态并写明单用户假设。

### ST-1 附带 deferred：响应构造与缓存头的完全统一

- 三栈缓存头策略不同（no-store / aiohttp 默认+ETag / immutable 长缓存），各有其场景语义；仅统一了白名单校验，响应构造统一收益低、回归面大，不做。

## 验证结果汇总

- `PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile` 逐一通过（9 文件）：`serving/auth.py`、`serving/network.py`、`serving/__init__.py`、`serving/http.py`、`visualization/backends/frontends.py`、`workspace_app.py`、`visualization/backends/world_realtime.py`、`app.py`、`visualization/backends/world.py`（还原后）。
- 导入冒烟（`PYTHONPATH=. python -c "import ..."`）：`worldfoundry.studio.serving`、`...backends.frontends`、`...backends.world_realtime`、`...backends.world` 全部 IMPORT_OK；`worldfoundry.studio.workspace_app` 缺 `fastapi`、`worldfoundry.studio.app` 缺 `gradio`（均为本机预存依赖缺口，非语法/名称错误——同文件 `py_compile` 通过）。
- auth 功能冒烟：`AUTH_FUNCTIONAL_SMOKE_OK`（loopback 判定、SystemExit、Bearer/query 校验、警告文案共 18 项断言）。
- 实机冒烟（会话内）：aiohttp middleware + 上传校验（401/200/400/残留清理）、`MediaViewerHandler`/`SparkViewerHandler` 线程服务（401/token 200/豁免路径 200/roots 收敛后 404）。
- rg 复核：无遗留对已删符号的引用（world.py 还原后不存在删除符号问题）；`shell=True` 在 `worldfoundry/studio/` 下 0 命中（仅存注释）；`VISUALIZER_MANAGED`/`SETTINGS` 写点均在锁内。
- `git status`（只读）确认改动集恰为：`app.py`、`serving/__init__.py`、`serving/network.py`、`visualization/backends/frontends.py`、`visualization/backends/world_realtime.py`、`workspace_app.py`（M）+ `serving/auth.py`（新增）；`world.py` 无改动。
