# studio 层评审

> 评审人：infra 代码评审（AI 辅助深度评审）
> 日期：2026-08-14
> 状态：已完成

## 评审范围与方法

- 范围：`worldfoundry/studio/` 全部（约 191 个 py 文件，~2.6 万行顶层 + 2.3 万行子目录）；`native/world_explorer` 只评 Python 绑定/服务接线，不评上游 C++/pybind11 vendored 代码风格。
- 方法：
  - 服务器/会话/流媒体核心全部精读：`serving/`（http/network/telemetry/realtime）、`visualization/backends/world_realtime.py`（WebRTC/WS 会话核心）、`world_realtime_client.py`、`app.py`、`workspace_app.py`、`execution.py`、`native/world_explorer/api/server_*.py` 与 `launcher.py`。
  - 页面/组件类代码（`ui/`、`visualization/plugins/`、`providers/`）抽查。
  - 每条发现均以 `路径:行号` + 代码摘录佐证。
- 严重度定义：P0=损坏/危险；P1=严重设计缺陷；P2=应修复；P3=改进建议。

## 发现（按主题分组）

> 说明：`worldfoundry/studio/serving/realtime/`（有界帧队列、背压策略、输入重采样）与 `world_realtime.py` 的会话生命周期整体是**高质量代码**（单会话互斥、drain、liveness watchdog、GPU 独立线程、显式背压/丢帧策略），下述问题不否定这一整体评价。

### 主题一：Web 框架混用与服务抽象

#### [ST-1] P2 三套并存的服务器栈缺少统一抽象，文件服务实现重复三份且安全语义不一致
- 位置 / 证据：
  - `serving/http.py` 提供了通用原语 `path_allowed()` / `send_file_response()` / `parse_byte_range()`：

```56:73:worldfoundry/studio/serving/http.py
def path_allowed(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    ...
        resolved.relative_to(root_resolved)
        return True
```
  - 但 `workspace_app.py`（FastAPI）完全不用它，自己重写了一份 range + 安全校验：

```2525:2537:worldfoundry/studio/workspace_app.py
def _safe_file_response(path_text: str | None, request: Request | None = None) -> Response:
    ...
    workspace_root = Path(MANAGER.workspace_root).resolve()
    registered = path in _registered_artifact_paths()
    if not registered:
        try:
            path.relative_to(workspace_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="file is outside Studio workspace") from exc
```
  - `world_realtime.py`（aiohttp）又自己实现第三份：

```2929:2936:worldfoundry/studio/visualization/backends/world_realtime.py
    async def file_response(request: Any) -> Any:
        raw = str(request.query.get("path") or "")
        path = Path(raw).expanduser().resolve()
        if not any(path == root or root in path.parents for root in file_roots):
            raise web.HTTPForbidden(reason="Path is outside allowed Studio roots.")
```
- 问题：三个前端栈（Gradio `app.py`、FastAPI `workspace_app.py`、aiohttp `world_realtime.py`、以及 `http.server` 的 `frontends.py`）各自实现文件服务/范围请求/路径白名单，语义与缓存头都不一致（`no-store` vs `private, max-age=86400`）。`serving/` 明明抽出了共享原语却只有 `frontends.py` 在用。
- 影响：安全检查散落多处，容易出现某一处遗漏；维护成本高。
- 建议：让 workspace_app / world_realtime 统一走 `serving.path_allowed` + 一个共享的 range/文件响应实现（可为 aiohttp / starlette 各写薄适配层，但白名单与校验逻辑单一来源）。

#### [ST-2] P3 pyproject `ui` extra 声明的框架与 studio 实际使用不符（flask / textual 冗余）
- 位置 / 证据：
```200:211:pyproject.toml
ui = [
    ...
    "fastapi",
    "flask",
    "flask-socketio",
    "gradio",
    "gradio-imageslider",
    ...
    "textual",
```
  - 但 studio 目录内对 flask / textual **零 import**（全仓 `^from flask` / `^from textual` 仅命中 `worldfoundry/cli/tui_app.py` 与 evaluation runners 的第三方 vendored 代码，不在 studio）。studio 内真实使用：`gradio`（`app.py`）、`fastapi`+`uvicorn`（`workspace_app.py`）、`aiohttp`（`world_realtime.py`）、`http.server`（`frontends.py`、`world.py`）。
- 问题：题述"fastapi+flask+gradio+textual 并存"在 studio 内部并不成立；`ui` extra 把 flask/flask-socketio/textual 也拉进来属冗余声明，会误导读者以为 studio 是四栈混合，并增加安装体积/依赖冲突面。
- 影响：依赖膨胀、认知误导；非功能缺陷。
- 建议：将 `ui` extra 拆分或裁剪，textual 归 CLI extra、flask 若确无 studio 用途则移除；用注释标注每个依赖服务于哪个前端。

### 主题二：安全

#### [ST-3] P1 所有服务器端点无认证，且显式支持绑定 0.0.0.0 / 公网
- 位置 / 证据：
  - bind host 默认 `127.0.0.1`，但可被 `--host` / 多个 env 覆盖为任意值：
```225:233:worldfoundry/studio/visualization/backends/frontends.py
def host_for_frontend(launch_config: StudioLaunchConfig) -> str:
    return (
        launch_config.host
        or env_first("WORLDFOUNDRY_STUDIO_HOST")
        or os.getenv("GRADIO_SERVER_NAME")
        or "127.0.0.1"
    ).strip() or "127.0.0.1"
```
  - `network.py` 还主动为 0.0.0.0 绑定打印"Network URL"，引导公网访问：
```17:21:worldfoundry/studio/serving/network.py
def public_url_for_bind(host: str, port: int) -> str:
    visible_host = get_external_ip() if host in {"", "0.0.0.0", "::"} else host
    return f"http://{visible_host}:{int(port)}"
```
  - 一旦绑到 0.0.0.0，以下端点全部无认证暴露：`POST /api/jobs`（运行任意 catalog 模型推理/评测，`workspace_app.py:2649`）、`POST /api/visualizers/{mode}/launch`（拉起子进程，`:2632`）、`POST /api/settings`（改全局，`:2570`）、`GET /api/artifacts/file`、`/api/runs/{id}/*`、world_realtime 的 `upload/offer/ws/file`。
- 问题：无任何 token/basic-auth/来源校验；安全性完全依赖"只绑 localhost"这一约定，但代码同时提供了公网绑定路径与引导。
- 影响：在共享节点/开发机上一旦按提示绑 0.0.0.0，等价于把"任意模型执行 + 子进程启动 + 工作区文件读取"开放给同网段任何人。
- 建议：对非 loopback 绑定强制要求 token（env 注入），或至少在绑定非 loopback 时打印显著告警并默认拒绝；`launch_config` 增加 `--allow-remote` 显式开关。

#### [ST-4] P2 world_realtime 上传端点缺少类型/大小的有效校验
- 位置 / 证据：
```2912:2927:worldfoundry/studio/visualization/backends/world_realtime.py
    async def upload_input(request: Any) -> Any:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            raise web.HTTPBadRequest(reason="Expected multipart field named 'file'.")
        filename = Path(field.filename or "input.bin").name
        suffix = Path(filename).suffix[:16]
        target = upload_root / f"{uuid.uuid4().hex}{suffix}"
        handle = await asyncio.to_thread(target.open, "wb")
        try:
            while chunk := await field.read_chunk(size=1024 * 1024):
                await asyncio.to_thread(handle.write, chunk)
```
  - `app = web.Application(client_max_size=2 * 1024**3)`（`:2810`）即单请求上限 2GB。
- 问题：无 MIME / 扩展名白名单（后缀直接取自用户文件名），无针对图像/视频的实际大小或内容校验；任意 2GB 文件都会落盘。虽然文件名用 uuid、后缀截断 16 字符避免了路径穿越，但缺类型校验。
- 影响：磁盘占用/DoS；下游 `Image.open` / av 解码非受控输入。
- 建议：限定允许的扩展名与内容类型，设置更小的 per-file 上限，落盘前校验图像/视频可解码。
- 缓解现状：已有 stale 清理（`on_startup` 删 24h 前文件，`:2993-2999`）与 `owned_uploads` 清理（`:2813-2820`）。

#### [ST-5] P2 `serve_rerun_frontend` 使用 `subprocess.Popen(shell=True)`
- 位置 / 证据：
```409:420:worldfoundry/studio/visualization/backends/frontends.py
    command = command_template.format(
        asset=shlex.quote(str(asset_path)),
        host=shlex.quote(host),
        port=port,
        grpc_port=grpc_port,
        ws_port=ws_port,
        model=shlex.quote(entry.model_id),
    )
    ...
    process = subprocess.Popen(command, shell=True)
```
- 问题：`command_template` 来自 `WORLDFOUNDRY_STUDIO_RERUN_COMMAND` env，`asset/host/model` 已 `shlex.quote`、端口为 int，因此注入面受控（仅能设置该 env 的操作者可改模板）。但 `shell=True` 属不必要的风险面。
- 影响：低（受控），但为将来引入用户可控字段留隐患。
- 建议：改用 `shlex.split(command)` + `Popen(list)`，去掉 `shell=True`。

#### [ST-11] P2 Spark 3DGS viewer 以 `Path.cwd()` 为允许根 + 接受任意 `?path`，可读当前工作目录整棵树
- 位置 / 证据：
```803:817:worldfoundry/studio/visualization/backends/frontends.py
def _spark_allowed_roots(asset_path: Path | None) -> tuple[Path, ...]:
    roots: list[Path] = [Path.cwd().resolve()]
    ...
    if asset_path is not None:
        roots.append(asset_path.parent.resolve())
    return tuple(dict.fromkeys(roots))
```
  - handler 接受任意 `?path=` 且只按上述白名单校验：
```555:563:worldfoundry/studio/visualization/backends/frontends.py
        if parsed.path == "/__worldfoundry__/asset":
            query = parse_qs(parsed.query)
            raw_path = (query.get("path") or [""])[0]
            path = Path(unquote(raw_path)).expanduser().resolve()
            if not path.exists() or not path_allowed(path, self.allowed_roots):
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found or not allowed.")
                return
            self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
```
- 问题：Studio 子进程通常以 `cwd=REPO_ROOT` 启动（`workspace_app.py:714`），Spark viewer 的允许根便包含整个仓库工作目录。客户端可 `GET /__worldfoundry__/asset?path=<cwd 下任意文件>` 读取该目录树下任意文件（源码、配置、数据）。
- 影响：结合 [ST-3]（可绑 0.0.0.0），构成对 cwd 整棵树的无认证任意文件读取；即便仅 localhost 也超出"只读点云资产"的应有范围。
- 建议：允许根收敛到显式 workspace/artifact/asset 目录，去掉 `Path.cwd()`；或要求 `?path` 必须位于所选资产父目录内。

#### 其它前端的路径遍历核查（正面）
- `workspace_app._safe_file_response`（工作区根限制 + 已注册 artifact 白名单）、world_realtime `file_response`（resolve 后父目录白名单）均为 resolve 后白名单比较，未见"用户输入直接拼接 open/send_file"的裸路径遍历。`media` viewer 的 `allowed_roots` 同样含 `Path.cwd()`（`frontends.py:366`），但只服务固定 `asset_path`、不接受 `?path`，不可利用（冗余授权，记为观察项）。
- Gradio 前端 `demo.launch` 设了 `allowed_paths` 白名单（asset/workspace/demo/spark 根）与 `queue(default_concurrency_limit=1, max_size=2)`，未开 `auth`（无认证，同 [ST-3]），`share` 由 env 默认关闭（`app.py:2815-2829`）。

### 主题三：实时流生命周期

#### [ST-6] P2 WebRTC 半开连接（建立 PeerConnection 但不开 DataChannel）无超时清理，可长期占用唯一会话
- 位置 / 证据：liveness watchdog 只在 DataChannel 打开后才创建：
```1857:1863:worldfoundry/studio/visualization/backends/world_realtime.py
                active.generation_task = asyncio.create_task(self._generation_worker(active))
                active.liveness_task = asyncio.create_task(self._liveness_watchdog(active))
                active.input_task = asyncio.create_task(self._input_worker(active))
```
  - `create_answer` 在设置 `self._active = active`（`:1840`）后即返回；此时 `self.active` 为真，会拒绝新会话（`:1770-1772`）。若客户端建立连接却始终不开 DataChannel，则既无 `liveness_task` 超时，`connectionstatechange` 也停留在 `connected`（不触发 `failed/closed/disconnected` 的 `close_active`）。
- 问题：单会话服务器可被一个半开 peer 永久占用（轻量 DoS）。
- 影响：其他用户无法建立实时会话，直至该 peer 底层超时/断开。
- 建议：在 `create_answer` 成功后对"未按时打开 DataChannel"启动一个独立超时（复用 `WORLDFOUNDRY_REALTIME_SOCKET_SETUP_TIMEOUT_SECONDS` 语义），到期 `close_active`。

#### 背压与内存（正面）
- 帧队列**有明确上限与背压策略**（`LatestFrameBuffer` 有界 + `latest-interactive`/`ordered-quality` 两档，`media.py:50-150`）；WebSocket 侧 `frame_packets` 亦为有界 `asyncio.Queue`（`world_realtime.py:2288-2294`）。`close_active` 会取消三类 task、关闭队列、`runtime.reset()` 并用 `_drain_done` 串行化（`:1708-1760`）。此项设计良好，非缺陷。

### 主题四：与推理层耦合

#### GPU 不阻塞事件循环（正面）
- world_realtime 将模型放到单线程执行器，`async def` 内以 `run_in_executor` 调度，不阻塞事件循环：
```744:744:worldfoundry/studio/visualization/backends/world_realtime.py
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world-realtime-runtime")
```
- workspace_app 的推理跑在 `StudioJobStore` 线程池（`jobs.py:130`）中的后台 job，FastAPI 同步端点由 Starlette 线程池执行，均不占事件循环。`execution.py` 顶部延迟加载 torch（`torch: Any | None = None`，`:41`），catalog 用 AST 发现，不在导入期拉起重模型。耦合处理得当。

### 主题五：状态管理

#### [ST-7] P2 进程级全局可变状态导致多用户/多标签页串扰
- 位置 / 证据：
```105:105:worldfoundry/studio/workspace_app.py
SETTINGS: dict[str, Any] = dict(DEFAULT_SETTINGS)
```
```2570:2574:worldfoundry/studio/workspace_app.py
    @app.post("/api/settings")
    def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
        SETTINGS.update({key: _coerce_setting_value(key, value) for key, value in payload.values.items()})
        _save_settings_to_disk()
        return dict(SETTINGS)
```
  - `VISUALIZER_MANAGED`（`:260`，每个 mode 全局单例进程）、`JOBS` / `MANAGER`（模块级单例）同理。
- 问题：`SETTINGS` 是全进程共享，任一客户端 `POST /api/settings` 会覆盖所有人的配置；每个 visualizer mode 仅允许一个子进程，第二个用户 launch 会 `_stop_visualizer` 抢占前一个。
- 影响：多人/多标签页同时使用时互相干扰，行为不可预测。
- 建议：明确文档化"单用户本地工具"假设；若需多会话，将 SETTINGS/visualizer 归入按会话/客户端隔离的容器。

### 主题六：子进程 / native 集成

#### [ST-8] P2 workspace_app 未注册 shutdown 钩子清理 viewer 子进程，主进程退出后遗留
- 位置 / 证据：`create_app()`（`workspace_app.py:2550-2882`）只注册路由，无 `app.on_event("shutdown")` / `add_event_handler`；`main()` 直接 `uvicorn.run(create_app())`（`:5419-5421`）。子进程以 `setsid` 脱离进程组启动：
```712:720:worldfoundry/studio/workspace_app.py
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
```
- 问题：`VISUALIZER_MANAGED` 中的 viser/rerun/media/spark 子进程没有随主进程关闭而清理（且因 `setsid` 不会随父进程组一起被信号带走）。对比 world_realtime 有完整 `on_shutdown`（`world_realtime.py:3001-3007`）。
- 影响：反复启动/停止 workspace 会遗留孤儿 viewer 进程，占端口与显存。
- 建议：`create_app` 注册 shutdown 事件，遍历 `VISUALIZER_MANAGED` 调 `_stop_visualizer`；或用 `atexit` 兜底。
- 正面：`_stop_visualizer` / `_terminate_process_group` 本身有 SIGINT→SIGTERM→SIGKILL 升级与超时（`:372-394` / `:442-459`），单次停止逻辑健壮。

#### native/world_explorer 接线核查（正面）
- Python 侧只通过 CLI 子进程接线（`launcher.py:50-66` 组 `python -m ...world_explorer client` 命令；`__main__.py` 子命令 setup/build/client），`api/` 下**无任何网络监听**（全目录搜索 `bind(`/`HTTPServer`/`uvicorn`/`websockets.serve`/`.listen(` 无命中）——`server_base.py` / `server_gen3c.py` / `server_lyra.py` 是运行在 native viewer 进程内的"模型 server"抽象（`asyncio.Task` 调度推理），不开独立端口。
- `backend_loader.backend_class` 用 `import_module` 加载 `module:Class`，但输入来自本地 CLI `--backend` / env（操作者可控），且校验 `issubclass(InferenceModel)`（`backend_loader.py:42-49`），非远程可控，属设计范围。
- pybind 对象生命周期：`__main__._client` 在 `finally: model.cleanup()` 释放（`__main__.py:100-103`）；C++/pybind11、fmt 等 `dependencies/` 为上游 vendored，未评。

### 主题七/八/九/十：错误处理、导入卫生、死代码

#### [ST-9] P2 `world.py` 内约 870 行同步 HTTP+WebSocket+会话实现为死代码
- 位置 / 证据：`serve_world_frontend` 唯一实际入口只转调 aiohttp 版：
```136:145:worldfoundry/studio/visualization/backends/world.py
    try:
        serve_realtime_world_frontend(
            entry=entry,
            launch_config=launch_config,
            manager=manager,
            host=host,
            port=port,
            demo_images=demo_images,
            allowed_roots=_world_allowed_roots(manager, launch_config),
        )
```
  - 而 `WorldFrontendHandler`（`:2518` 起）、`WorldFrontendState`（`:94`）、`WorldSession`、`_run_model`、自研 WebSocket 掩码帧编解码（`:2770-2803`）、会话 prune（`:2957-2966`）等构成一整套并行实现；全 studio 搜索 `WorldFrontendHandler(` / `WorldFrontendState(` 均**无实例化点**，`StudioThreadingHTTPServer` 仅 `frontends.py` 使用。
- 问题：这套实现不可达（dead code），却包含独立的路径白名单、会话管理与手写 WS 协议，仍会被读者误认为在用，并需要跟着维护安全逻辑。
- 影响：约 870 行死代码；维护/审计负担、误导。
- 建议：删除 `WorldFrontendHandler` 及其专属 State/Session/helpers，或如仍为 HTTP 回退保留则接线并加测试；当前状态两者皆非。

#### [ST-10] P3 静默吞错若干处
- 位置 / 证据：
```1188:1190:worldfoundry/studio/visualization/backends/world_realtime.py
            except Exception:
                pass
```
（`ResidentWorldRuntime.reset` 中 runtime reset 失败被完全吞掉）
  - 另 `_load_settings_from_disk` 对损坏配置 `except (OSError, json.JSONDecodeError): return`（`workspace_app.py:306`）静默回退默认值，用户无感知。
- 问题：关键清理/加载失败无日志，排障困难。
- 建议：至少 `logger.warning(..., exc_info=True)` 记录后再吞。

#### 可选依赖降级（正面）
- 实时依赖 aiortc/av/aiohttp 缺失时给出安装 `worldfoundry[studio_realtime]` 的清晰 `SystemExit`（`world_realtime.py:2704-2719`）；unified 前端对缺失 UI 依赖同样有引导（`cli.py:49-61`）；viser 用 `importlib.util` 探测可用性再降级（`viser.py:6,40`）。

#### 宽 except 回退的整体倾向（观察）
- 全 studio（排除 vendored `dependencies/`）**无裸 `except:`**（均为 `except Exception`/具体异常），这是加分项。但 `except Exception:` 后直接 `return None/默认值/continue` 的模式非常密集（env 解析、AST 解析、文件探测、best-effort 清理等数十处，如 `execution.py`、`conda_dispatch.py`、`app.py` 回调）。多数属可接受的 best-effort，但整体倾向于"吞掉非预期异常并回退"，会掩盖真实 bug。建议对"清理/加载/回调"这三类关键路径改为记录后再回退。

## 汇总

### 严重度统计

| 严重度 | 数量 | 条目 |
| --- | --- | --- |
| P0 | 0 | — |
| P1 | 1 | ST-3 |
| P2 | 8 | ST-1, ST-4, ST-5, ST-6, ST-7, ST-8, ST-9, ST-11 |
| P3 | 2 | ST-2, ST-10 |
| 合计 | 11 | — |

> 另有多处正面结论（实时流生命周期/背压、GPU 不阻塞事件循环、native 无网络监听、无裸 except、依赖降级、Gradio allowed_paths），已在各主题内标注，不计入缺陷统计。

### Top 5 问题

1. **[ST-3 · P1] 端点全程无认证且显式支持绑定 0.0.0.0/公网**——绑非 loopback 后，`POST /api/jobs`（任意模型推理/评测）、`/api/visualizers/*/launch`（拉子进程）、文件读取端点全部无认证暴露，`network.py` 还主动引导公网 URL。是本层最高优先级风险。
2. **[ST-9 · P2] `world.py` 约 870 行同步 HTTP+WebSocket+会话实现是死代码**——`serve_world_frontend` 只走 aiohttp 版，`WorldFrontendHandler`/`WorldFrontendState`/`WorldSession` 全无实例化点，却携带独立的路径白名单与手写 WS 协议，误导且增加审计面。
3. **[ST-11 · P2] Spark viewer 以 `Path.cwd()` 为允许根 + 任意 `?path`**——Studio 以 `cwd=REPO_ROOT` 启动时，等于把仓库整棵树开放为可下载文件，叠加 ST-3 即无认证任意文件读取。
4. **[ST-6 · P2] WebRTC 半开连接无超时清理**——建立 PeerConnection 但不开 DataChannel 时无 liveness watchdog，可用一个连接永久占用唯一实时会话（轻量 DoS）。
5. **[ST-1 · P2] 三套前端各自实现文件服务/范围/白名单，安全语义不一致**——`serving/http.py` 已抽出共享原语却仅 `frontends.py` 使用，workspace_app、world_realtime 各写一份，易漏校验、难维护。（并列关注 [ST-7] 子进程无 shutdown 清理导致孤儿 viewer。）
