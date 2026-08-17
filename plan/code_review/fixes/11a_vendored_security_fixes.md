# vendored 集成安全修复日志（VI-22 + VI-23 base_models 子集）

> 修复人：vendored 安全修复 agent；日期：2026-08-14
> 对应评审报告：`plan/code_review/11_vendored_integration.md`（主题十：[VI-22]、[VI-23]）
> 约束：只改 7 个指定文件（dust3r/mast3r/cut3r 的 model.py、splatt3r worldfoundry_runtime.py、unik3d.py、unidepthv2.py、wan/resident.py）；另一 agent 并发修其他模块，不得越界；编辑前逐一 `stat -c '%y'` 核对 mtime（7 个文件均为 2026-07-21/22，无并发修改冲突）；无 GPU/真实 checkpoint 端到端条件；pypi 不可用（环境有 torch 2.7 + safetensors，无 huggingface_hub/transformers，不装新依赖）。
> 验证手段：`python3 -m py_compile` 全部 7 文件；/tmp 下一次性脚本对三套白名单解析器跑 accept/reject 用例（提取已交付源码中的 helper 执行，覆盖 `inf`/`-inf`/`nan`、真实 ckpt 字符串格式、两条 landscape_only 手术分支、注入类字符串拒绝）；用合成 argparse.Namespace checkpoint 实测 `load_torch_checkpoint(..., allow_unsafe_pickle_fallback=True)` 在 torch 2.7 下的回退语义；`rg` 全仓核查被改函数无外部调用方、7 文件内无残留 `weights_only=False`/`eval(`。

## 已修复

### [VI-22] P1 DUSt3R/MASt3R 加载器 `eval(checkpoint 字符串)` 任意代码执行
- 文件：
  - `worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r/model.py`
  - `worldfoundry/base_models/three_dimensions/general_3d/mast3r/mast3r/model.py`
- 改动：
  1. **消除 eval（不可妥协项）**：`net = eval(args)` 改为白名单构造分发 `_instantiate_model_from_checkpoint_args`——用 `ast.parse(args, mode="eval")` 解析，要求表达式恰为一个 `ast.Call` 且 callee 是白名单内的裸类名（dust3r：`{AsymmetricCroCo3DStereo}`；mast3r：`{AsymmetricMASt3R, AsymmetricCroCo3DStereo}`，后者保留原 eval 作用域内两类均可构造的语义）；仅接受关键字实参，每个值经 `_CheckpointLiteralNames`（把裸名 `inf`/`nan` 重写为常量，`-inf` 由 literal_eval 对数字的一元负号支持覆盖）后走 `ast.literal_eval`；位置实参、`**kwargs` 展开、非字面量值、未知类名均抛带 checkpoint 字段名（`'args.model'`）与违规关键字名的 `ValueError`。
  2. **保留上游字符串手术**：`.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")`（注意它同时改写字符串字面量 `patch_embed_cls='ManyAR_PatchEmbed'` 的值——上游既有行为，原样保留）、`landscape_only` 缺失时追加 `, landscape_only=False)` / 存在时去空格替换 `True→False`、`assert "landscape_only=False" in args`，全部不动。
  3. **torch.load 改走集中安全加载器**：`from worldfoundry.core.model_loading.file import load_torch_checkpoint`（函数内惰性 import，避免 vendored 顶层模块 import 时耦合宿主包），调用 `load_torch_checkpoint(model_path, map_location='cpu', allow_unsafe_pickle_fallback=True)`——先试 `weights_only=True`，仅在 `pickle.UnpicklingError("Weights only load failed...")` 时显式回退 unsafe pickle（真实 DUSt3R/MASt3R ckpt 的 `'args'` 是 argparse.Namespace，必然走回退，功能不变）。dust3r 原有的 `except TypeError` 旧版 torch 兼容分支随之删除（集中加载器本身不支持无 `weights_only` 形参的远古 torch，与仓库基线一致）。
  4. 按仓库惯例在每处改动上方加 `# Modified by WorldFoundry:` 注释并引用报告条目；上游 Naver 版权头保持原样。
- 额外修复（顺带）：mast3r 原来是**裸 `torch.load(model_path, map_location='cpu')` 不带 weights_only**——在 torch>=2.6 默认 `weights_only=True` 下加载真实 MASt3R ckpt（含 Namespace）本来就会直接失败；改走集中加载器的显式回退后恢复可用。
- 验证：
  - `py_compile` 通过。
  - /tmp 解析器用例（对已交付源码提取执行）：接受真实上游格式 `AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, ..., patch_embed_cls='ManyAR_PatchEmbed')` 与 `AsymmetricMASt3R(..., two_confs=True)`（经两条 landscape 手术分支后解析，`-inf/inf` 正确转为 float，ManyAR 替换后 kwargs 值为 `'PatchEmbedDust3R'`）；拒绝 `__import__('os').system('id')`、`os.system('id')`、未知类、位置实参、非字面量 kwarg（`open(...)`/`os.getcwd()`）、`**{...}` 展开、attribute call、非 Call 表达式、嵌套 call kwarg——全部以 ValueError 拒绝且报错含字段/关键字名。
  - torch 2.7 实测：Namespace ckpt 在 `weights_only=True` 下抛 `_pickle.UnpicklingError`、消息含 "Weights only load failed"，`load_torch_checkpoint(..., allow_unsafe_pickle_fallback=True)` 回退成功且纯 state_dict 文件留在安全路径。
  - `rg`：`load_model` 仅被各自模块内 `from_pretrained` 调用（`from dust3r.model import` 处只 import 类），签名未变，外部调用方（vmem preprocessor、geometry_priors、splatt3r_runtime）不受影响。
- 风险：
  - 若存在非常规第三方微调 ckpt，其 `args.model` 字符串包含白名单类之外的表达式（如引用其他类、算术表达式），加载将由"静默执行"变为显式 ValueError——这是修复的目标行为，报错信息已指明违规字段便于排查。
  - `allow_unsafe_pickle_fallback=True` 对恶意文件不构成实质防线（攻击者可故意让 weights-only 失败），真正的收益是消除 eval、收敛到集中策略点；报告对此定位一致（"仅对确知可信的内部权重显式开 fallback"）。

### [VI-23] P2 base_models 子集：绕过集中加载器的 `torch.load(..., weights_only=False)`（5 处）
- `worldfoundry/base_models/three_dimensions/point_clouds/cut3r/model.py:80`
  - 改动：`load_model` 的 `torch.load(..., weights_only=False)` 改走 `load_torch_checkpoint(..., map_location="cpu", allow_unsafe_pickle_fallback=True)`（ckpt 含 argparse.Namespace，确需 pickle 对象，同 VI-22 第 3 点，惰性 import + 注释）。
  - **顺带消除同函数内同型 eval（超出 VI-23 字面范围，但与 VI-22 同类同函数）**：cut3r 的 `net = eval(args)` 与 dust3r 完全同源，且其字符串是嵌套调用 `ARCroco3DStereo(ARCroco3DStereoConfig(...))`（`args[:-2] + ", landscape_only=False))"` 手术印证）；白名单解析器为此做了递归版 `_instantiate_checkpoint_call`：实参值允许"白名单类的嵌套构造调用"或字面量，白名单 `{ARCroco3DStereo, ARCroco3DStereoConfig}`，其余一律拒绝。只修 torch.load 而留着一行之隔的 eval 会让该文件的"安全修复"形同虚设，故一并处理并在此记录。
  - 验证：py_compile；/tmp 用例接受真实嵌套格式（外层 1 个位置实参=内层 config 实例、`landscape_only=False` 注入内层 kwargs、`-inf/inf` 正确），拒绝 `ARCroco3DStereo(os.system('id'))`、`ARCroco3DStereoConfig(x=__import__('os'))`、`**{...}` 等；`rg` 确认 `load_model` 仅被本文件 `ARCroco3DStereo.from_pretrained` 调用。
- `worldfoundry/base_models/three_dimensions/general_3d/splatt3r/worldfoundry_runtime.py:146`
  - 改动：`_ensure_model` 内 `torch.load(..., weights_only=False)` 改为 `load_torch_checkpoint(self._resolve_checkpoint(), map_location="cpu", allow_unsafe_pickle_fallback=True)`（Lightning ckpt 的 `hyper_parameters` 载荷含 pickled 配置对象，确需回退；import 放在 `_ensure_model` 内与既有 `import torch` 同位，维持模块的惰性重依赖设计）。仓库自写 wrapper，不加 vendored 标记，注释说明信任边界。
  - 验证：py_compile；与 pi3 `loger_representation.py:144`/`infinite_vggt_representation.py:83` 的既有集中加载器用法同构（它们传 `weights_only=False`，本处用更严的"先安全后显式回退"）。
- `worldfoundry/base_models/three_dimensions/depth/unik3d/unik3d.py:366`、`worldfoundry/base_models/three_dimensions/depth/unidepth/models/unidepthv2/unidepthv2.py:341`
  - 改动：两处 `load_pretrained` 直接改 `weights_only=True`——加载结果仅被当作 state_dict 消费（可选嵌套在 `'model'` 键下，随后即 `load_state_dict`），不需要 pickle 对象；不向 vendored 树引入宿主包 import。加 `# Modified by WorldFoundry:` 标记（两文件均为 Luigi Piccinelli 上游 vendored 代码，版权头未动）。
  - 验证：py_compile；`rg` 全仓无这两个方法的调用方（dvlt 的同名 `load_pretrained` 是无关类），属上游兼容入口，风险面最小。
  - 风险：若有人用它加载"完整训练 checkpoint"（除 `'model'` 外还含非张量 pickled 配置对象），将显式失败并提示 weights_only——属预期收紧；此时应改用官方 `from_pretrained`（safetensors）或集中加载器。
- `worldfoundry/base_models/diffusion_model/models/encoders/wan/resident.py:38`
  - 改动：UMT5 文本编码器权重 `models_t5_umt5-xxl-enc-bf16.pth` 是纯张量 state dict 且直接喂 `load_state_dict`，改 `weights_only=True` 并注释。仓库自写 wrapper，不加 vendored 标记。
  - 验证：py_compile；调用方 `WanTextEncoder.__init__` 契约不变。

## Deferred（有意不做及原因）

1. **monst3r / stable_virtual_camera 内的同型 `eval(args)` 副本未修**：`worldfoundry/base_models/three_dimensions/general_3d/monst3r/dust3r/model.py:47` 与 `.../stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r/dust3r/model.py:57` 是 dust3r loader 的两份 vendored 副本，携带同样的 `eval(ckpt['args'].model)`（rg 普查确认）。不在本次允许编辑的文件清单内（并发 agent 约束），且属 [VI-8~11] 副本治理范畴；修法可直接照搬本次 dust3r 的白名单解析器。**→ 已在续跑轮完成，见下方"续跑"节。**
2. **representations 层的 4 处裸 `torch.load(..., weights_only=False)` 未修**（depth_anything v1/v2、flash_world、lingbot_map，见报告 [VI-23] 位置清单）：任务范围限定为 VI-23 的 base_models 子集，representations/ 不在允许清单内。
3. **未把 unik3d/unidepthv2 接到集中加载器**：两处选择了任务给出的方案 A（`weights_only=True`），理由：消费方纯 state_dict、无仓内调用方、避免向 vendored 树增加宿主包依赖；如后续要统一收敛到 `load_torch_checkpoint`，是一行等价替换。
4. **未做真实 checkpoint 端到端加载验证**：环境无 huggingface_hub/transformers（cut3r 的 model.py 顶层 import transformers，dust3r/mast3r 顶层 import huggingface_hub，模块本体在本环境不可 import），也无已下载的 DUSt3R/MASt3R/CUT3R/Splatt3R 权重。已用"提取已交付源码 helper + 真实格式字符串"与"合成 Namespace ckpt 实测集中加载器回退"两条路径覆盖新增逻辑；建议在具备完整依赖的运行环境跑一次 `AsymmetricCroCo3DStereo.from_pretrained(<本地 .pth>)` 冒烟。
5. **报告建议的 CI lint（wrapper 层禁裸 `torch.load`）未加**：涉及 CI/工具链配置，超出本次文件白名单。

## 续跑（2026-08-14 第二轮）：两份 sibling 副本的 eval 消除

> 范围：仅上面 Deferred 第 1 条的两个文件 + 本日志追加；编辑前 `stat -c '%y'` 复核（两文件 mtime 均为 2026-07-21，无并发修改，未跳过）。

### [VI-22 续] monst3r fork 副本 `eval(args)` 消除
- 文件：`worldfoundry/base_models/three_dimensions/general_3d/monst3r/dust3r/model.py`
- fork 差异核对（VI-8 提示该文件为 MonST3R 5 个分叉文件之一，逐行对照已修的 canonical dust3r）：本文件相对 canonical 的分叉点为 `from third_party.raft import load_RAFT` import、`repo_url` 指向 monst3r、`torch.load` 无 `weights_only` 形参且无 `except TypeError` 兼容分支；`load_model` 的字符串手术与 canonical 完全一致（`args[:-1] + ', landscape_only=False)'` 单层格式），模块内唯一可合法构造的模型类仍只有 `AsymmetricCroCo3DStereo`（MonST3R ckpt 的 `args.model` 即构造它，常带 `patch_embed_cls='ManyAR_PatchEmbed'`）。fork 特有行为（RAFT import、类体内所有方法）一律未动。
- 改动：与 canonical dust3r 完全同型——`eval(args)` → 严格白名单解析器（仅 `{AsymmetricCroCo3DStereo}`、仅关键字实参、值仅字面量、`inf`/`-inf`/`nan` 裸名映射）；裸 `torch.load(model_path, map_location='cpu')` → `load_torch_checkpoint(..., allow_unsafe_pickle_fallback=True)`（惰性 import；原裸调在 torch>=2.6 默认 weights_only=True 下加载含 Namespace 的真实 ckpt 本就会失败，此改动同时恢复可用性，与第一轮 mast3r 情况相同）；`# Modified by WorldFoundry:` 标记；上游 Naver 版权头未动。
- 解析器 helper 按任务要求**逐副本复制、保持自包含**（不跨副本 import：三棵 dust3r 副本靠 sys.path 时序解析同名顶层包，跨副本 import 恰是报告主题八批评的隔离脆弱点）。

### [VI-22 续] stable_virtual_camera third_party 旧版副本 `eval(args)` 消除
- 文件：`worldfoundry/base_models/three_dimensions/general_3d/stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r/dust3r/model.py`
- 版本差异核对：这是更旧的 dust3r 版本（`set_freeze` 无 `encoder_and_decoder` 项、`from_pretrained` HF 分支带 try/except、双引号代码风格），但 `load_model` 与 canonical 同构：同一套 landscape 手术、裸 `torch.load(model_path, map_location="cpu")`、唯一模型类 `AsymmetricCroCo3DStereo`。类体与其余逻辑未动。
- 改动：同上（白名单解析器 + 集中加载器显式回退 + 标记注释），字符串风格跟随本文件的双引号惯例。
- 验证（两文件共同）：
  - `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile` 通过（不在仓内留 pyc）。
  - /tmp 提取式解析器用例（`/tmp/wf_vi22_siblings_test.py`）：接受 MonST3R 真实格式（`patch_embed_cls='ManyAR_PatchEmbed'` 经手术替换为 `'PatchEmbedDust3R'`、`-inf/inf` 转 float、手术分支 1）与旧版含 `landscape_only=True` 格式（手术分支 2）；拒绝 `__import__('os').system('id')`、`os.system('id')`、未知类、位置实参、非字面量 kwarg、`**{...}` 展开、非 Call 表达式——两文件 2×10 用例全过。
  - `rg` 复核：全仓 `net = eval(args)` 活代码清零（5 处命中均为 "Modified by WorldFoundry" 注释行）；两文件的 `load_model` 签名未变，仅被各自模块内 `from_pretrained` 调用。
- 风险：与第一轮 dust3r/mast3r 条目相同（非常规 ckpt 字符串由静默执行变显式 ValueError；fallback 不构成对恶意文件的防线，收益在 eval 消除与策略收敛）。
