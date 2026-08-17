# vendored 代码集成卫生评审（base_models/synthesis/representations）

> 评审对象：`worldfoundry/base_models/`（3240 py）、`worldfoundry/synthesis/`（2369 py）、`worldfoundry/representations/`（40 py），合计 5649 个 vendored/半 vendored Python 文件。
> 评审提交：`077fe858c72c6cd3c0bc81e63f0d5d69653caa64`（feat: update training, runtimes, tests, and docs）
> 评审日期：2026-08-14。评审人视角：资深 infra 工程师，关注"vendoring 集成卫生"而非上游代码风格。

## 评审范围与方法

本报告不评审上游模型代码本身的质量，只评审仓库对近 6000 个 vendored 文件的**管理方式**，共 10 个检查点：

| # | 检查点 | 方法 |
|---|--------|------|
| 1 | sys.path 操纵 | `rg 'sys.path.(insert|append)'` 全仓分布 + 逐个判断是否 import 时执行 |
| 2 | import 副作用 | 审读各级 `__init__.py`，检查顶层重依赖与懒加载（`__getattr__`）机制 |
| 3 | vendored 副本重复 | 目录名聚类 + 文件级 diff/hash 对比（vggt×3、depth_anything 多版本等） |
| 4 | fork 漂移管理 | 抽查 20 个 vendored 子目录的 UPSTREAM/LICENSE/THIRD_PARTY 覆盖率 |
| 5 | 本地补丁可追踪性 | `git log --follow` 抽查 vendored 目录的提交历史 |
| 6 | monkey-patching | `rg` 扫运行时属性覆写模式 |
| 7 | 打包正确性 | pyproject.toml exclude 列表逐条与实际目录比对 + package-data 审计 |
| 8 | 命名空间卫生 | 汇总 sys.path hack 暴露的顶层模块名，找冲突 |
| 9 | 集成接口 | 审读 `base_models/capabilities.py` 与各子树 wrapper 层 |
| 10 | 安全 | 只查 wrapper 实际调用路径上的 `torch.load(weights_only=False)`/pickle/eval |

所有证据格式：`路径:行号` + 摘录/统计。验证脚本置于 /tmp，不触碰仓库源码。

## 发现（按主题分组）

### 主题一：sys.path 操纵（检查点 1）

**总量与分布**：`rg 'sys\.path\.(insert|append)'` 在 `worldfoundry/` 包内共 **149 处**（排除 `__pycache__`）。AST 分类（脚本 `/tmp/classify_syspath.py`）：**74 处在模块顶层**（import 即执行，其中 1 处是注释，实际 73 处生效），75 处在函数内。按二级目录分布：

| 目录 | 命中数 |
|---|---|
| evaluation/tasks | 55 |
| synthesis/visual_generation | 49 |
| base_models/three_dimensions | 17 |
| base_models/llm_mllm_core | 13 |
| base_models/perception_core | 12 |
| representations / pipelines / evaluation/utils.py | 各 1 |

#### [VI-1] P1 核心模块 `evaluation/utils.py` 在 import 时顶层插入 REPO_ROOT，全局暴露仓库根

- 位置：`worldfoundry/evaluation/utils.py:208-210`
- 证据：

```python
# Side effect: benchmarks and model-registry discovery rely on repo-root imports.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- 问题：这不是 vendored 脚本，而是评测框架的**核心工具模块**。任何 `import worldfoundry.evaluation.utils`（几乎所有评测路径都会触发）都会把仓库根目录插到 `sys.path[0]`。仓库根下有 `test/`、`tests/`、`scripts/`、`dataset/`、`configs/`、`thirdparty/` 等目录，全部变成可 import 的顶层包，且**优先级高于 site-packages**——若第三方依赖恰好有同名顶层模块（如 `datasets` vs `dataset`、`test` 与 CPython 自带 test 包），解析结果被静默改写。注释自己承认这是 side effect。
- 影响：进程级全局污染；任何嵌入 worldfoundry 的宿主进程（服务、notebook）都被动接受该路径注入。
- 建议：改为显式包内 import（`worldfoundry.*` 已可达所有代码）；benchmark 发现逻辑改用 `importlib.import_module` + 包相对路径，删除该顶层插入。

#### [VI-2] P1 74 处顶层 sys.path 修改在 import 时执行，其中"remove+insert(0) 强制置顶"模式导致顺序敏感

- 位置（代表性）：
  - `worldfoundry/base_models/three_dimensions/general_3d/dust3r/__init__.py:25`（`IMPORT_PATHS = ensure_import_paths()` 在包 `__init__` 顶层执行）
  - `worldfoundry/base_models/three_dimensions/general_3d/mast3r/__init__.py:35`（`dust3r = reexport_dust3r()` 顶层执行，连带触发 dust3r 插入）
  - `worldfoundry/synthesis/visual_generation/vmem/runtime_env.py:203-216`（`_prepend_sys_path`）
  - `worldfoundry/base_models/llm_mllm_core/mllm/open_flamingo/__init__.py:49-57`
- 证据（dust3r `__init__.py:14-25`）：

```python
def ensure_import_paths() -> tuple[str, str]:
    paths = (str(SOURCE_ROOT), str(CROCO_ROOT))
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        if path not in sys.path:
            sys.path.insert(0, path)
    return paths

IMPORT_PATHS = ensure_import_paths()
```

- 问题：`remove + insert(0)` 不是幂等追加而是**强制抢占**：每次 import 一个新的 wrapper，`sys.path[0]` 的归属就换一次。当多个 vendored 树暴露同名顶层包（见 [VI-14] `utils3d`/`dust3r`/`croco` 冲突）时，`import dust3r` 的解析结果取决于**哪个 wrapper 最后被 import**；而 `sys.modules` 缓存又让第一个完成的 import 永久生效——两种"谁赢"规则叠加，行为对 import 顺序高度敏感且难以复现。多线程下 `remove/insert` 非原子，存在竞态。
- 影响：同进程串多个模型（pipeline 场景是本仓库核心卖点）时，3D 组件解析到错误副本的概率随模型数上升；错误表现为 `isinstance` 失败、属性缺失等远端症状，极难归因。
- 建议：vendored 包一律通过包内相对 import 使用（上游代码 `import dust3r` 的地方，vendor 时批量改写为绝对包路径，或用 `sys.modules["dust3r"] = <包模块>` 的显式别名注册一次并断言冲突），淘汰 path 抢占。

#### [VI-3] P2 跨树相对路径 reach：synthesis 脚本按目录层级硬编码引用 base_models

- 位置：
  - `worldfoundry/synthesis/visual_generation/wonderjourney/wonderjourney_runtime/run.py:19`
  - `worldfoundry/synthesis/visual_generation/wonderworld/wonderworld_runtime/util/midas_utils.py:7`
- 证据：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "base_models" / "three_dimensions" / "depth"))
from midas.model_loader import load_model
```

- 问题：用 `parents[4]`/`parents[5]` 这种"数目录层级"的方式从 synthesis 树伸手到 base_models 树，把两棵 vendored 树的**磁盘相对位置**变成了隐式契约。任何一侧目录改名/移动（本仓库频繁重组，见 [VI-16] exclude 规则失配）都会静默断链；且暴露的顶层名 `midas` 是全局的。
- 影响：目录重构即断；wheel 安装后若 exclude 规则剔除了 depth 树，运行时才发现 ImportError。
- 建议：改为 `from worldfoundry.base_models.three_dimensions.depth.midas.model_loader import load_model`（包内绝对 import），不再依赖磁盘布局。

#### [VI-4] P2 死路径与 CWD 相对路径：占位符路径、`sys.path.append('.')` 等已损坏或环境敏感的插入

- 位置与证据：
  - `worldfoundry/evaluation/tasks/execution/runners/stevo_bench/runtime/stevo_bench/generation/world_models/generate_lingbot_poses.py:16`：`sys.path.insert(0, '<path_to_your_HY-WorldPlay_repo>/hyvideo')` —— 字面占位符，永远不可能存在，import 该模块即注入一条死路径（后续 `import hyvideo` 直接失败）。
  - `worldfoundry/synthesis/visual_generation/ac3d/ac3d_runtime/inference/cli_demo_camera.py:3`：`sys.path.append('..')` —— 解析结果取决于进程 CWD。
  - `worldfoundry/synthesis/visual_generation/open_sora_plan/open_sora_plan_runtime/opensora/models/frame_interpolation/interpolation.py:14`、`.../sample/rec_image.py:2`：`sys.path.append('.')`。
  - `worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/camera_tracking_scripts/test_demo.py:28`：`sys.path.append("base/droid_slam")`——CWD 相对；目标目录 `mega_sam_runtime/base/droid_slam/` 存在，但只有进程 CWD 恰好是 `mega_sam_runtime/` 时才可解析（顺带暴露：`slam/droid_slam/` 与 `slam/mega_sam_runtime/base/droid_slam/` 是两份 DROID-SLAM 副本，见主题三）。
- 问题：这些是上游脚本原样带入的路径 hack，在本仓库目录布局下已失效或依赖特定 CWD。它们大多在"脚本模式"下才执行，但只要被误 import（如批量 import 做冒烟测试、文档工具收集），就会注入垃圾路径。
- 影响：低概率但难排查的 import 异常；同时说明 vendor 时未做"路径 hack 清扫"这道工序。
- 建议：vendor 清单中加入一道检查：对 `sys.path.append('.'|'..'|相对路径|占位符)` 全量清理或改为显式失败。

#### [VI-5] P3 好实践与坏实践并存：已有集中式 helper 与防冲突断言，但未推广

- 位置与证据：
  - 好：`worldfoundry/evaluation/tasks/metrics/_shared/imports.py:9-13` 提供集中式 `prepend_import_path`（幂等，不强制抢占）；`open_flamingo/__init__.py:26-46` 的 `_assert_top_level_package_not_shadowed` 在暴露顶层名前检测已有同名模块并显式报错——这是全仓库唯一一处 shadow 检测。
  - 坏：其余 140+ 处各自手写，风格从 `append` 到 `insert(0)` 到 `remove+insert(0)` 不一，绝大多数无冲突检测。
- 问题：仓库已经发明了正确的轮子（集中 helper + shadow 断言），但没有作为强制规范推广到三棵 vendored 树的 wrapper 层。
- 影响：新增 vendored 模型时大概率继续复制粘贴坏模式。
- 建议：把 `prepend_import_path` + shadow 断言提升为 `worldfoundry.core` 级公共设施，wrapper 层强制走该入口；CI 加 lint 规则禁止 vendored wrapper 内裸写 `sys.path.insert`。

### 主题二：import 副作用与懒加载（检查点 2）

**实测**（脚本 `/tmp/test_import_side_effects.py`，子进程隔离测量）：

| import 目标 | 耗时 | 拉起 torch/transformers/diffusers/numpy/cv2 | sys.path 增量 |
|---|---|---|---|
| `worldfoundry` | 0.005s | 无 | 0 |
| `worldfoundry.base_models` | 0.008s | 无 | 0 |
| `worldfoundry.base_models.diffusion_model` | 0.010s | 无 | 0 |
| `worldfoundry.synthesis` / `.visual_generation` / `.action_generation` | ≤0.018s | 无 | 0 |
| `worldfoundry.representations` | 0.015s | 无 | 0 |
| `worldfoundry.evaluation` | 0.007s | 无 | 0 |
| `...general_3d.dust3r` | 0.040s | 无 | **+2 条** |
| `...general_3d.mast3r` | 0.032s | 无 | **+3 条** |

#### [VI-6] P3（正面确认）聚合层 `__init__.py` 统一采用 `__getattr__` 懒加载，`import worldfoundry.base_models` 不会加载一切

- 位置：`worldfoundry/__init__.py`（纯 docstring）、`worldfoundry/base_models/__init__.py:3-9`（仅 `__all__`）、`worldfoundry/base_models/diffusion_model/__init__.py:50-57`、`worldfoundry/synthesis/__init__.py:27-60`、`worldfoundry/synthesis/visual_generation/__init__.py:58-64`（均为 `_EXPORTS` 表 + `__getattr__` + `__dir__` 模式）。
- 证据：见上表实测，全部聚合 import 在 20ms 内完成、零重依赖。
- 结论：这是本仓库 vendoring 集成做得最好的一环，`_EXPORTS` 懒加载模式统一且有效。担心的"import 即加载 3000 文件"不成立。唯一例外是 dust3r/mast3r 这类 wrapper 包 `__init__` 顶层执行 `ensure_import_paths()` 产生 sys.path 副作用（已在 [VI-2] 记录，实测确认 +2/+3 条路径）。
- 建议：保持；可在 CI 加"聚合 import 不得加载 torch、不得增 sys.path"的守护测试，防回归。

#### [VI-7] P2 `perception_core`、`llm_mllm_core` 等 436 个目录无 `__init__.py`，依赖隐式命名空间包

- 位置：`worldfoundry/base_models/perception_core/`（无 `__init__.py`）、`worldfoundry/base_models/llm_mllm_core/`（无 `__init__.py`）；统计：三棵树内 1545 个含代码目录中 **436 个（28%）缺 `__init__.py`**。
- 问题：同一棵树内常规包与 PEP 420 命名空间包混用。`base_models/__init__.py` 的 `__all__` 列出了 `perception_core`/`llm_mllm_core`，但它们实际是命名空间包——运行时可用，代价是：(a) 打包必须依赖 `find_namespace_packages` 语义，否则这些目录被 wheel 静默丢弃（与检查点 7 的 `where=["."]` 配置直接相关，见 [VI-17]）；(b) 命名空间包无法承载 per-package 懒加载/断言逻辑；(c) 工具链（mypy、部分 IDE、pkgutil 遍历）对命名空间包处理不一致。
- 影响：wheel 完整性风险 + 行为不一致。
- 建议：为三棵树内所有"确属本包"的目录补 `__init__.py`（一行 docstring 即可），把命名空间包语义留给真正需要的场景。

### 主题三：vendored 副本重复（检查点 3）

**方法**：目录名聚类（`/tmp/dup_clusters.py`）+ 同相对路径文件 md5 对比；对比前先剔除本仓库批量注入的自动 docstring 行（`"""Module for ... functionality."""`，见 [VI-12]），避免把工具噪声当成语义差异（`/tmp/dup_compare2.py`）。

**先说做对的**：仓库存在明确的收敛机制，且部分已落地——
- `worldfoundry/base_models/diffusion_model/models/networks/wan/variants/` 下收敛了 40+ 个 WAN 衍生变体（`forcing`、`dreamzero`、`moverse`、`fantasy_world.py` 等），synthesis 侧同名目录只是 wrapper；
- FantasyWorld 的 VGGT fork 已折叠进 canonical 树：`point_clouds/vggt/vggt/variants/fantasy_world/`；
- `depth/dvlt/dvlt_runtime/src/dvlt/model/vggt/model.py:27-29` 直接 `from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT`——嵌入目录只是转发 wrapper，不是副本；
- `representations/` 树整体是薄 wrapper（每模型 2-9 个 py），不含模型代码副本。

**但以下真实重复仍然存在**：

#### [VI-8] P1 DUSt3R/CroCo 全量三副本，其中一份 100% 纯重复、一份是无标记 fork、一份是不同版本

- 位置与统计（同相对路径文件数 / docstring 归一化后完全相同 / 语义差异）：
  - canonical：`worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r`（37 py）
  - `general_3d/monst3r/dust3r`（35 py）：common 32，**identical 27，真实差异仅 5 个文件**（`cloud_opt/base_opt.py`、`cloud_opt/optimizer.py`、`model.py`、`utils/misc.py`、`utils/vo_eval.py`）——MonST3R 对 DUSt3R 的 fork 改动就藏在这 5 个文件里，无任何 patch 标记；
  - `general_3d/stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r/dust3r`（25 py）：common 24，identical 仅 2——是**另一个上游版本**；
  - CroCo 同样三份：`dust3r/croco`、`monst3r/croco`（9/9 common 文件 docstring 归一化后**逐字节相同，纯重复**）、`stable_virtual_camera/.../dust3r/croco`，另有 `perception_core/.../uniception/models/libs/croco`（3 py 迷你版）。
- 问题：canonical dust3r 已有 wrapper（`dust3r/__init__.py` 的 `ensure_import_paths`）和 mast3r 的复用先例（`mast3r/__init__.py:25-35` 显式 reexport canonical），说明收敛机制可用，但 monst3r/svc 两份没有走这条路。三份副本借 sys.path 都暴露同一个顶层名 `dust3r`（见 [VI-2]/[VI-14]），加载哪份取决于 import 顺序。
- 影响：升级 DUSt3R 要改三处；monst3r 的 5 文件 fork 无标记，升级时改动会被静默覆盖或漏合；顶层名互踩导致运行错版本。
- 建议：以 canonical 为唯一实现，monst3r 的 5 个差异文件改为 variant 注入（仿照 `wan/variants` 模式）；svc 的旧版本若必须保留，改名顶层包（如 `dust3r_svc`）并记录版本差异。

#### [VI-9] P2 YOLO-World 完整双副本，差异纯属格式化噪声

- 位置：`worldfoundry/base_models/perception_core/detection/yolo_world/yolo_world`（36 py）与 `worldfoundry/base_models/perception_core/video_text/opens2v_nexus/eval/utils/yoloworld/yolo_world`（36 py）。
- 证据：36 个文件相对路径完全一致；md5 对比 34 个"不同"，但抽查 `datasets/mm_dataset.py` 的 89 行 diff 全部是 black/ruff 风格重排（引号、换行、尾逗号），无语义差异——即**同一上游快照，一份被 reformat 过**。
- 问题：这是最坏的重复形态：文本 diff 完全失效，只有 AST 级对比才能确认等价。另外 `detection/yolo_world` 还整包携带了 `mmyolo`（428 py，含 `configs/yolov5` 45 py 等纯配置目录）。
- 影响：升级/修 CVE 时极易只改一份；`opens2v_nexus` 的评测结果与 detection 路径的行为可能随时间静默分叉。
- 建议：删除 `opens2v_nexus` 内嵌副本，改 import canonical；把 mmyolo 的 configs/projects 目录从 vendor 范围剔除。

#### [VI-10] P2 DROID-SLAM、UniDepth、Depth-Anything 在 mega_sam 内嵌成套旧 fork，与 canonical 并存

- 位置与统计：
  - `slam/droid_slam`（23 py）vs `slam/mega_sam_runtime/base/droid_slam`（27 py）：19 个同名文件中 **17 个真实差异**——mega-sam 上游魔改过 DROID-SLAM，属预期 fork，但无差异记录；
  - `slam/mega_sam_runtime/UniDepth/unidepth`（40 py）vs `depth/unidepth`（14 py）：10 个同名文件全部真实差异（不同版本）；
  - `slam/mega_sam_runtime/Depth-Anything/depth_anything`（3 py）vs `depth/depth_anything/depth_anything_v1`（8 py）：3 个同名文件全部真实差异。
- 问题：mega-sam 上游本身就 vendor 了它依赖的三个项目，本仓库照单全收，于是同一上游在树内有"canonical 版"与"mega_sam 特化版"两个谱系，且**没有任何文件说明两者的版本关系与差异原因**。
- 影响：修 Depth-Anything/UniDepth 的 bug 或安全问题时，mega_sam 内嵌版必然被遗漏；反向地，误"去重"会破坏 mega-sam 的魔改语义。
- 建议：在 mega_sam_runtime 顶层放 UPSTREAM 说明，声明内嵌三件套的来源 commit 与"不要与 canonical 合并"的原因；长期看把 mega-sam 的魔改提炼为对 canonical 的 patch/variant。

#### [VI-11] P2 VGGT 变体收敛不彻底、HunyuanWorldMirror 两版本并存

- 位置与统计：
  - `point_clouds/vggt/vggt`（46 py，含 `variants/fantasy_world`）与兄弟目录 `point_clouds/vggt_omega/vggt_omega`（22 py；5 个同名文件 4 个真实差异）、`point_clouds/infinite_vggt`（9 py；6 个同名文件全部真实差异，是对 vggt 层的局部魔改副本）；
  - `point_clouds/hunyuan_mirror`（29 py）vs `point_clouds/hyworldmirror_2p0`（27 py）：17 个同名文件 14 个真实差异——同一模型两个版本平铺为两个顶层目录。
- 问题：同是"VGGT 家族 fork"，fantasy_world 走了 `variants/` 收敛，vggt_omega/infinite_vggt 却保持独立全拷贝——一套代码库里两种治理标准并存，说明收敛是个别工程师行为而非制度。
- 影响：`vggt_omega` 又被 wrbench 的评测脚本用 sys.path 引用（`wrbench/eval/d1/run_vggt_omega_batch.py:42`），形成跨评测/模型的隐式耦合；升级 VGGT 主干时 omega/infinite 两份不会跟着动。
- 建议：把 vggt_omega/infinite_vggt 的 attention/aggregator 差异迁入 `vggt/vggt/variants/`；hunyuan_mirror 与 hyworldmirror_2p0 合并为单包多版本（或至少 UPSTREAM 注明两者版本与差异）。

### 主题四：fork 漂移管理——上游来源记录（检查点 4）

**抽查方法**：选 20 个有代表性的 vendored 子目录（dust3r、monst3r、mast3r、vipe、vggt、vggt_omega、cut3r、gaussian_splatting、mega_sam_runtime、depth_anything、midas、yolo_world、grounding_dino、grit、open_flamingo、modelscope_swift、open_sora_plan、animatediff、cogvideox、openvla），检查目录内 UPSTREAM/PROVENANCE 文件、LICENSE、含上游 URL 的 README、commit 记录、根级 `THIRD-PARTY-NOTICES` 收录情况（脚本 `/tmp/provenance_audit.py`）。

#### [VI-13] P1 vendored 树的上游来源记录接近于零：20 个抽查目录 0 个有快照 commit，全包仅 1 个 UPSTREAM.md

- 证据（统计）：
  - 抽查 20 目录：**UPSTREAM 文件 0/20、目录内 LICENSE 1/20（仅 mega_sam_runtime）、含上游 URL 的 README 0/20、commit 记录 0/20、根级 NOTICES 收录 0/20**；
  - 全 `worldfoundry/` 包内 UPSTREAM 文件仅 1 个：`worldfoundry/synthesis/visual_generation/matrix_game/matrix_game_3p5_runtime/UPSTREAM.md`；三棵树内 LICENSE 文件仅 5 个；
  - 根级 `THIRD-PARTY-NOTICES`（350 行）只有 29 个 `Source:` 条目（23 个带 revision），内容集中在训练代码改编（AnyFlow、SenseFlow、rCM 等），**dust3r/vggt/mega-sam/yolo-world/open_sora/cogvideox/depth-anything 等大树全部未收录**；
  - 对照组：`thirdparty/UPSTREAM_PROVENANCE.md` 对 4 个 CUDA 扩展做到了 `upstream_url` + `fork_status` + `MODIFICATIONS.md` + `source_commit`（gsplat 甚至有 `.worldfoundry_upstream_commit` 标记文件）——**正确做法在仓库里存在，只是没有应用到 5649 个 py 文件的三棵树**；
  - 部分弥补：`worldfoundry/data/models/catalog/**.yaml`（274 个）记录了 `official_sources.github` URL 与许可证（如 `three_d_four_d/dust3r.yaml:10,18`），161 个含 40 位 SHA；但其中 105 个 `head_sha` 的语义是"catalog 校验时 `git ls-remote` 的上游 HEAD"（`video/framepack.yaml:52` 自述），**不是 vendored 快照的源 commit**；dust3r/monst3r 等重度 fork 的条目完全没有 SHA。
- 问题：没有快照 commit，"升级"的流程只能是：手工猜版本 → 全量 diff（又被 [VI-12] 的批量 docstring 注入污染）→ 人肉分辨"本地补丁"与"上游演进"。对 monst3r 内嵌 dust3r 这类**已确认存在 5 个文件本地/上游改动**的目录（[VI-8]），这实际上不可完成。
- 影响：任何上游安全修复（如 transformers/detectron2 类 CVE 波及的 vendored 推理代码）无法批量定位受影响副本；升级成本随时间单调上升。
- 建议：把 `thirdparty/` 的模式制度化：每个 vendored 子树根放 `UPSTREAM.md`（url + source_commit + 快照日期 + fork_status + 本地改动清单），用脚本从 model catalog 反向生成骨架；CI 校验新增 vendored 目录必须带该文件。

#### [VI-19] P1 限制性许可证代码未随目录携带 LICENSE：Inria 3DGS、CC-BY-NC 的 DUSt3R 等

- 位置与证据：
  - `worldfoundry/base_models/three_dimensions/point_clouds/gaussian_splatting/`（arguments/gaussian_renderer/scene/utils 全套 Inria 3DGS 训练/渲染代码）——**无 LICENSE、无 NOTICES 条目**。Inria/MPII 的 Gaussian-Splatting License 是研究用途限定且要求随代码分发许可证文本；
  - DUSt3R 为 CC-BY-NC-SA-4.0（catalog `dust3r.yaml:18` 自己记录了），但三份树内副本（[VI-8]）同样无一携带 LICENSE 文本；
  - 三棵树 5649 个 py 文件总共只有 5 个 LICENSE 文件。
- 问题：与 [VI-16]（exclude 失配）联动后风险放大：一旦 wheel 意外携带这些目录，就是在以 Apache-2.0（`pyproject.toml` 声明）名义再分发非商用许可证代码。
- 影响：法务合规风险；对外发布 wheel/镜像时构成实际的许可证违规。
- 建议：为每个限制性许可证子树补 LICENSE 原文；在 NOTICES 中列出"非 Apache 兼容"清单；打包侧显式断言这些路径不进 wheel（见 [VI-16] 建议）。

### 主题五：本地补丁可追踪性（检查点 5）

#### [VI-15] P1 vendored 代码到达即巨型 squash 提交，git 历史无法区分"上游原样"与"本地补丁"

- 证据：
  - 全仓库 git 历史仅 **42 个提交**；初始导入提交 `6f33a761`（提交名 "Avoid eager metric imports in CLI"）一次性引入 **15,947 个文件 / 3,405,627 行**——提交信息与内容完全无关；
  - 其后的大提交同样混装：`077fe858`"feat: update training, runtimes, tests, and docs"改 4,717 文件（+610,718/−480,815），`30a0c4d8`"refactor: consolidate inference-only base runtimes"改 1,959 文件，`e46c095f` 改 1,729 文件——vendored 树的改动与框架开发混在一起，无法按目录审计；
  - 抽查已确认有本地/上游差异的文件 `monst3r/dust3r/cloud_opt/base_opt.py`（[VI-8] 的 5 个真实差异文件之一）：`git log --follow` 仅 1 个提交（初始导入），即**这些差异是导入前在仓库外产生的，来源不可考**；
  - 三棵树内 patch/diff 文件数：**0**；`MODIFICATIONS.md` 仅存在于 `thirdparty/`（3 个）；
  - 好模式存在但覆盖极低：21 个文件带 "Modified by WorldFoundry"/"WorldFoundry modifications" 头部标记（如 `synthesis/action_generation/spirit_v15/modeling.py:5`、`x_wam/modeling.py:3`），对照已知的无标记修改（monst3r 5 文件、mega_sam droid_slam 17 文件）只是零头；
  - 唯一规范的 vendoring 提交是 thirdparty 的 `37cb1913` "Vendor modified Gaussian rasterization CUDA forks into thirdparty."。
- 问题：git 层面（squash 混装）、文件层面（无 patch/无标记）、目录层面（无 UPSTREAM，[VI-13]）三层追踪手段全部缺失，本地补丁事实上散落且不可枚举。
- 影响：升级=重新 vendor + 人肉三方对比；回答"我们改过这个文件吗"需要找到上游精确版本做 diff，而 [VI-12] 让 diff 也失效。
- 建议：最低成本方案：对每个 vendored 子树生成"导入基线"（把纯上游快照放一个 orphan 分支或 tarball 存档 + 记 commit），此后所有本地改动以普通提交落在 vendored 路径上，即可用 `git log -- <dir>` 枚举补丁。

#### [VI-12] P1 批量自动 docstring 注入改写了上千个 vendored 文件，摧毁与上游的可 diff 性

- 证据：
  - 模块级：**1,186 个文件**含自动生成的 `"""Module for <路径回显> functionality."""` docstring（rg 统计）；例：`worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r/cloud_opt/commons.py:7`——上游 Naver 版权头（1-6 行）之后被插入一行机器 docstring；
  - 函数级：**975 个文件、12,267 行**内容为空话的参数 docstring（形如 `i: The i.`，正则 `^\s+(\w+): The \1\.$` 统计）；同文件 14-20 行：

```python
def edge_str(i, j):
    """Edge str.

    Args:
        i: The i.
        j: The j.
    """
```

- 问题：这是对 vendored 代码的**最大规模本地修改**，却没有任何语义价值。它把"与上游 diff"从可行操作变成噪声海洋（本评审的副本对比必须先写归一化脚本剔除这类行，见 `/tmp/dup_compare2.py`）；同时每个被改文件都没有修改声明，与部分上游许可证的"修改需声明"条款相抵触。
- 影响：升级对比成本倍增；docstring 生成器若再次运行，还会产生新一轮全树 churn；[VI-9] 的 yolo_world 双副本"format-only 差异"同理。
- 建议：对三棵 vendored 树批量剔除机器 docstring（正则可逆向）；把"禁止对 vendored 目录跑格式化/docstring/lint 自动修复"写进 CI 守护（ruff 虽已排除这三个目录，但显然曾有工具越界执行过）。

### 主题六：monkey-patching（检查点 6）

**扫描面**：`sys.modules[...] =` 赋值 75 处；`types.MethodType` 实例级方法替换 52 个文件；类级 `forward` 覆写与第三方模块属性覆写若干（rg 全仓扫描）。分三个风险等级：

#### [VI-18] P1 framepack 在 import 时全局覆写 `torch.nn.LayerNorm.forward` 与 diffusers 归一化类，污染同进程所有其它模型的数值行为

- 位置：`worldfoundry/synthesis/visual_generation/framepack/framepack_runtime/diffusers_helper/model.py:23-69`
- 证据（全部在模块顶层执行）：

```python
accelerate.accelerator.convert_outputs_to_fp32 = lambda x: x   # L23

LayerNorm.forward = _layer_norm_forward                        # L30 (diffusers)
torch.nn.LayerNorm.forward = _layer_norm_forward               # L31 (torch 全局！)
FP32LayerNorm.forward = _fp32_layer_norm_forward               # L45
RMSNorm.forward = _rms_norm_forward                            # L59
AdaLayerNormContinuous.forward = _ada_layer_norm_continuous_forward  # L69
```

- 问题：这是上游 FramePack 自带的 hack，被原样 vendor 进来。`torch.nn.LayerNorm` 是**进程全局类**：一旦任何代码 import 了 framepack 的这个模块，整个进程内所有模型（包括其它 vendored 模型、用户宿主代码）的 LayerNorm 数值路径都被替换（`.to(x)` 的 dtype 语义与原生实现不同，autocast 行为改变）；`accelerate.convert_outputs_to_fp32` 被改成恒等函数，影响所有 accelerate 用户。WorldFoundry 的卖点是同框架内多模型编排，这类补丁使"进程内模型组合"结果不可信。
- 影响：跨模型数值污染，症状（精度轻微劣化/评测分数漂移）几乎不可能被归因到这里。
- 建议：wrapper 层禁止 import 该模块的全局补丁段；把补丁改为局部类（子类化后仅 framepack 自己用），或至少改成显式 `enable_framepack_patches()` + 进程隔离运行说明。CI 加守护：import 任意 synthesis 模块后断言 `torch.nn.LayerNorm.forward` 未被替换。

- 相关（较低风险，记录在案）：
  - `worldfoundry/core/distributed/sequence_parallel/cuda_utils.py:34-46` 顶层覆写 `torch.cuda.set_stream`（vLLM 式缓存 current_stream 的已知模式，有注释说明假设条件）——框架自有代码，风险可控但同样是进程全局，建议文档化；
  - 实例级 `MethodType` 注入 52 个文件（longvie/lingbot_world/ati/hydra 等的 USP sequence-parallel attention 替换，如 `synthesis/visual_generation/longvie/longvie_runtime/native_pipeline.py:418-428`）——限定在自有模型实例上，属可接受模式；
  - `worldfoundry/base_models/diffusion_model/models/networks/sana/optional_fla.py:23-70`：为绕过 flash-linear-attention 包根 `__init__` 的全量注册，**在 sys.modules 中伪造 `fla`/`fla.modules`/`fla.ops` 空命名空间**再精准加载单个子模块。文档清晰，但副作用是此后同进程内任何人 `import fla` 都拿到被掏空的假包（loader=None），全量 API 用户会莫名 AttributeError——应在伪造前检查"进程内是否还有其它 FLA 用户"，或在文档中列为已知限制。

### 主题七：打包正确性（检查点 7）

**验证方法**：在 /tmp 用与 `pyproject.toml:509-539` 完全一致的 include/exclude 参数实际运行 `setuptools.find_packages`（脚本 `/tmp/pkgfind_audit.py`）；用最小合成工程实测 MANIFEST `prune` 对 wheel 的效力（`/tmp/mani_test`）；抽查 package-data/MANIFEST 引用路径存在性；扫描进 wheel 的三棵树包内未被 package-data 覆盖的非 py 资产（`/tmp/pkgdata_gap.py`）。

**实测基线**：无 exclude 时发现 838 个包，带 exclude 后 712 个（净排除 126 个）；wheel 内三棵树约 3,695 个 py 文件（base_models 887 / synthesis 1,173 / representations 35 / 其余为框架）。

#### [VI-16] P0 打包治理三重失效：19/30 条 exclude 是死规则、MANIFEST"license-gated"prune 挡不住 wheel、Apache-2.0 wheel 实际携带 CC-BY-NC 代码

- 证据：
  1. **exclude 死规则**：对发现的 838 个包逐条 fnmatch，`pyproject.toml:509-539` 的 30 条 exclude 中 **19 条匹配 0 个包**。确证过时的如 `worldfoundry.base_models.*.vggt_fantasy_world.vggt`（L529-530）——`vggt_fantasy_world` 目录已在重构中并入 `vggt/vggt/variants/fantasy_world/`（[VI-11]），规则残留；其余死规则（`depth_anything`、`gaussian_splatting.*`、`vggt.vggt` 等）之所以匹配 0，是因为目标目录缺 `__init__.py` 本来就不会被 find 发现——规则沦为对目录布局的错误记忆。真正生效的只有 `*_runtime` 家族（119 个包）、vggt_omega（5）、pixelsplat（1）等。
  2. **MANIFEST prune 对 wheel 无效**：`MANIFEST.in:70-81` 注释明确写着 "License-gated upstream runtimes are allowed as ignored local checkouts only" 并 prune 了 `general_3d/dust3r`、`monst3r`、`mast3r`、`gaussian_splatting`、`cut3r`、`pi3` 等目录；但 `prune` 只影响 sdist。/tmp 合成工程实测：`prune pkg/vendored` 后 `bdist_wheel` 产物**仍包含 `pkg/vendored/core.py`**——包内 .py 模块由 packages 列表决定，与 MANIFEST 无关。
  3. **许可证泄漏坐实**：find 实测 `worldfoundry.base_models.three_dimensions.general_3d.dust3r` 与 `...dust3r.dust3r` **均在打包集合内**（dust3r 有 `__init__.py`，且无任何 exclude 命中它）。DUSt3R 是 CC-BY-NC-SA-4.0（[VI-19]），而 `pyproject.toml:10` 声明 `license = "Apache-2.0"`。即：**从源码树直接 `pip wheel` 得到的官方 wheel 以 Apache-2.0 名义分发非商用代码**。`synthesis/visual_generation/hunyuan_world`（117 py，Tencent 社区许可）同样实测在包集合内。
  4. 反向问题：若走 sdist→wheel 流程，prune 使 dust3r 文件缺失但包名仍在 packages 列表——装出来的 `mast3r` wrapper（顶层 reexport dust3r，[VI-2]）import 即崩。两条构建路径产物不一致。
- 问题：exclude 列表是"手工维护的目录布局快照"，目录一动就失效，且无 CI 校验"每条 exclude 至少命中一个包"；"哪些代码可以进 wheel"这个法务决策实际由 `__init__.py` 的有无这种偶然因素决定。
- 影响：许可证违规风险（P0 定级原因）；wheel 内容随构建路径漂移。
- 建议：(a) CI 增加 wheel 内容审计：构建后解包断言禁运目录名单（dust3r/gaussian_splatting/hunyuan\*/...）不出现；(b) exclude 列表由脚本从"许可证清单"生成而非手写；(c) 每条 exclude 加"必须命中 ≥1 包"的守护测试；(d) 修正 license 元数据或剥离非商用代码到 extras 仓库。

#### [VI-17] P2 用 `find` 而非 `find_namespace`，`perception_core`/`llm_mllm_core` 两棵子树整体不进 wheel

- 证据：`pyproject.toml:509` 用 `packages = { find = ... }`（经典 find_packages，要求逐级 `__init__.py`）；实测包发现结果中 `worldfoundry.base_models.perception_core`、`worldfoundry.base_models.llm_mllm_core` 及其全部子包 **不在 838 个包里**（两目录均无 `__init__.py`，[VI-7]）；同理 `point_clouds/vggt/`、`gaussian_splatting/`、`depth_anything/`、`monst3r/` 等缺 `__init__.py` 的目录整体不可发现。
- 问题：源码 checkout 下这些树靠 PEP 420 命名空间包可以 import，`pip install` 后则 ImportError——`base_models/capabilities.py` 及 evaluation 对 perception 模型的引用在安装态全部断裂。部分目录（gaussian_splatting）"恰好"因此没进 wheel，与 exclude 死规则互为掩护：**打包结果碰巧接近意图，但机制是错的**——任何人补一个 `__init__.py`（比如为了修 [VI-7]）就会静默改变 wheel 内容与许可证暴露面。
- 影响：安装态与源码态行为分叉；修 [VI-7] 时若不同步改打包策略会引爆 [VI-16]。
- 建议：明确"wheel 支持哪种安装形态"：若支持完整安装，改 `find_namespace` + 显式许可证 exclude 清单 + wheel 内容断言；若 wheel 只发框架层，把三棵树显式整体 exclude，别依赖 `__init__.py` 缺失的巧合。

#### [VI-20] P2 package-data/MANIFEST 引用大量幽灵路径，同时遗漏若干运行必需资产

- 证据（存在性抽查 11 条，6 条失效）：
  - 幽灵引用（文件从未入库，`git log --all` 为空）：`pyproject.toml:583-585` 与 `MANIFEST.in:35-37` 引用的 `vipe/LICENSE`、`vipe/THIRD_PARTY_LICENSES.md`、`vipe/UPSTREAM.md` 均不存在（vipe 的 THIRD_PARTY_LICENSES.md 实际在，但 UPSTREAM.md/LICENSE 缺失）；`rolling_forcing/UPSTREAM.md`、`LEGAL.md`（pyproject:598-600）不存在；`moverse/UPSTREAM.md`（pyproject:603、MANIFEST:14）不存在；`diffusion_model/diffsynth/tokenizer_configs/**`（pyproject:576、MANIFEST:47）——`diffsynth` 目录已整个移除（`diffusion_model/__init__.py` 文档自述"deliberately does not wrap DiffSynth"）；`mira_wm/.worldfoundry_upstream_commit`（MANIFEST:17）不存在；
  - 遗漏资产（进 wheel 的包内、未被任何 package-data 模式覆盖）：`general_3d/dust3r/croco/models/curope/{kernels.cu,curope.cpp}`（运行时 JIT 编译 curope 扩展的源文件）、`mmaudio/mmaudio/ext/bigvgan/bigvgan_vocoder.yml`、`ext/synchformer/divided_224_16x4.yaml`、`world_model/vid2world/{csgo_utils/test_split.txt,eval_inputs/val_file_list.json,nvm_utils/data_config.yaml}` 共 7 个文件（依赖 `include-package-data + setuptools-scm` 的兜底才可能进包，语义不明确）。
- 问题：声明式打包清单与真实目录持续脱钩，既有"引用不存在的许可证/来源文件"（合规记录比实际乐观），也有"运行必需文件靠兜底机制碰运气"。
- 影响：wheel 里缺许可证文件；离线安装态 JIT 编译/配置加载可能失败。
- 建议：CI 校验 package-data/MANIFEST 每条模式至少匹配 1 个真实文件；对运行必需资产改为显式列举。

### 主题八：命名空间卫生（检查点 8）

#### [VI-14] P1 多个 vendored 树经 sys.path 暴露相同顶层模块名（`dust3r`×3、`opensora`×2、`utils`×5、`models`×3 等），同进程互相覆盖

- 证据（静态解析 sys.path 插入目标目录后枚举可 import 顶层名，脚本 `/tmp/ns_conflicts.py`；仅解析出 24 个目标目录已发现 5 组冲突）：
  - `dust3r`：canonical `general_3d/dust3r/`、`general_3d/monst3r/`、svc `third_party/dust3r/` 三份都自带 `dust3r/utils/path_to_croco.py`（模块顶层执行）各自把**自己的** croco 目录插到 `sys.path[0]`；全仓库裸 `import dust3r`/`from dust3r ...` 共 **171 处**——具体解析到哪份，取决于哪条 path hack 先跑 + `sys.modules` 先占；
  - `croco`：三份副本（monst3r 的与 canonical 逐字节相同，svc 的是**不同版本**，[VI-8]）竞争同一顶层名——若 svc 先加载，canonical dust3r 后续拿到的是旧版 croco；
  - `opensora`：`open_sora/open_sora_runtime/opensora` 与 `open_sora_plan/open_sora_plan_runtime/opensora` 是**两个不同上游**（Open-Sora vs Open-Sora-Plan）的同名顶层包，裸 `opensora` import 共 **133 处**；同进程先后评测两个模型时，后者拿到前者的 `sys.modules["opensora"]`，且无任何 purge/防护；
  - 通用名污染：`utils` 由 ≥5 个根目录暴露（croco、wonderworld_runtime、genie_envisioner_runtime、wbench、visual_chronometer），裸 `utils` import **138 处**；`models` ≥3 个根、61 处；`run` 由 wonderjourney/wonderworld 两个 runtime 同名暴露；
  - vmem 的"解法"证明了问题存在：`vmem/runtime_env.py:41` 定义 `_RUNTIME_TOP_LEVEL_MODULES = ("modeling", "utils", "add_ckpt_path", "models", "dust3r")` 并在初始化时**驱逐 sys.modules 中所有来源不在 vmem 根下的同名模块**（L160-184）——即 vmem 每次初始化都会把别人已加载的 `utils`/`models`/`dust3r` 踢出缓存，此后其它组件的重新 import 又会解析到 vmem 的副本；旧模块对象仍被先前持有者引用，形成**同名类双实例**（isinstance/单例/enum 全部失效的经典温床）；
  - `fantasy_world/runtime_env.py:182-185` 同样以驱逐+强绑（`sys.modules["utils3d"] = vendored_utils3d`）的方式抢占 `utils3d`——若宿主进程装了 PyPI 的 `utils3d`，会被静默替换。
- 问题：三棵树的隔离靠"路径插入顺序 + 模块缓存先占 + 各自驱逐"三套互相冲突的机制拼凑，进程内多模型组合的名字解析是时序函数。
- 影响：WorldFoundry 的 pipeline/evaluation 场景（一个进程串多个模型与 metric）随组合规模增长必然踩中；症状是隐蔽的错版本加载与类身份分裂。
- 建议：制度化唯一防线：所有 vendored 顶层名注册进一张全局表，wrapper 层统一走"检测冲突→显式别名（`sys.modules['pkg'] = 包内模块`）→冲突即 fail-fast"的公共设施（open_flamingo 的 `_assert_top_level_package_not_shadowed` 已是原型，[VI-5]）；对 `utils`/`models` 这类通用名，vendor 时必须重命名或包内改写 import。

### 主题九：capabilities.py 与集成接口（检查点 9）

#### [VI-21] P2 wrapper 契约总体统一，但存在两处结构破绽：API 分支游离于 ABC 之外、action 基类反向依赖 evaluation 层

- 正面证据（先记录做得好的）：
  - `worldfoundry/base_models/capabilities.py`（4,698 行）是设计良好的声明式能力注册表：65 个 `BaseModelCapability` + 33 个 `BaseModelStack`，每条含 `canonical_owner`（import 路径）、`canonical_path`（目录）、`package_imports`/`install_packages` 成对映射、`non_pip_imports` + 人工安装指引（如 `capabilities.py:945-977` 的 droid_slam 条目）、资产声明（env 覆盖链 + `hf_repo_id`/`hf_revision`/`git_revision` + `min_size_bytes`/`required_files` 校验）。抽验 65 条 `canonical_path` **全部存在**（0 失效）——与 pyproject exclude 的腐烂（[VI-16]）形成对照，说明"有 consumer 持续消费的清单"才活得下来。
  - wrapper 类层次基本统一：AST 统计三棵树 197 个具体 `*Synthesis`/`*Representation` 类，主干都收敛到 `BaseSynthesis`（`synthesis/base_synthesis.py:38`，`from_pretrained`/`api_init`/`predict` 契约）或 `BaseRepresentation`（`representations/base_representation.py:4`）：`RuntimeVideoSynthesis`/`RuntimeFacadeSynthesis`/`RuntimeAdapterSynthesis` 直接继承（`runtime_facade.py:56,99`、`runtime_video_synthesis.py:244`），52 个 action 模型经 `ActionModelSynthesis` 归于同谱系。
- 问题证据：
  1. **API 模型分支脱离契约**：`worldfoundry/synthesis/visual_generation/api_video_client.py:34` 的 `CredentialedSynthesis` 是普通类（不继承 BaseSynthesis），其下 `ApiVideoSynthesis`（同文件 L69）及 kling/luma/minimax/runway/sora/veo 等 6 个实现只与本地模型 duck-type 兼容——`isinstance(x, BaseSynthesis)` 分派对 API 模型失灵。
  2. **分层倒置**：`worldfoundry/synthesis/action_generation/base_action_synthesis.py:14` 的 `ActionModelSynthesis` 继承自 `worldfoundry/evaluation/models/runtime/profiles.py:942` 的 `RuntimeProfileSynthesis(BaseSynthesis)`——**模型库（synthesis）的基类依赖评测框架（evaluation）**；单独使用 synthesis 树必须连带加载 evaluation。
- 影响：接口分派不可靠（问题 1）；模型库无法独立于评测框架发布/复用，也让 [VI-16] 的"拆分发布"更难落地（问题 2）。
- 建议：`CredentialedSynthesis` 挂到 `BaseSynthesis` 下；把 `RuntimeProfileSynthesis` 下沉到 synthesis（或 core），evaluation 侧只做注册与消费。

### 主题十：安全——被调用路径上的反序列化/eval（检查点 10）

**范围限定**：只看 wrapper 实际调用路径上的入口，不全量审 6000 文件。全仓计数：`weights_only=False` 38 处、`torch.load(` 537 处、`pickle.load(s)` 39 处、`trust_remote_code=True` 40 处（22 个文件在 llm_mllm_core）。

**先确认存在集中式安全设施**：`worldfoundry/core/model_loading/file.py:173-208` 的 `load_torch_checkpoint` 默认 `weights_only=True`，unsafe pickle 回退需显式 `allow_unsafe_pickle_fallback=True` 且带警告（L188-197 docstring 明确风险）——设计正确。问题在于多数 vendored wrapper **绕过它**裸调 `torch.load`。

#### [VI-22] P1 DUSt3R/MASt3R 加载器在 wrapper 调用路径上执行 `eval(checkpoint_string)`，构成"下载 checkpoint 即任意代码执行"

- 位置：
  - `worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r/model.py:38-49`
  - `worldfoundry/base_models/three_dimensions/general_3d/mast3r/mast3r/model.py:33-42`
- 证据（dust3r `model.py:38-49`）：

```python
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
...
net = eval(args)                    # ← checkpoint 内的字符串直接进 eval
s = net.load_state_dict(ckpt['model'], strict=False)
```

- 调用路径（确认在实际执行链上，非死代码）：
  - `worldfoundry/pipelines/geometry_priors/pipeline_geometry_priors.py:37` `DUSt3RPipeline` → `GeometryPriorSynthesis`/`GeometryPriorOperator` → dust3r 推理；catalog `dust3r.yaml` 标注 `integration.status: verified`；
  - `worldfoundry/synthesis/visual_generation/vmem/vmem_runtime/modeling/modules/preprocessor.py:46` `AsymmetricCroCo3DStereo.from_pretrained(...)`（VMem 相机预处理器）；
  - `worldfoundry/base_models/three_dimensions/general_3d/splatt3r/splatt3r_runtime/model.py` `from mast3r.model import ...`（Splatt3R 有 `worldfoundry_runtime.py` wrapper）。
- 问题：`weights_only=False` + `eval(ckpt['args'].model)` 双重危险。即使不谈 `eval`，`weights_only=False` 本身允许 pickle 在反序列化时执行任意 `__reduce__`。DUSt3R/MASt3R 权重是 CC-BY-NC 的第三方 checkpoint（用户从外部 URL 下载），一个被篡改的 .pth 在 `torch.load` 阶段或 `eval` 阶段即可 RCE。
- 影响：下载即执行的攻击面，命中本仓库主打的 3D 几何 pipeline。
- 建议：`load_model` 的 `eval(args)` 改为白名单类名 dispatch（`{"AsymmetricCroCo3DStereo": ...}[cls_name](**parsed_kwargs)`）；checkpoint 加载改走 `load_torch_checkpoint`（默认 weights_only），仅对确知可信的内部权重显式开 fallback。

#### [VI-23] P2 12+ 处 wrapper 直接 `torch.load(..., weights_only=False)` 绕过集中安全加载器

- 位置（wrapper/representations 层，节选）：
  - `worldfoundry/representations/depth_generation/depth_anything/depth_anything_v2_representation.py:183`、`depth_anything_v1_representation.py:106`
  - `worldfoundry/representations/point_clouds_generation/flash_world/flash_world_representation.py:404`
  - `worldfoundry/representations/point_clouds_generation/lingbot_map/lingbot_map_representation.py:242`
  - `worldfoundry/base_models/three_dimensions/general_3d/splatt3r/worldfoundry_runtime.py:146`
  - `worldfoundry/base_models/three_dimensions/depth/unik3d/unik3d.py:366`、`.../unidepth/models/unidepthv2/unidepthv2.py:341`
  - `worldfoundry/base_models/three_dimensions/point_clouds/cut3r/model.py:80`、`worldfoundry/base_models/diffusion_model/models/encoders/wan/resident.py:38`
- 证据：`representations/` 是仓库自写的最薄 wrapper 层（每模型 2-9 py），本应最容易统一走安全加载器，但其中 4 个 representation 直接 `torch.load(..., weights_only=False)`；而 `pi3/loger_representation.py:144` 与 `infinite_vggt_representation.py:83` 已改用集中式 `load_torch_checkpoint(..., weights_only=False)`——**同一层里两种写法并存**，迁移做了一半。
- 问题：`weights_only=False` 在 wrapper 层随手写，安全默认值形同虚设；分散的裸 `torch.load` 无法通过改一处集中加载器来收敛策略。
- 影响：反序列化攻击面散布在最外层 API；无法集中加固。
- 建议：representations/wrapper 层禁止裸 `torch.load`，统一改 `load_torch_checkpoint`；确需 unsafe 的显式传 `allow_unsafe_pickle_fallback=True` 并标注可信来源；CI lint 拦截 wrapper 层的 `weights_only=False`。

## 汇总

### 严重度统计表

| 严重度 | 数量 | 编号 |
|---|---|---|
| P0（损坏/危险） | 1 | VI-16 |
| P1（严重设计缺陷） | 10 | VI-1, VI-2, VI-8, VI-12, VI-13, VI-14, VI-15, VI-18, VI-19, VI-22 |
| P2（应修复） | 10 | VI-3, VI-4, VI-7, VI-9, VI-10, VI-11, VI-17, VI-20, VI-21, VI-23 |
| P3（改进建议） | 2 | VI-5, VI-6（其中 VI-6 为正面确认） |
| **合计** | **23** | |

按检查点分布：CP1 sys.path（VI-1~5）、CP2 import 副作用（VI-6,7）、CP3 副本重复（VI-8~11）、CP4 来源记录（VI-13,19）、CP5 补丁可追踪（VI-15,12）、CP6 monkey-patch（VI-18）、CP7 打包（VI-16,17,20）、CP8 命名空间（VI-14）、CP9 接口（VI-21）、CP10 安全（VI-22,23）。

### 一句话总览

这套 vendoring 的**内层 API 卫生做得好**（聚合层 `__getattr__` 懒加载统一有效、capabilities 声明式注册表健康、有集中安全加载器与 shadow 断言原型），但**外层治理系统性失效**：来源不可考（20/20 无快照 commit）、补丁不可追踪（squash 巨提交 + 上千文件被机器 docstring 污染）、隔离靠时序运气（顶层名互撞 + sys.path 抢占 + sys.modules 驱逐三套机制打架）、打包与许可证脱钩（exclude 死规则 + wheel 携带 CC-BY-NC 代码）。共性根因是：**正确的模式在仓库里都已存在，但没有一个被制度化（CI 守护）成强制规范**，于是每新增一个 vendored 模型就重新赌一次运气。

### Top 5 问题

1. **[VI-16] P0 打包治理三重失效 → wheel 许可证泄漏**：30 条 exclude 有 19 条是死规则（实测匹配 0 包），MANIFEST 的 `prune`（自称挡住 license-gated runtime）对 wheel 完全无效（/tmp 合成工程实测坐实），CC-BY-NC 的 DUSt3R、Tencent 许可的 hunyuan_world 实测在打包集合内，而 `pyproject.toml:10` 声明 Apache-2.0——直接 `pip wheel` 即以 Apache 名义分发非商用代码。唯一拦住部分目录的竟是"缺 `__init__.py`"这种偶然（[VI-17]），补一个文件就会扩大泄漏面。

2. **[VI-22] P1 DUSt3R/MASt3R 加载器 `eval(checkpoint_string)` + `weights_only=False`**：`dust3r/model.py:38-49`、`mast3r/model.py:33-42` 在已确认的 pipeline/VMem/Splatt3R 调用路径上，从第三方下载的 .pth 里取字符串直接 `eval`，构成"下载 checkpoint 即 RCE"；仓库明明有默认 `weights_only=True` 的集中加载器却被绕过（[VI-23] 另有 12+ 处裸 `torch.load(weights_only=False)`）。

3. **[VI-14] P1 顶层命名空间互撞，多模型同进程解析靠时序**：`dust3r`×3、`opensora`×2（两个不同上游）、`utils`×5、`models`×3 经 sys.path 暴露同名顶层包（裸 import 分别达 171/133/138 处）；隔离依赖"路径抢占顺序 + sys.modules 先占 + vmem/fantasy_world 主动驱逐"三套互相冲突的机制，正是 WorldFoundry 主打的 pipeline/评测多模型编排场景必然踩中的雷，症状是隐蔽的错版本加载与类身份分裂。

### 值得肯定（避免以偏概全）

- 聚合层 `__init__.py` 懒加载统一且实测有效（[VI-6]，`import worldfoundry.base_models` <10ms、零重依赖）；
- `capabilities.py` 声明式注册表健康（65 条 canonical_path 0 失效，[VI-21]）；
- `core/model_loading/file.py` 的安全加载器、`open_flamingo` 的 shadow 断言、`thirdparty/UPSTREAM_PROVENANCE.md` 的 provenance 范式、`wan/variants` 的收敛模式——**正确做法都已存在，缺的是制度化推广与 CI 守护**。
