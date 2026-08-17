# vendored 来源/许可证记录修复日志（11d：provenance）

> 修复人：infra 修复 agent；日期：2026-08-14
> 对应评审报告：`plan/code_review/11_vendored_integration.md`（主题四 [VI-13]/[VI-19]，联动 [VI-8]/[VI-9]/[VI-10]/[VI-11]）
> 约束：**只允许新建文件**（LICENSE / LICENSE.MISSING.md / UPSTREAM.md），零改动任何既有文件（.py、pyproject.toml、MANIFEST.in、THIRD-PARTY-NOTICES 等一律不碰）。
> 诚实性纪律：不编造事实。catalog 的 `head_sha` 语义是"catalog 校验时 `git ls-remote` 的上游 HEAD"（`worldfoundry/data/models/catalog/video/framepack.yaml:52` 自述），**不是 vendored 快照 commit**，因此所有 UPSTREAM.md 的 source_commit 一律如实写 `unknown — needs backfill`；许可证名逐一与 catalog yaml / 树内源码头 / 树内既有许可证记录交叉核对后才写入。

## 已修复

### [VI-19] 限制性许可证 vendored 树补 LICENSE（4 个 CC-BY-NC-SA-4.0 + 1 个 MISSING 声明）

**CC-BY-NC-SA-4.0 legalcode 全文来源**：仓库内已存在完整标准文本——`worldfoundry/base_models/three_dimensions/general_3d/vipe/THIRD_PARTY_LICENSES.md` 的 UniK3D 段（941-1377 行，含 Creative Commons Corporation 前言与 Section 1-8 全文）。用 `sed -n '941,1377p'` **逐字节提取**（拒绝手工重打法律文本），4 份输出 md5 一致（`fb5d051e53001fdff7fec0f368f47190`，437 行）。新建：

1. `worldfoundry/base_models/three_dimensions/general_3d/dust3r/LICENSE`
   - 依据：catalog `three_d_four_d/dust3r.yaml:18` `license: CC-BY-NC-SA-4.0`；上游 naver/dust3r。同时覆盖同目录下 `croco/`（源码头自述 "Licensed under CC BY-NC-SA 4.0"，评审 [VI-8] 确认与上游同源）。
2. `worldfoundry/base_models/three_dimensions/general_3d/monst3r/LICENSE`
   - 依据：catalog `three_d_four_d/monst3r.yaml:15` `license: CC-BY-NC-SA-4.0`（confirmation 字段引述上游 LICENSE 原文名）。覆盖其内嵌 `dust3r/`（fork）与 `croco/`。
3. `worldfoundry/base_models/three_dimensions/general_3d/mast3r/LICENSE`
   - 依据：catalog **无** mast3r 条目（已核实 `three_d_four_d/` 全目录列表），改用树内源码头直接证据：`mast3r/mast3r/model.py:1-2` "Copyright (C) 2024-present Naver Corporation ... Licensed under CC BY-NC-SA 4.0"。与任务陈述（Naver 官方 CC-BY-NC-SA-4.0）一致。
4. `worldfoundry/base_models/three_dimensions/general_3d/stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r/LICENSE`
   - 依据：该子树是 DUSt3R（旧版本）副本，源码头同样自述 CC BY-NC-SA 4.0；许可证跟随 DUSt3R 而非外层 SVC（SVC 本体是 Stability AI Non-Commercial，`stable-virtual-camera.yaml:21`，不适用于此子树）。

5. `worldfoundry/base_models/three_dimensions/point_clouds/gaussian_splatting/LICENSE.MISSING.md`
   - Inria/MPII "Gaussian-Splatting License" **全文不在仓库内**（核实过程：`rg -l "Gaussian-Splatting License"` 仅命中评审报告与 `thirdparty/THIRD_PARTY_LICENSES.md` 的摘要行 10-12/30-31；`thirdparty/` 三个 gsplat 系 fork 目录内均无 LICENSE 文件；全文特征短语检索零命中）。按纪律不凭记忆重写法律文本，故创建 MISSING 声明：写明许可证名、版权方（Inria GRAPHDECO + MPII）、树内源码头证据（`scene/gaussian_model.py:1-10`）、官方原文 URL（https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md）、"必须逐字拷贝为本目录 LICENSE.md"的待办，以及在此之前"仅限研究/评估用途"的保守解读。

### [VI-13]/[VI-10]/[VI-11]（含 [VI-8]/[VI-9] 证据落地）7 个 UPSTREAM.md

每个文件统一记录：upstream_url、license（含出处行号）、source_commit（一律 `unknown — needs backfill`，理由见抬头）、fork_status、评审已坐实的本地差异证据、"禁止盲目去重"警告与交叉引用。格式对齐仓库内范式 `thirdparty/UPSTREAM_PROVENANCE.md`。

6. `worldfoundry/base_models/three_dimensions/general_3d/monst3r/UPSTREAM.md`
   - 上游 https://github.com/Junyi42/monst3r；内嵌 `dust3r/` 是 DUSt3R fork，真实改动恰为 5 个文件（`cloud_opt/base_opt.py`、`cloud_opt/optimizer.py`、`model.py`、`utils/misc.py`、`utils/vo_eval.py`，[VI-8]），明示"勿与 canonical general_3d/dust3r 盲目去重"；`croco/` 为纯重复（docstring 归一化后逐字节相同）；记录顶层名 `dust3r`/`croco` 冲突隐患（[VI-14]）。
7. `worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/UPSTREAM.md`
   - 上游 https://github.com/mega-sam/mega-sam；**该目录已有 LICENSE（Apache-2.0 全文），是 20 个抽查目录中唯一的 1/20，未重复添加**（已核实 `ls`）。表格记录三件内嵌 fork 与 canonical 的差异（DROID-SLAM 19 个同名文件 17 个差异、UniDepth 10/10、Depth-Anything 3/3，[VI-10]），声明"这是 mega-sam 上游自己的魔改，勿合并"；内嵌件的上游身份/许可证引用树内既有记录（vipe/THIRD_PARTY_LICENSES.md:3-8 DROID-SLAM BSD-3-Clause、:522-527 UniDepth CC-BY-NC-4.0；catalog depth-anything-v1.yaml:10-11 Apache-2.0）。
8. `worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror/UPSTREAM.md`
9. `worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/UPSTREAM.md`
   - 两版本互相交叉引用（[VI-11]：17 个同名文件 14 个真实差异）。**哪个目录对应哪个版本已从消费方确证**：`hunyuan_mirror` 被 `pipelines/hunyuan_world/pipeline_hunyuan_mirror.py`（catalog `hunyuanworld-mirror` 的 pipeline_binding）与 `hunyuan_world_mirror_representation.py` 消费；`hyworldmirror_2p0`（类名 `WorldMirror`）被 HY-World 2.0 栈（`hy_world_2p0/worldmirror_runtime.py`、neoverse runtime）消费，对应 catalog `hy-world-2.0.yaml:16-17,34` 的 "WorldMirror 2.0 reconstruction component"。2p0 的精确上游仓库（HunyuanWorld-Mirror 新版 or HY-World-2.0 内置件）不可考，如实记两个候选 URL + unknown。**新发现的删除/合并雷区已写入双方**：`hyworldmirror_2p0/models/models/worldmirror.py:15` 反向 import `hunyuan_mirror` 的 utils，两目录在运行时不独立。许可证均为 `other:tencent-hunyuanworld-community`（两个 yaml 的 license 字段）。
10. `worldfoundry/base_models/three_dimensions/general_3d/stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r/UPSTREAM.md`
    - 与 canonical 是**不同（更旧）上游版本**（24 个同名文件仅 2 个相同，[VI-8]）；经 Stability-AI/stable-virtual-camera 的 third_party 间接 vendor 进来；顶层名 `dust3r`/`croco` 与 canonical、monst3r 副本三方相撞的时序性风险单列一节；建议若保留应改名（`dust3r_svc`）。
11. `worldfoundry/base_models/perception_core/detection/yolo_world/UPSTREAM.md`
12. `worldfoundry/base_models/perception_core/video_text/opens2v_nexus/eval/utils/yoloworld/UPSTREAM.md`
    - 同一上游快照的双副本，34/36 文件差异纯格式化噪声（[VI-9]），互相交叉引用；明示"第二轮去重前，语义修改必须双写"。上游 URL（AILab-CVC/YOLO-World）在仓库内无独立记录（catalog 无 yolo-world 条目，已核实），文件中如实标注"URL 来自上游项目身份、待回填 commit 时确认"；许可证 GPL-3.0 的树内证据：内嵌 `mmyolo/setup.py:181` 自述 `license='GPL License 3.0'`，yolo_world 源码头仅有 Tencent Inc. 版权行。opens2v 副本额外记录宿主 OpenS2V-Nexus（https://github.com/PKU-YuanGroup/OpenS2V-Nexus，`docs/fumadocs/mdx/partials/metrics-usage.mdx:228` 有引用）。

## 验证记录

- **许可证文本完整性**：4 份 LICENSE md5 全部为 `fb5d051e53001fdff7fec0f368f47190`（437 行），首行 `Attribution-NonCommercial-ShareAlike 4.0 International`、末行 `Creative Commons may be contacted at creativecommons.org.`，与源段（vipe/THIRD_PARTY_LICENSES.md:941-1377，UniK3D 段收录的官方 legalcode）逐字节一致。
- **目录存在性**：写入前对全部 9 个目标目录 `ls` 核实存在；mega_sam_runtime 已有 LICENSE 故只加 UPSTREAM.md。
- **许可证名交叉核对**：dust3r/monst3r ← catalog yaml `license` 字段；mast3r/croco/svc-dust3r ← 源码头 "Licensed under CC BY-NC-SA 4.0"；两个 mirror ← 两个 catalog yaml 的 `license: other:tencent-hunyuanworld-community`；yolo_world ← mmyolo setup.py + 上游仓库声明（已标注推断边界）；gaussian_splatting ← 源码头 + thirdparty 摘要（全文缺失故走 MISSING）。
- **commit 诚实性**：全部 source_commit 写 unknown；依据 framepack.yaml:52 对 head_sha 语义的自述，catalog SHA 不可用作快照 commit；dust3r/monst3r/mirror 等条目本身无 SHA。
- **零改动既有文件（本修复线）**：本线全部产出均以"新建文件"落地，`git status` 未跟踪清单与本文件清单一一对应；本线使用的 shell 命令仅为只读（`sed -n` 提取、`md5sum`/`diff`/`ls`/`cat`/`rg`）。注意：验证时 `git status --short | grep -v '^??'` 显示 33 个既有文件处于 modified 状态，抽查 diff（如 `monst3r/dust3r/utils/path_to_croco.py` 带 "Modified by WorldFoundry: idempotent insert" 标记、`dust3r/__init__.py` 等）确认是**并行修复线（[VI-2]/[VI-3]/[VI-4]/[VI-18] sys.path/monkey-patch 修复）的在途改动**，与本线创建的 13 个文件零交集，未回滚（回滚会毁掉他人工作）。

## Deferred（超出本轮"只新建文件"范围，移交第二轮）

- **[VI-13] 系统性修复**：为三棵树全部 vendored 子目录补 UPSTREAM.md、用脚本从 model catalog 反向生成骨架、CI 校验"新增 vendored 目录必须携带 UPSTREAM.md"——本轮只覆盖评审点名的 7 个目录，制度化属第二轮。
- **[VI-19] 打包侧断言**：wheel 构建后解包断言限制性许可证目录（dust3r/gaussian_splatting/hunyuan\* 等）不进产物（联动 [VI-16]），需改 CI/pyproject，本轮禁改。
- **THIRD-PARTY-NOTICES 增补**：根级 NOTICES 未收录 dust3r/vggt/mega-sam/yolo-world 等大树；该共享文件由其他修复线持有，本轮不碰，防冲突。
- **gaussian_splatting/LICENSE.md 原文回填**：需要从官方仓库逐字拷贝（本环境不引入外部文本），LICENSE.MISSING.md 中已留明确操作指引。
- **快照 commit 回填**：全部 9 处 `unknown — needs backfill` 需要与上游逐版本 diff 定位（受 [VI-12] docstring 污染需先归一化），属第二轮考古工作。
