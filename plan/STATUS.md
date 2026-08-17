# 状态看板

最后更新：2026-08-14 03:12 (UTC+8) — 首次全面巡检（Cycle 1 中段）

| TID | 状态 | 阶段 | 关键情况 |
|-----|------|------|----------|
| T01_frankl | RUNNING | 复现 SOTA 中 | SOTA 核实=0.382709 (Liu, arXiv:2306.08824)；已提取 9 维优化结构，正写复现代码 |
| T02_hadwiger_nelson | RUNNING | Cycle 2 | C1 完成(PROGRESS)：G510 复现+DRAT；**从零构造 5-色图 G_4069（零外部依赖）**。C2=DRAT-core 顶点缩减冲 <509 纪录 + 显式 χ_gf>4 见证图（公开缺口） |
| T03_ramsey_r55 | RUNNING | Cycle 3 | C2 完成(PROGRESS)：R55 排除墙推至 \|W\|=34+2态对合结构；R46 流水线平移（37证书全验证、交换群 Cayley 零解、地板=12 冲突、半数 12-态 \|Aut\|=4 对称吸引子）。C3=对称子空间攻势+PAWS 引擎升级 |
| T04_lonely_runner | RUNNING | Cycle 3 | 三路验证汇齐：**数学成立，新颖性降级**（Kravitz 2021 已有 j=1 族；J–K 2026 框架被遗漏）。C3=对照 J–K 重定位（j=2 首例待核）+ 诚实定位版短文 + 谱理论推进 |
| T05_erdos_straus | RUNNING | Cycle 3 | C2 完成(PROGRESS)：**定理1（SYM 核心反演不变，已证）+ 定理3（U_q 反演封闭 q≤23，已证）**；强形式被反例证书证伪（102481/112561）；#R₃=34 双路线否定；C5 假说证伪。C3=证 C1' 普适化 + G₈ 复核 + 逃逸素数解谱 |
| T06_graceful | RUNNING | 写求解器 | 前沿确认 ≤35 顶点 (Fang 2010)；36–40 带证书验证是空白记录；主攻 lobster |
| T07_no_three_in_line | RUNNING | Cycle 2 | C1 完成(PROGRESS)：管线+标度数据扎实但 n=71 差 12 点（算子平台效应确认）。C2 双轨=混合破坏修复+CP-SAT 补尾冲 71（时间盒 50%）∥ torus 变体首表（14 年空白，保底产出） |
| T08_sunflowers | RUNNING | 任务修正+ILP | f(3,3)=20 已知；转攻 i(4)∈[27,77]：i(4)≥32 即改进 53 年未动的 AHS 下界常数 |
| T09_one_third | RUNNING | 接管重启 | 前任 agent 上下文耗尽（数据完好：n≤9普查+断梯15-21+退火搜索全落盘）；新 agent 接管收尾 verdict + 主攻宽度3 n≥10 空白 |
| T10_reconstruction | RUNNING | Cycle 2 | C1 完成(PROGRESS)：SC(16/17) 零碰撞验证 ⚠️ 新颖性被侦察兵推翻（arXiv:1012.5995 于 2010 已做 ≤17）→ 降级为独立复现；McKay 补遗对仍成立。C2 主攻（自逆有向图 9-10 猎反例）不受影响（McKay 只做到 8） |
| T11_hadamard_668 | RUNNING | Cycle 2 | C1 完成(PROGRESS)：复现 H(428) 端到端；TT(56) 纯 DFS 判不可行(~10⁷年)。C2=BDKR 两阶段法 + **直攻 TT(46)（偶数 TT 已知≤44，独立记录目标，不受 668 声明影响）** |
| T12_zarankiewicz | RUNNING | 写代码 | z4 表已认证 (Tan SAT)；n=14 开放；LP 上界 2024 有改进；ortools 就绪 |
| T13_scout | RUNNING | Cycle 5 | C5=观察哨扫描（重点盯 T04 谱定理撞车风险）+ T04 论文引文包 + T16 下批速胜目标预研 |
| T14_borsuk62 | RUNNING | Cycle 2 | C1 完成(PROGRESS)：63 维纪录第五套独立验证（优先权更正为 Grinsztajn 2026-05）+ 15015 witness 平面新事实。C2=穷举 [320,63] 码低重词（定理化 62 排除 or 复活 62）+ SRG 母体机扫 |
| T15_multicolor_ramsey | RUNNING | 1 | 03:15 新开（侦察兵 Top2）：R(3,3,3,3) ≥ 52，Chung 下界 55 年未动 |
| T16_quickwins | RUNNING | 1 | 03:48 新开：Erdős #340 Mian–Chowla 33、皇后支配 γ(Q20)、WS(6)→651 障碍量化 |

## 侦察兵重点情报（待其 findings.md 正式确认）

- **2026 AI 反例潮**：Jacobian 猜想（维度≥3）被反例推翻（Lean/Isabelle 双验证）；Erdős 单位距离猜想被反例推翻；sum-product 猜想（R 上）被推翻；HRT 猜想被反例推翻；Sendov 猜想被解决。
- **高价值可攻记录**（候选新线程）：Borsuk 最小反例维度刚降到 63（2026-08，321点三距离集），62 维全新前沿；bunkbed 反例 7222 顶点"unlikely optimal"→ 最小化反例；cap set dim9 >1082；R(3,3,3,3)≥51 下界 55 年未动；WS(6)、W(2,7)>3703、3-MOLS(10)、kissing dim11/12。
- **影响现有线程**：T04（k≤12 已证）、T07（n≤76 已达）、T11（668 疑似已解）需在 Cycle 2 换向或转验证角色。

## 突破 / 重要事件

- **05:30 [最终认定：数学成立，突破降级] T04 定理 A/B**：三路对抗验证汇齐——V1 CONFIRMED（逻辑审计，22 条不等式独立重推）、V2 CONFIRMED（盲复现至 r=1000 零不一致）、V3 复跑与证书 CONFIRMED 但新颖性 FLAWED：k=7 无限族已有 Kravitz 2021 Thm 3.1（j=1 族）先例；9/65 在 Fan–Sun 穷举域内未被列出而非域外；最相关文献 Jain–Kravitz 2026（arXiv:2411.12684，谱结构定理，n=7 留作 future work）被遗漏。**定性修正：已发表纲领内的首批 k=7 具体算例（j=2 无限族可能仍为首例，待 T04 C3 对照 J–K 框架核实），增量贡献级，非 BREAKTHROUGH。** 全部报告归档 breakthroughs/T04_familyAB/。协议按设计运作：自产声明经三重敌意审查后才定性。

- **04:04 [外部突破确认] Hadamard 猜想 n<2000 全部成立**：侦察兵定位到 Anthropic 声明的原始数据（github.com/foocker/Hadamard668），本机独立验证 H·Hᵀ=668·I 及全部 10 个先前未知阶（668–1916）通过；最小未知阶现为 2004。材料与验证脚本存档于 threads/T13_scout/results/hadamard668/。T11 原目标死亡（死因有我方证书），其 Cycle 2 主攻 TT(46) 序列记录不受影响（独立价值，但权重下调，收官时评估换向 dossier）。
- **04:47 [首批自产定理] T05：Salez 判据系统的反演对称定理**（定理 1：SYM 核心覆盖集对 x↦x⁻¹ 不变，sympy 三对偶引理已证；定理 3：U_q 反演封闭对 q∈{11..23} 已证）；强形式在 L≥120120 被端到端反例证书证伪。C3 正冲击普适化（C1'）。
- （其他已验证阶段性成果：T03 循环构造排除、T10 SC(16/17) 独立复现+McKay 补遗对、T04 无限族下界、T14 63 维第五套独立验证）

## 待办交接（编排器备忘）

- T08 下次 resume 时注入：arXiv:2606.30593（sunflower-free capacity 上界侧，2026-06）。
- T10 下次 resume 时注入：arXiv:2608.11930（重构牌组 census 理论接口，48h 内新文）。
- T14 C1 收官时使用 dossier（T13 findings.md C3-B）：主攻 [320,63] 码最小重量精确计算。
- T07 C1 收官时使用 dossier：若 71/73/75 无进展则转 torus 变体首表 + f₃(n) census。
- T10 下次 resume 时注入：SC(16/17) 新颖性更正（arXiv:1012.5995，2010 已验证 ≤17）——findings/技术报告须改口径为"独立复现+方法交叉验证"；自逆有向图方向不受影响。
- T09 下次 resume 时注入：侦察兵警报——Gupta 2026 全量普查覆盖 n≤14，宽度 3 的 n≤14 最小值可能可从其数据推出；建议改口径为"宽度 3 结构刻画+n>14 外推"或按 verdict 换向。
- T07 下次 resume 时注入：竞争强度=六项之最（Heule/Prellberg 活跃，缺口存活期 ~3.5 周）；71 时间盒 ≤2 周、只打 rct4 之外低对称类，否则转 torus 首表（dossier 备用方向）。
- T15 下次 resume 时注入：2–3 周内抢完 block-Cayley 枚举（Wesley 管线已对准该区域）。
- T13 侦察兵 Cycle 5：观察哨常设职能，下次心跳评估是否续期（当前载体上下文尚新，可直接 resume）。

## 编排器日志

- 02:41 目录创建；128 核 / 2TB / Python 3.12。
- 02:42 13 线程启动（Cycle 1）。
- 03:04 修复 python-sat 安装（pypi SSL 抖动 → 阿里云镜像）；env_notes.md 建立。
- 03:12 首次全面巡检：13/13 线程过文献关进入实验期；4 个线程发现任务书前提过时并已自我纠偏（这正是要求先联网核实 SOTA 的原因）；无 verdict，等待 Cycle 1 各线程收尾。
