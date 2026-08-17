# WorldFoundry Math Lab — 自主数学研究流水线

- 启动时间：2026-08-14 02:41 (UTC+8)
- 编排：主会话 orchestrator（本目录由其维护），13 个并行研究线程（后台 subagent）
- 原则：允许联网调研、允许失败；goal 永不停止 —— 每个线程完成一轮（cycle）后由编排器立即布置下一轮目标。

## 目录结构

```
plan/
  README.md            # 本文件
  STATUS.md            # 滚动状态看板（每次巡检更新）
  threads/Txx_*/       # 每线程独立工作区
    GOAL.md            # 问题、经核实的 SOTA（带出处）、攻击计划
    log.md             # 追加式工作日志
    findings.md        # 结果汇总（严格区分已验证 vs 启发式）
    verdict.md         # 每轮结束：STATUS: PROGRESS|BREAKTHROUGH|BLOCKED + 下轮方向
    code/  results/    # 脚本与数据
  breakthroughs/       # 疑似突破存档 + 对抗验证报告
```

## 线程索引（Cycle 1）

| TID | 问题 | Cycle 1 目标 |
|-----|------|--------------|
| T01_frankl | Frankl 并封闭集猜想 | 熵方法常数数值优化 + 小基集极值族图谱 |
| T02_hadwiger_nelson | 平面色数 χ(R²)∈{5,6,7} | 单位距离图 + SAT 流水线；缩小 5-色图 / 探 6-色 gadget |
| T03_ramsey_r55 | R(5,5)，已知 43–46 | 搜 43 顶点无单色 K5 二染色（下界 →44） |
| T04_lonely_runner | 孤独跑者猜想（k=7 开放） | 严格验证器 + 紧例图谱 + k=7 有界速度系统验证 |
| T05_erdos_straus | 4/n=1/x+1/y+1/z | 覆盖同余系统改进，缩小困难剩余类 |
| T06_graceful | 优雅树 / Bermond lobster 猜想 | 高性能标号搜索器 + lobster 大规模验证与归纳规则提炼 |
| T07_no_three_in_line | 网格 2n 点无三点共线 | 攻 n=47+ 新纪录（退火 + 对称类） |
| T08_sunflowers | Erdős–Rado 向日葵猜想 | 小 k 精确值（ILP/SAT）+ 下界构造常数改进 |
| T09_one_third | 1/3–2/3 猜想 | n≤9 穷举极值偏序集 + n≤14 反例候选搜索 |
| T10_reconstruction | 图重构猜想（n≤11 已验证） | n=12 高危子类（正则/点传递/自补）deck 碰撞验证 |
| T11_hadamard_668 | Hadamard 阶 668（最小开放阶） | Turyn 型/good matrices 搜索 + 参数类排除 |
| T12_zarankiewicz | z(n,n;4,4) / ex(n,K44)（指数开放） | ILP+局部搜索冲击已知表值；代数构造对比 |
| T13_scout | 元研究：侦察可攻破目标 | arXiv 2025–26 扫描，产出 Top15 可攻清单 |

## 协议

- 每轮 verdict.md 首行 `STATUS: PROGRESS | BREAKTHROUGH | BLOCKED`。
- BREAKTHROUGH ⇒ 编排器立即向用户汇报，并并行派 ≥3 个对抗验证 agent 专门找漏洞，报告归档 `breakthroughs/`。
- BLOCKED ⇒ 总结失败原因 + 3 个替代方向，线程换向重启，不消亡。
- 若某方向被评估为更有潜力，允许多线程收敛集中攻击。
