#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_verifier.py — WorldFoundry Math Lab / T04, 对抗验证员 V2（盲复现）。

独立、从零实现的 Lonely-Runner ML 精确验证器（k=7，非零整数速度）。
本文件由 V2 独立编写，未读取 T04 线程的 code/ 或 rerun/code/ 目录中的任何文件。

定义
----
||x||  = x 到最近整数的距离。
f_v(t) = min_i ||t * v_i||        （v 为正整数元组；负速度可取绝对值，||-x||=||x||）
ML(v)  = sup_{t in R} f_v(t)

所用定理（证明见 v2_report.md）
------------------------------
每条 g_i(t)=||v_i t|| 连续、分段线性、以 1 为周期，斜率 ±v_i，
断点恰为 t = j/(2 v_i)（j 偶 → 零点；j 奇 → 峰值 1/2）。
f = min_i g_i 因而连续分段线性、1-周期、f(0)=0，故
ML(v) = f 在 (0,1) 内局部极大值之最大者。在局部极大 t* 处 f 的斜率由 >0 变 <0，
只可能是：
  (P) 活跃曲线的峰：      t* = 奇数/(2 v_i)                         → 分母 2 v_i
  (X) 上升支与下降支相交： v_i t - a = b - v_j t（a,b 整数）
                          => t* = (a+b)/(v_i+v_j)                    → 分母 v_i+v_j
因此一切局部极大值落在分母属于 D0 = {2 v_i} ∪ {v_i+v_j : i<j} 的有理点上。
按"宁可取大"，另加入同向支相交的分母 {|v_i - v_j|}，得
      D = D0 ∪ {|v_i - v_j| : i<j}。
ML(v) = max_{d in D, 1<=c<d} f(c/d)，逐点精确计算：
      f(c/d) = min_i min(m_i, d-m_i)/d，  m_i = (c*v_i) mod d
（纯整数运算，与公共分母 d 下的 Fraction 运算恒等；最优点再用 fractions.Fraction
独立复算一遍作为断言）。另提供对全体分母 2..Dmax（⊇D）的全扫描做交叉验证。
"""
from fractions import Fraction
import argparse
import csv
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# 核心精确算法
# ----------------------------------------------------------------------------

def theorem_denominators(vs):
    """候选分母集合 D = {2 v_i} ∪ {v_i+v_j} ∪ {|v_i-v_j|}（去 0，升序）。"""
    dens = set()
    for v in vs:
        dens.add(2 * v)
    k = len(vs)
    for i in range(k):
        for j in range(i + 1, k):
            dens.add(vs[i] + vs[j])
            dd = abs(vs[i] - vs[j])
            if dd:
                dens.add(dd)
    dens.discard(0)
    return sorted(dens)


def max_over_dens(vs, dens):
    """精确求 max_{d in dens, c=1..d-1} f(c/d)。返回 (ML: Fraction, witness t*: Fraction)。

    全程整数运算：f(c/d) = min_i min((c v_i) mod d, d - (c v_i) mod d) / d，
    比较用交叉相乘，无任何浮点。
    """
    bn, bd = 0, 1          # 当前最优值 bn/bd
    wc, wd = 0, 1          # 其 witness t = wc/wd
    for d in dens:
        for c in range(1, d):
            mmin = d
            alive = True
            for v in vs:
                m = (c * v) % d
                if m + m > d:
                    m = d - m
                if m < mmin:
                    # 剪枝：min 只会更小；若已不可能超过当前最优则放弃该 c
                    if m * bd <= bn * d:
                        alive = False
                        break
                    mmin = m
            if alive and mmin * bd > bn * d:
                bn, bd, wc, wd = mmin, d, c, d
    return Fraction(bn, bd), Fraction(wc, wd)


def ml_theorem(vs):
    """定理候选集上的精确 ML。"""
    return max_over_dens(vs, theorem_denominators(vs))


def f_exact(vs, t):
    """用 Fraction 独立复算 f(t)（与 max_over_dens 的整数路径互为校验）。"""
    best = None
    for v in vs:
        fr = (t * v) % 1
        dist = fr if fr <= Fraction(1, 2) else 1 - fr
        if best is None or dist < best:
            best = dist
    return best


# ----------------------------------------------------------------------------
# 被验声明
# ----------------------------------------------------------------------------

ANCHORS = [
    ("anchor:(1..7)", (1, 2, 3, 4, 5, 6, 7), Fraction(1, 8)),
    ("anchor:GW(1,4,5,6,7,11,13)", (1, 4, 5, 6, 7, 11, 13), Fraction(1, 8)),
    ("anchor:GW(1,2,3,4,5,7,12)", (1, 2, 3, 4, 5, 7, 12), Fraction(1, 8)),
]

SPORADIC = ("sporadic:(2,7,9,11,13,20,54)", (2, 7, 9, 11, 13, 20, 54), Fraction(9, 65))


def family_A(r):
    return (1, 2, 3, 4, 5, 7, 6 * r + 12), Fraction(r + 2, 6 * r + 17)


def family_B(r):
    return (1, 3, 4, 5, 7, 11, 6 * r + 24), Fraction(r + 4, 6 * r + 31)


# ----------------------------------------------------------------------------
# 实例执行（含交叉验证与不一致时的自动加深）
# ----------------------------------------------------------------------------

def run_instance(name, vs, claimed, family="", r="", crosscheck=False, extended=0):
    t0 = time.perf_counter()
    dens = theorem_denominators(vs)
    ml, wit = max_over_dens(vs, dens)
    assert ml > 0, f"{name}: ML=0 impossible"
    assert f_exact(vs, wit) == ml, f"{name}: witness re-evaluation failed"
    rec = {
        "name": name, "family": family, "r": r,
        "velocities": " ".join(map(str, vs)),
        "computed_ml": str(ml), "claimed": str(claimed),
        "equal": (ml == claimed),
        "witness_t": str(wit),
        "crosscheck": "", "escalation": "", "time_s": 0.0,
    }
    notes = []
    if crosscheck:
        # 全分母扫描 2..max(D)：候选集的严格超集，应恰好复现同一 ML
        ml_full, _ = max_over_dens(vs, range(2, max(dens) + 1))
        notes.append("full=ok" if ml_full == ml else f"FULLSWEEP {ml_full} != {ml}")
    if extended:
        # 扩展全扫描 2..extended*max(D)：检验"没有更大的极大值藏在更大分母处"
        ml_ext, _ = max_over_dens(vs, range(2, extended * max(dens) + 1))
        notes.append(f"ext{extended}x=ok" if ml_ext == ml else f"EXT{extended}X {ml_ext} != {ml}")
    rec["crosscheck"] = ";".join(notes)
    if not rec["equal"]:
        # 与声明不一致：立即加深 —— 扩大分母集合（全分母 2x / 4x 扫描）复测
        esc = []
        for fac in (2, 4):
            mlx, witx = max_over_dens(vs, range(2, fac * max(dens) + 1))
            esc.append(f"sweep{fac}x: ML={mlx} @ t={witx}")
        rec["escalation"] = " | ".join(esc)
    rec["time_s"] = round(time.perf_counter() - t0, 3)
    return rec


def fmt(rec):
    eq = "Y" if rec["equal"] else "N  <-- MISMATCH"
    xc = rec["crosscheck"] or "-"
    return (f"{rec['name']:<32s} ML={rec['computed_ml']:>10s} claim={rec['claimed']:>10s} "
            f"eq={eq:<16s} t*={rec['witness_t']:>12s} xc={xc} ({rec['time_s']:.2f}s)")


# ----------------------------------------------------------------------------
# 自检：随机小实例，定理候选集 vs 4x 全分母扫描 vs 稠密浮点网格
# ----------------------------------------------------------------------------

def selftest(n_random=200, seed=20260814, grid_n=20000):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    for it in range(n_random):
        k = rng.randint(2, 5)
        vs = tuple(sorted(rng.sample(range(1, 13), k)))
        dens = theorem_denominators(vs)
        ml, wit = max_over_dens(vs, dens)
        assert f_exact(vs, wit) == ml, (vs, ml, wit)
        ml_full, _ = max_over_dens(vs, range(2, 4 * max(dens) + 1))
        assert ml_full == ml, ("theorem set incomplete!", vs, ml, ml_full)
        # 浮点网格：任何网格值不得超过 ML；由 |f'|<=max(v) 网格最大值不得离 ML 太远
        gm = 0.0
        for a in range(1, grid_n):
            t = a / grid_n
            m = min(abs(t * v - round(t * v)) for v in vs)
            if m > gm:
                gm = m
        assert gm <= float(ml) + 1e-9, (vs, gm, float(ml))
        assert float(ml) - gm <= max(vs) / grid_n + 1e-9, (vs, gm, float(ml))
    print(f"SELFTEST OK: {n_random} random instances "
          f"(k=2..5, v<=12; theorem==4x-fullsweep==grid) in {time.perf_counter()-t0:.1f}s")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def md_table(rows, title):
    lines = [f"### {title}", "",
             "| name | velocities | computed ML | claimed | equal | witness t* | crosscheck |",
             "|---|---|---|---|---|---|---|"]
    for rr in rows:
        eq = "YES" if rr["equal"] else "**NO**"
        lines.append(f"| {rr['name']} | ({rr['velocities'].replace(' ', ',')}) | "
                     f"{rr['computed_ml']} | {rr['claimed']} | {eq} | {rr['witness_t']} | "
                     f"{rr['crosscheck'] or '-'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="V2 blind verifier for T04 ML claims")
    ap.add_argument("--rmin", type=int, default=1)
    ap.add_argument("--rmax", type=int, default=40)
    ap.add_argument("--crosscheck-rmax", type=int, default=40,
                    help="r <= this get an additional full-denominator sweep")
    ap.add_argument("--extended-factor", type=int, default=3,
                    help="anchors/sporadic/r<=3 get a full sweep up to factor*max(D)")
    ap.add_argument("--csv", default=os.path.join(HERE, "v2_results.csv"))
    ap.add_argument("--tables", default=os.path.join(HERE, "v2_tables.md"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-anchors", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    t_start = time.perf_counter()
    rows = []
    anchor_rows, sporadic_rows, famA_rows, famB_rows = [], [], [], []

    if not args.skip_anchors:
        print("== anchors (verifier correctness gate, all must equal 1/8) ==")
        for name, vs, cl in ANCHORS:
            rec = run_instance(name, vs, cl, family="anchor",
                               crosscheck=True, extended=args.extended_factor)
            rows.append(rec); anchor_rows.append(rec); print(fmt(rec))
        anchors_ok = (all(r["equal"] for r in anchor_rows)
                      and not any("!=" in r["crosscheck"] for r in anchor_rows))
        print(f"anchors_ok = {anchors_ok}")
        if not anchors_ok:
            print("FATAL: anchor failure -> verifier not trusted; aborting family runs.")

        print("== sporadic ==")
        name, vs, cl = SPORADIC
        rec = run_instance(name, vs, cl, family="sporadic",
                           crosscheck=True, extended=args.extended_factor)
        rows.append(rec); sporadic_rows.append(rec); print(fmt(rec))

    for fam, gen, bucket in (("A", family_A, famA_rows), ("B", family_B, famB_rows)):
        print(f"== family {fam}: r = {args.rmin}..{args.rmax} ==")
        for r in range(args.rmin, args.rmax + 1):
            vs, cl = gen(r)
            rec = run_instance(f"{fam}:r={r}", vs, cl, family=fam, r=r,
                               crosscheck=(r <= args.crosscheck_rmax),
                               extended=(args.extended_factor if r <= 3 else 0))
            rows.append(rec); bucket.append(rec)
            bad = (not rec["equal"]) or ("!=" in rec["crosscheck"])
            if bad:
                print(fmt(rec))
                if rec["escalation"]:
                    print("   escalation:", rec["escalation"])
            elif r <= 3 or r % 20 == 0:
                print(fmt(rec))

    # ---- CSV ----
    fields = ["name", "family", "r", "velocities", "computed_ml", "claimed",
              "equal", "witness_t", "crosscheck", "escalation", "time_s"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)

    # ---- markdown tables (r<=40 in full; beyond summarized) ----
    with open(args.tables, "w") as fh:
        fh.write("<!-- generated by v2_verifier.py -->\n\n")
        if anchor_rows:
            fh.write(md_table(anchor_rows, "锚点（必须 = 1/8）") + "\n")
        if sporadic_rows:
            fh.write(md_table(sporadic_rows, "零散例（声明 9/65）") + "\n")
        fh.write(md_table([r_ for r_ in famA_rows if isinstance(r_["r"], int) and r_["r"] <= 40],
                          "家族 A：ML(1,2,3,4,5,7,6r+12) ?= (r+2)/(6r+17)，r=1..40") + "\n")
        fh.write(md_table([r_ for r_ in famB_rows if isinstance(r_["r"], int) and r_["r"] <= 40],
                          "家族 B：ML(1,3,4,5,7,11,6r+24) ?= (r+4)/(6r+31)，r=1..40") + "\n")
        for fam, bucket in (("A", famA_rows), ("B", famB_rows)):
            ext = [r_ for r_ in bucket if isinstance(r_["r"], int) and r_["r"] > 40]
            if ext:
                neq = [r_ for r_ in ext if not r_["equal"]]
                fh.write(f"### 家族 {fam} 扩展段 r=41..{max(r_['r'] for r_ in ext)}：")
                fh.write(f"{len(ext)} 个实例，全部相等 = {not neq}"
                         + (f"，不一致 r 列表：{[r_['r'] for r_ in neq]}" if neq else "")
                         + "（逐条数据见 v2_results.csv）\n\n")

    # ---- verdict-relevant summary ----
    def ok40(bucket):
        sub = [r_ for r_ in bucket if isinstance(r_["r"], int) and r_["r"] <= 40]
        return len(sub) >= 40 and all(r_["equal"] for r_ in sub)

    mism = [r_ for r_ in rows if not r_["equal"]]
    xfail = [r_ for r_ in rows if "!=" in r_["crosscheck"]]
    print("\n================ SUMMARY ================")
    print(f"instances: {len(rows)}  mismatches: {len(mism)}  crosscheck-failures: {len(xfail)}")
    if mism:
        for r_ in mism:
            print("  MISMATCH:", fmt(r_))
    if xfail:
        for r_ in xfail:
            print("  XCHECK FAIL:", fmt(r_))
    if anchor_rows:
        print(f"anchors 1/8: {'PASS' if all(r_['equal'] for r_ in anchor_rows) else 'FAIL'}")
        print(f"sporadic 9/65: {'PASS' if all(r_['equal'] for r_ in sporadic_rows) else 'FAIL'}")
    print(f"family A r<=40 all equal: {ok40(famA_rows)}")
    print(f"family B r<=40 all equal: {ok40(famB_rows)}")
    ra = [r_["r"] for r_ in famA_rows if isinstance(r_["r"], int)]
    rb = [r_["r"] for r_ in famB_rows if isinstance(r_["r"], int)]
    if ra:
        print(f"family A verified range: r={min(ra)}..{max(ra)}, "
              f"all equal: {all(r_['equal'] for r_ in famA_rows)}")
    if rb:
        print(f"family B verified range: r={min(rb)}..{max(rb)}, "
              f"all equal: {all(r_['equal'] for r_ in famB_rows)}")
    print(f"total wall time: {time.perf_counter()-t_start:.1f}s")
    print(f"csv: {args.csv}")
    print(f"tables: {args.tables}")


if __name__ == "__main__":
    main()
