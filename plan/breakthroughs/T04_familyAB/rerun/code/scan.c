/* scan.c - exhaustive exact Lonely-Runner scan (thread T04).
 *
 * Enumerates all speed sets 1 <= v_1 < ... < v_k = m with gcd(v)=1, for each
 * layer m in [m_lo, m_hi].  For each set:
 *   phase 1: fast bitset test - does some t=c/d (2<=d<=2m) give
 *            f(t) = min_i ||t v_i|| >= theta ?  If yes: ML >= theta, skip.
 *   phase 2: otherwise compute ML exactly (all-integer arithmetic) over the
 *            full candidate grid {c/d : 2<=d<=2m, 1<=c<=d/2}, which contains
 *            the critical denominators D(v) = {2 v_i} u {v_i+v_j}; by the
 *            critical-time theorem the maximum of f equals max over D(v),
 *            hence the grid max IS the exact sup.  Print "m,v...,num,den".
 *
 * If any exact ML < 1/(k+1) is found, prints COUNTEREXAMPLE and exits 2.
 * All arithmetic is integer; no floating point anywhere in the decision path.
 *
 * usage: scan k m_lo m_hi theta_num theta_den outfile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAXK 9
#define WORDS 4                 /* bitsets up to d = 256 -> max speed 128 */

static int K;
static long THN, THD;           /* theta = THN/THD */
static int DMAX;
static uint64_t *bad;           /* bad[(off[d]+w)*WORDS ...] */
static size_t *off;
static uint64_t full_mask[257][WORDS];

static long long n_combos, n_hard, n_printed;
static FILE *out;
static int v[MAXK + 1];
static int m_cur;

static long long gcdll(long long a, long long b) {
    while (b) { long long t = a % b; a = b; b = t; }
    return a;
}

static void build_tables(void) {
    off = malloc((size_t)(DMAX + 1) * sizeof(size_t));
    size_t tot = 0;
    for (int d = 2; d <= DMAX; d++) { off[d] = tot; tot += (size_t)d; }
    bad = calloc(tot * WORDS, sizeof(uint64_t));
    if (!bad) { fprintf(stderr, "alloc fail\n"); exit(1); }
    for (int d = 2; d <= DMAX; d++) {
        memset(full_mask[d], 0, sizeof full_mask[d]);
        for (int c = 0; c < d; c++)
            full_mask[d][c >> 6] |= 1ULL << (c & 63);
        for (int w = 0; w < d; w++) {
            uint64_t *b = bad + (off[d] + (size_t)w) * WORDS;
            int r = 0;                        /* r = c*w mod d, incremental */
            for (int c = 0; c < d; c++) {
                int rr = (r <= d - r) ? r : d - r;
                if (THD * (long)rr < THN * (long)d)   /* f(c/d) < theta at this runner */
                    b[c >> 6] |= 1ULL << (c & 63);
                r += w; if (r >= d) r -= d;
            }
        }
    }
}

/* 1 iff exists c,d with f(c/d) >= theta (then ML >= theta, instance passes) */
static inline int phase1_pass(void) {
    const int dmax = 2 * m_cur;
    for (int d = 2; d <= dmax; d++) {
        const uint64_t *fm = full_mask[d];
        const size_t o = off[d];
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i = 0; i < K; i++) {
            const uint64_t *b = bad + (o + (size_t)(v[i] % d)) * WORDS;
            a0 |= b[0]; a1 |= b[1]; a2 |= b[2]; a3 |= b[3];
        }
        if (a0 != fm[0] || a1 != fm[1] || a2 != fm[2] || a3 != fm[3])
            return 1;   /* some c is good for every runner at this d */
    }
    return 0;
}

/* exact ML as reduced fraction (pn/pd) */
static void exact_ml(long long *pn, long long *pd) {
    long best_n = 0, best_d = 1;
    const int dmax = 2 * m_cur;
    int step[MAXK], r[MAXK];
    for (int d = 2; d <= dmax; d++) {
        for (int i = 0; i < K; i++) { step[i] = v[i] % d; r[i] = 0; }
        const int half = d / 2;                 /* f(c/d)=f((d-c)/d) symmetry */
        for (int c = 1; c <= half; c++) {
            for (int i = 0; i < K; i++) { r[i] += step[i]; if (r[i] >= d) r[i] -= d; }
            int mn = d;
            for (int i = 0; i < K; i++) {
                int rr = (r[i] <= d - r[i]) ? r[i] : d - r[i];
                if (rr < mn) {
                    mn = rr;
                    if ((long)mn * best_d <= best_n * (long)d) break; /* can't beat best */
                }
            }
            if ((long)mn * best_d > best_n * (long)d) { best_n = mn; best_d = d; }
        }
    }
    long long g = gcdll(best_n, best_d);
    *pn = best_n / g; *pd = best_d / g;
}

static void leaf(long long g_prefix) {
    n_combos++;
    long long g = (g_prefix == 1) ? 1 : gcdll(g_prefix, m_cur);
    if (g != 1) return;             /* scaled copy of a smaller-max instance */
    if (phase1_pass()) return;
    n_hard++;
    long long pn, pd;
    exact_ml(&pn, &pd);
    n_printed++;
    fprintf(out, "%d", m_cur);
    for (int i = 0; i < K; i++) fprintf(out, ",%d", v[i]);
    fprintf(out, ",%lld,%lld\n", pn, pd);
    if (pn * (K + 1) < pd) {        /* ML < 1/(k+1): counterexample!! */
        fprintf(out, "COUNTEREXAMPLE,%d\n", m_cur);
        fflush(out);
        fprintf(stderr, "*** COUNTEREXAMPLE at m=%d ***\n", m_cur);
        exit(2);
    }
}

static void rec(int pos, int lo, long long g) {
    if (pos == K - 1) { leaf(g); return; }
    /* leave room for remaining slots below m_cur */
    const int hi = m_cur - (K - 1 - pos);
    for (int x = lo; x <= hi; x++) {
        v[pos] = x;
        rec(pos + 1, x + 1, (g == 1) ? 1 : gcdll(g, x));
    }
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s k m_lo m_hi theta_num theta_den outfile\n", argv[0]);
        return 1;
    }
    K = atoi(argv[1]);
    int mlo = atoi(argv[2]), mhi = atoi(argv[3]);
    THN = atol(argv[4]); THD = atol(argv[5]);
    DMAX = 2 * mhi;
    if (K < 1 || K > MAXK || DMAX > 256) { fprintf(stderr, "bad params\n"); return 1; }
    build_tables();
    out = fopen(argv[6], "w");
    if (!out) { perror("fopen"); return 1; }
    for (int m = mlo; m <= mhi; m++) {
        m_cur = m; v[K - 1] = m;
        n_combos = n_hard = n_printed = 0;
        clock_t t0 = clock();
        if (m >= K) rec(0, 1, 0);
        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        fprintf(out, "STATS,m=%d,combos=%lld,hard=%lld,printed=%lld,secs=%.2f\n",
                m, n_combos, n_hard, n_printed, secs);
        fflush(out);
    }
    fprintf(out, "DONE\n");
    fclose(out);
    return 0;
}
