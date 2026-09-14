# *********************************************************************************
# Dynamic programming vs. brute force
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************

import copy
import math
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

FIXED_N = 3
FIXED_K = 3

FIXED_COSTS = [
    [[4, 1, 9]],                             # costs[0]: S -> stage 1
    [[8,  8,  5], [4, 18,  3], [19, 14, 2]], # costs[1]: stage 1 -> stage 2
    [[1,  3,  7], [8, 17, 20], [ 1, 18, 7]], # costs[2]: stage 2 -> stage 3
    [[18], [14], [8]],                        # costs[3]: stage 3 -> T
]


def fixed_costs() -> tuple:
    """Return (N, k, costs) for the fixed graph -- always the same problem."""
    return FIXED_N, FIXED_K, copy.deepcopy(FIXED_COSTS)


def random_costs(N: int, k: int, max_cost: int = 20,
                 seed: Optional[int] = None) -> list:
    """Generate random edge costs (kept for the experiments in Part 3)."""
    rng = random.Random(seed)
    costs = []
    costs.append([[rng.randint(1, max_cost) for _ in range(k)]])
    for _ in range(1, N):
        costs.append([[rng.randint(1, max_cost) for _ in range(k)]
                      for _ in range(k)])
    costs.append([[rng.randint(1, max_cost)] for _ in range(k)])
    return costs


# ==============================================================================
# Part 1 -- Brute force
# ==============================================================================

@dataclass
class BFResult:
    best_cost: float
    best_path: List       # ['S', node_idx, ..., 'T']
    ops: int              # total comparison operations


def brute_force(N: int, k: int, costs: list) -> BFResult:
    best_cost = math.inf
    best_path: List = []
    op_count = [0]
    def recurse(stage: int, node: int, current_cost: float, path: list):
        nonlocal best_cost, best_path
        op_count[0] += 1

        if stage == N:
            total = current_cost + costs[N][node][0]
            op_count[0] += 1
            if total < best_cost:
                best_cost = total
                best_path = path + ['T']
            return

        for nxt in range(k):
            recurse(
                stage + 1,
                nxt,
                current_cost + costs[stage][node][nxt],
                path + [nxt],
            )

    for j in range(k):
        recurse(1, j, costs[0][0][j], ['S', j])

    return BFResult(best_cost=best_cost, best_path=best_path,
                    ops=op_count[0])


# ==============================================================================
# Part 2 -- Dynamic programming: Bellman backward induction
# ==============================================================================

@dataclass
class DPResult:
    best_cost: float
    best_path: List
    ops: int              # total evaluation operations
    V: list               # value table  V[stage][node]
    policy: list          # policy table  policy[stage][node]


def dynamic_programming(N: int, k: int, costs: list) -> DPResult:
    op_count = [0]

    # Initialize tables
    V      = [[math.inf] * k for _ in range(N + 1)]
    policy = [[-1]       * k for _ in range(N + 1)]

    for j in range(k):
        V[N][j] = costs[N][j][0]
        op_count[0] += 1

    for s in range(N - 1, 0, -1):
        for j in range(k):
            best_val  = math.inf
            best_next = -1
            for l in range(k):
                val = costs[s][j][l] + V[s + 1][l]  # use already-solved V[s+1]
                op_count[0] += 1
                if val < best_val:
                    best_val  = val
                    best_next = l
            V[s][j]      = best_val
            policy[s][j] = best_next
    best_cost  = math.inf
    start_node = -1
    for j in range(k):
        c = costs[0][0][j] + V[1][j]
        op_count[0] += 1
        if c < best_cost:
            best_cost  = c
            start_node = j

    path = ['S', start_node]
    cur  = start_node
    for s in range(1, N):
        cur = policy[s][cur]
        path.append(cur)
    path.append('T')

    return DPResult(best_cost=best_cost, best_path=path,
                    ops=op_count[0], V=V, policy=policy)


# ==============================================================================
# Part 3 -- Graph
# ==============================================================================

@dataclass
class ExpResult:
    N: int
    k: int
    bf_ops:    float   # average BF operations
    dp_ops:    float   # average DP operations
    bf_ms:     float   # average BF time (ms)
    dp_ms:     float   # average DP time (ms)
    ratio:     float   # bf_ops / dp_ops
    bf_theory: float   # theoretical k^(N+1)
    dp_theory: float   # theoretical N * k^2


def run_experiment(configs: list, reps: int = 6) -> List[ExpResult]:
    results = []
    for N, k in configs:
        bf_ops_acc = dp_ops_acc = 0.0
        bf_ms_acc  = dp_ms_acc  = 0.0

        for seed in range(reps):
            costs = random_costs(N, k, seed=seed)

            t0 = time.perf_counter()
            bf = brute_force(N, k, costs)
            bf_ms_acc  += (time.perf_counter() - t0) * 1000
            bf_ops_acc += bf.ops

            t1 = time.perf_counter()
            dp = dynamic_programming(N, k, costs)
            dp_ms_acc  += (time.perf_counter() - t1) * 1000
            dp_ops_acc += dp.ops

            assert bf.best_cost == dp.best_cost, (
                f"Correctness error: BF={bf.best_cost} != DP={dp.best_cost} "
                f"for N={N}, k={k}, seed={seed}"
            )

        results.append(ExpResult(
            N=N, k=k,
            bf_ops    = bf_ops_acc / reps,
            dp_ops    = dp_ops_acc / reps,
            bf_ms     = bf_ms_acc  / reps,
            dp_ms     = dp_ms_acc  / reps,
            ratio     = bf_ops_acc / dp_ops_acc,
            bf_theory = k ** (N + 1),
            dp_theory = N * k ** 2,
        ))
        r = results[-1]

    return results


def plot_experiments(results: List[ExpResult]):
    C_BF_K3 = "#fd7a39"   
    C_BF_K5 = "#ff68bd"   
    C_DP_K3 = "#07bdff"   
    C_DP_K5 = "#00c69e"  
    C_RT_K3 = "#a6e23d"   
    C_RT_K5 = "#a890ff"   

    k3 = [r for r in results if r.k == 3]
    k5 = [r for r in results if r.k == 5]
    n_k3, n_k5 = len(k3), len(k5)
    x_k3 = np.arange(n_k3)
    x_k5 = np.arange(n_k3 + 1, n_k3 + 1 + n_k5)
    bw = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Brute force vs. Dynamic programming",
                 fontsize=13, fontweight="bold", y=0.99)

    def add_group_labels(ax):
        ax.axvline(x=n_k3 - 0.5, color="#ccc", lw=1, linestyle="--", zorder=0)
        ax.text(np.mean(x_k3), -0.22, "k = 3",
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, color="#555555", fontweight="bold")
        ax.text(np.mean(x_k5), -0.22, "k = 5",
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, color="#555555", fontweight="bold")
        ax.set_xticks(list(x_k3) + list(x_k5))
        ax.set_xticklabels([f"N={r.N}" for r in k3] + [f"N={r.N}" for r in k5],
                           fontsize=8)


    ax = axes[0, 0]
    ax.bar(x_k3 - bw/2, [r.bf_ops for r in k3], bw,
           label="BF k=3", color=C_BF_K3, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k3 + bw/2, [r.dp_ops for r in k3], bw,
           label="DP k=3", color=C_DP_K3, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k5 - bw/2, [r.bf_ops for r in k5], bw,
           label="BF k=5", color=C_BF_K5, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k5 + bw/2, [r.dp_ops for r in k5], bw,
           label="DP k=5", color=C_DP_K5, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.set_ylabel("Operations (average)"); ax.set_title("Operations performed")
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0); add_group_labels(ax)

    ax = axes[0, 1]
    ax.bar(x_k3 - bw/2, [r.bf_ms for r in k3], bw,
           label="BF k=3", color=C_BF_K3, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k3 + bw/2, [r.dp_ms for r in k3], bw,
           label="DP k=3", color=C_DP_K3, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k5 - bw/2, [r.bf_ms for r in k5], bw,
           label="BF k=5", color=C_BF_K5, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k5 + bw/2, [r.dp_ms for r in k5], bw,
           label="DP k=5", color=C_DP_K5, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.set_ylabel("Time (ms)"); ax.set_title("Average runtime")
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0); add_group_labels(ax)

    ax = axes[1, 0]
    ax.bar(x_k3, [r.ratio for r in k3],
           label="Ratio k=3", color=C_RT_K3, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.bar(x_k5, [r.ratio for r in k5],
           label="Ratio k=5", color=C_RT_K5, edgecolor="#ccc", lw=0.5, zorder=3)
    ax.set_ylabel("Ratio (times)"); ax.set_title("Operation ratio BF / DP")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for i, r in zip(list(x_k3) + list(x_k5), k3 + k5):
        ax.text(i, r.ratio * 1.15, f"{r.ratio:.0f}x",
                ha="center", va="bottom", fontsize=7, color="#555555")
    add_group_labels(ax)

    ax = axes[1, 1]
    ns_k3 = [r.N for r in k3]
    ns_k5 = [r.N for r in k5]
    ax.plot(ns_k3, [r.bf_theory for r in k3], "o--",
            color=C_BF_K3, label=r"BF theory k=3: $k^{N+1}$", lw=2, mec="#999")
    ax.plot(ns_k3, [r.bf_ops    for r in k3], "s-",
            color=C_BF_K3, label="BF empirical k=3", lw=1.5, alpha=0.7, mec="#999")
    ax.plot(ns_k5, [r.bf_theory for r in k5], "o--",
            color=C_BF_K5, label=r"BF theory k=5: $k^{N+1}$", lw=2, mec="#999")
    ax.plot(ns_k5, [r.bf_ops    for r in k5], "s-",
            color=C_BF_K5, label="BF empirical k=5", lw=1.5, alpha=0.7, mec="#999")
    ax.plot(ns_k3, [r.dp_theory for r in k3], "o--",
            color=C_DP_K3, label=r"DP theory k=3: $N\cdot k^2$", lw=2, mec="#999")
    ax.plot(ns_k3, [r.dp_ops    for r in k3], "s-",
            color=C_DP_K3, label="DP empirical k=3", lw=1.5, alpha=0.7, mec="#999")
    ax.plot(ns_k5, [r.dp_theory for r in k5], "o--",
            color=C_DP_K5, label=r"DP theory k=5: $N\cdot k^2$", lw=2, mec="#999")
    ax.plot(ns_k5, [r.dp_ops    for r in k5], "s-",
            color=C_DP_K5, label="DP empirical k=5", lw=1.5, alpha=0.7, mec="#999")
    ax.set_xlabel("N (stages)")
    ax.set_ylabel("Operations")
    ax.set_title("Theoretical vs. empirical order")
    ax.set_yscale("log"); ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

def fmt_path(path: list) -> str:
    """Format a path list as a human-readable string (1-indexed nodes)."""
    parts = []
    for p in path:
        parts.append(p if p in ('S', 'T') else str(p + 1))
    return " -> ".join(parts)

def count_bf_visits(N: int, k: int) -> dict:
    visits: dict = {(s, j): 0 for s in range(1, N + 1) for j in range(k)}

    def rec(stage: int, node: int):
        visits[(stage, node)] += 1
        if stage < N:
            for nxt in range(k):
                rec(stage + 1, nxt)

    for j in range(k):
        rec(1, j)
    return visits


def _graph_helpers(N: int, k: int, opt_path: list, costs: list):

    X_GAP, Y_GAP, R = 2.0, 1.4, 0.40

    def node_pos(stage: int, node: int = 0):
        x = stage * X_GAP
        y = (k - 1) * Y_GAP / 2 if stage in (0, N + 1) else node * Y_GAP
        return x, y

    def on_optimal_path(stage: int, node: int) -> bool:
        if stage in (0, N + 1):
            return True
        return len(opt_path) > stage and opt_path[stage] == node

    def draw_edge(ax_, xs, ys, xt, yt, cost, highlight):
        dx, dy = xt - xs, yt - ys
        d = math.hypot(dx, dy)
        nx, ny = dx / d, dy / d
        x1, y1 = xs + nx * R, ys + ny * R
        x2, y2 = xt - nx * R, yt - ny * R
        color = "#fd7a39" if highlight else "#DDDDDD"
        lw    = 2.2       if highlight else 0.6
        ax_.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=color,
                                     lw=lw, mutation_scale=11),
                     zorder=3 if highlight else 1)
        if highlight:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset_y = 0.13 if dy >= 0 else -0.13
            ax_.text(mx, my + offset_y, str(cost),
                     ha="center", va="center", fontsize=7,
                     color="#c05000", fontweight="bold", zorder=4)

    def draw_node(ax_, x, y, label, sublabel=None,
                  fc="#07bdff", ec="#0080cc"):
        ax_.add_patch(plt.Circle((x, y), R, fc=fc, ec=ec, lw=1.5, zorder=4))
        if sublabel:
            ax_.text(x, y + 0.11, label, ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white", zorder=5)
            ax_.text(x, y - 0.13, sublabel, ha="center", va="center",
                     fontsize=7, color="white", alpha=0.95, zorder=5)
        else:
            ax_.text(x, y, label, ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white", zorder=5)

    def draw_column_labels(ax_):
        ax_.text(node_pos(0)[0],     -0.65, "S",
                 ha="center", fontsize=9, color="#666", fontstyle="italic")
        for s in range(1, N + 1):
            ax_.text(node_pos(s, 0)[0], -0.65, f"Stage {s}",
                     ha="center", fontsize=9, color="#666", fontstyle="italic")
        ax_.text(node_pos(N + 1)[0], -0.65, "T",
                 ha="center", fontsize=9, color="#666", fontstyle="italic")

    def draw_all_edges(ax_):
        # S -> stage 1
        xs0, ys0 = node_pos(0)
        for j in range(k):
            draw_edge(ax_, xs0, ys0, *node_pos(1, j),
                      costs[0][0][j], on_optimal_path(1, j))
        # stage s -> stage s+1
        for s in range(1, N):
            for j in range(k):
                for l in range(k):
                    draw_edge(ax_, *node_pos(s, j), *node_pos(s + 1, l),
                              costs[s][j][l],
                              on_optimal_path(s, j) and on_optimal_path(s + 1, l))
        # stage N -> T
        xt_T, yt_T = node_pos(N + 1)
        for j in range(k):
            draw_edge(ax_, *node_pos(N, j), xt_T, yt_T,
                      costs[N][j][0], on_optimal_path(N, j))

    def draw_all_edges_gray(ax_):
        xs0, ys0 = node_pos(0)
        for j in range(k):
            draw_edge(ax_, xs0, ys0, *node_pos(1, j), costs[0][0][j], False)
        for s in range(1, N):
            for j in range(k):
                for l in range(k):
                    draw_edge(ax_, *node_pos(s, j), *node_pos(s + 1, l),
                              costs[s][j][l], False)
        xt_T, yt_T = node_pos(N + 1)
        for j in range(k):
            draw_edge(ax_, *node_pos(N, j), xt_T, yt_T,
                      costs[N][j][0], False)

    xlim = (-0.6, (N + 1) * X_GAP + 0.6)
    ylim = (-0.95, (k - 1) * Y_GAP + 0.75)

    return (node_pos, on_optimal_path, draw_node,
            draw_column_labels, draw_all_edges, draw_all_edges_gray, xlim, ylim)


def plot_bf(N: int, k: int, costs: list, opt_path: list):
    visits       = count_bf_visits(N, k)
    max_visits   = max(visits.values())
    total_visits = sum(visits.values())
    total_paths  = k ** N

    (node_pos, on_opt, draw_node,
     draw_col_labels, _, _2, xlim, ylim) = _graph_helpers(N, k, opt_path, costs)

    R = 0.40

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_aspect("auto"); ax.axis("off")
    ax.set_title(
        f"Brute force -- N={N}, k={k}\n",
        fontsize=13, fontweight="bold", y=0.99,
    )

    def draw_edge(xs, ys, xt, yt, cost, is_optimal):
        dx, dy = xt - xs, yt - ys
        d = math.hypot(dx, dy)
        nx, ny = dx / d, dy / d
        x1, y1 = xs + nx * R, ys + ny * R
        x2, y2 = xt - nx * R, yt - ny * R
        if is_optimal:
            color, lw, zo = "#fd7a39", 2.5, 4
        else:
            color, lw, zo = "#CCCCCC", 0.8, 1
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=lw, mutation_scale=10), zorder=zo)
        if is_optimal:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            oy = 0.14 if dy >= 0 else -0.14
            ax.text(mx, my + oy, str(cost), ha="center", va="center",
                    fontsize=7, color="#c05000", fontweight="bold", zorder=5)

    # Draw all edges
    xs0, ys0 = node_pos(0)
    for j in range(k):
        draw_edge(xs0, ys0, *node_pos(1, j), costs[0][0][j], on_opt(1, j))
    for s in range(1, N):
        for j in range(k):
            for l in range(k):
                draw_edge(*node_pos(s, j), *node_pos(s + 1, l),
                          costs[s][j][l], on_opt(s, j) and on_opt(s + 1, l))
    xtT, ytT = node_pos(N + 1)
    for j in range(k):
        draw_edge(*node_pos(N, j), xtT, ytT, costs[N][j][0], on_opt(N, j))

    # Draw nodes -- label shows visit count and formula
    draw_node(ax, *node_pos(0),     "S", fc="#a890ff", ec="#7050cc")
    draw_node(ax, *node_pos(N + 1), "T", fc="#00c69e", ec="#008060")
    for s in range(1, N + 1):
        for j in range(k):
            x, y = node_pos(s, j)
            v = visits[(s, j)]   # = k^(s-1) for this staged graph
            t = (v - 1) / max(max_visits - 1, 1)
            if   v == 1:   fc, ec = "#a6e23d", "#70a010"   # lime = visited once
            elif t < 0.5:  fc, ec = "#ff68bd", "#cc3090"   # pink = few repeats
            else:          fc, ec = "#a890ff", "#7050cc"   # purple = many repeats
            # sublabel: "visits=V (=k^s-1)"
            sublabel = f"visits={v}"
            draw_node(ax, x, y, f"n{j+1}", sublabel, fc=fc, ec=ec)

    draw_col_labels(ax)
    ax.legend(handles=[
        mpatches.Patch(fc="#a6e23d", ec="#70a010", lw=1,
                       label=r"visits=1  (stage 1, $k^0=1$)"),
        mpatches.Patch(fc="#ff68bd", ec="#cc3090", lw=1,
                       label=f"visits={k}  (stage 2, $k^1={k}$)"),
        mpatches.Patch(fc="#a890ff", ec="#7050cc", lw=1,
                       label=f"visits={max_visits}  (stage {N}, $k^{{N-1}}$={k**(N-1)})"),
        mpatches.Patch(fc="#fd7a39", ec="#c05000", lw=1,
                       label="Optimal path (orange)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.14),
       ncol=2, fontsize=9, framealpha=0.9, edgecolor="#ddd")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.tight_layout()


def plot_pd(N: int, k: int, costs: list, opt_path: list, dp_result: DPResult):
    visits   = count_bf_visits(N, k)
    total_bf = sum(visits.values())
    total_dp = N * k

    (node_pos, on_opt, draw_node,
     draw_col_labels, draw_all_edges, draw_all_edges_gray, xlim, ylim) = \
        _graph_helpers(N, k, opt_path, costs)

    X_GAP, Y_GAP, R = 2.0, 1.4, 0.40

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(
        f"Dynamic programming (Bellman backward) N={N}, k={k}\n"
        f"(DP visits: {total_dp}  vs BF: {total_bf})",
        fontsize=13, fontweight="bold", y=0.99,
    )

    draw_all_edges_gray(ax)
    draw_node(ax, *node_pos(0),     "S", fc="#a890ff", ec="#7050cc")
    draw_node(ax, *node_pos(N + 1), "T", fc="#00c69e", ec="#008060")

    for s in range(N, 0, -1):
        for j in range(k):
            x, y = node_pos(s, j)
            v = int(dp_result.V[s][j])
            if on_opt(s, j): fc, ec = "#fd7a39", "#c05000"
            else:             fc, ec = "#07bdff", "#0080cc"
            draw_node(ax, x, y, f"n{j+1}", f"V={v}", fc=fc, ec=ec)

    draw_col_labels(ax)
    path_nodes = [] 
    for i, p in enumerate(opt_path):
        if p == 'S':
            path_nodes.append((0, None))
        elif p == 'T':
            path_nodes.append((N + 1, None))
        else:
            path_nodes.append((i, p))

    def draw_backward_arrow(x_from, y_from, x_to, y_to, label):
        dx, dy = x_to - x_from, y_to - y_from
        d = math.hypot(dx, dy)
        nx, ny = dx / d, dy / d
        perp_x, perp_y = -ny * 0.18, nx * 0.18
        x1 = x_from + nx * R + perp_x
        y1 = y_from + ny * R + perp_y
        x2 = x_to - nx * R + perp_x
        y2 = y_to - ny * R + perp_y
        ax.plot([x1, x2], [y1, y2],
                color="#fd7a39", lw=2.0, linestyle="--", zorder=6)
        ax.annotate("", xy=(x2, y2),
                    xytext=(x1 * 0.05 + x2 * 0.95, y1 * 0.05 + y2 * 0.95),
                    arrowprops=dict(arrowstyle="->", color="#fd7a39",
                                   lw=2.0, mutation_scale=14), zorder=7)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + perp_x * 1.5, my + perp_y * 1.5,
                label, ha="center", va="center",
                fontsize=7, color="#C05800", fontweight="bold", zorder=7)

    xt_T, yt_T = node_pos(N + 1)

    opt_N = opt_path[N]
    xN, yN = node_pos(N, opt_N)
    draw_backward_arrow(xt_T, yt_T, xN, yN,
                        str(costs[N][opt_N][0]))
    for s in range(N, 1, -1):
        j_right = opt_path[s]
        j_left  = opt_path[s - 1]
        draw_backward_arrow(*node_pos(s, j_right), *node_pos(s - 1, j_left),
                            str(costs[s - 1][j_left][j_right]))

    opt_1 = opt_path[1]
    draw_backward_arrow(*node_pos(1, opt_1), *node_pos(0),
                        str(costs[0][0][opt_1]))

    ax.legend(handles=[
        mpatches.Patch(fc="#fd7a39", ec="#c05000", lw=1,
                       label=f"Optimal path (cost = {dp_result.best_cost})"),
        mpatches.Patch(fc="#00c69e", ec="#008060", lw=1,
                       label="Solved once (V = min cost to T)"),
        plt.Line2D([0], [0], color="#fd7a39", lw=2, linestyle="--",
                   label="Bellman backward arrows (T -> S)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.12),
       ncol=2, fontsize=8, framealpha=0.85, edgecolor="#ddd")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    plt.tight_layout()


def plot_graph(N: int, k: int, costs: list,
               opt_path: list, dp_result: DPResult):
    plot_bf(N, k, costs, opt_path)
    plot_pd(N, k, costs, opt_path, dp_result)

def ask_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input. Please enter an integer.")


def ask_costs(N: int, k: int) -> list:
    print("\n  Enter each edge cost (positive integer):")
    costs = []
    row = []
    for j in range(k):
        c = ask_int(f"    S -> node {j+1} (stage 1): ", 1, 9999)
        row.append(c)
    costs.append([row])
    for s in range(1, N):
        stage_rows = []
        for j in range(k):
            row = []
            for l in range(k):
                c = ask_int(
                    f"    Stage {s} node {j+1} -> stage {s+1} node {l+1}: ",
                    1, 9999)
                row.append(c)
            stage_rows.append(row)
        costs.append(stage_rows)
    last = []
    for j in range(k):
        c = ask_int(f"    Stage {N} node {j+1} -> T: ", 1, 9999)
        last.append([c])
    costs.append(last)
    return costs


def choose_graph() -> tuple:
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  Graph setup")
    print(SEP)
    print("  1. Use the fixed graph  (N=3, k=3, predefined costs)")
    print("  2. Random graph         (choose N and k)")
    print("  3. Enter costs manually (choose N and k, then type each cost)")
    choice = ask_int("  Choose an option [1/2/3]: ", 1, 3)

    if choice == 1:
        N, k, costs = fixed_costs()
        return N, k, costs

    N = ask_int("  Number of stages N (1-8): ", 1, 8)
    k = ask_int("  Nodes per stage k  (1-6): ", 1, 6)

    if choice == 2:
        seed_input = input("  Random seed (press Enter for random): ").strip()
        seed = int(seed_input) if seed_input.isdigit() else None
        costs = random_costs(N, k, seed=seed)
        label = f"seed={seed}" if seed is not None else "no seed"
        return N, k, costs

    costs = ask_costs(N, k)
    return N, k, costs


def main():
    SEP  = "=" * 70
    SEP2 = "-" * 70
    N, k, costs = choose_graph()
    bf = brute_force(N, k, costs)
    dp = dynamic_programming(N, k, costs)
    assert bf.best_cost == dp.best_cost, "Correctness error!"
    plot_graph(N, k, costs, bf.best_path, dp)
    configs = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),  # k=3, N=2..6
        (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),  # k=5, N=2..6
    ]
    results = run_experiment(configs, reps=6)
    plot_experiments(results)


if __name__ == "__main__":
    main()
    plt.show()


"""
CONCLUSION                                                      
  ---------------------                                                       
  Dijkstra finds the shortest path from a source node S to all other nodes   
  in a graph with non-negative edge weights.  It maintains a set of         
  "settled" nodes and a priority queue with tentative distances.  At each    
  step it extracts the node with the smallest tentative distance.            
                                                                                                
  SIMILARITIES WITH DP                                           
  ----------------------------------                                          
  1. Optimal substructure: both exploit the fact that every sub-path of an   
     optimal path is itself optimal (Bellman's principle).                    
  2. No recomputation: each sub-problem is solved once and reused.            
  3. Non-negative weights: both require costs >= 0 for correctness.           
  4. On a staged graph, Dijkstra would settle nodes in exactly the same       
     stage order as backward DP (all edges point forward, so distance        
     increases monotonically with stage index). 


  DIFFERENCES WITH DP                                           
  ----------------------------------     
  1. Bellman goes from T → S (backward). Dijkstra goes from S → T (forward).
  2. Bellman requires a staged graph (layered DAG). Dijkstra works on any graph.
  3. Bellman processes nodes in a fixed order (stage N, N-1, ..., 1). 
    Dijkstra dynamically selects the next node based on the smallest tentative
    distance.
  4. Bellman uses a value table V[N][k]. Dijkstra uses a priority queue (min-heap).
  5. Bellman is O(N·k²). Dijkstra is O((V+E) log V) — slower by a logarithmic factor.
  6. Bellman computes the minimum cost from each node to T. 
     Dijkstra computes the minimum cost from S to each node.
"""
