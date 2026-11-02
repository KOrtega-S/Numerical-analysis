# *********************************************************************************
# FEM
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************

import numpy as np
import matplotlib.pyplot as plt

nInternal = 20   # number of internal nodes, not counting the two endpoints
seed = 5        # fixes the mesh so it comes out the same on every run

def f(x):
    return np.pi**2 * np.sin(np.pi * x)

def uExact(x):
    return np.sin(np.pi * x)

# Mesh

# Random internal nodes in (0, 1) plus the two endpoints
rng = np.random.default_rng(seed)
x = np.sort(np.concatenate(([0.0], rng.uniform(0, 1, nInternal), [1.0])))
h = np.diff(x)
n = len(x)

# Assembly and solution
K = np.zeros((n, n))
F = np.zeros(n)

# 2-point Gauss quadrature on [-1, 1]
xi = np.array([-1, 1]) / np.sqrt(3)
weights = np.array([1.0, 1.0])

for e in range(n - 1):
    he = h[e]
    nodes = [e, e + 1]

    # Element stiffness matrix
    K[np.ix_(nodes, nodes)] += np.array([[1, -1], [-1, 1]]) / he

    # Load vector: integral of f times the basis functions, using Gauss
    xg = x[e] + (xi + 1) * he / 2
    phi = np.array([1 - (xg - x[e]) / he, (xg - x[e]) / he])
    F[nodes] += phi @ (weights * f(xg)) * he / 2

# Since u(0) = u(1) = 0, only the internal nodes are solved
u = np.zeros(n)
u[1:-1] = np.linalg.solve(K[1:-1, 1:-1], F[1:-1])

# Plots: the three figures share a single window and the x axis
ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": grid, "axes.labelcolor": muted, "axes.titlecolor": ink,
                     "axes.titleweight": "semibold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
                     "xtick.color": muted, "ytick.color": muted, "legend.frameon": False,
                     "legend.labelcolor": ink, "font.size": 10})
dot = dict(ms=7, mec="#fcfcfb", mew=1.5, zorder=3)   # markers with a ring in the background color

xx = np.linspace(0, 1, 400)
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="tight",
                        gridspec_kw={"height_ratios": [3, 3, 1.2]})

# 1. Solution
ax = axs[0]
if uExact is not None:
    ax.plot(xx, uExact(xx), "--", color=muted, lw=1.5, label="exact")
ax.plot(x, u, "-o", color=blue, lw=2, label="finite elements", **dot)
ax.set_ylabel("u(x)")
maxError = np.max(np.abs(u - uExact(x)))   # maximum error at the nodes
ax.set_title(f"Solution of -u'' = f with linear finite elements (max error = {maxError:.2e})")
ax.legend(loc="lower center")

# 2. Load function
ax = axs[1]
fx = f(xx) * np.ones_like(xx)   # ones_like in case f is constant
ax.plot(xx, fx, color=orange, lw=2)
ax.fill_between(xx, fx, color=orange, alpha=0.08, lw=0)
ax.plot(x, f(x) * np.ones_like(x), "o", color=orange, **dot)
ax.set_ylabel("f(x)")
ax.set_title("Load function f(x)")

# 3. Node distribution
ax = axs[2]
ax.axhline(0, color=grid, lw=2, zorder=1)
ax.plot(x[1:-1], np.zeros(n - 2), "o", color=aqua, label="internal nodes", **dot)
ax.plot(x[[0, -1]], [0, 0], "s", color=ink, label="boundary", **dot)
ax.set(yticks=[], ylim=(-1, 1.6), xlabel="x")
ax.spines["left"].set_visible(False)
ax.grid(False, axis="y")
ax.legend(loc="upper center", ncol=2, fontsize=9)
ax.set_title("Distribution of the random nodes")
plt.show()
