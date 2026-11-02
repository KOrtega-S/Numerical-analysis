# *********************************************************************************
# FDM
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

imagePath = r"C:\Users\gaezn\Downloads\Cristo.jpg"
vCloud = 1.0
maxSide = 500    # large photos are downscaled to this size, otherwise memory runs out
refine = 2       # nodes per pixel in each direction
nBolts = 10      # number of lightning bolts to simulate
eta = 2.0        # breakdown model exponent: higher = straighter bolts
neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Segmentation
img = Image.open(imagePath).convert("RGB")
scale = min(1.0, maxSide / max(img.size)) * refine
if scale != 1:
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
rgb = np.asarray(img)
gray = ndi.gaussian_filter(np.asarray(img.convert("L")).astype(float), refine)
ny, nx = gray.shape

smooth = ndi.gaussian_filter(rgb.astype(float), (refine, refine, 0))
blueness = smooth[..., 2] - smooth[..., 0]
brightness = smooth.mean(axis=2)

if np.mean(blueness[: ny // 10] > 40) > 0.5:
    air = (blueness > 40) | ((blueness > 6) & (brightness > 185))
    objMask = ndi.binary_opening(~air, iterations=refine)
    labels, n = ndi.label(objMask)
    ids = np.unique(labels[-1][labels[-1] > 0])
    sizes = ndi.sum(objMask, labels, ids)
    # the largest region touching the ground
    statue = ndi.binary_closing(labels == ids[sizes.argmax()], iterations=2 * refine)
    statue = ndi.binary_fill_holes(statue)
else:
    # Without a blue sky, binarize the grayscale image using Otsu's threshold
    hist, _ = np.histogram(gray, 256, (0, 256))
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    threshold = np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12))

    # The background is what lies at the top of the photo; the statue is the opposite
    background = np.median(gray[: ny // 20])
    lightBackground = background > threshold
    threshold += 0.3 * (background - threshold)
    objMask = gray < threshold if lightBackground else gray > threshold
    objMask = ndi.binary_opening(ndi.binary_closing(objMask, iterations=2 * refine), iterations=refine)
    labels, n = ndi.label(objMask)
    sizes = ndi.sum(objMask, labels, range(1, n + 1))
    statue = ndi.binary_fill_holes(labels == (1 + sizes.argmax()))

    # Extend the pedestal down to the edge so the statue rests on the ground
    rows = np.nonzero(statue.any(axis=1))[0]
    height = rows[-1] - rows[0]
    base = statue[rows[-1] - max(2, height // 20): rows[-1] + 1].any(axis=0)
    statue[rows[-1]:, base] = True
    print(f"Grayscale segmentation, threshold = {threshold:.0f}")

statue[-1, :] = True
statue[0, :] = False

# Mesh and linear system: known nodes are the statue, the bottom edge and the cloud
fixed = statue.copy()
fixed[0, :] = True
vFixed = np.zeros((ny, nx))
vFixed[0, :] = vCloud

free = ~fixed
N = free.sum()
idx = -np.ones((ny, nx), dtype=int)
idx[free] = np.arange(N)

# 5-point stencil: 4 V(i,j) - (sum of the 4 neighbors) = 0
I, J = np.nonzero(free)
p = idx[I, J]
rows, cols, vals = [p], [p], [np.full(N, 4.0)]
b = np.zeros(N)
for di, dj in neighbors:
    ii = I + di
    # Neumann on the sides using a ghost node: V(-1) = V(1), V(nx) = V(nx-2)
    jj = (nx - 1) - np.abs((nx - 1) - np.abs(J + dj))
    known = fixed[ii, jj]
    b += np.where(known, vFixed[ii, jj], 0.0)
    rows.append(p[~known])
    cols.append(idx[ii[~known], jj[~known]])
    vals.append(np.full((~known).sum(), -1.0))

A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                  shape=(N, N))
V = vFixed.copy()
V[free] = spla.spsolve(A, b)

# Electric field: lightning strikes where |E| is largest next to the statue
Ey, Ex = np.gradient(-V)
E = np.hypot(Ex, Ey) * scale
edge = ndi.binary_dilation(statue) & ~statue
edge[0, :] = False
i0, j0 = np.unravel_index(np.argmax(np.where(edge, E, 0)), E.shape)
eUniform = vCloud / (ny / scale)   # the field there would be without the statue


# Plots
figSize = (9, 9 * ny / nx) if nx > ny else (7 * nx / ny + 1, 7)

def finishAxes(ax, title, mappable=None, **kw):
    if mappable is not None:
        # colorbar with the same height as the image
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
        ax.figure.colorbar(mappable, cax=cax, label="V")
    ax.set(title=title, xticks=[], yticks=[], **kw)

fig, axs = plt.subplots(2, 2, figsize=(1.5 * figSize[0] + 2, 1.3 * figSize[1]), layout="tight")

# 1. Segmentation
ax = axs[0, 0]
ax.imshow(rgb)
ax.contour(statue, [0.5], colors="r", linewidths=1.5)
finishAxes(ax, "Statue segmentation")

# 2. Mesh: 0 air, 1 sides, 2 statue, 3 cloud
nodeType = np.zeros((ny, nx))
nodeType[:, [0, -1]] = 1
nodeType[statue] = 2
nodeType[0, :] = 3
nodeColors = ListedColormap(["#dfefff", "#2ca02c", "#555555", "#ff7f0e"])

# With every node drawn nothing is visible, so only one in every k is plotted
k = max(1, nx // 50)
t = nodeType[::k, ::k].copy()
t[:, [0, -1]] = np.where(t[:, [0, -1]] == 0, 1, t[:, [0, -1]])
t[0] = 3

ax = axs[0, 1]
ax.pcolormesh(np.arange(0, nx, k), np.arange(0, ny, k), t, cmap=nodeColors, vmin=0, vmax=3,
              edgecolors="k", linewidth=0.15, shading="nearest")
ax.plot(j0, i0, "c*", ms=16, mec="k")
ax.invert_yaxis()
patches = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k") for c in nodeColors.colors]
ax.legend(patches, ["air (unknown)", "side: dV/dx = 0", "statue: V = 0", "cloud: V = 1"],
          loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
finishAxes(ax, f"Mesh: {ny} x {nx} nodes (showing 1 in every {k})", aspect="equal")

# 3. Potential over the silhouette
ax = axs[1, 0]
m = ax.imshow(np.ma.masked_where(statue, V), cmap="viridis", vmin=0, vmax=vCloud)
ax.contour(V, levels=np.linspace(0.05, 0.95, 19) * vCloud, colors="w", linewidths=0.6)
ax.imshow(np.ma.masked_where(~statue, statue), cmap=ListedColormap(["#444444"]))
ax.contour(statue, [0.5], colors="k", linewidths=1.2)
ax.plot(j0, i0, "r*", ms=16, mec="w")
finishAxes(ax, "Potential V(x, y)", m)

# 4. Equipotentials and electric field
ax = axs[1, 1]
ax.imshow(rgb, alpha=0.5)
c = ax.contour(V, levels=np.linspace(0, vCloud, 21), cmap="plasma", linewidths=1)
step = 12 * refine
Y, X = np.mgrid[0:ny:step, 0:nx:step]
# Near the head the field is huge; it is clipped so the arrows remain visible
U, W = Ex[::step, ::step], -Ey[::step, ::step]
mag = np.hypot(U, W)
f = np.minimum(1, np.percentile(mag, 75) / (mag + 1e-12))
f[statue[::step, ::step]] = 0   # no arrows inside the statue
ax.quiver(X, Y, U * f, W * f, color="k", width=0.0025)
ax.plot(j0, i0, "c*", ms=16, mec="k")
finishAxes(ax, "Equipotentials and electric field E", c)
fig.savefig("1-4_results.png", dpi=130, bbox_inches="tight")

# 5. Lightning (Niemeyer-Pietronero): the bolt grows from the cloud with probability ~ |E|^eta
g = max(1, ny // 150)   # coarser mesh, otherwise each bolt takes too long
statueCoarse = statue[::g, ::g].copy()
statueCoarse[-1, :] = True
statueCoarse[0, :] = False
nyCoarse, nxCoarse = statueCoarse.shape
vCoarse = V[::g, ::g].copy()
cross = ndi.generate_binary_structure(2, 1)
rng = np.random.default_rng()

def relax(V, fixed, it):
    # Jacobi; the "reflect" pad is the Neumann ghost node V(-1) = V(1)
    for _ in range(it):
        P = np.pad(V, ((0, 0), (1, 1)), mode="reflect")
        vNew = 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])
        V[1:-1] = np.where(fixed[1:-1], V[1:-1], vNew)

def neighborIn(i, j, mask):
    # first 4-neighbor of (i, j) that belongs to the mask, or None
    return next(((i + di, j + dj) for di, dj in neighbors
                 if 0 <= j + dj < nxCoarse and mask[i + di, j + dj]), None)

bolts = []
for r in range(nBolts):
    vBolt = vCoarse.copy()
    fixedBolt = statueCoarse.copy()
    fixedBolt[0, :] = True
    channel = np.zeros_like(statueCoarse)
    jStart = rng.integers(nxCoarse)
    channel[0, jStart] = True
    parent = {(0, jStart): None}
    impact = None
    while impact is None:
        ci, cj = np.nonzero(ndi.binary_dilation(channel, cross) & ~fixedBolt)
        w = np.clip(vCloud - vBolt[ci, cj], 0, None) ** eta + 1e-12
        k = rng.choice(len(ci), p=w / w.sum())
        i, j = ci[k], cj[k]
        parent[(i, j)] = neighborIn(i, j, channel)
        channel[i, j] = fixedBolt[i, j] = True
        vBolt[i, j] = vCloud
        impact = neighborIn(i, j, statueCoarse)
        relax(vBolt, fixedBolt, 20)
    hitPlace = "ground" if impact[0] == nyCoarse - 1 else "statue"
    bolts.append((parent, (i, j), impact, hitPlace))

nOnStatue = sum(d == "statue" for *_, d in bolts)

fig, ax = plt.subplots(figsize=figSize)
ax.imshow(rgb)
for (parent, tip, hit, _), color in zip(bolts, plt.cm.tab10(np.arange(nBolts) % 10)):
    # all branches thin and the main channel (cloud -> impact) thick
    branches = [[(q[1] * g, q[0] * g), (p[1] * g, p[0] * g)] for p, q in parent.items() if q]
    ax.add_collection(LineCollection(branches, colors=[color], linewidths=0.6, alpha=0.5))
    path = [hit, tip]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    yy, xx = np.array(path).T * g
    ax.plot(xx, yy, color=color, lw=2)
    ax.plot(hit[1] * g, hit[0] * g, "o", color=color, mec="k", ms=8)
ax.plot(j0, i0, "c*", ms=16, mec="k")
finishAxes(ax, f"{nBolts} simulated lightning bolts ({nOnStatue} on the statue)")
plt.savefig("5_lightning.png", dpi=130, bbox_inches="tight")

# Show all figures at the same time
plt.show()
