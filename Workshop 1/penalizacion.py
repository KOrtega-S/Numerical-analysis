# *********************************************************************************
# Implementation of the Exterior Penalty Method for Constrained Optimization
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************
import numpy as np
import matplotlib.pyplot as plt

#---------------- Problem ----------------#

def f(x):
    """Objective function."""
    return x[0] ** 2 + x[1] ** 2


def h(x):
    """Equality constraint."""
    return np.array([x[0] + x[1] - 4])


def g(x):
    """Inequality constraint."""
    return np.array([x[0] - x[1] - 1])

def gradient(F, x, h=1e-6):
    """Approximate the gradient using finite differences."""
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = 1
        g[i] = (F(x + h * e) - F(x - h * e)) / (2 * h)
    return g

def backtracking(F, x, p, a=1, rho=0.5, c=1e-4):
    """Find a step size that satisfies the Armijo condition."""
    fx = F(x)
    g = gradient(F, x)
    while F(x + a * p) > fx + c * a * np.dot(g, p):
        a *= rho
    return a

def bfgs(F, x0, tol=1e-8, itmax=300):
    """Minimize a function using the BFGS method."""
    x = x0.astype(float).copy()
    H = np.eye(len(x))
    cam = [x.copy()]

    for _ in range(itmax):
        g = gradient(F, x)
        if np.linalg.norm(g) < tol:
            break

        p = -H @ g
        a = backtracking(F, x, p)

        s = a * p
        xn = x + s

        y = gradient(F, xn) - g
        ys = y @ s

        if ys > 1e-12:
            rho = 1 / ys
            I = np.eye(len(x))
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = xn
        cam.append(x.copy())

    return x, np.array(cam)

#--------------- Penalty method ----------------#

def penalizacion(x0, r0=1, beta=10, tol=1e-6):
    """Solve the constrained problem using the exterior penalty method."""
    x = x0.astype(float).copy()
    r = r0

    trajectory = [x.copy()]  # Store all iterations

    while r < 1e6:
        def penalized_objective(z):
            equality_violation = np.sum(h(z) ** 2)
            inequality_violation = np.sum(np.maximum(0, g(z)) ** 2)
            return f(z) + 0.5 * r * (equality_violation + inequality_violation)

        x, cam = bfgs(penalized_objective, x)
        trajectory.extend(cam[1:])

        if np.linalg.norm(h(x)) < tol and np.max(g(x)) <= tol:
            break

        r *= beta

    return x, np.array(trajectory)

#---------------- Run ----------------#

x0 = np.array([0.0, 0.0])
x, hist = penalizacion(x0)

print("Solution =", x)
print("f(x) =", f(x))
print("h(x) =", h(x))
print("g(x) =", g(x))

#--------------- Plot ----------------#

xx = np.linspace(-1, 5, 300)
yy = np.linspace(-1, 5, 300)

X, Y = np.meshgrid(xx, yy)
Z = X ** 2 + Y ** 2

plt.figure(figsize=(7, 6))
plt.contourf(X, Y, Z, 20, cmap="plasma", alpha=0.35)
plt.contour(X, Y, Z, 20, colors="k", alpha=0.3)

# Equality constraint
plt.plot(xx, 4 - xx, "r", lw=2, label="x1 + x2 = 4")

# Inequality constraint
plt.plot(xx, xx - 1, "g", lw=2, label="x1 - x2 - 1 = 0")

plt.plot(
    hist[:, 0],
    hist[:, 1],
    "-o",
    color="dodgerblue",
    linewidth=2,
    ms=4,
    markerfacecolor="deepskyblue",
    markeredgecolor="black",
    markeredgewidth=0.7,
    label="Penalty path",
)

plt.scatter(x0[0], x0[1], c="k", s=80, label="Start")
plt.scatter(
    x[0],
    x[1],
    marker="*",
    s=220,
    color="red",
    edgecolors="black",
    linewidths=1.2,
    zorder=10,
    label="Solution",
)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Exterior Penalty Method")
plt.grid(True)
plt.legend()
plt.show()