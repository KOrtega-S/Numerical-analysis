import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

x1, x2 = sp.symbols("x1 x2")
objective_symbolic = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

gradient_symbolic = sp.Matrix([
    sp.diff(objective_symbolic, x1),
    sp.diff(objective_symbolic, x2)
])

hessian_symbolic = sp.hessian(objective_symbolic, (x1, x2))

objective_function = sp.lambdify((x1, x2), objective_symbolic, "numpy")
gradient_function = sp.lambdify((x1, x2), gradient_symbolic, "numpy")
hessian_function = sp.lambdify((x1, x2), hessian_symbolic, "numpy")


def objective(x):
    """Evaluate the Rosenbrock objective function."""
    return float(objective_function(x[0], x[1]))


def gradient(x):
    """Evaluate the gradient of the objective function."""
    return np.array(
        gradient_function(x[0], x[1]), dtype=float
    ).flatten()


def hessian(x):
    """Evaluate the Hessian matrix of the objective function."""
    return np.array(
        hessian_function(x[0], x[1]), dtype=float
    )


def backtracking(
    objective_value,
    gradient_value,
    x,
    direction,
    alpha=1,
    rho=0.5,
    c=1e-4
):
    """Find a step size that satisfies the Armijo condition."""
    current_value = objective_value(x)
    gradient_at_x = gradient_value(x)

    while objective_value(x + alpha * direction) > (
        current_value
        + c * alpha * np.dot(gradient_at_x, direction)
    ):
        alpha *= rho

    return alpha


def gradient_descent(x0, tol=1e-8, max_iter=10000):
    x = x0.copy().astype(float)
    gradient_norms = []
    objective_values = []
    table = []
    trajectory = [x.copy()]

    for iteration in range(max_iter):
        gradient_at_x = gradient(x)
        norm_gradient = np.linalg.norm(gradient_at_x)

        gradient_norms.append(norm_gradient)
        objective_values.append(objective(x))

        if norm_gradient < tol:
            table.append(
                [iteration, *x, objective(x), norm_gradient, 0]
            )
            break

        direction = -gradient_at_x
        step_size = backtracking(
            objective,
            gradient,
            x,
            direction
        )

        table.append(
            [iteration, *x, objective(x), norm_gradient, step_size]
        )

        x = x + step_size * direction
        trajectory.append(x.copy())

    return (
        x,
        gradient_norms,
        objective_values,
        table,
        np.array(trajectory)
    )


def newton_method(x0, tol=1e-8, max_iter=100):
    x = x0.copy().astype(float)
    gradient_norms = []
    objective_values = []
    table = []
    trajectory = [x.copy()]

    for iteration in range(max_iter):
        gradient_at_x = gradient(x)
        norm_gradient = np.linalg.norm(gradient_at_x)

        gradient_norms.append(norm_gradient)
        objective_values.append(objective(x))

        if norm_gradient < tol:
            table.append(
                [iteration, *x, objective(x), norm_gradient, 0]
            )
            break

        direction = np.linalg.solve(
            hessian(x),
            -gradient_at_x
        )

        step_size = backtracking(
            objective,
            gradient,
            x,
            direction
        )

        table.append(
            [iteration, *x, objective(x), norm_gradient, step_size]
        )

        x = x + step_size * direction
        trajectory.append(x.copy())

    return (
        x,
        gradient_norms,
        objective_values,
        table,
        np.array(trajectory)
    )


def bfgs(x0, tol=1e-8, max_iter=500):
    x = x0.copy().astype(float)
    inverse_hessian = np.eye(2)

    gradient_norms = []
    objective_values = []
    table = []
    trajectory = [x.copy()]

    for iteration in range(max_iter):
        gradient_at_x = gradient(x)
        norm_gradient = np.linalg.norm(gradient_at_x)

        gradient_norms.append(norm_gradient)
        objective_values.append(objective(x))

        if norm_gradient < tol:
            table.append(
                [iteration, *x, objective(x), norm_gradient, 0]
            )
            break

        direction = -inverse_hessian @ gradient_at_x

        step_size = backtracking(
            objective,
            gradient,
            x,
            direction
        )

        table.append(
            [iteration, *x, objective(x), norm_gradient, step_size]
        )

        step = step_size * direction
        x_new = x + step

        y = gradient(x_new) - gradient_at_x
        ys = y @ step

        if ys > 1e-12:
            rho = 1 / ys
            identity = np.eye(2)

            inverse_hessian = (
                (identity - rho * np.outer(step, y))
                @ inverse_hessian
                @ (identity - rho * np.outer(y, step))
                + rho * np.outer(step, step)
            )

        x = x_new
        trajectory.append(x.copy())

    return (
        x,
        gradient_norms,
        objective_values,
        table,
        np.array(trajectory)
    )


# --------------------------------------------------
# Convergence analysis
# --------------------------------------------------

def convergence_order(errors):
    """
    Estimate the experimental order p_k from consecutive errors.
    """
    errors = np.asarray(errors, dtype=float)
    orders = []

    for k in range(2, len(errors)):
        if (
            errors[k] > 0
            and errors[k - 1] > 0
            and errors[k - 2] > 0
        ):
            denominator = np.log(
                errors[k - 1] / errors[k - 2]
            )

            if abs(denominator) > 1e-14:
                order = np.log(
                    errors[k] / errors[k - 1]
                ) / denominator
            else:
                order = np.nan

            orders.append(order)
        else:
            orders.append(np.nan)

    return np.array(orders)


def convergence_ratio(errors):
    ratios = []

    for k in range(len(errors) - 1):
        if errors[k] > 0:
            ratios.append(errors[k + 1] / errors[k])
        else:
            ratios.append(np.nan)

    return np.array(ratios)


# --------------------------------------------------
# Run the three methods
# --------------------------------------------------

x0 = np.array([-1.2, 1.0])

(
    x_gradient,
    gradient_norms_gd,
    objective_values_gd,
    table_gd,
    trajectory_gd
) = gradient_descent(x0)

(
    x_newton,
    gradient_norms_newton,
    objective_values_newton,
    table_newton,
    trajectory_newton
) = newton_method(x0)

(
    x_bfgs,
    gradient_norms_bfgs,
    objective_values_bfgs,
    table_bfgs,
    trajectory_bfgs
) = bfgs(x0)


order_gd = convergence_order(gradient_norms_gd)
order_newton = convergence_order(gradient_norms_newton)
order_bfgs = convergence_order(gradient_norms_bfgs)

ratio_gd = convergence_ratio(gradient_norms_gd)
ratio_newton = convergence_ratio(gradient_norms_newton)
ratio_bfgs = convergence_ratio(gradient_norms_bfgs)


print("\nConvergence analysis:")

print("\nGradient descent:")
print("Last convergence orders:", order_gd[-5:])
print("Last error ratios:", ratio_gd[-5:])

print("\nNewton:")
print("Last convergence orders:", order_newton[-5:])
print("Last error ratios:", ratio_newton[-5:])

print("\nBFGS:")
print("Last convergence orders:", order_bfgs[-5:])
print("Last error ratios:", ratio_bfgs[-5:])


def regression_order(errors, window=8, min_error=1e-10):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors) & (errors > min_error)]

    if len(errors) < 4:
        return np.nan

    if len(errors) > window:
        errors = errors[-window:]

    x = np.log(errors[:-1])
    y = np.log(errors[1:])

    if len(x) < 3:
        return np.nan

    p, _ = np.polyfit(x, y, 1)
    return p


p_gd = regression_order(gradient_norms_gd, window=8)
p_newton = regression_order(gradient_norms_newton, window=8)
p_bfgs = regression_order(gradient_norms_bfgs, window=8)

fig, ax = plt.subplots(figsize=(12, 7))

# Gradient descent
ax.plot(
    range(2, len(order_gd) + 2),
    order_gd,
    color="C0",
    linewidth=0.8,
    alpha=0.35,
    label="Gradient descent"
)

# Newton
ax.plot(
    range(2, len(order_newton) + 2),
    order_newton,
    color="C1",
    marker="s",
    markersize=4,
    linewidth=1.5,
    label="Newton"
)

# BFGS
ax.plot(
    range(2, len(order_bfgs) + 2),
    order_bfgs,
    color="C2",
    marker="^",
    markersize=4,
    linewidth=1.5,
    label="BFGS"
)

# Expected theoretical orders
ax.axhline(
    1,
    color="C0",
    linestyle="--",
    linewidth=1.5,
    label="p = 1 (linear)"
)

ax.axhline(
    2,
    color="C2",
    linestyle="--",
    linewidth=1.5,
    label="p = 2 (quadratic)"
)

# Main graph
ax.set_xlim(
    1,
    max(
        len(order_gd),
        len(order_newton),
        len(order_bfgs)
    ) + 2
)

ax.set_ylim(-2, 3)

ax.set_xlabel("Iteration $k$")
ax.set_ylabel("Estimated convergence order $p_k$")
ax.set_title("Experimental Convergence Order")

ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")


# --------------------------------------------------
# Colored p labels
# --------------------------------------------------

if np.isfinite(p_gd):
    ax.text(
        0.78,
        0.62,
        f"Gradient: p = {p_gd:.2f}",
        transform=ax.transAxes,
        color="C0",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="C0",
            boxstyle="round,pad=0.3"
        )
    )

if np.isfinite(p_newton):
    ax.text(
        0.78,
        0.52,
        f"Newton: p = {p_newton:.2f}",
        transform=ax.transAxes,
        color="C1",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="C1",
            boxstyle="round,pad=0.3"
        )
    )

if np.isfinite(p_bfgs):
    ax.text(
        0.78,
        0.42,
        f"BFGS: p = {p_bfgs:.2f}",
        transform=ax.transAxes,
        color="C2",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="C2",
            boxstyle="round,pad=0.3"
        )
    )


# --------------------------------------------------
# Small zoom of the first 20 iterations
# --------------------------------------------------

zoom = 20

ax_zoom = ax.inset_axes(
    [0.72, 0.08, 0.24, 0.24]
)

ax_zoom.plot(
    range(2, min(len(order_gd), zoom) + 2),
    order_gd[:zoom],
    color="C0",
    marker="o",
    markersize=3,
    linewidth=1,
    label="Gradient"
)

ax_zoom.plot(
    range(2, min(len(order_newton), zoom) + 2),
    order_newton[:zoom],
    color="C1",
    marker="s",
    markersize=3,
    linewidth=1,
    label="Newton"
)

ax_zoom.plot(
    range(2, min(len(order_bfgs), zoom) + 2),
    order_bfgs[:zoom],
    color="C2",
    marker="^",
    markersize=3,
    linewidth=1,
    label="BFGS"
)

ax_zoom.axhline(
    1,
    color="C0",
    linestyle="--",
    linewidth=1
)

ax_zoom.axhline(
    2,
    color="C2",
    linestyle="--",
    linewidth=1
)

ax_zoom.set_xlim(1, zoom + 1)
ax_zoom.set_ylim(-2, 3)

ax_zoom.set_title(
    "First iterations",
    fontsize=8
)

ax_zoom.set_xlabel(
    "$k$",
    fontsize=8
)

ax_zoom.set_ylabel(
    "$p_k$",
    fontsize=8
)

ax_zoom.tick_params(
    axis="both",
    labelsize=7
)

ax_zoom.grid(True, alpha=0.25)


plt.tight_layout()
plt.show()


# --------------------------------------------------
# SECOND GRAPH: trajectories over the Rosenbrock function
# This graph is kept as in the original code.
# --------------------------------------------------

plt.figure(figsize=(8, 6))
xx = np.linspace(-2, 2, 400)
yy = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(xx, yy)
Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

contour = plt.contourf(
    X, Y, Z,
    levels=np.logspace(-1, 3, 25),
    cmap="plasma",
    alpha=0.40
)
plt.contour(
    X, Y, Z,
    levels=np.logspace(-1, 3, 25),
    colors="black",
    linewidths=0.4,
    alpha=0.5
)
plt.colorbar(contour, label="f(x)")

plt.plot(
    trajectory_gd[:, 0],
    trajectory_gd[:, 1],
    "-",
    color="dodgerblue",
    linewidth=1.5
)
plt.scatter(
    trajectory_gd[:, 0],
    trajectory_gd[:, 1],
    s=12,
    color="deepskyblue",
    edgecolors="black",
    linewidths=0.4,
    zorder=5,
    label=f"Gradient descent ({len(trajectory_gd)} iter.)"
)

plt.plot(
    trajectory_newton[:, 0],
    trajectory_newton[:, 1],
    "-",
    color="orange",
    linewidth=2
)
plt.scatter(
    trajectory_newton[:, 0],
    trajectory_newton[:, 1],
    marker="s",
    s=35,
    color="lightcoral",
    edgecolors="black",
    linewidths=0.7,
    zorder=6,
    label=f"Newton ({len(trajectory_newton)} iter.)"
)

plt.plot(
    trajectory_bfgs[:, 0],
    trajectory_bfgs[:, 1],
    "-",
    color="limegreen",
    linewidth=2
)
plt.scatter(
    trajectory_bfgs[:, 0],
    trajectory_bfgs[:, 1],
    marker="^",
    s=35,
    color="lightgreen",
    edgecolors="black",
    linewidths=0.7,
    zorder=6,
    label=f"BFGS ({len(trajectory_bfgs)} iter.)"
)

plt.scatter(
    x0[0], x0[1],
    marker="o",
    s=120,
    color="black",
    edgecolors="white",
    linewidths=1.5,
    zorder=10,
    label="Start"
)
plt.scatter(
    1, 1,
    marker="*",
    s=300,
    color="red",
    edgecolors="black",
    linewidths=1.5,
    zorder=11,
    label="Optimum"
)

plt.title("Trajectories of the methods on the Rosenbrock function")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

