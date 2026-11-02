# Numerical Analysis — Workshop Projects

This repository contains three independent workshops developed for a numerical analysis course, covering classical optimization algorithms, optimal control theory, and numerical methods for differential equations (finite differences and finite elements).

---

## Workshop 1 — Optimization Project

Implementation of optimization algorithms from scratch, divided into three main parts: linear programming, unconstrained nonlinear optimization, and constrained nonlinear optimization.

### Project Structure

**Linear Optimization**
Implementation of the Simplex method using reduced costs to solve linear programming problems.

**Unconstrained Nonlinear Optimization**
Three optimization algorithms for minimizing differentiable functions:
- Gradient Descent
- Damped Newton Method
- BFGS (Broyden–Fletcher–Goldfarb–Shanno)

**Constrained Nonlinear Optimization**
Implementation of the exterior penalty method for solving optimization problems with equality and inequality constraints.

### How to Run

```bash
# Linear Optimization
python Simplex.py

# Unconstrained Nonlinear Optimization
python rosenbrock_optimization.py

# Constrained Nonlinear Optimization
python penalization.py
```

### Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
sympy>=1.12
```

```bash
pip install -r requirements.txt
```

---

## Workshop 2 — Dynamic Programming & Optimal Control

Two independent implementations that compare classical algorithms and apply dynamic programming to a real control problem.

### Part A — Dynamic Programming vs. Brute Force: Shortest Paths in a Staged Graph

Comparison of two algorithms for finding the minimum-cost path from S to T in a directed staged graph with N stages and k nodes per stage.

**Algorithms implemented:**
- Brute force: enumerates all k^N paths — complexity O(k^(N+1))
- Bellman backward induction (discrete DP): solves each node exactly once — complexity O(N·k²)

```bash
python DinámicaFuerza.py
```

### Part B — LQR Optimal Control: Drone Landing

Implementation of a Linear Quadratic Regulator (LQR) for the vertical landing of a drone modeled as a double integrator. The optimal control law is derived by solving the Riccati differential equation backward in time.

```bash
python Riccati.py
```

### Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

```bash
pip install numpy matplotlib scipy
```

---

## Workshop 3 — Finite Differences & Finite Elements

Implementations of classical discretization methods for differential equations, plus a stability analysis of an explicit scheme.

### Part A — Finite Difference Method: Lightning Strike on a Statue

Solves the Laplace equation ∇²V = 0 on a 2D grid built from a photograph of a statue (e.g. Cristo Redentor). The image is segmented to separate the statue from the sky, and Dirichlet conditions are imposed on the statue (grounded), the bottom edge, and the cloud at the top.

**Steps implemented:**
- Image segmentation (blue-sky detection, with Otsu thresholding as fallback)
- Mesh generation and assembly of the sparse linear system using the 5-point stencil
- Computation of the potential, equipotential lines and electric field |E|
- Lightning simulation with the Niemeyer–Pietronero dielectric breakdown model, where the bolt grows with probability proportional to |E|^η

**Outputs:** `1-4_results.png` (segmentation, mesh, potential and field) and `5_lightning.png` (simulated bolts).

> Before running, set `imagePath` at the top of `FDMSimulation.py` to the location of your image.

```bash
python FDMSimulation.py
```

### Part B — 1D Finite Element Method

Solves the boundary value problem −u'' = π² sin(πx) on [0, 1] with u(0) = u(1) = 0 using linear finite elements on a random non-uniform mesh. The load vector is integrated with 2-point Gauss quadrature, and the numerical solution is compared against the exact solution u(x) = sin(πx), reporting the maximum nodal error.

```bash
python FEM1D.py
```

### Part C — Von Neumann Stability of the Explicit Heat Equation

Derivation of the amplification factor G(θ) = 1 − 4r sin²(θ/2) for the explicit (FTCS) scheme of the heat equation u_t = κ u_xx, showing that it is conditionally stable under r = κΔt/Δx² ≤ 1/2.

```bash
jupyter notebook solution_problem2.ipynb
```

### Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pillow>=10.0.0
```

```bash
pip install numpy matplotlib scipy pillow
```
