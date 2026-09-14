# Numerical Analysis — Workshop Projects

This repository contains two independent workshops developed for a numerical analysis course, covering classical optimization algorithms and optimal control theory.

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

