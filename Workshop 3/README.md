# Workshop 3 — Finite Differences & Finite Elements

This workshop contains implementations of classical discretization methods for differential equations, developed for a numerical analysis course. It is divided into three parts: a 2D finite difference simulation, a 1D finite element solver, and a von Neumann stability analysis.

## Requirements

The project was developed in Python and uses the following libraries:

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pillow>=10.0.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

**Finite Difference Method — Lightning Strike on a Statue** (`FDMSimulation.py`)

Solves the Laplace equation ∇²V = 0 on a 2D grid built from a photograph of a statue (e.g. Cristo Redentor). Dirichlet conditions are imposed on the statue (grounded), the bottom edge and the cloud at the top.
- Image segmentation (blue-sky detection, with Otsu thresholding as fallback)
- Assembly of the sparse linear system using the 5-point stencil
- Potential, equipotential lines and electric field |E|
- Lightning simulation with the Niemeyer–Pietronero dielectric breakdown model, where the bolt grows with probability proportional to |E|^η

Outputs: `1-4_results.png` (segmentation, mesh, potential and field) and `5_lightning.png` (simulated bolts).

**1D Finite Element Method** (`FEM1D.py`)

Solves −u'' = π² sin(πx) on [0, 1] with u(0) = u(1) = 0 using linear finite elements on a random non-uniform mesh. The load vector is integrated with 2-point Gauss quadrature, and the result is compared against the exact solution u(x) = sin(πx), reporting the maximum nodal error.

**Von Neumann Stability of the Explicit Heat Equation** (`solution_problem2.ipynb`)

Step-by-step derivation of the amplification factor G(θ) = 1 − 4r sin²(θ/2) for the explicit (FTCS) scheme of the heat equation, showing that it is conditionally stable under r = κΔt/Δx² ≤ 1/2.

## How to Run

> Before running the FDM simulation, set `imagePath` at the top of `FDMSimulation.py` to the location of your image.

```bash
# Finite Difference Method
python FDMSimulation.py

# Finite Element Method
python FEM1D.py

# Stability analysis
jupyter notebook solution_problem2.ipynb
```
