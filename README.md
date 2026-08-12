Optimization Project
This repository contains the implementation of optimization algorithms developed for an numerical analysis course. 
The project is divided into three main parts: linear programming, unconstrained nonlinear optimization, and constrained nonlinear optimization.

Requirements

The project was developed in Python and uses the following libraries:

numpy>=1.24.0

matplotlib>=3.7.0

pandas>=2.0.0

sympy>=1.12

Install all dependencies with:
pip install -r requirements.txt

Project Structure

•	Linear Optimization: Implementation of the Simplex method using reduced costs.

•	Unconstrained Nonlinear Optimization: Implementation of the descent method.
This part implements three optimization algorithms for minimizing differentiable functions:
  -	Gradient Descent
  -	Damped Newton Method
  -	BFGS
    
•	Constrained Nonlinear Optimization: Implementation of penalty method.
This part implements the exterior penalty method for solving optimization problems with equality and inequality constraints.

How to Run
Examples:

Linear Optimization: python Simplex.py

Unconstrained Nonlinear Optimization: python rosenbrock_optimization.py

Constrained Nonlinear Optimization: python penalization.py

The goal of this project is to implement classical optimization algorithms from scratch and compare their behavior on linear, nonlinear, and constrained optimization problems.
