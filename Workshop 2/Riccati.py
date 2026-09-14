# *********************************************************************************
# Riccati Equation for Optimal Control
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

A = np.array([[0., 1.], [0., 0.]])
B = np.array([[0.], [1.]])
Q = 2 * np.eye(2)
R = np.array([[1.]])
x0 = np.array([100., -20.])
T  = 8.0

PALETTE = {
    'blue':   '#07bdff',
    'purple': '#a890ff',
    'orange': '#fd7a39',
    'teal':   '#00c69e',
    'pink':   '#ff68bd',
    'lime':   '#a6e23d',
}

def riccati_derivative(P):
    R_inv = np.linalg.inv(R)
    return -A.T @ P - P @ A + P @ B @ R_inv @ B.T @ P - Q

def integrate_rk4(dt=0.01):
    n_steps = int(round(T / dt))
    times   = np.linspace(T, 0., n_steps + 1)
    P       = np.zeros((2, 2))
    Ps      = [P.copy()]
    for _ in range(n_steps):
        k1 = riccati_derivative(P)
        k2 = riccati_derivative(P - (dt/2)*k1)
        k3 = riccati_derivative(P - (dt/2)*k2)
        k4 = riccati_derivative(P - dt    *k3)
        P  = P - (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        Ps.append(P.copy())
    return times[::-1], Ps[::-1]   

def propagate_forward(times, Ps, dt):
    n_steps = len(times) - 1
    R_inv   = np.linalg.inv(R)

    xs = np.zeros((n_steps + 1, 2))
    us = np.zeros(n_steps)
    Js = np.zeros(n_steps + 1)

    xs[0] = x0  

    def f(x, u):
        """Drone dynamics: x_dot = A x + B u"""
        return A @ x + B.flatten() * u

    for i in range(n_steps):
        P = Ps[i]
        x = xs[i]
        u = -(R_inv @ B.T @ P @ x).flatten()[0]
        us[i] = u
        Js[i+1] = Js[i] + dt * (x @ Q @ x + R[0,0] * u**2)

        # RK4 
        k1 = f(x,              u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt   * k3, u)
        xs[i+1] = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return xs, us, Js

def plot_punto3(times, xs, us, Js, dt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    # --- Altitude x1*(t) 
    ax = axes[0]
    ax.plot(times, xs[:, 0], color=PALETTE['blue'], lw=2.5,
            label=r'$x_1^*(t)$ — altitude')
    ax.axhline(0, color='gray', lw=1.2, linestyle='--', label='Ground ($x_1=0$)')
    ax.fill_between(times, xs[:, 0], 0, alpha=0.12, color=PALETTE['blue'])
    ax.scatter([0], [x0[0]], color=PALETTE['lime'], s=80, zorder=5,
               label=f'$x_1(0)={x0[0]}$ m')
    ax.scatter([T], [xs[-1, 0]], color=PALETTE['orange'], s=80, zorder=5,
               label=f'$x_1(T)={xs[-1,0]:.2f}$ m')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title(r'Altitude $x_1^*(t)$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- Velocity x2*(t) 
    ax = axes[1]
    ax.plot(times, xs[:, 1], color=PALETTE['purple'], lw=2.5,
            label=r'$x_2^*(t)$ — velocity')
    ax.axhline(0, color='gray', lw=1.2, linestyle='--', label='Zero velocity')
    ax.fill_between(times, xs[:, 1], 0, alpha=0.12, color=PALETTE['purple'])
    ax.scatter([0], [x0[1]], color=PALETTE['lime'], s=80, zorder=5,
               label=f'$x_2(0)={x0[1]}$ m/s')
    ax.scatter([T], [xs[-1, 1]], color=PALETTE['orange'], s=80, zorder=5,
               label=f'$x_2(T)={xs[-1,1]:.2f}$ m/s')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title(r'Velocity $x_2^*(t)$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- Control u*(t) 
    ax = axes[2]
    ax.plot(times[:-1], us, color=PALETTE['orange'], lw=2.5,
            label=r'$u^*(t) = -R^{-1}B^T P(t)\,x^*(t)$')
    ax.axhline(0, color='gray', lw=1.2, linestyle='--')
    ax.fill_between(times[:-1], us, 0, alpha=0.12, color=PALETTE['orange'])
    ax.scatter([0], [us[0]], color=PALETTE['lime'], s=80, zorder=5,
               label=f'$u^*(0)={us[0]:.2f}$ m/s²')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Acceleration [m/s$^2$]')
    ax.set_title(r'Control $u^*(t)$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        rf'Forward propagation with optimal feedback  ',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    dt = 0.01
    times, Ps = integrate_rk4(dt=dt)
    R_inv = np.linalg.inv(R)
    K0 = -(R_inv @ B.T @ Ps[0]).flatten()
    xs, us, Js = propagate_forward(times, Ps, dt=dt)
    plot_punto3(times, xs, us, Js, dt)

def integrate_scipy_rk45(dt_eval=0.001):
    def riccati_flat(tau, p_flat):
        P = p_flat.reshape(2, 2)
        return (-riccati_derivative(P)).flatten()

    sol = solve_ivp(riccati_flat, (0., T), np.zeros(4),
                    method='RK45', t_eval=np.linspace(0., T, int(round(T/dt_eval))+1),
                    rtol=1e-10, atol=1e-12)
    times_sp = T - sol.t[::-1]
    Ps_sp    = [sol.y[:, i].reshape(2, 2) for i in range(sol.y.shape[1]-1, -1, -1)]
    return times_sp, Ps_sp

def interp_P(times_ref, Ps_ref, times_query):
    Ps_arr = np.array(Ps_ref)
    out    = np.zeros((len(times_query), 2, 2))
    for i in range(2):
        for j in range(2):
            out[:, i, j] = np.interp(times_query, times_ref, Ps_arr[:, i, j])
    return out

def plot_comparison_errors(times_ref, Ps_ref, times_rk, Ps_rk, dts, err_rk_dt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    ax = axes[0]
    ax.plot(times_ref, np.array(Ps_ref)[:, 0, 0],
            color=PALETTE['teal'], lw=2.5, label='scipy RK45')
    ax.plot(times_rk,  np.array(Ps_rk)[:, 0, 0],
            color=PALETTE['purple'], lw=1.8, linestyle='--',
            label='Own RK4 (dt=0.01)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel(r'$P_{11}(t)$')
    ax.set_title(r'$P_{11}(t)$ — Own RK4 vs scipy RK45', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    P_ref_on_rk = interp_P(times_ref, Ps_ref, times_rk)
    err_time    = np.abs(np.array(Ps_rk) - P_ref_on_rk).max(axis=(1, 2))
    ax.semilogy(times_rk, np.maximum(err_time, 1e-16),
                color=PALETTE['orange'], lw=2,
                label=r'$|$Own RK4 $-$ scipy RK45$|$ (max over $P$ elements)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Max absolute error')
    ax.set_title(r'Error $|$Own RK4 $-$ scipy RK45$|$ over $[0,T]$', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

    ax = axes[2]
    ax.loglog(dts, err_rk_dt, 's-',
              color=PALETTE['purple'], lw=2, ms=6, label='Own RK4')
    dt_arr = np.array(dts, dtype=float)
    scale4 = err_rk_dt[2] / dts[2]**4
    ax.loglog(dt_arr, scale4 * dt_arr**4, '--',
              color=PALETTE['pink'], lw=1.5, label=r'$O(\Delta t^4)$')
    ax.set_xlabel(r'Step size $\Delta t$')
    ax.set_ylabel(r'Max $|$error$|$ at $t=0$')
    ax.set_title(r'Convergence as $\Delta t \to 0$', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

    fig.suptitle(
        'Error comparison: Own RK4 vs scipy RK45',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

def main_comparacion():
    dt_own = 0.01
    dt_ref = 0.001

    times_ref, Ps_ref = integrate_scipy_rk45(dt_eval=dt_ref)
    times_rk,  Ps_rk  = integrate_rk4(dt=dt_own)

    err0 = np.abs(Ps_rk[0] - Ps_ref[0]).max()
    dts = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    err_rk_dt = []
    for dt in dts:
        _, Ps_r = integrate_rk4(dt)
        e = np.abs(Ps_r[0] - Ps_ref[0]).max()
        err_rk_dt.append(e)

    plot_comparison_errors(times_ref, Ps_ref, times_rk, Ps_rk, dts, err_rk_dt)

def plot_adaptive_step():
    def riccati_flat(tau, p_flat):
        P = p_flat.reshape(2, 2)
        return (-riccati_derivative(P)).flatten()

    sol = solve_ivp(riccati_flat, (0., T), np.zeros(4),
                    method='RK45', rtol=1e-8, atol=1e-10)

    tau_steps = sol.t
    t_steps   = T - tau_steps          
    step_sizes = np.diff(tau_steps)    

    sol_ref = solve_ivp(riccati_flat, (0., T), np.zeros(4),
                        method='RK45', rtol=1e-12, atol=1e-14,
                        t_eval=np.linspace(0, T, 1000))
    P_ref = {tau: sol_ref.y[:, i].reshape(2, 2)
             for i, tau in enumerate(sol_ref.t)}
    t_ref  = T - sol_ref.t[::-1]
    P11_ref = sol_ref.y[0, ::-1] 

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    ax = axes[0]
    ax.plot(t_ref, P11_ref, color=PALETTE['teal'], lw=2, label=r'$P_{11}(t)$', zorder=2)
    ax.scatter(t_steps, np.interp(t_steps, t_ref, P11_ref),
               color=PALETTE['orange'], s=35, alpha=0.75, zorder=4,
               label=f'RK45 steps ({len(t_steps)} total)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel(r'$P_{11}(t)$')
    ax.set_title(r'Adaptive steps of RK45 on $P_{11}(t)$', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    t_mid = T - (tau_steps[:-1] + tau_steps[1:]) / 2
    order = np.argsort(t_mid)
    t_mid, step_sizes = t_mid[order], step_sizes[order]
    ax.bar(t_mid, step_sizes, width=step_sizes * 0.8,
           color=PALETTE['purple'], alpha=0.8, label='Adaptive step size')
    dt_fixed = T / 800
    ax.axhline(dt_fixed, color=PALETTE['orange'], lw=1.5,
               linestyle='--', label=f'Fixed step dt={dt_fixed:.4f} (800 steps)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel(r'Step size $\Delta t$')
    ax.set_yscale('log')
    ax.set_title('Adaptive step size: large where smooth,\nsmall where rapidly changing',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

    dts_fixed  = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    n_evals_fixed = [int(T/dt)*4 for dt in dts_fixed]   
    errs_fixed = []
    _, Ps_ref_fine = integrate_scipy_rk45(dt_eval=0.001)
    for dt in dts_fixed:
        _, Ps_r = integrate_rk4(dt)
        errs_fixed.append(np.abs(Ps_r[0] - Ps_ref_fine[0]).max())

    tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    n_evals_adaptive, errs_adaptive = [], []
    for tol in tols:
        s = solve_ivp(riccati_flat, (0., T), np.zeros(4),
                      method='RK45', rtol=tol, atol=tol*0.01)
        P0_adapt = s.y[:, -1].reshape(2, 2)   # tau=T => t=0
        err = np.abs(P0_adapt - Ps_ref_fine[0]).max()
        n_evals_adaptive.append(s.nfev)
        errs_adaptive.append(err)

    ax = axes[2]
    ax.loglog(n_evals_fixed, errs_fixed, 'o-',
              color=PALETTE['purple'], lw=2, ms=6, label='RK4 fixed step')
    ax.loglog(n_evals_adaptive, errs_adaptive, 's-',
              color=PALETTE['teal'], lw=2, ms=6, label='RK45 adaptive (scipy)')
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Max error in P(0)')
    ax.set_title('Efficiency: error vs computational cost\n(fewer evaluations = more efficient)',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')
    ax.text(0.97, 0.95,
            'Lower-left = more efficient\n(same accuracy for less cost)',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.85))

    fig.suptitle(
        'Adaptive step (RK45) vs fixed step (RK4)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])


if __name__ == "__main__":
    main()
    main_comparacion()
    plot_adaptive_step()
    plt.show()

    """
  CONCLUSION                                                      
  ---------------------                                                       
  A step-adaptive method automatically adjusts the step size Δt at each 
  iteration based on how fast the solution is changing, making it large 
  where the solution is smooth and small where it changes rapidly. 
  The mechanism works by computing two solutions of different orders 
  (4 and 5) using the same set of function evaluations — their difference 
  serves as a local error estimator, and if that error exceeds the 
  user-specified tolerance the step is rejected and retried with a smaller Δt,
  otherwise it is accepted and the next optimal step size is computed 
  automatically. 
  RK45 is adaptive precisely because it uses an embedded pair: with only 
  6 evaluations of f it produces both a 4th-order and a 5th-order approximation 
  at no extra cost, which is what drives the step-size control. 
  The practical advantage over a fixed-step method like RK4 is significant.
  More importantly, the accuracy is guaranteed by the specified tolerance 
  rather than requiring the user to manually verify convergence through
  trial and error.

"""
