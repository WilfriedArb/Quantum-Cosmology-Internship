"""
Quantum trajectory in affine quantum cosmology — asymmetric biverse state (n0 != n1)
With Wavefunction Probability Density Plot
"""

import numpy as np
import math
from scipy.integrate import solve_ivp, quad
from scipy.special import gamma, genlaguerre
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────
nu      = 1       # affine parameter
n0      = 1       # quantum number for state 0
n1      = 1       # quantum number for state 1
r       = 2.0     # energy ratio  E1/E0
Dtau    = 50.0    # bounce-time separation  Δτ
rho     = 1.0     # amplitude ratio  ρ
delta   = 0.0     # relative phase  δ

# ── Function to compute ξ_{ν,n} dynamically (Eq 21) ──
# ξ is the normalization/overlap constant of the n-th affine coherent state;
# unlike the n=0 case it has no closed form in general, so it is obtained
# by numerically integrating the weighted squared Laguerre polynomial
def compute_xi(n_val, nu_val):
    Ln_func = genlaguerre(n_val, nu_val)
    def integrand(y):
        return y**(nu_val + 0.5) * (Ln_func(y))**2 * np.exp(-y)
    integral_val, _ = quad(integrand, 0, np.inf)
    Gn = (math.factorial(n_val) / gamma(nu_val + n_val + 1)) * integral_val
    return Gn**2

xi0 = compute_xi(n0, nu)
xi1 = compute_xi(n1, nu)

# Energies  (dimensionless units, E0 = 1)
E0, E1 = 1.0, r

# Bounce amplitudes  qB,  frequencies  om,  bounce times tB
qB0, qB1 = xi0/np.sqrt(E0),  xi1/np.sqrt(E1)
om0, om1 = 2*E0/xi0,         2*E1/xi1
tB0, tB1 = 0.0,              Dtau

print(f"--- States initialized ---")
print(f"State 0 (n={n0}) : ξ = {xi0:.6f}, q_B = {qB0:.4f}, ω = {om0:.4f}, τ_B = {tB0}")
print(f"State 1 (n={n1}) : ξ = {xi1:.6f}, q_B = {qB1:.4f}, ω = {om1:.4f}, τ_B = {tB1}")

# ── Precompute the constant prefactors for the ratio ψ₁/ψ₀ (Eq 26) ──
coeff_N0 = math.sqrt(math.factorial(n0) / gamma(nu + n0 + 1))
coeff_N1 = math.sqrt(math.factorial(n1) / gamma(nu + n1 + 1))
ratio_N = coeff_N1 / coeff_N0
ratio_xi = (xi1 / xi0)**((nu + 1) / 2.0)
constant_ratio = ratio_N * ratio_xi

# ─────────────────────────────────────────────
# 2.  SEMICLASSICAL TRAJECTORIES  (eq. 15)
# ─────────────────────────────────────────────
def q_sc(t, qB, om, tB):
    return qB * np.sqrt(1.0 + om**2 * (t - tB)**2)

def p_sc(t, qB, om, tB):
    dt = t - tB
    return 0.5 * qB * om**2 * dt / np.sqrt(1.0 + om**2 * dt**2)

# ─────────────────────────────────────────────
# 3.  GUIDANCE EQUATION
# ─────────────────────────────────────────────
def get_alpha_q(tau, qB, om, tB, xi_val):
    q = q_sc(tau, qB, om, tB)
    p = p_sc(tau, qB, om, tB)
    return xi_val - 1j*q*p,  q          

# Pre-compute Laguerre polynomials for the ODE
L_n0   = genlaguerre(n0, nu)
L_n0_m1 = genlaguerre(n0 - 1, nu + 1) if n0 > 0 else lambda y: np.zeros_like(y)

L_n1   = genlaguerre(n1, nu)
L_n1_m1 = genlaguerre(n1 - 1, nu + 1) if n1 > 0 else lambda y: np.zeros_like(y)

def guidance_rhs(tau, state):
    # Bohmian velocity field dx/dtau = 2 Im[d(ln Psi)/dx] for the biverse
    # superposition Psi = psi_0 + w * psi_1, expanded in terms of the
    # per-branch log-derivatives f0, f1 and the branch amplitude ratio w
    x = state[0]

    al0, q0 = get_alpha_q(tau, qB0, om0, tB0, xi0)
    al1, q1 = get_alpha_q(tau, qB1, om1, tB1, xi1)

    y0 = xi0 * (x**2) / (q0**2)
    y1 = xi1 * (x**2) / (q1**2)

    eps = 1e-300  # avoids a 0/0 division right at the nodes of the Laguerre polynomials
    val_L0 = L_n0(y0) + eps
    val_L1 = L_n1(y1) + eps

    # Logarithmic derivative of the Laguerre factor, d/dx[ln L_n(y)]
    term_L0 = 2.0 * xi0 * (x / q0**2) * L_n0_m1(y0) / val_L0 if n0 > 0 else 0.0
    term_L1 = 2.0 * xi1 * (x / q1**2) * L_n1_m1(y1) / val_L1 if n1 > 0 else 0.0

    # f_a = d(ln psi_a)/dx, combining the power-law, Laguerre and Gaussian factors
    f0 = (nu + 0.5)/x - term_L0 - al0 * x / q0**2
    f1 = (nu + 0.5)/x - term_L1 - al1 * x / q1**2

    power_phase0 = (2*n0 + nu + 1) / 2.0
    power_phase1 = (2*n1 + nu + 1) / 2.0

    ratio_q = (q0/q1)**(nu + 1)
    ratio_alpha = (al1 / np.conj(al1))**power_phase1 / (al0 / np.conj(al0))**power_phase0
    poly_ratio = val_L1 / val_L0
    exp_arg = 0.5 * x**2 * (al0/q0**2 - al1/q1**2)

    # w = psi_1/psi_0, weighted by the biverse amplitude ratio rho and phase delta
    w   = rho * np.exp(-1j*delta) * constant_ratio * ratio_q * ratio_alpha * poly_ratio * np.exp(exp_arg)
    den = 1.0 + w

    return [2.0 * np.imag((f0 + w*f1) / den)]

# ─────────────────────────────────────────────
# 4.  SOLVE ODE
# ─────────────────────────────────────────────
tau_i, tau_f = -300.0, 300.0
x_i = q_sc(tau_i, qB0, om0, tB0)
print(f"\nInitial condition : x({tau_i:.0f}) = {x_i:.4f}")

tau_eval = np.linspace(tau_i, tau_f, 15000)

sol = solve_ivp(
    guidance_rhs, (tau_i, tau_f), [x_i],
    method='RK45', t_eval=tau_eval,
    rtol=1e-10, atol=1e-12,
    max_step=0.1           
)
print(f"ODE solver : {sol.message}  —  {sol.t.shape[0]} points")

print("Computing momentum along trajectory …")
dxdtau = np.array([guidance_rhs(sol.t[i], [sol.y[0][i]])[0] for i in range(len(sol.t))])
p_traj = 0.5 * dxdtau

q0_ref = q_sc(tau_eval, qB0, om0, tB0)
q1_ref = q_sc(tau_eval, qB1, om1, tB1)


# ─────────────────────────────────────────────
# 5.  PLOT TRAJECTORIES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 12))

ax = axes[0]
ax.plot(tau_eval, sol.y[0], color='royalblue', lw=2.0, label=r'$x(\tau)$ — quantum trajectory')
ax.plot(tau_eval, q0_ref,  color='forestgreen', lw=1.5, ls='--', label=rf'$q_0(\tau)$ — semiclassical ($E_0$, n={n0})')
ax.plot(tau_eval, q1_ref,  color='dimgray',     lw=1.5, ls=':',  label=rf'$q_1(\tau)$ — semiclassical ($E_1$, n={n1})')
ax.axvline(tB0, color='salmon', ls=':', lw=1.3)
ax.axvline(tB1, color='orchid', ls=':', lw=1.3)
ax.set_ylabel(r'$x(\tau)$', fontsize=14)
ax.set_xlim(tau_i, tau_f)
ax.legend(fontsize=11, loc='upper right')
ax.set_title(f'Quantum trajectory — Asymmetric Biverse (n0={n0}, n1={n1})', fontsize=13)
ax.grid(True, alpha=0.3)

ax3 = axes[1]
p0_ref = p_sc(tau_eval, qB0, om0, tB0)
p1_ref = p_sc(tau_eval, qB1, om1, tB1)
ax3.plot(sol.y[0], p_traj, color='royalblue', lw=1.5, label=r'$(x, p)$ — quantum trajectory')
ax3.plot(q0_ref, p0_ref, color='forestgreen', lw=1.2, ls='--', label=r'$(q_0, p_0)$ — semiclassical')
ax3.plot(q1_ref, p1_ref, color='dimgray', lw=1.2, ls=':', label=r'$(q_1, p_1)$ — semiclassical')
ax3.axhline(0, color='black', lw=0.5, alpha=0.5)
ax3.set_xlabel(r'$x(\tau)$', fontsize=14)
ax3.set_ylabel(r'$p(\tau)$', fontsize=14)
ax_p_max = min(5.0, max(np.max(np.abs(p_traj)), 3.0))
ax3.set_ylim(-ax_p_max, ax_p_max)
ax3.legend(fontsize=11, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_title(r'Phase Space', fontsize=13)

plt.tight_layout()
fig.savefig(f'quantum_trajectory_n0_{n0}_n1_{n1}.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved → quantum_trajectory_n0_{n0}_n1_{n1}.png")
plt.close()

# ─────────────────────────────────────────────
# 6.  PLOT PROBABILITY DENSITIES (|Ψ|²)
# ─────────────────────────────────────────────
print("Computing wavefunctions for probability density plot...")

def calc_complex_psi(x_vals, tau_val, n_val, qB, om, tB, xi_val):
    """Compute the exact complex wavefunction of one branch at a given tau."""
    q = q_sc(tau_val, qB, om, tB)
    p = p_sc(tau_val, qB, om, tB)
    al = xi_val - 1j*q*p
    
    c_a = np.sqrt(2.0 * math.factorial(n_val) / gamma(nu + n_val + 1.0))
    phase_power = (2.0 * n_val + nu + 1.0) / 2.0
    phase_term = (al / np.conj(al)) ** phase_power
    amp_term = (xi_val**((nu + 1.0)/2.0)) / (q**(nu + 1.0))
    
    y = xi_val * (x_vals**2) / (q**2)
    L_n_func = genlaguerre(n_val, nu)
    lag_term = L_n_func(y)
    
    exp_term = np.exp(-0.5 * al * (x_vals**2) / (q**2))
    
    return c_a * phase_term * amp_term * (x_vals**(nu + 0.5)) * lag_term * exp_term

# ── Instant at which the wavefunction is plotted ──
tau_plot = 0.0

# Determine an appropriate x-axis scale
q0_plot = q_sc(tau_plot, qB0, om0, tB0)
q1_plot = q_sc(tau_plot, qB1, om1, tB1)
x_max = max(q0_plot, q1_plot) * (4.0 + 0.5 * max(n0, n1))
x_vals = np.linspace(1e-4, x_max, 5000)

psi0_complex = calc_complex_psi(x_vals, tau_plot, n0, qB0, om0, tB0, xi0)
psi1_complex = calc_complex_psi(x_vals, tau_plot, n1, qB1, om1, tB1, xi1)

# Superposition totale
Psi_biverse = psi0_complex + rho * np.exp(-1j * delta) * psi1_complex

# Numerical normalization of the biverse state
prob_density_biverse_raw = np.abs(Psi_biverse)**2
norm_factor = np.trapezoid(prob_density_biverse_raw, x_vals)
Psi_biverse_normalized = Psi_biverse / np.sqrt(norm_factor)

pdf0 = np.abs(psi0_complex)**2
pdf1 = np.abs(psi1_complex)**2
pdf_biverse = np.abs(Psi_biverse_normalized)**2

fig_wf, ax_wf = plt.subplots(1, 1, figsize=(15, 6))

ax_wf.plot(x_vals, pdf0, color='forestgreen', ls='--', lw=1.5, label=rf'$|\psi_0|^2$ (n={n0})')
ax_wf.plot(x_vals, pdf1, color='dimgray', ls=':', lw=1.5, label=rf'$|\psi_1|^2$ (n={n1})')
ax_wf.plot(x_vals, pdf_biverse, color='royalblue', lw=2.0, label=r'$|\Psi_{biverse}|^2$ (Superposition)')

ax_wf.set_xlabel(r'Scale factor $x$', fontsize=14)
ax_wf.set_ylabel(r'Probability density $|\Psi|^2$', fontsize=14)
ax_wf.set_title(f'Squared wavefunction moduli at $\\tau = {tau_plot}$', fontsize=14)
ax_wf.set_xlim(0, x_max)
ax_wf.grid(True, alpha=0.3)
ax_wf.legend(fontsize=12)

# Inset — zoom on [0, 10]
ax_ins = ax_wf.inset_axes([0.33, 0.36, 0.35, 0.50])
mask = (x_vals >= 0) & (x_vals <= 10)
ax_ins.plot(x_vals[mask], pdf_biverse[mask],
                color='royalblue',     lw=2.0)
ax_ins.plot(x_vals[mask], pdf0[mask],
            color='forestgreen',   lw=1.5, ls=':')
ax_ins.plot(x_vals[mask], pdf1[mask],
                color='dimgray', lw=1.5, ls='--')
ax_ins.set_xlim(0, 10)
ax_ins.set_ylim(0, 1)
ax_ins.set_title(r'Zoom on $x \in [0,\,10]$', fontsize=9)
ax_ins.grid(True, which='both', alpha=0.3)
ax.indicate_inset_zoom(ax_ins, edgecolor='black', alpha=0.4)

plt.tight_layout()
fig_wf.savefig(f'wavefunctions_density_n0_{n0}_n1_{n1}.png', dpi=150, bbox_inches='tight')
print(f"Figure saved → wavefunctions_density_n0_{n0}_n1_{n1}.png\n")