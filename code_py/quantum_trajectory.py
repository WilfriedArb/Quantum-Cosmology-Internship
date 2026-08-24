"""
Quantum trajectory in affine quantum cosmology — biverse state
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────
nu      = 1       # affine parameter
r       = 2.0     # energy ratio  E1/E0
Dtau    = 50.0    # bounce-time separation  Δτ
rho     = 1.0     # amplitude ratio  ρ
delta   = 0.0     # relative phase  δ


# xi (ξ) is the n=0 normalization/overlap constant of the affine coherent state;
# it has this closed form only for the ground state (n=0), and equals 9π/16 when nu=1
xi = (gamma(nu + 1.5) / gamma(nu + 1))**2

# Energies  (dimensionless units, E0 = 1)
E0, E1 = 1.0, r

# Bounce amplitudes  qB,  frequencies  om,  bounce times tB
qB0, qB1 = xi/np.sqrt(E0),  xi/np.sqrt(E1)
om0, om1 = 2*E0/xi,         2*E1/xi
tB0, tB1 = 0.0,             Dtau

print(f"ξ  = {xi:.6f}   (check 9π/16 = {9*np.pi/16:.6f})")
print(f"State 0 :  q_B = {qB0:.4f},  ω = {om0:.4f},  τ_B = {tB0}")
print(f"State 1 :  q_B = {qB1:.4f},  ω = {om1:.4f},  τ_B = {tB1}")

# ─────────────────────────────────────────────
# 2.  SEMICLASSICAL TRAJECTORIES 
# ─────────────────────────────────────────────
def q_sc(t, qB, om, tB):
    return qB * np.sqrt(1.0 + om**2 * (t - tB)**2)

def p_sc(t, qB, om, tB):
    dt = t - tB
    return 0.5 * qB * om**2 * dt / np.sqrt(1.0 + om**2 * dt**2)

# ─────────────────────────────────────────────
# 3.  GUIDANCE EQUATION  (eq. 39)
#
#  dx/dτ = 2 Im[ (f₀ + w f₁) / (1 + w) ]
#
#  where  fₐ = ∂ₓ ln ψₐ = 3/(2x) − αₐ x/qₐ²
#  and    w  = ρ e^{-iδ} (ψ₁/ψ₀)
#             = ρ e^{-iδ} · (C₁/C₀) · exp((α₀/q₀² − α₁/q₁²) x²/2)
#
#  C₁/C₀ = (q₀/q₁)² · (α₁/α₁*) · (α₀*/α₀)    [x^{3/2} cancels]
# ─────────────────────────────────────────────
def get_alpha_q(tau, qB, om, tB):
    # alpha (α) is the complex Gaussian spread parameter of the wavefunction
    # branch: Re(α) sets its width via q, Im(α) = -q*p carries the phase curvature
    q = q_sc(tau, qB, om, tB)
    p = p_sc(tau, qB, om, tB)
    return xi - 1j*q*p,  q          # returns (αₐ, qₐ)

def guidance_rhs(tau, state):
    x = state[0]

    al0, q0 = get_alpha_q(tau, qB0, om0, tB0)
    al1, q1 = get_alpha_q(tau, qB1, om1, tB1)

    # Log-derivatives  fₐ = ∂ₓ ln ψₐ
    f0 = 1.5/x - al0*x/q0**2
    f1 = 1.5/x - al1*x/q1**2

    # Ratio ψ₁/ψ₀  =  (C₁/C₀) · exp(Δ)
    ratio_C  = (q0/q1)**2 * (al1/np.conj(al1)) * (np.conj(al0)/al0)
    exp_arg  = 0.5 * x**2 * (al0/q0**2 - al1/q1**2)

    w   = rho * np.exp(-1j*delta) * ratio_C * np.exp(exp_arg)
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
    max_step=0.5           # small step for oscillatory region
)
print(f"ODE solver : {sol.message}  —  {sol.t.shape[0]} points")

# Compute momentum  p = ẋ/2  by calling the RHS at each point
print("Computing momentum along trajectory …")
dxdtau = np.array([guidance_rhs(sol.t[i], [sol.y[0][i]])[0]
                   for i in range(len(sol.t))])
p_traj = 0.5 * dxdtau

# Reference semiclassical trajectories for plotting
q0_ref = q_sc(tau_eval, qB0, om0, tB0)
q1_ref = q_sc(tau_eval, qB1, om1, tB1)

# ─────────────────────────────────────────────
# 5.  PLOT 
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 18))

# ── Top panel : x(τ) ──────────────────────────
ax = axes[0]
ax.plot(tau_eval, sol.y[0], color='royalblue', lw=2.0,
            label=r'$x(\tau)$ — quantum trajectory')
ax.plot(tau_eval, q0_ref,  color='forestgreen', lw=1.5,
            ls='--', label=r'$q_0(\tau)$ — semiclassical ($E_0$)')
ax.plot(tau_eval, q1_ref,  color='dimgray',     lw=1.5,
            ls=':',  label=r'$q_1(\tau)$ — semiclassical ($E_1 = 2E_0$)')

ax.axvline(tB0, color='salmon', ls=':', lw=1.3)
ax.axvline(tB1, color='orchid', ls=':', lw=1.3)
ax.text(tB0+3, 2.5, r'$\tau_{B,0}=0$',  fontsize=10, color='salmon')
ax.text(tB1+3, 2.5, r'$\tau_{B,1}=50$', fontsize=10, color='orchid')

# Inset — zoom on [−20, 80]
ax_ins = ax.inset_axes([0.33, 0.06, 0.35, 0.50])
mask = (tau_eval >= -20) & (tau_eval <= 80)
ax_ins.plot(tau_eval[mask], sol.y[0][mask],
            color='royalblue',   lw=1.5)
ax_ins.plot(tau_eval[mask], q0_ref[mask],
                color='forestgreen', lw=1.0, ls='--')
ax_ins.plot(tau_eval[mask], q1_ref[mask],
                color='dimgray',     lw=1.0, ls=':')
ax_ins.axvline(tB0, color='salmon', ls=':', lw=1)
ax_ins.axvline(tB1, color='orchid', ls=':', lw=1)
ax_ins.set_xlim(-20, 80)
ax_ins.set_ylim(1.5, 150)
ax_ins.set_title(r'Zoom on $\tau \in [-20,\,80]$', fontsize=9)
ax_ins.grid(True, which='both', alpha=0.3)
ax.indicate_inset_zoom(ax_ins, edgecolor='black', alpha=0.4)

ax.set_ylabel(r'$x(\tau)$', fontsize=14)
ax.set_xlim(tau_i, tau_f)
ax.set_ylim(1.5, 700)
ax.legend(fontsize=11, loc='upper right')
ax.set_title(
    r'Quantum trajectory — biverse state'
    r' ($r=2,\ \Delta\tau=50,\ \rho=1,\ \delta=0,\ \nu=1$)',
    fontsize=13)
ax.grid(True, which='both', alpha=0.3)

# ── Middle panel : momentum p(τ) ────────────────────────────
ax2 = axes[1]
p0_ref = p_sc(tau_eval, qB0, om0, tB0)
p1_ref = p_sc(tau_eval, qB1, om1, tB1)

ax2.plot(sol.t, p_traj, color='royalblue', lw=1.5,
         label=r'$p(\tau) = \dot{x}/2$ — quantum trajectory')
ax2.plot(tau_eval, p0_ref, color='forestgreen', lw=1.2,
         ls='--', label=r'$p_0(\tau)$ — semiclassical')
ax2.plot(tau_eval, p1_ref, color='dimgray', lw=1.2,
         ls=':', label=r'$p_1(\tau)$ — semiclassical')

ax2.axvline(tB0, color='salmon', ls=':', lw=1.3)
ax2.axvline(tB1, color='orchid', ls=':', lw=1.3)
ax2.axhline(0,   color='black',  ls='-', lw=0.5, alpha=0.5)

ax2.set_xlabel(r'$\tau$', fontsize=14)
ax2.set_ylabel(r'$p(\tau)$', fontsize=14)
ax2.set_xlim(tau_i, tau_f)
ax2.set_ylim(-3, 3)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_title(r'Momentum of the quantum trajectory', fontsize=13)

# ── Bottom panel : momentum phase space ────────────────────────────
ax3 = axes[2]
p0_ref = p_sc(tau_eval, qB0, om0, tB0)
p1_ref = p_sc(tau_eval, qB1, om1, tB1)

ax3.plot(sol.y[0], p_traj, color='royalblue', lw=1.5,
         label=r'$p(\tau) = \dot{x}/2$ — quantum trajectory')
ax3.plot(q0_ref, p0_ref, color='forestgreen', lw=1.2,
         ls='--', label=r'$p_0(\tau)$ — semiclassical')
ax3.plot(q1_ref, p1_ref, color='dimgray', lw=1.2,
         ls=':', label=r'$p_1(\tau)$ — semiclassical')

ax3.axvline(tB0, color='salmon', ls=':', lw=1.3)
ax3.axvline(tB1, color='orchid', ls=':', lw=1.3)
ax3.axhline(0,   color='black',  ls='-', lw=0.5, alpha=0.5)

ax3.set_xlabel(r'$x(tau)$', fontsize=14)
ax3.set_ylabel(r'$p(\tau)$', fontsize=14)
ax3.set_xlim(tau_i, tau_f)
ax3.set_ylim(-3, 3)
ax3.legend(fontsize=11, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_title(r'Momentum of the quantum trajectory', fontsize=13)

plt.tight_layout()
fig.savefig('quantum_trajectory.png',
            dpi=150, bbox_inches='tight')
print("\nFigure saved → quantum_trajectory.png")
plt.close()
