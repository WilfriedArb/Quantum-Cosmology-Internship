"""
3D Visualization of the Quantum Potential Q(x, tau)
Axes: x (Scale factor), tau (Time), Q (Quantum Potential)
"""

import numpy as np
import math
from scipy.integrate import quad
from scipy.special import gamma, genlaguerre
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. MODEL PARAMETERS
# ─────────────────────────────────────────────
nu      = 1       # affine parameter
n0      = 0       # quantum number of state 0
n1      = 0       # quantum number of state 1
r       = 2.0     # energy ratio E1/E0
Dtau    = 50.0    # bounce-time separation
rho     = 1.0     # amplitude ratio
delta   = 0.0     # relative phase

# ξ is the normalization/overlap constant of the n-th affine coherent state,
# obtained by numerically integrating the weighted squared Laguerre polynomial
def compute_xi(n_val, nu_val):
    Ln_func = genlaguerre(n_val, nu_val)
    def integrand(y):
        return y**(nu_val + 0.5) * (Ln_func(y))**2 * np.exp(-y)
    integral_val, _ = quad(integrand, 0, np.inf)
    Gn = (math.factorial(n_val) / gamma(nu_val + n_val + 1)) * integral_val
    return Gn**2

xi0 = compute_xi(n0, nu)
xi1 = compute_xi(n1, nu)

E0, E1 = 1.0, r
qB0, qB1 = xi0/np.sqrt(E0),  xi1/np.sqrt(E1)
om0, om1 = 2*E0/xi0,         2*E1/xi1
tB0, tB1 = 0.0,              Dtau

print("Biverse parameters computed.")


def q_sc(t, qB, om, tB):
    return qB * np.sqrt(1.0 + om**2 * (t - tB)**2)

def p_sc(t, qB, om, tB):
    dt = t - tB
    return 0.5 * qB * om**2 * dt / np.sqrt(1.0 + om**2 * dt**2)

# ─────────────────────────────────────────────
# 2. VECTORIZED COMPUTATION OF THE QUANTUM POTENTIAL Q(x, tau)
# ─────────────────────────────────────────────

tau_surf = np.linspace(-50, 100, 400)  # time range of interest
x_surf = np.linspace(0.1, 200, 4000)
X, TAU = np.meshgrid(x_surf, tau_surf)

def Q_tot_vectorized(x, tau, amp=1.0, delta=0.0):
    """
    Compute the total effective quantum potential Q(x, tau), including V_nu.

    Q is obtained from the logarithmic derivative g = Re(f'/f) of the
    (x-power-law-stripped) complex wavefunction amplitude f = psi_0 + w*psi_1,
    via Q = -(2*nu+1)*g/x - g**2 - g', where the -(2*nu+1)*g/x term is the
    affine/radial-measure correction to the plain 1D Bohm quantum potential
    -R''/R. Differentiating analytically in f avoids a noisy numerical
    second derivative of the probability density.
    """
    q0, p0 = q_sc(tau, qB0, om0, tB0), p_sc(tau, qB0, om0, tB0)
    q1, p1 = q_sc(tau, qB1, om1, tB1), p_sc(tau, qB1, om1, tB1)

    z0 = (xi0 - 1j*q0*p0)/q0**2
    z1 = (xi1 - 1j*q1*p1)/q1**2

    A0 = np.exp(-1j*(nu+1)*np.arctan(q0*p0/xi0))/q0**(nu+1)
    A1 = np.exp(-1j*(nu+1)*np.arctan(q1*p1/xi1))/q1**(nu+1)

    e0 = A0 * np.exp(-z0 * x**2 / 2)
    e1 = amp * np.exp(-1j*delta) * A1 * np.exp(-z1 * x**2 / 2)

    f   = e0 + e1        # amplitude (biverse superposition, x^(nu+0.5) prefactor omitted: cancels in g)
    fp  = -x * (z0*e0 + z1*e1)
    fpp = (z0**2 * x**2 - z0)*e0 + (z1**2 * x**2 - z1)*e1

    g  = np.real(fp / f)
    gp = np.real(fpp / f - (fp / f)**2)

    return -(2*nu+1) * g / x - g**2 - gp

# Allocate the (tau, x) matrix
Q_matrix = np.zeros((len(tau_surf), len(x_surf)))

# Loop over time, passing the whole spatial vector at once (much faster)
for idx_tau, t_val in enumerate(tau_surf):
    Q_matrix[idx_tau, :] = Q_tot_vectorized(x_surf, t_val, amp=rho, delta=delta)

# Handling singularities: clip the infinities (deep wells and spikes)
Q_min = -3.0
Q_max = 10.0

# Q_matrix already includes V_classique via Q_tot_vectorized.
# Nothing more to add here, just clip it.
Q_matrix_clipped = np.clip(Q_matrix, Q_min, Q_max)

# ─────────────────────────────────────────────
# 3. 3D FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, TAU, Q_matrix_clipped,
                       cmap='viridis',      # 'plasma' or 'magma' also render nicely
                       edgecolor='black',
                       linewidth=0.3,
                       antialiased=True)

ax.set_xlabel(r'Scale factor $x$', fontsize=12)
ax.set_ylabel(r'Time $\tau$', fontsize=12)
ax.set_zlabel(r'Quantum potential $Q(x,\tau)$', fontsize=12)

# Adjust the viewing angle (elevation, azimuth) to see the bounce valley clearly
ax.view_init(elev=30, azim=-70)
fig.savefig('quantum_potential_3D_analytique.png', dpi=150, bbox_inches='tight')
print("Figure saved -> quantum_potential_3D_analytique.png")
