# Quantum Cosmology — The Biverse Model

M1 internship project, Institut d'Astrophysique de Paris (IAP, CNRS) — May–July 2026.

**Author:** Wilfried Aribaud (Université Paris Cité, Magistère de Physique Fondamentale)
**Supervisor:** Patrick Peter (CNRS Research Director) · **Reviewer:** Pierre Auclair (CNRS)

Full report: [`rapport_de_stage_WilfriedARIBAUD.pdf`](rapport_de_stage_WilfriedARIBAUD.pdf)

## Overview

This project studies what happens to the very early Universe's geometry when it is
placed in a **quantum superposition of two bouncing states** instead of a single
quasi-classical one — a toy background we call the **Biverse**. The two ingredients are:

1. **Affine quantum cosmology.** Because the scale factor is strictly positive, its phase
   space is a half-plane, not a plane, and the usual canonical (Weyl–Heisenberg)
   quantization doesn't apply. Quantizing instead with the affine group generates a
   repulsive $1/\hat X^2$ term that forbids the Universe from reaching zero volume: the
   Big Bang singularity is replaced by a smooth **bounce**. We use the exact affine
   coherent states of Bergeron et al., whose mean values reproduce the semiclassical
   bouncing trajectory exactly.
2. **De Broglie–Bohm pilot-wave mechanics.** A single coherent state's mean value already
   equals its semiclassical trajectory, so the pilot-wave formalism adds nothing new by
   itself. It becomes essential once the background is a **superposition of two coherent
   states** (a Biverse): the mean value $\langle \hat X\rangle$ then no longer defines a
   unique background trajectory, while the Bohmian guidance equation still does, once an
   initial condition is fixed.

The Biverse is built from two affine coherent states — bouncing at different times,
with different energies, weights and relative phase — and its Bohmian trajectory,
mean value, and quantum potential are all derived in closed form and explored
numerically here.

## The model

$$|\Psi_{\textsc{b}}\rangle = \mathcal{N}_2\left(|0\rangle + \alpha e^{-i\delta}|1\rangle\right),\qquad
|0\rangle \equiv |E_0,\tau_{\textsc{b},0}\rangle,\quad |1\rangle \equiv |rE_0,\tau_{\textsc{b},0}+\Delta\tau\rangle$$

Four parameters control the dynamics:

| Parameter | Role |
|---|---|
| $\alpha$ (`rho` in the code) | relative weight between the two branches |
| $\delta$ (`delta`) | relative phase between the branches |
| $r$ (`r`) | energy ratio $E_1/E_0$ between the branches |
| $\Delta\tau$ (`Dtau`) | time delay between the two bounces |

The Bohmian velocity field follows from the guidance equation
$\dot x = 2\,\partial_x S = 2\,\Im(\partial_x\Psi_{\textsc{b}}/\Psi_{\textsc{b}})$,
which the code integrates numerically (`scipy.integrate.solve_ivp`) starting from an
initial condition on one branch's classical asymptote.

## Key results

- **A closed-form guidance field.** The gradient $\partial_x S$ splits into each branch's
  own probability current plus an interference term, giving the Bohmian velocity field
  analytically at every point (no numerical differentiation needed).
- **A genuine double bounce with a renormalized asymptotic expansion rate.** The Bohmian
  trajectory follows branch 0 through its bounce, gets captured by branch 1's bounce,
  and only then settles onto an asymptotic slope $2c\sqrt{E_0}$ with $c\neq1$ — neither
  branch's own rate. The two wave packets never fully separate, so this interference
  imprint on the expansion rate persists at all times, unlike a position offset (which
  would become unobservably small compared to $q_0(\tau)$).
- **The mean value $\langle \hat X\rangle$ is not a viable background.** It has a
  closed form (`x_mean.ipynb`) but (i) statistically, it describes an ensemble average,
  meaningless for a single, unobserved Universe, and (ii) dynamically, it obeys no
  closed evolution equation — Ehrenfest's theorem only closes for at-most-quadratic
  Hamiltonians, and the affine $\hat X^{-2}$ term breaks that. The Bohmian trajectory,
  by contrast, satisfies an exact (quantum-corrected) Friedmann equation.
- **The two pictures disagree on the effective radiation density.** Both the mean-value
  and the Bohmian trajectory recover the classical $\langle a(\eta)\rangle\propto\eta$
  law at late times, but with different asymptotic slopes — so the interpretive choice,
  invisible in a lab, becomes a computable cosmological quantity.
- **A regular total quantum potential.** $Q_{\mathrm{tot}}=V_\nu+Q$ is finite everywhere,
  including at $x=0$, once the centrifugal divergence is cancelled analytically against
  the quantum potential's own divergence — a cancellation that is numerically unstable
  to compute directly and must be done in closed form first (`quantum_pot_analytique.py`).
  In the overlap region it develops deep interference wells, whose gradient drives the
  trajectory's sudden accelerations at each bounce.

See the [report](rapport_de_stage_WilfriedARIBAUD.pdf) for the full derivations
(exact coherent-state construction, Ehrenfest constraint, asymptotic expansions) and
for the observational outlook (expected imprints on the CMB power spectrum through the
Mukhanov–Sasaki $a''/a$ term).

## Repository structure

```
rapport_de_stage_WilfriedARIBAUD.pdf   full internship report (French)
code_py/                                standalone Python scripts
code_ipynb/                             Jupyter notebooks
image/, animation/                      example output figures and GIFs
```

`code_py/` and `code_ipynb/` also contain earlier exploratory/development notebooks and
scripts (parameter tests, alternate derivations) kept for reference; the files below are
the cleaned, documented entry points.

### `code_py/`

| File | What it does |
|---|---|
| `quantum_trajectory.py` | Baseline Biverse trajectory ($n_0=n_1=0$): solves the guidance equation and plots $x(\tau)$ and phase space against the two semiclassical branches $q_0,q_1$. |
| `quantum_n_general.py` | Same, generalized to arbitrary excitation levels $n_0,n_1$: trajectory, phase space, and wavefunction probability densities. |
| `quantum_func_3d_analytic.py` | The central figure of the report: 3D $|\Psi_{\textsc{b}}|^2$ surface, Bohmian trajectory, and the closed-form mean value $\langle\hat X\rangle$, overlaid on a high-contrast 2D projection. |
| `quantum_func_anim.py` | Animates $|\psi_0|^2$, $|\psi_1|^2$ and $|\Psi_{\textsc{b}}|^2$ evolving through both bounces (GIF). |
| `quantum_pot_analytique.py` | Computes and plots the total effective quantum potential $Q_{\mathrm{tot}}(x,\tau)=V_\nu+Q$ via the analytic log-derivative form, avoiding the numerically unstable direct second derivative of $|\Psi_{\textsc{b}}|$. |

### `code_ipynb/`

| File | What it does |
|---|---|
| `quantum_n0_3d.ipynb` | 3D sweep of the trajectory over the excitation number $n_0$ (0 to 39) of one branch. |
| `quantum_n0_n1_3d_anim.ipynb` | Same idea, animated: sweeps $n_1$ while showing every $n_0$ trajectory at once (GIF). |
| `quantum_parameter_anim_2.ipynb` | Animates the trajectory's response to each of $\alpha,\delta,\Delta\tau,r$ individually (one GIF per parameter). |
| `quantum_parameter_image.ipynb` | Static companion: side-by-side comparison at the two extreme values of each parameter. |
| `x_mean.ipynb` | Derivation of the closed-form mean value $\langle\hat X\rangle(\tau)$, checked against the numerical Bohmian trajectory and the naive arithmetic mean $(q_0+q_1)/2$. |

## Running the code

```bash
pip install numpy scipy matplotlib pillow
python code_py/quantum_trajectory.py
# or, for the notebooks:
jupyter notebook code_ipynb/
```

`pillow` is only needed for the scripts/notebooks that export GIF animations.

## References

The model builds on the exact affine coherent states of Bergeron, Gazeau, Małkiewicz &
Peter, *New Class of Exact Coherent States: Enhanced Quantization of Motion on the Half
Line* (Phys. Rev. D, 2024), and extends the two-state superposition ("Biverse")
background studied in Mazde, Mickel & Peter, *Quantum Cosmological Background
Superposition and Perturbation Predictions* (arXiv, 2025). Full citations are in the
report's bibliography.
