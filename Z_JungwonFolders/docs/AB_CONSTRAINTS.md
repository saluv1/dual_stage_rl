# Backup-CBF `A(x), b(x)` construction

The implementation is `bcbf/official_ab.py`; the validation entry point is
`commands/05_evaluate_ab_constraints.sh`.

## Backup policy

The fixed composed policy is

\[
\pi_b(x)=
\begin{cases}
\pi_B(x),&h_B(x)\ge0,\\
\pi_{SA}(x),&h_B(x)<0.
\end{cases}
\]

It uses the same trained actor, physical action conversion, DLQR, base set,
gravity, and Euler dynamics used in Phase I.

## Flow and sensitivity

For rollout nodes \(t_i=i\Delta t\),

\[
x_i^b=\Phi_{\pi_b}(x,t_i),
\qquad
\Psi_i=\frac{\partial\Phi_{\pi_b}(x,t_i)}{\partial x},
\qquad
\Psi_0=I.
\]

The numerical sensitivity update is

\[
J_i=\frac{\partial f_{\pi_b}}{\partial x}(x_i^b),
\qquad
\Psi_{i+1}=\exp(\Delta tJ_i)\Psi_i.
\]

## Safe-set rows

For \(h_S(x)=z_{max}-p_z\), every rollout node contributes

\[
A_i^S=-\nabla h_S(x_i^b)^\top\Psi_i g(x),
\]

\[
b_i^S=\alpha_Sh_S(x_i^b)+
\nabla h_S(x_i^b)^\top
\left[\Psi_i f(x)-f_{\pi_b}(x_i^b)\right].
\]

The term \(-f_{\pi_b}(x_i^b)\) is the relative-time correction.

## Terminal base row

At \(T=N\Delta t\),

\[
A^B=-\nabla h_B(x_N^b)^\top\Psi_Ng(x),
\]

\[
b^B=\alpha_Bh_B(x_N^b)+
\nabla h_B(x_N^b)^\top\Psi_Nf(x).
\]

The terminal row does not contain the relative-time subtraction.

Stacking gives

\[
A_{BCBF}(x)u\le b_{BCBF}(x).
\]

For \(N=100\), one scalar safe barrier, one terminal base barrier, and four
controls,

\[
A_{BCBF}\in\mathbb R^{102\times4},
\qquad
b_{BCBF}\in\mathbb R^{102}.
\]

Adding

\[
\begin{bmatrix}I\\-I\end{bmatrix}u
\le
\begin{bmatrix}u_{max}\\-u_{min}\end{bmatrix}
\]

produces \(110\) hard rows.

## Compute matrices in Python

```python
import numpy as np
from bcbf.official_ab import ABConfig, ABConstraintBuilder

cfg = ABConfig(
    dt=0.02,
    num_steps=100,
    gravity=9.81,
    base_pair_mode="official",
)

builder = ABConstraintBuilder.from_checkpoint(
    "Trained Models/001/checkpoints/best.pt",
    cfg,
)

x = np.array([
    0.0, 0.0, 2.60,
    1.5, 0.0, -0.5,
    0.9659258, 0.0, 0.2588190, 0.0,
])

A_bcbf, b_bcbf, info = builder.compute_bcbf_rows(x)
A_hard, b_hard, info = builder.compute_hard_constraints(x)
```

The returned matrices use physical actions. For a normalized task actor, call

```python
A_norm, b_norm, info = builder.compute_normalized_action_constraints(x)
```

instead of inserting normalized actions into physical-action constraints.

## Shared-slack QP

```python
Q, q, Aeq, beq, G, h, info = builder.compute_qp_matrices(
    x,
    u_reference,
)
```

uses \(z=[u^\top,\delta]^\top\) and

\[
A_{BCBF}u-\delta\mathbf1\le b_{BCBF},
\qquad
\delta\ge0,
\]

while actuator limits remain hard.

## Validation

```bash
./commands/05_evaluate_ab_constraints.sh
```

checks:

- dimensions and finite values;
- the identity between direct derivatives and \(b-Au\);
- \(\dot x=f(x)+g(x)u\);
- one-step and terminal flow sensitivities;
- automatic-differentiation and finite-difference Jacobians;
- PyTorch/JAX actor parity;
- backup feasibility on sampled states verified to be in \(C_T(\pi_b)\);
- shared-slack QP feasibility;
- the official DLQR reconstruction.

A negative backup margin for an arbitrary state outside \(C_T(\pi_b)\) is not,
by itself, an implementation failure. Avoid finite-difference checks exactly on
the nonsmooth hard-switch boundary \(h_B=0\).
