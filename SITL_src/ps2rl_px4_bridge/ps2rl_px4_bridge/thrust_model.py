"""Map PS2-RL's ``a_cmd`` [m/s^2] onto PX4's normalized collective thrust [0, 1].

PS2-RL commands a mass-normalized thrust acceleration along the body +z axis.
PX4's ``VehicleRatesSetpoint.thrust_body`` wants a unitless [0, 1] value whose
relationship to actual acceleration is airframe-specific and *not* linear:
rotor thrust goes roughly with the square of RPM, and PX4's own mixer applies
``THR_MDL_FAC`` on top.

Two models are provided.

``linear``
    ``a = g * T / hover_thrust``.  Zero calibration required, exact at hover,
    and increasingly optimistic above it (it under-commands at high thrust).
    Fine for a first bring-up, not for the powerloop.

``quadratic``
    ``a = k2*T^2 + k1*T + k0``, fitted from a thrust sweep (see
    ``thrust_calib_node.py``).  Inverted analytically at runtime.  This is what
    you want before running an aggressive reference.

Both clamp to ``[thrust_min, thrust_max]``.  Saturation is reported so the
bridge can log how often the policy is asking for more thrust than the airframe
can deliver — a silently saturating actuator invalidates the CBF guarantee,
since the projection assumed the commanded action was applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ThrustModel:
    model: str = "linear"
    gravity: float = 9.81
    hover_thrust: float = 0.60          # PX4 MPC_THR_HOVER for your airframe
    k0: float = 0.0                      # quadratic coefficients: a = k2 T^2 + k1 T + k0
    k1: float = 0.0
    k2: float = 0.0
    thrust_min: float = 0.02
    thrust_max: float = 1.0

    def __post_init__(self) -> None:
        model = str(self.model).strip().lower()
        if model not in ("linear", "quadratic"):
            raise ValueError(f"model must be 'linear' or 'quadratic', got {self.model!r}")
        self.model = model
        if not (0.0 < self.hover_thrust < 1.0):
            raise ValueError(f"hover_thrust must be in (0, 1), got {self.hover_thrust}")
        if model == "quadratic" and abs(self.k2) < 1e-12 and abs(self.k1) < 1e-12:
            raise ValueError("quadratic model needs non-zero k1/k2 — run the calibration first")

    # ------------------------------------------------------------------ #

    def accel_from_thrust(self, thrust: float) -> float:
        """Forward model, used by the calibration fit and for sanity checks."""
        t = float(thrust)
        if self.model == "linear":
            return self.gravity * t / self.hover_thrust
        return self.k2 * t * t + self.k1 * t + self.k0

    def thrust_from_accel(self, a_cmd: float) -> tuple[float, bool]:
        """Inverse model. Returns ``(thrust_normalized, saturated)``."""
        a = float(a_cmd)

        if self.model == "linear":
            t = self.hover_thrust * a / self.gravity
        else:
            t = self._invert_quadratic(a)

        clamped = min(max(t, self.thrust_min), self.thrust_max)
        return clamped, (abs(clamped - t) > 1e-9)

    # ------------------------------------------------------------------ #

    def _invert_quadratic(self, a: float) -> float:
        """Solve k2 T^2 + k1 T + (k0 - a) = 0 for the physical root."""
        c = self.k0 - a

        if abs(self.k2) < 1e-12:                      # degenerate: affine
            return -c / self.k1

        disc = self.k1 * self.k1 - 4.0 * self.k2 * c
        if disc < 0.0:
            # Requested acceleration is outside the achievable envelope; return
            # the vertex, i.e. the closest thrust the model can produce.
            return -self.k1 / (2.0 * self.k2)

        sqrt_disc = math.sqrt(disc)
        r1 = (-self.k1 + sqrt_disc) / (2.0 * self.k2)
        r2 = (-self.k1 - sqrt_disc) / (2.0 * self.k2)

        # Pick the root inside [0, 1]; prefer the smaller feasible one, since
        # the physical branch is monotonically increasing from T = 0.
        candidates = sorted(r for r in (r1, r2) if -1e-6 <= r <= 1.0 + 1e-6)
        if candidates:
            return candidates[0]
        return max(r1, r2)

    # ------------------------------------------------------------------ #

    @property
    def max_accel(self) -> float:
        """Acceleration achievable at ``thrust_max`` — compare against 4g."""
        return self.accel_from_thrust(self.thrust_max)

    def describe(self) -> str:
        if self.model == "linear":
            return (
                f"linear (hover_thrust={self.hover_thrust:.3f}), "
                f"a_max={self.max_accel:.2f} m/s^2 = {self.max_accel / self.gravity:.2f} g"
            )
        return (
            f"quadratic (k2={self.k2:.3f}, k1={self.k1:.3f}, k0={self.k0:.3f}), "
            f"a_max={self.max_accel:.2f} m/s^2 = {self.max_accel / self.gravity:.2f} g"
        )


def fit_quadratic(thrusts, accels, gravity: float = 9.81) -> ThrustModel:
    """Least-squares fit of ``a = k2 T^2 + k1 T + k0`` from a calibration sweep."""
    import numpy as np

    t = np.asarray(thrusts, dtype=np.float64).reshape(-1)
    a = np.asarray(accels, dtype=np.float64).reshape(-1)
    if t.size < 3:
        raise ValueError("need at least 3 calibration points")

    k2, k1, k0 = np.polyfit(t, a, 2)
    return ThrustModel(model="quadratic", gravity=gravity, k0=float(k0), k1=float(k1), k2=float(k2))
