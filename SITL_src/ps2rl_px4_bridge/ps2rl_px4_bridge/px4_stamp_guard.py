"""Keep PX4 message timestamps on one continuous clock across uXRCE-DDS time-sync resets.

What happens without it (Gazebo SITL, 2026-11-18, three policy flights aborted this way): PX4's DDS client
keeps its clock in step with the agent's (the computer's wall clock). When the simulation's real-time
factor wanders, that time sync declares a jump ("[timesync] time jump detected. Resetting time
synchroniser") and, while it re-converges, PX4 stamps a few outgoing messages in the agent's clock
(Unix time, ~1.79e9 s) instead of its own (seconds since PX4 started). The next message is back on
PX4's clock, i.e. 1.79e9 s "earlier", and the landing bridge took that as an autopilot restart and
aborted, although the vehicle and its estimate were fine.

``StampGuard`` maps one topic's stamps onto a continuous clock: ordinary steps pass through unchanged
(apart from a constant offset after a jump), a step that cannot be real time - backwards by more than
``max_back_s``, or forwards by more than the wall time since the previous message allows - is treated
as a change of time base: the guard re-anchors so that the mapped clock continues from where it was.
No ROS imports: the logic is tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StampJump:
    wall_s: float
    raw_step_s: float     # the step in the raw stamps that was absorbed


@dataclass
class StampGuard:
    max_back_s: float = 0.5      # a topic's own stamps never go back more than this in normal operation
    max_rtf: float = 3.0         # simulation faster than this x wall time is not believed
    slack_s: float = 1.0         # plus this much forward slack (odometry gaps, scheduling)
    offset_us: int = 0
    last_raw_us: int | None = None
    last_out_us: int | None = None
    last_wall_s: float | None = None
    jumps: list[StampJump] = field(default_factory=list)

    def __call__(self, raw_us: int, wall_s: float) -> tuple[int, StampJump | None]:
        """Mapped stamp for a raw one, and the jump if one was absorbed."""
        raw_us = int(raw_us)
        if raw_us <= 0:
            return raw_us, None
        if self.last_out_us is None:
            self.last_raw_us, self.last_out_us, self.last_wall_s = raw_us, raw_us, wall_s
            return raw_us, None
        elapsed = max(0.0, wall_s - (self.last_wall_s if self.last_wall_s is not None else wall_s))
        step_us = raw_us - self.last_raw_us
        max_fwd_us = int((self.max_rtf * elapsed + self.slack_s) * 1e6)
        jump = None
        if step_us < -int(self.max_back_s * 1e6) or step_us > max_fwd_us:
            # a new time base: continue the mapped clock by (at most 0.1 s of) the wall time that passed
            cont_us = self.last_out_us + int(min(elapsed, 0.1) * 1e6)
            self.offset_us = cont_us - raw_us
            jump = StampJump(wall_s=wall_s, raw_step_s=step_us * 1e-6)
            self.jumps.append(jump)
        out = raw_us + self.offset_us
        self.last_raw_us, self.last_wall_s = raw_us, wall_s
        self.last_out_us = max(self.last_out_us, out)
        return out, jump


__all__ = ["StampGuard", "StampJump"]
