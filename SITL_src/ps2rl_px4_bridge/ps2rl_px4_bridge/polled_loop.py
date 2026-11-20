"""Run a bridge node without an rclpy executor (``io: poll``).

Why: on a Raspberry Pi 4B the obstacle bridge spent about three quarters of its CPU in rclpy's executor, not in its own
work (Pi-side profile, 2026-11-20): every PX4 message and every timer is one pass of ``wait_for_ready_callbacks``,
which rebuilds the wait set, the callback bookkeeping and a task per callback in Python, about 200-250 times a second
(odometry 100 Hz, local position 50 Hz, two 50 Hz timers, status). The bridge process filled its core and the control
steps fell behind (72 % of them done, step p95 133 ms against a 40 ms period).

This loop does the node's work in one thread and wakes only when there is something to do:

  - Controller phase. After a control step it sleeps until shortly before the vehicle clock (the odometry stamps)
    reaches the next one, then waits in a one-entry wait set on the odometry subscription and makes the step as soon
    as the odometry message that reaches it arrives - the rule of ``LandingBridge._policy_loop``, whose thread is not
    used. The other subscriptions are read right before each step, the flight-logic tick runs in the same pass, and
    the setpoint stream stays quiet while the steps publish their commands (about 50 wake-ups a second at 25 Hz,
    against about 225 executor passes before).
  - Other phases: it wakes at the timer deadlines only (flight logic at the control rate, the setpoint stream at
    ``setpoint_rate``, the health check every 5 s) and reads every subscription then.
  - Messages are taken with ``take_message`` (the executor's own call) and handed to the node's callbacks; of
    odometry and local position only the newest is handed on. The arrival of the odometry message that will reach the
    next step is predicted from the newest one's stamp and arrival time (the middleware's receive stamp) and the
    odometry period; the loop wakes half an odometry period (at least 3 ms) before it and waits for that message.

No parameter services: ``ros2 param get`` does not answer this node (nothing here needs it).
"""

from __future__ import annotations

from collections import deque
import gc
import statistics
import time

import numpy as np


def _next(deadline: float, period: float, now: float) -> float:
    """The next deadline of a periodic task; after a stall start again from now instead of catching up in a burst."""
    deadline += period
    return deadline if deadline > now else now + period


class PolledLoop:
    def __init__(self, node, *, wake_topic: str, latest_only_topics: tuple = (), early_s: float = 0.003,
                 max_wait_s: float = 0.02, poll_s: float = 0.002, slack_s: float = 0.002) -> None:
        self.node = node
        self.early_s = float(early_s)          # wake this long before the predicted control step, then wait for odometry
        self.max_wait_s = float(max_wait_s)
        self.poll_s = float(poll_s)
        self.slack_s = float(slack_s)          # timers due this soon run in the current pass

        def norm(name: str) -> str:
            return "/" + str(name).strip().lstrip("/")

        subs = list(node.subscriptions)
        self.wake = next((s for s in subs if norm(s.topic_name) == norm(wake_topic)), None)
        if self.wake is None:
            raise RuntimeError(f"io poll: the node has no subscription on {wake_topic}")
        self.others = [s for s in subs if s is not self.wake]
        latest = {norm(t) for t in latest_only_topics if t}
        self.others_latest_only = [norm(s.topic_name) in latest for s in self.others]

        self._ws = None
        self._sigint = None
        try:
            from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
            from rclpy.signals import SignalHandlerGuardCondition
            self._ws = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, node.context.handle)
            self._sigint = SignalHandlerGuardCondition(node.context)
        except Exception as exc:   # an rclpy without these internals: poll on a short sleep instead
            self._ws = None
            node.get_logger().warn(f"io poll: no rclpy wait set ({exc!r}); checking for odometry every "
                                   f"{self.poll_s * 1e3:.0f} ms near a control step instead")

        # passes during the controller phase (for the flight record)
        self.passes_ms: list[float] = []
        self._prev_stamp = 0
        self._spacing: deque = deque(maxlen=50)     # odometry stamp spacing [us]
        self.policy_t0: float | None = None
        self.policy_t1: float | None = None

    # ------------------------------------------------------------------ pieces
    def _wait_odometry(self, timeout_s: float) -> bool:
        """Sleep until odometry arrives or timeout_s passes; False on Ctrl-C / shutdown."""
        timeout_s = max(0.0, timeout_s)
        if self._ws is None:
            time.sleep(min(timeout_s, self.poll_s))
            return True
        ws = self._ws
        ws.clear_entities()
        ws.add_subscription(self.wake.handle)
        ws.add_guard_condition(self._sigint.handle)
        ws.wait(int(timeout_s * 1e9))
        guards = ws.get_ready_entities("guard_condition")
        return not (guards and self._sigint.handle.pointer in guards)

    def _sleep(self, timeout_s: float) -> bool:
        """Sleep without waking for messages (they queue); False on Ctrl-C / shutdown."""
        timeout_s = max(0.0, timeout_s)
        if self._ws is None:
            time.sleep(timeout_s)
            return True
        ws = self._ws
        ws.clear_entities()
        ws.add_guard_condition(self._sigint.handle)
        ws.wait(int(timeout_s * 1e9))
        guards = ws.get_ready_entities("guard_condition")
        return not (guards and self._sigint.handle.pointer in guards)

    @staticmethod
    def _take(sub, latest_only: bool):
        """Hand the queued messages to the callback (only the newest if latest_only); the newest's message info."""
        last = None
        while True:
            try:
                with sub.handle:
                    info = sub.handle.take_message(sub.msg_type, sub.raw)
            except Exception:      # destroyed during shutdown
                info = None
            if info is None:
                break
            if latest_only:
                last = info
            else:
                sub.callback(info[0])
        if last is not None:
            sub.callback(last[0])
        return last

    def _take_odometry(self) -> None:
        """The newest odometry, when it arrived (the middleware's receive stamp, in monotonic time) and the odometry
        period (median spacing of the stamps): the control step is predicted from these, not from when this loop got
        round to the message."""
        last = None
        while True:
            try:
                with self.wake.handle:
                    info = self.wake.handle.take_message(self.wake.msg_type, self.wake.raw)
            except Exception:      # destroyed during shutdown
                info = None
            if info is None:
                break
            stamp = int(getattr(info[0], "timestamp", 0))
            if self._prev_stamp and 0 < stamp - self._prev_stamp < 200_000:
                self._spacing.append(stamp - self._prev_stamp)
            self._prev_stamp = stamp
            last = info
        if last is None:
            return
        self.wake.callback(last[0])
        now_mono, now_wall = time.monotonic(), time.time()
        rx = last[1].get("received_timestamp", 0) if isinstance(last[1], dict) else 0
        node = self.node
        node._odom_mono = now_mono - min(max(now_wall - rx * 1e-9, 0.0), 0.5) if rx else now_mono
        if len(self._spacing) >= 5:
            node._odom_period = statistics.median(self._spacing) * 1e-6

    def _take_others(self) -> None:
        for sub, latest_only in zip(self.others, self.others_latest_only):
            self._take(sub, latest_only)

    def summary(self) -> str | None:
        """One line on the controller phase: wake-ups per second and the length of a pass."""
        if not self.passes_ms or self.policy_t0 is None or self.policy_t1 is None:
            return None
        span = max(self.policy_t1 - self.policy_t0, 1e-6)
        p = np.asarray(self.passes_ms)
        return (f"bridge loop (io poll) during the controller phase: {len(p) / span:.0f} wake-ups/s; a pass takes median "
                f"{np.median(p):.2f} / p95 {np.percentile(p, 95):.2f} / max {p.max():.1f} ms (control steps included); "
                f"busy {100.0 * p.sum() * 1e-3 / span:.0f} % of the wall time")

    # ------------------------------------------------------------------ the loop
    def run(self) -> None:
        node = self.node
        policy = node._Phase.POLICY
        logic_dt = 1.0 / float(node.policy_rate)
        stream_dt = 1.0 / float(node.setpoint_rate)
        quiet_dt = 1.5 * float(node.runtime.dt)        # the stream's silence after a published command
        now = time.monotonic()
        next_logic, next_stream, next_health = now, now, now + 5.0
        node._loop_stats = self
        gc.collect()
        gc.freeze()           # what exists now is never garbage: keep the collector from walking it every pass
        while node.context.ok() and not (node.exit_when_done and node.done_event.is_set()):
            now = time.monotonic()
            t_timer = min(next_logic, next_stream, next_health)
            in_policy = node.flight.phase is policy and not node._logic_failed
            due = node._policy_due_mono() if in_policy else None
            early = max(self.early_s, 0.5 * (node._odom_period or 0.0))   # stamps jitter: up to half a period early
            mode_odom = in_policy and (due is None or due - early <= now)
            if mode_odom:
                ok = self._wait_odometry(min(t_timer, now + self.max_wait_s) - now)
            else:
                wake = t_timer if not in_policy else min(t_timer, due - early)
                ok = self._sleep(min(wake, now + 0.1) - now)
            if not ok:
                break

            t0 = time.monotonic()
            self._take_odometry()
            others_read = False

            def read_others() -> None:
                nonlocal others_read
                if not others_read:
                    self._take_others()
                    others_read = True

            stepped = node._policy_poll(before_step=read_others) if in_policy else False
            now = time.monotonic()
            if stepped:
                # the flight-logic tick rides on the step (in the controller phase it only checks for failures and
                # starts / ends the CPU record); the stream has nothing to add while steps publish
                node._logic_tick()
                next_logic = now + logic_dt
                next_stream = max(next_stream, now + quiet_dt)
            elif now + self.slack_s >= next_logic:
                read_others()
                node._logic_tick()
                next_logic = _next(next_logic, logic_dt, now)
            if now + self.slack_s >= next_stream:
                node._stream_tick()
                next_stream = _next(next_stream, stream_dt, now)
            if now + self.slack_s >= next_health:
                read_others()
                node._health_tick()
                next_health = now + 5.0
            if in_policy:
                t1 = time.monotonic()
                if self.policy_t0 is None:
                    self.policy_t0 = t0
                self.policy_t1 = t1
                if len(self.passes_ms) < 200_000:
                    self.passes_ms.append((t1 - t0) * 1e3)


__all__ = ["PolledLoop"]
