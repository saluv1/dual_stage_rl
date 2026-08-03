"""Definition of utility dataclasses and environment registries."""

import chex
from acme import types

from src.envs.inverted_pendulum import InvertedPendulumEnv
from src.envs.pendulum import PendulumEnv
from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.env_powerloop import QuadrotorPowerLoopEnv
from src.envs.reacher import ReacherEnv


@chex.dataclass
class ParamState:
    policy: types.NestedArray

    q1: types.NestedArray
    q2: types.NestedArray

    q1_target: types.NestedArray
    q2_target: types.NestedArray

    log_alpha: types.NestedArray


@chex.dataclass
class OptState:
    policy: types.NestedArray

    q1: types.NestedArray
    q2: types.NestedArray

    alpha: types.NestedArray


@chex.dataclass
class LearnerState:
    params: ParamState
    opt_state: OptState


@chex.dataclass
class Transitions:
    observations: types.NestedArray
    actions: types.NestedArray
    next_observations: types.NestedArray
    rewards: chex.ArrayNumpy
    dones: chex.ArrayNumpy


environments = {
    0: PendulumEnv,
    1: InvertedPendulumEnv,
    2: ReacherEnv,
    3: QuadrotorEnv,
    4: QuadrotorPowerLoopEnv,
}


env_names = {
    0: "PendulumEnv",
    1: "InvertedPendulumEnv",
    2: "ReacherEnv",
    3: "Quadrotor",
    4: "QuadrotorPowerLoop",
}