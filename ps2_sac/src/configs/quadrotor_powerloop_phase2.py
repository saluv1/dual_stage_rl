"""PS2-RL Phase-II power-loop configuration.

Requires a vanilla tracker checkpoint, frozen Phase-I backup policy,
certified base set/controller, and state-dependent BCBF provider.
"""

from src.configs.quadrotor_powerloop_vanilla import get_config as vanilla_config


def get_config():
    config = vanilla_config()

    config.use_cil = True
    config.cil_mode = "bcbf"
    config.terminate_on_ceiling = True

    config.warm_start_path = (
        "results/powerloop_vanilla/"
        "powerloop_vanilla_mm.pickle"
    )
    config.backup_policy_path = ""
    config.base_set_path = ""

    config.backup_horizon_seconds = 2.0
    config.backup_steps = 100
    config.alpha_s = 4.0
    config.alpha_b = 2.0
    config.qp_slack_penalty = 1e6

    return config
