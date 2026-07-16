import numpy as np
from bcbf.set_indicator import SetIndicator
from bcbf.lqrgain import LQRGain

def test_indicator():

    gains = LQRGain()
    K, P_ = gains.gain()
    sets = SetIndicator(P=P_, c_b=8.0, zceil=3.0)

    # Case 1: exactly at hover/base center
    full_state = np.array([
        0.0, 0.0, 2.0,   # position
        0.0, 0.0, 0.0,   # velocity
        1.0, 0.0, 0.0, 0.0  # quaternion
    ])

    reduced_state = np.array([
        0.0,  # pz - z_des
        0.0, 0.0, 0.0,  # velocity
        0.0, 0.0, 0.0   # attitude error
    ])

    indicator = sets.compute_indicator(full_state, reduced_state)

    print("Case 1: hover center")
    print("hb:", sets.hb)
    print("hs:", sets.hs)
    print("indicator:", indicator)
    print()

    # Case 2: safe but outside base set
    full_state = np.array([
        0.0, 0.0, 2.8,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    ])

    reduced_state = np.array([
        0.8,  # pz - z_des
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    indicator = sets.compute_indicator(full_state, reduced_state)

    print("Case 2: safe but probably outside base")
    print("hb:", sets.hb)
    print("hs:", sets.hs)
    print("indicator:", indicator)
    print()

    # Case 3: failure, above ceiling
    full_state = np.array([
        0.0, 0.0, 3.2,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    ])

    reduced_state = np.array([
        1.2,  # pz - z_des
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    indicator = sets.compute_indicator(full_state, reduced_state)

    print("Case 3: failure")
    print("hb:", sets.hb)
    print("hs:", sets.hs)
    print("indicator:", indicator)
    print()


if __name__ == "__main__":
    test_indicator()