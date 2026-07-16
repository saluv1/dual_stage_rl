"""
Define the certified base set and safe set with control barrier functions
"""

import numpy as np
from .lqrgain import LQRGain

class SetIndicator:

    def __init__(self, P, c_b=8.0, zceil=3.0):

        self.P = P
        self.c_b = c_b
        self.zceil = zceil

        self.indicator = 1  # 0: Base B, 1: Continuation H, 2: Failure F

        self.hb = 0.0
        self.hs = 0.0

    def update_indicator(self):

        # Failure set has priority for safety checking
        if self.hs < 0:
            self.indicator = 2
        elif self.hb >= 0:
            self.indicator = 0
        else:
            self.indicator = 1

        return self.indicator

    def compute_hb(self, reduced_state):

        reduced_state = np.array(reduced_state, dtype=float)
        self.hb = self.c_b - reduced_state.T @ self.P @ reduced_state

        return self.hb

    def compute_hs(self, full_state):

        full_state = np.array(full_state, dtype=float)
        pz = full_state[2]
        self.hs = self.zceil - pz

        return self.hs

    def compute_indicator(self, full_state, reduced_state):

        self.compute_hb(reduced_state)
        self.compute_hs(full_state)
        self.update_indicator()

        return self.indicator