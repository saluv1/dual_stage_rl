"""Official reset-library curriculum sampling in the original local style."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from official_phase1_evaluation.official_resets import (
    OFFICIAL_REGIONS,
    ResetLibrary,
    load_reset_library,
    normalize_region_name,
    perturb_state,
)
from .state_action import compute_bfc

# The paper/checkpoint selection emphasizes the difficult trace regions.
MU_SA_WEIGHTS = {
    "general_trace": 1.0,
    "near_ceiling": 2.0,
    "bridge": 2.5,
    "base_shell": 1.0,
}

DEFAULT_RESET_LIBRARY = (
    Path(__file__).resolve().parents[2]
    / "official_phase1_evaluation"
    / "assets"
    / "reset_library.pkl"
)


@dataclass
class OfficialPhase1Sampler:
    library: ResetLibrary
    split: str = "train"

    @classmethod
    def load(cls, path=DEFAULT_RESET_LIBRARY, split="train"):
        return cls(load_reset_library(path), split=str(split))

    @property
    def regions(self):
        pools = self.library.split_pools.get(self.split)
        if pools:
            return pools
        return self.library.all_pools

    @property
    def region_names(self):
        return [r for r in OFFICIAL_REGIONS if r in self.regions and len(self.regions[r]) > 0]

    def curriculum_weights(self, s):
        cfg = self.library.library_config
        s = float(np.clip(s, 0.0, 1.0))
        values = {}
        for region in self.region_names:
            stem = {
                "general_trace": "general",
                "near_ceiling": "near_ceiling",
                "bridge": "bridge",
                "base_shell": "base_shell",
            }[region]
            low = float(cfg[f"mix_{stem}_low"])
            high = float(cfg[f"mix_{stem}_high"])
            values[region] = (1.0 - s) * low + s * high
        total = sum(values.values())
        if total <= 0.0:
            raise RuntimeError("Official curriculum has no positive region weights")
        return {name: value / total for name, value in values.items()}

    def sample(self, sets, s, rng, *, force_region=None, return_region=False, max_tries=10000):
        weights = self.curriculum_weights(s)
        names = self.region_names
        if force_region is not None:
            requested = normalize_region_name(force_region)
            if requested not in names:
                raise ValueError(f"Region {requested!r} is unavailable in split {self.split!r}")
            names = [requested]
            probs = np.array([1.0])
        else:
            probs = np.asarray([weights[name] for name in names], dtype=float)
            probs /= probs.sum()

        for _ in range(int(max_tries)):
            region = str(rng.choice(names, p=probs))
            pool = np.asarray(self.regions[region], dtype=float)
            base = pool[int(rng.integers(0, len(pool)))]
            state = perturb_state(
                self.library,
                base,
                rng=rng,
                curriculum_scale=float(s),
                region=region,
            )
            _, f, _ = compute_bfc(sets, state)
            # The official collector accepts every safe reset, including states
            # already inside the LQR handoff set.  Those episodes naturally
            # generate a real LQR-applied transition and an exact goal flag.
            if f == 0.0:
                return (state, region) if return_region else state
        raise RuntimeError(f"Failed to sample safe state at s={float(s):.3f}")

    def fixed_set(self, sets, *, seed, episodes_per_region, difficulty=1.0):
        count = int(episodes_per_region)
        heldout = self.library.heldout_reset_sets.get(self.split)
        if heldout is not None:
            labels = np.asarray(heldout["region"]).astype(str)
            output = {}
            for region in self.region_names:
                states = np.asarray(heldout["states"])[labels == region]
                if len(states) < count:
                    raise ValueError(
                        f"Held-out split {self.split!r} has only {len(states)} "
                        f"states for {region!r}, requested {count}"
                    )
                output[region] = np.asarray(states[:count], dtype=np.float64).copy()
            return output

        rng = np.random.default_rng(int(seed))
        output = {}
        for region in self.region_names:
            states = [
                self.sample(
                    sets,
                    difficulty,
                    rng,
                    force_region=region,
                    return_region=False,
                )
                for _ in range(count)
            ]
            output[region] = np.asarray(states, dtype=np.float64)
        return output


def load_official_sampler(reset_library=DEFAULT_RESET_LIBRARY, split="train"):
    return OfficialPhase1Sampler.load(reset_library, split=split)


def get_curriculum_weights(s, sampler=None):
    sampler = sampler or load_official_sampler()
    return sampler.curriculum_weights(s)


def sample_initial_state(sets, regions, s, rng, max_tries=10000, return_region=False, force_region=None):
    """Compatibility wrapper; ``regions`` is an OfficialPhase1Sampler."""
    if not isinstance(regions, OfficialPhase1Sampler):
        raise TypeError("regions must be an OfficialPhase1Sampler in the official-aligned version")
    return regions.sample(
        sets,
        s,
        rng,
        force_region=force_region,
        return_region=return_region,
        max_tries=max_tries,
    )


def inspect_sampler(sets, regions, s, rng, n_samples=500):
    counts = {name: 0 for name in regions.region_names}
    hb, hs, speed = [], [], []
    for _ in range(int(n_samples)):
        state, region = regions.sample(sets, s, rng, return_region=True)
        compute_bfc(sets, state)
        counts[region] += 1
        hb.append(float(sets.hb))
        hs.append(float(sets.hs))
        speed.append(float(np.linalg.norm(state[3:6])))
    print("---------------------------------------")
    print(f"Official sampler inspection, s={float(s):.2f}")
    print("Weights:", regions.curriculum_weights(s))
    print("Counts:", counts)
    print(f"h_B mean/min/max: {np.mean(hb):.3f}, {np.min(hb):.3f}, {np.max(hb):.3f}")
    print(f"h_S mean/min/max: {np.mean(hs):.3f}, {np.min(hs):.3f}, {np.max(hs):.3f}")
    print(f"|v| mean/min/max: {np.mean(speed):.3f}, {np.min(speed):.3f}, {np.max(speed):.3f}")
    print("---------------------------------------")