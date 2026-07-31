"""
Networks for value, Q, and policy functions.
"""

from typing import Optional, Sequence, Tuple

from acme import specs
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class ValueNetwork(hk.Module):
    """Legacy value network.

    Modern twin-target-Q SAC does not use this network, but it is retained
    so imports from other agents do not break.
    """

    def __init__(
        self,
        output_sizes: Sequence[int],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._output_sizes = output_sizes

    def __call__(
        self,
        observations: chex.Array,
    ) -> chex.Array:
        h = observations

        for output_size in self._output_sizes:
            h = hk.Linear(output_size)(h)
            h = jax.nn.relu(h)

        return hk.Linear(1)(h)[..., 0]


class SoftQNetwork(hk.Module):
    """Plain MLP critic without LayerNorm."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._output_sizes = output_sizes

    def __call__(
        self,
        observations: chex.Array,
        actions: chex.Array,
    ) -> chex.Array:
        # Concatenate along the feature dimension.
        h = jnp.concatenate(
            [observations, actions],
            axis=-1,
        )

        for output_size in self._output_sizes:
            h = hk.Linear(output_size)(h)
            h = jax.nn.relu(h)

        return hk.Linear(
            1,
            b_init=hk.initializers.Constant(0.0),
        )(h)[..., 0]


class PolicyNetwork(hk.Module):
    """Gaussian policy network.

    The second output is interpreted as log standard deviation by SAC.
    """

    def __init__(
        self,
        output_sizes: Sequence[int],
        action_spec: specs.BoundedArray,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        self._output_sizes = output_sizes
        self._action_spec = action_spec

    def __call__(
        self,
        observations: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = int(np.prod(action_shape))

        h = observations

        for output_size in self._output_sizes:
            h = hk.Linear(output_size)(h)
            h = jax.nn.relu(h)

        mu = hk.Linear(
            action_dims,
            name="mu_head",
        )(h)

        log_sigma = hk.Linear(
            action_dims,
            name="log_sigma_head",
        )(h)

        return (
            hk.Reshape(action_shape)(mu),
            hk.Reshape(action_shape)(log_sigma),
        )