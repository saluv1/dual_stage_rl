"""Backup-control-barrier-function utilities."""

# NOTE: official_ab imports jax. Import it lazily so that pulling in
# jax-free submodules (e.g. bcbf.lqrgain for Phase-I training) does not
# require jax to be installed / numpy-2-compatible.

__all__ = ["ABConfig", "ABConstraintBuilder"]


def __getattr__(name):
    if name in ("ABConfig", "ABConstraintBuilder"):
        from .official_ab import ABConfig, ABConstraintBuilder
        return {"ABConfig": ABConfig, "ABConstraintBuilder": ABConstraintBuilder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")