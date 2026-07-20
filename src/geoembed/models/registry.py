"""Model registry — maps string names to Embedder implementations."""

from typing import Any

from geoembed.models.base import Embedder

_REGISTRY: dict[str, type[Embedder]] = {}


def register_model(name: str, cls: type[Embedder]) -> None:
    """
    Register an Embedder implementation under a given name.

    Args:
        name: Short identifier (e.g., "dofa", "clay").
        cls: The Embedder subclass to register.

    Raises:
        TypeError: If cls is not a subclass of Embedder.
    """
    if not (isinstance(cls, type) and issubclass(cls, Embedder)):
        msg = f"Expected a subclass of Embedder, got {cls}"
        raise TypeError(msg)
    _REGISTRY[name.lower()] = cls


def get_embedder(name: str, **kwargs: Any) -> Embedder:
    """
    Instantiate a registered Embedder by name.

    Args:
        name: Model name (case-insensitive).
        **kwargs: Arguments passed to the Embedder constructor.

    Returns:
        An instantiated Embedder.

    Raises:
        KeyError: If name is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = list(_REGISTRY.keys())
        msg = f"Model '{name}' not registered. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[key](**kwargs)


def _register_builtins() -> None:
    """Register built-in models on import."""
    from geoembed.models.dofa import DofaEmbedder

    register_model("dofa", DofaEmbedder)


_register_builtins()
