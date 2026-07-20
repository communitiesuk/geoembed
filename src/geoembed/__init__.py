"""geoembed — Generate vector embeddings from geospatial imagery at scale.

Core modules (metadata, chipping, COG conversion) import without torch.
Model modules (models, backends) require torch — import them explicitly:

    from geoembed.models import DofaEmbedder, get_embedder, register_model
"""

__all__ = ["__version__"]
__version__ = "0.1.0-dev"
