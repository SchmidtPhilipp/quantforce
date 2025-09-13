from typing import Optional, Tuple

import numpy as np

from qf.unsupervised.pca import PCA


def create_memory_optimized_pca(
    data_shape: Tuple[int, int],
    n_components: Optional[int] = None,
    memory_limit_mb: Optional[float] = None,
    **kwargs,
) -> PCA:
    """
    Create a memory-optimized PCA instance with automatic configuration.

    Args:
        data_shape: Shape of the data (n_samples, n_features)
        n_components: Number of components to keep
        memory_limit_mb: Memory limit in MB
        **kwargs: Additional arguments for PCA

    Returns:
        PCA instance configured for memory optimization
    """
    n_samples, n_features = data_shape

    # Auto-configure based on data size
    if n_components is None:
        n_components = min(10, min(n_samples, n_features))

    # Auto-detect memory limit if not provided
    if memory_limit_mb is None:
        try:
            import psutil

            available_memory = psutil.virtual_memory().available / 1024 / 1024
            memory_limit_mb = available_memory * 0.5  # Use 50% of available memory
        except ImportError:
            memory_limit_mb = 1000  # Default to 1GB

    # Determine if incremental PCA should be used
    estimated_memory = n_samples * n_features * 8 / 1024 / 1024  # Rough estimate
    use_incremental = estimated_memory > memory_limit_mb * 0.8

    # Auto-detect chunk size
    chunk_size = max(
        1, min(100, int(memory_limit_mb / (n_features * 8 / 1024 / 1024) / 2))
    )

    return PCA(
        n_components=n_components,
        memory_optimized=True,
        memory_limit_mb=memory_limit_mb,
        use_incremental=use_incremental,
        chunk_size=chunk_size,
        dtype=np.float32,  # Use float32 for memory efficiency
        **kwargs,
    )
