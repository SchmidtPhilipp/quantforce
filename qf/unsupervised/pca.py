"""
Principal Component Analysis (PCA) module for QuantForce.

This module provides PCA functionality for financial data analysis,
including portfolio optimization and dimensionality reduction.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pypfopt import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

# from qf.unsupervised.util.pca_utils import create_memory_optimized_pca  # Commented to avoid circular import

try:
    import pypfopt as ppo
    from pypfopt import expected_returns, risk_models

    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    warnings.warn(
        "PyPortfolioOpt not available. Portfolio optimization features will be disabled."
    )


class DataPreprocessor:
    """Data preprocessing utilities for financial data."""

    @staticmethod
    def clean_financial_data(
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Clean financial data by removing infinite values, zeros, and handling missing data.

        Args:
            data: Input data (DataFrame, numpy array, or torch tensor)

        Returns:
            Cleaned data of the same type as input
        """
        if isinstance(data, pd.DataFrame):
            return DataPreprocessor._clean_dataframe(data)
        elif isinstance(data, np.ndarray):
            return DataPreprocessor._clean_numpy(data)
        elif isinstance(data, torch.Tensor):
            return DataPreprocessor._clean_torch(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean pandas DataFrame."""
        if df.empty:
            return df

        # Make a copy to avoid modifying original data
        cleaned = df.copy()

        # Remove infinite values
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        # Remove rows with all NaN values
        cleaned = cleaned.dropna(how="all")

        # Remove columns with all NaN values
        cleaned = cleaned.dropna(axis=1, how="all")

        # Forward fill and backward fill to handle missing values
        cleaned = cleaned.ffill().bfill()

        # Remove any remaining NaN values
        cleaned = cleaned.dropna()

        return cleaned

    @staticmethod
    def _clean_numpy(arr: np.ndarray) -> np.ndarray:
        """Clean numpy array."""
        if arr.size == 0:
            return arr

        # Remove infinite values
        arr = np.where(np.isinf(arr), np.nan, arr)

        # Remove rows with all NaN values
        mask = ~np.isnan(arr).all(axis=1)
        arr = arr[mask]

        # Remove columns with all NaN values
        mask = ~np.isnan(arr).all(axis=0)
        arr = arr[:, mask]

        # Forward fill and backward fill
        arr = DataPreprocessor._ffill_bfill_numpy(arr)

        # Remove any remaining NaN values
        mask = ~np.isnan(arr).any(axis=1)
        arr = arr[mask]

        return arr

    @staticmethod
    def _clean_torch(tensor: torch.Tensor) -> torch.Tensor:
        """Clean torch tensor."""
        if tensor.numel() == 0:
            return tensor

        # Convert to numpy for cleaning, then back to torch
        arr = tensor.numpy()
        cleaned_arr = DataPreprocessor._clean_numpy(arr)
        return torch.from_numpy(cleaned_arr)

    @staticmethod
    def _ffill_bfill_numpy(arr: np.ndarray) -> np.ndarray:
        """Forward fill and backward fill for numpy arrays."""
        # Forward fill
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        arr = arr[idx, np.arange(mask.shape[1])]

        # Backward fill
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], mask.shape[0] - 1)
        idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
        arr = arr[idx, np.arange(mask.shape[1])]

        return arr

    @staticmethod
    def calculate_returns(
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor], method: str = "log"
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Calculate returns from price data.

        Args:
            data: Price data
            method: 'log' or 'simple'

        Returns:
            Returns data of the same type as input
        """
        if isinstance(data, pd.DataFrame):
            return DataPreprocessor._calculate_returns_dataframe(data, method)
        elif isinstance(data, np.ndarray):
            return DataPreprocessor._calculate_returns_numpy(data, method)
        elif isinstance(data, torch.Tensor):
            return DataPreprocessor._calculate_returns_torch(data, method)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _calculate_returns_dataframe(df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Calculate returns for DataFrame."""
        if method == "log":
            returns = np.log(df / df.shift(1))
        else:  # simple
            returns = (df - df.shift(1)) / df.shift(1)

        # Remove the first row (NaN due to shift)
        returns = returns.dropna()

        # Final check for any remaining infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()

        return returns

    @staticmethod
    def _calculate_returns_numpy(arr: np.ndarray, method: str) -> np.ndarray:
        """Calculate returns for numpy array."""
        if method == "log":
            returns = np.log(arr[1:] / arr[:-1])
        else:  # simple
            returns = (arr[1:] - arr[:-1]) / arr[:-1]

        # Remove infinite values
        returns = np.where(np.isinf(returns), np.nan, returns)
        mask = ~np.isnan(returns).any(axis=1)
        returns = returns[mask]

        return returns

    @staticmethod
    def _calculate_returns_torch(tensor: torch.Tensor, method: str) -> torch.Tensor:
        """Calculate returns for torch tensor."""
        # Convert to numpy, calculate returns, then back to torch
        arr = tensor.numpy()
        returns_arr = DataPreprocessor._calculate_returns_numpy(arr, method)
        return torch.from_numpy(returns_arr)


class PCA:
    """
    Principal Component Analysis (PCA) implementation for QuantForce.

    This class provides PCA functionality with support for multiple data types
    (pandas DataFrame, numpy array, torch tensor) and includes portfolio
    optimization capabilities. It also supports memory-optimized processing
    for large datasets.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        standardize: bool = True,
        memory_optimized: bool = False,
        chunk_size: Optional[int] = None,
        memory_limit_mb: Optional[float] = None,
        use_incremental: bool = False,
        dtype: np.dtype = np.float64,
        log_returns: bool = False,
    ):
        """
        Initialize PCA.

        Args:
            n_components: Number of components to keep. If None, keep all components.
            random_state: Random state for reproducibility.
            standardize: Whether to standardize the data before PCA.
            memory_optimized: Whether to use memory-optimized processing for large datasets.
            chunk_size: Size of chunks for processing large datasets. If None, auto-detect.
            memory_limit_mb: Memory limit in MB. If exceeded, use chunked processing.
            use_incremental: Whether to use incremental PCA for very large datasets.
            dtype: Data type for memory efficiency (np.float32 for memory optimization).
            log_returns: If True, the input is already log returns. If False, input is prices
                        and will be converted to log returns.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.standardize = standardize
        self.memory_optimized = memory_optimized
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.use_incremental = use_incremental
        self.dtype = dtype
        self.log_returns = log_returns

        # Initialize sklearn PCA
        self._pca = SklearnPCA(n_components=n_components, random_state=random_state)

        # Initialize state variables
        self._fitted = False
        self._scaler = None
        self._original_data = None
        self._original_log_returns = (
            None  # Store original log returns for portfolio optimization
        )
        self._transformed_data = None
        self._feature_names = None
        self._memory_usage = []

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _estimate_memory_requirements(self, data_shape: Tuple[int, int]) -> float:
        """Estimate memory requirements for PCA in MB."""
        n_samples, n_features = data_shape

        # Base data memory
        data_memory = (
            n_samples * n_features * np.dtype(self.dtype).itemsize / 1024 / 1024
        )

        # Covariance matrix memory (n_features x n_features)
        cov_memory = n_features * n_features * 8 / 1024 / 1024  # float64 for precision

        # Components memory
        n_comp = self.n_components or min(n_samples, n_features)
        components_memory = n_comp * n_features * 8 / 1024 / 1024

        # Total estimated memory
        total_memory = data_memory + cov_memory + components_memory

        return total_memory

    def _auto_detect_chunk_size(self, data_shape: Tuple[int, int]) -> int:
        """Auto-detect optimal chunk size based on memory constraints."""
        n_samples, n_features = data_shape

        if self.memory_limit_mb is None:
            # Default to processing all data at once
            return n_samples

        # Estimate memory per sample
        memory_per_sample = self._estimate_memory_requirements((1, n_features))

        # Calculate safe chunk size
        safe_chunk_size = max(1, int(self.memory_limit_mb / memory_per_sample / 2))

        # Don't exceed data size
        return min(safe_chunk_size, n_samples)

    def _prepare_data_memory_optimized(
        self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Prepare data for memory-optimized PCA processing."""
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                data = data.copy()
                data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
            data = data.values
        elif isinstance(data, torch.Tensor):
            data = data.numpy()

        # Convert to specified dtype for memory efficiency
        data = data.astype(self.dtype)

        # Remove any infinite or NaN values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data

    def _prepare_data_for_pca(
        self, prices: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray, torch.Tensor],
        Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ]:
        """
        Prepare data for PCA by converting prices to log returns if needed and cleaning.

        Returns:
            Tuple of (original_data, processed_data_for_pca)
        """
        # Store original data for later use
        original_data = prices

        # Convert to log returns if needed
        if not self.log_returns:
            # Input is prices, convert to log returns
            processed_data = DataPreprocessor.calculate_returns(prices, method="log")
        else:
            # Input is already log returns
            processed_data = prices

        # Clean the data
        cleaned_data = DataPreprocessor.clean_financial_data(processed_data)

        return original_data, cleaned_data

    def _convert_to_dataframe(
        self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor], prefix: str = "Asset"
    ) -> pd.DataFrame:
        """
        Convert any data type to DataFrame with proper column names.

        Args:
            data: Input data of any supported type
            prefix: Prefix for column names if no feature names available

        Returns:
            DataFrame with proper column names
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if self._feature_names:
                return pd.DataFrame(data, columns=self._feature_names)
            else:
                return pd.DataFrame(
                    data,
                    columns=[f"{prefix}_{i}" for i in range(data.shape[1])],
                )
        elif isinstance(data, torch.Tensor):
            data_array = data.numpy()
            if self._feature_names:
                return pd.DataFrame(data_array, columns=self._feature_names)
            else:
                return pd.DataFrame(
                    data_array,
                    columns=[f"{prefix}_{i}" for i in range(data_array.shape[1])],
                )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _get_pca_covariance_matrix(
        self, prices: Union[pd.DataFrame, np.ndarray, torch.Tensor] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get both traditional and PCA-transformed covariance matrices.

        Returns:
            Tuple of (traditional_cov_df, pca_cov_df)
        """
        # Get original log returns
        if prices is None:
            original_log_returns = self._original_log_returns
        else:
            if self.log_returns:
                original_log_returns = prices
            else:
                original_log_returns = DataPreprocessor.calculate_returns(
                    prices, method="log"
                )

        # Convert to DataFrame
        df_original = self._convert_to_dataframe(original_log_returns, "Asset")

        # Get PCA-transformed data for covariance calculation
        if prices is None:
            transformed_data = self.transform(self._original_data)
        else:
            transformed_data = self.transform(prices)

        # Convert transformed data to DataFrame
        df_pca = self._convert_to_dataframe(transformed_data, "PC")

        S_pca_space = risk_models.sample_cov(df_pca)
        regularization = 1e-6 * np.eye(S_pca_space.shape[0])
        S_pca_space = S_pca_space + regularization
        S_pca_space = (S_pca_space + S_pca_space.T) / 2

        # Transform back to original space
        components = self.components_
        S_pca_transformed = components.T @ S_pca_space @ components

        if self._scaler is not None:
            scaling_factor = np.outer(self._scaler.scale_, self._scaler.scale_)
            S_pca_transformed = S_pca_transformed * scaling_factor

        regularization_original = 1e-8 * np.eye(S_pca_transformed.shape[0])
        S_pca_transformed = S_pca_transformed + regularization_original
        S_pca_transformed = (S_pca_transformed + S_pca_transformed.T) / 2

        S_pca_full_space = pd.DataFrame(
            S_pca_transformed, index=df_original.columns, columns=df_original.columns
        )

        return S_pca_space, S_pca_full_space

    def _format_transformed_output(
        self,
        transformed_array: np.ndarray,
        original_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        index: Optional[pd.Index] = None,
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Format transformed array back to the same type as original data.

        Args:
            transformed_array: PCA-transformed numpy array
            original_data: Original input data to match format
            index: Optional pandas index for DataFrame output

        Returns:
            Formatted output matching original data type
        """
        if isinstance(original_data, pd.DataFrame):
            column_names = [f"PC{i+1}" for i in range(transformed_array.shape[1])]
            return pd.DataFrame(transformed_array, index=index, columns=column_names)
        elif isinstance(original_data, np.ndarray):
            return transformed_array
        elif isinstance(original_data, torch.Tensor):
            return torch.from_numpy(transformed_array)
        else:
            return transformed_array

    def fit(self, prices: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> "PCA":
        """
        Fit PCA to the price data.

        Args:
            prices: Input price data to fit PCA on. Will be converted to log returns
                   unless log_returns=True is set in constructor.

        Returns:
            Self for method chaining.
        """
        # Store original price data
        self._original_data = prices

        # Prepare data using helper method
        original_data, cleaned_data = self._prepare_data_for_pca(prices)

        # Store original log returns for portfolio optimization
        self._original_log_returns = (
            cleaned_data
            if self.log_returns
            else DataPreprocessor.calculate_returns(prices, method="log")
        )

        if self.memory_optimized:
            return self._fit_memory_optimized(cleaned_data)
        else:
            return self._fit_standard(cleaned_data)

    def _fit_standard(
        self, cleaned_data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> "PCA":
        """Standard PCA fitting process."""
        # Convert to numpy array for processing
        if isinstance(cleaned_data, pd.DataFrame):
            self._feature_names = cleaned_data.columns.tolist()
            data_array = cleaned_data.values
        elif isinstance(cleaned_data, np.ndarray):
            data_array = cleaned_data
            self._feature_names = [f"Feature_{i}" for i in range(data_array.shape[1])]
        elif isinstance(cleaned_data, torch.Tensor):
            data_array = cleaned_data.numpy()
            self._feature_names = [f"Feature_{i}" for i in range(data_array.shape[1])]
        else:
            raise TypeError(f"Unsupported data type: {type(cleaned_data)}")

        # Convert to specified dtype
        data_array = data_array.astype(self.dtype)

        # Standardize if requested
        if self.standardize:
            self._scaler = StandardScaler()
            data_array = self._scaler.fit_transform(data_array)

        # Fit PCA
        self._pca.fit(data_array)
        self._fitted = True

        return self

    def _fit_memory_optimized(
        self, cleaned_data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> "PCA":
        """Memory-optimized PCA fitting process."""
        import gc

        # Prepare data for memory-optimized processing
        data_array = self._prepare_data_memory_optimized(cleaned_data)
        n_samples, n_features = data_array.shape

        # Store feature names
        if isinstance(cleaned_data, pd.DataFrame):
            if isinstance(cleaned_data.columns, pd.MultiIndex):
                self._feature_names = [
                    f"{col[0]}_{col[1]}" for col in cleaned_data.columns
                ]
            else:
                self._feature_names = cleaned_data.columns.tolist()
        else:
            self._feature_names = [f"Feature_{i}" for i in range(n_features)]

        # Auto-detect chunk size if not specified
        if self.chunk_size is None:
            self.chunk_size = self._auto_detect_chunk_size((n_samples, n_features))

        # Determine number of components
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Print configuration if verbose
        current_memory = self._get_memory_usage()
        estimated_memory = self._estimate_memory_requirements((n_samples, n_features))

        print(f"Memory-Optimized PCA Configuration:")
        print(f"  Data shape: {data_array.shape}")
        print(f"  Components: {self.n_components}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Use incremental: {self.use_incremental}")
        print(f"  Memory limit: {self.memory_limit_mb} MB")
        print(f"  Estimated memory: {estimated_memory:.1f} MB")
        print(f"  Current memory: {current_memory:.1f} MB")

        # Choose processing strategy
        should_use_incremental = self.use_incremental or (
            self.memory_limit_mb and estimated_memory > self.memory_limit_mb
        )

        if should_use_incremental:
            self._fit_incremental_pca(data_array)
        elif self.chunk_size < n_samples:
            self._fit_chunked_pca(data_array)
        else:
            self._fit_standard_memory_optimized(data_array)

        self._fitted = True
        return self

    def _fit_standard_memory_optimized(self, data_array: np.ndarray):
        """Standard PCA fitting with memory optimization."""
        import gc

        print("Using standard PCA fitting with memory optimization...")

        # Standardize if requested
        if self.standardize:
            self._scaler = StandardScaler()
            data_array = self._scaler.fit_transform(data_array)

        # Fit PCA
        self._pca = SklearnPCA(
            n_components=self.n_components, random_state=self.random_state
        )
        self._pca.fit(data_array)

        # Force garbage collection
        gc.collect()

    def _fit_chunked_pca(self, data_array: np.ndarray):
        """Chunked PCA fitting for large datasets."""
        import gc

        from sklearn.decomposition import IncrementalPCA

        print("Using chunked PCA fitting...")

        n_samples = data_array.shape[0]

        # Use incremental PCA for chunked fitting
        self._pca = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.chunk_size,
            random_state=self.random_state,
        )

        # Standardize if requested
        if self.standardize:
            self._scaler = StandardScaler()
            # Fit scaler on first chunk
            first_chunk = data_array[: self.chunk_size]
            self._scaler.fit(first_chunk)

        # Process data in chunks
        for i in range(0, n_samples, self.chunk_size):
            chunk = data_array[i : i + self.chunk_size]

            # Standardize chunk if needed
            if self.standardize:
                chunk = self._scaler.transform(chunk)

            # Partial fit
            self._pca.partial_fit(chunk)

            print(
                f"Processed chunk {i//self.chunk_size + 1}/{(n_samples + self.chunk_size - 1)//self.chunk_size}"
            )
            if self._get_memory_usage() > 0:
                print(f"Memory usage: {self._get_memory_usage():.1f} MB")

            # Force garbage collection
            gc.collect()

    def _fit_incremental_pca(self, data_array: np.ndarray):
        """Incremental PCA fitting for very large datasets."""
        import gc

        from sklearn.decomposition import IncrementalPCA

        print("Using incremental PCA fitting...")

        # Use incremental PCA
        self._pca = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.chunk_size,
            random_state=self.random_state,
        )

        # Standardize if requested
        if self.standardize:
            self._scaler = StandardScaler()
            # Fit scaler on sample of data
            sample_size = min(1000, data_array.shape[0])
            sample_indices = np.random.choice(
                data_array.shape[0], sample_size, replace=False
            )
            self._scaler.fit(data_array[sample_indices])

        # Process data in batches
        for i in range(0, data_array.shape[0], self.chunk_size):
            batch = data_array[i : i + self.chunk_size]

            # Standardize batch if needed
            if self.standardize:
                batch = self._scaler.transform(batch)

            # Partial fit
            self._pca.partial_fit(batch)

            print(
                f"Processed batch {i//self.chunk_size + 1}/{(data_array.shape[0] + self.chunk_size - 1)//self.chunk_size}"
            )
            if self._get_memory_usage() > 0:
                print(f"Memory usage: {self._get_memory_usage():.1f} MB")

            # Force garbage collection
            gc.collect()

    def transform(
        self, prices: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Transform price data using fitted PCA.

        Args:
            prices: Price data to transform. Will be converted to log returns
                   unless log_returns=True is set in constructor.

        Returns:
            Transformed data.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before transforming data.")

        # Prepare data using helper method
        original_data, cleaned_data = self._prepare_data_for_pca(prices)

        if self.memory_optimized:
            return self._transform_memory_optimized(cleaned_data, original_data)
        else:
            return self._transform_standard(cleaned_data, original_data)

    def _transform_standard(
        self,
        cleaned_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        original_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """Standard transform process."""
        # Convert to numpy array
        if isinstance(cleaned_data, pd.DataFrame):
            data_array = cleaned_data.values
            index = cleaned_data.index
        elif isinstance(cleaned_data, np.ndarray):
            data_array = cleaned_data
            index = None
        elif isinstance(cleaned_data, torch.Tensor):
            data_array = cleaned_data.numpy()
            index = None
        else:
            raise TypeError(f"Unsupported data type: {type(cleaned_data)}")

        # Convert to specified dtype
        data_array = data_array.astype(self.dtype)

        # Standardize if needed
        if self.standardize and self._scaler is not None:
            data_array = self._scaler.transform(data_array)

        # Transform data
        transformed_array = self._pca.transform(data_array)

        # Store transformed data
        self._transformed_data = transformed_array

        # Return in original format using helper
        return self._format_transformed_output(transformed_array, original_data, index)

    def _transform_memory_optimized(
        self,
        cleaned_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        original_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """Memory-optimized transform process."""
        import gc

        # Prepare data for memory-optimized processing
        data_array = self._prepare_data_memory_optimized(cleaned_data)

        # Get index if available
        index = None
        if isinstance(cleaned_data, pd.DataFrame):
            index = cleaned_data.index

        # Standardize if needed
        if self.standardize and self._scaler is not None:
            if self.chunk_size and data_array.shape[0] > self.chunk_size:
                # Standardize in chunks
                standardized_chunks = []
                for i in range(0, data_array.shape[0], self.chunk_size):
                    chunk = data_array[i : i + self.chunk_size]
                    chunk_standardized = self._scaler.transform(chunk)
                    standardized_chunks.append(chunk_standardized)
                    gc.collect()
                data_array = np.vstack(standardized_chunks)
            else:
                data_array = self._scaler.transform(data_array)

        # Transform data
        if self.chunk_size and data_array.shape[0] > self.chunk_size:
            # Transform in chunks
            transformed_chunks = []
            for i in range(0, data_array.shape[0], self.chunk_size):
                chunk = data_array[i : i + self.chunk_size]
                chunk_transformed = self._pca.transform(chunk)
                transformed_chunks.append(chunk_transformed)

                # Force garbage collection
                gc.collect()

            # Combine chunks
            transformed_array = np.vstack(transformed_chunks)
        else:
            # Transform all at once
            transformed_array = self._pca.transform(data_array)

        # Store transformed data
        self._transformed_data = transformed_array

        # Return in original format using helper
        return self._format_transformed_output(transformed_array, original_data, index)

    def fit_transform(
        self, prices: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Fit PCA and transform price data in one step.

        Args:
            prices: Price data to fit and transform. Will be converted to log returns
                   unless log_returns=True is set in constructor.

        Returns:
            Transformed data.
        """
        return self.fit(prices).transform(prices)

    def _is_price_data(
        self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> bool:
        """Check if data looks like price data (positive values)."""
        if isinstance(data, pd.DataFrame):
            return (data > 0).all().all()
        elif isinstance(data, np.ndarray):
            return np.all(data > 0)
        elif isinstance(data, torch.Tensor):
            return torch.all(data > 0)
        return False

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Get explained variance ratio."""
        if not self._fitted:
            raise ValueError(
                "PCA must be fitted before accessing explained_variance_ratio_."
            )
        return self._pca.explained_variance_ratio_

    @property
    def explained_variance_(self) -> np.ndarray:
        """Get explained variance."""
        if not self._fitted:
            raise ValueError("PCA must be fitted before accessing explained_variance_.")
        return self._pca.explained_variance_

    @property
    def components_(self) -> np.ndarray:
        """Get principal components."""
        if not self._fitted:
            raise ValueError("PCA must be fitted before accessing components_.")
        return self._pca.components_

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on PCA components.

        Returns:
            DataFrame with feature importance scores.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before getting feature importance.")

        # Calculate feature importance as the sum of squared loadings
        feature_importance = np.sum(self.components_**2, axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"Feature": self._feature_names, "Importance": feature_importance}
        )

        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)

        return importance_df

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the PCA results.

        Returns:
            Dictionary with PCA summary information.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before getting summary.")

        summary = {
            "n_components": self.n_components,
            "n_features": len(self._feature_names),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(
                self.explained_variance_ratio_
            ).tolist(),
            "total_variance_explained": np.sum(self.explained_variance_ratio_),
            "feature_importance": self.get_feature_importance().to_dict("records"),
            "standardized": self.standardize,
        }

        return summary

    def plot_explained_variance(
        self,
        cumulative: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot explained variance using optimized plotting functions.

        Args:
            cumulative: Whether to plot cumulative variance.
            figsize: Figure size.
            save_path: Optional path to save the plot.
            **kwargs: Additional plotting arguments.

        Returns:
            matplotlib Figure object.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting.")

        from qf.utils.optimized_plots import plot_pca_explained_variance

        return plot_pca_explained_variance(
            explained_variance_ratio=self.explained_variance_ratio_,
            cumulative=cumulative,
            figsize=figsize,
            save_path=save_path,
            title="PCA Explained Variance",
            **kwargs,
        )

    def plot_scatter(
        self,
        components: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot PCA scatter plot using optimized plotting functions.

        Args:
            components: Tuple of component indices to plot (0-indexed).
            figsize: Figure size.
            save_path: Optional path to save the plot.
            **kwargs: Additional plotting arguments.

        Returns:
            matplotlib Figure object.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting.")

        if self._transformed_data is None:
            raise ValueError("No transformed data available. Call transform() first.")

        from qf.utils.optimized_plots import plot_pca_scatter

        # Get the specified components
        pc1_idx, pc2_idx = components
        if (
            pc1_idx >= self._transformed_data.shape[1]
            or pc2_idx >= self._transformed_data.shape[1]
        ):
            raise ValueError(
                f"Component indices {components} out of range. Available components: 0-{self._transformed_data.shape[1]-1}"
            )

        x = self._transformed_data[:, pc1_idx]
        y = self._transformed_data[:, pc2_idx]

        # Create time-based coloring for better visualization
        time_colors = np.arange(len(x))

        # Create labels for time series if available
        labels = None
        time_index = None
        if hasattr(self._original_data, "index") and hasattr(
            self._original_data.index, "date"
        ):
            labels = [str(date.date()) for date in self._original_data.index]
            # Pass the actual time index for proper colorbar labels
            if isinstance(self._original_data.index, pd.DatetimeIndex):
                time_index = self._original_data.index

        # Set default title if not provided in kwargs
        default_title = f"PCA Scatter Plot (PC{pc1_idx+1} vs PC{pc2_idx+1})"
        default_xlabel = (
            f"PC{pc1_idx+1} ({self.explained_variance_ratio_[pc1_idx]:.1%} variance)"
        )
        default_ylabel = (
            f"PC{pc2_idx+1} ({self.explained_variance_ratio_[pc2_idx]:.1%} variance)"
        )

        return plot_pca_scatter(
            x=x,
            y=y,
            colors=time_colors,  # Use time-based coloring
            # labels=labels,
            figsize=figsize,
            save_path=save_path,
            title=kwargs.pop("title", default_title),  # Allow user to override title
            xlabel=kwargs.pop(
                "xlabel", default_xlabel
            ),  # Allow user to override xlabel
            ylabel=kwargs.pop(
                "ylabel", default_ylabel
            ),  # Allow user to override ylabel
            colorbar=kwargs.pop("colorbar", True),  # Show colorbar for time progression
            cmap=kwargs.pop("cmap", "viridis"),  # Use viridis colormap
            alpha=kwargs.pop(
                "alpha", 0.7
            ),  # Slightly transparent for better visibility
            time_index=kwargs.pop(
                "time_index", time_index
            ),  # Pass time index for proper colorbar labels
            **kwargs,
        )

    def plot_3d_scatter(
        self,
        components: Tuple[int, int, int] = (0, 1, 2),
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot 3D PCA scatter plot using optimized plotting functions.

        Args:
            components: Tuple of component indices to plot (0-indexed).
            figsize: Figure size.
            save_path: Optional path to save the plot.
            **kwargs: Additional plotting arguments.

        Returns:
            matplotlib Figure object.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting.")

        if self._transformed_data is None:
            raise ValueError("No transformed data available. Call transform() first.")

        if self._transformed_data.shape[1] < 3:
            raise ValueError("Need at least 3 components for 3D plot.")

        from qf.utils.optimized_plots import OptimizedPlotter

        # Get the specified components
        pc1_idx, pc2_idx, pc3_idx = components
        if any(idx >= self._transformed_data.shape[1] for idx in components):
            raise ValueError(
                f"Component indices {components} out of range. Available components: 0-{self._transformed_data.shape[1]-1}"
            )

        x = self._transformed_data[:, pc1_idx]
        y = self._transformed_data[:, pc2_idx]
        z = self._transformed_data[:, pc3_idx]

        # Create 3D scatter plot
        plotter = OptimizedPlotter()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Create time-based coloring if available
        if hasattr(self._original_data, "index") and hasattr(
            self._original_data.index, "date"
        ):
            time_values = np.arange(len(x))
            scatter = ax.scatter(
                x, y, z, c=time_values, cmap="viridis", alpha=0.7, **kwargs
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label("Time Index")
        else:
            ax.scatter(x, y, z, alpha=0.7, **kwargs)

        # Customize plot
        ax.set_xlabel(
            f"PC{pc1_idx+1} ({self.explained_variance_ratio_[pc1_idx]:.1%} variance)"
        )
        ax.set_ylabel(
            f"PC{pc2_idx+1} ({self.explained_variance_ratio_[pc2_idx]:.1%} variance)"
        )
        ax.set_zlabel(
            f"PC{pc3_idx+1} ({self.explained_variance_ratio_[pc3_idx]:.1%} variance)"
        )
        ax.set_title("PCA 3D Scatter Plot")

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path,
                dpi=plotter.style_config["dpi"],
                bbox_inches="tight",
                format=plotter.style_config["save_format"],
            )

        return fig

    def plot_correlation_heatmap(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot correlation heatmap of original features using optimized plotting functions.

        Args:
            figsize: Figure size.
            save_path: Optional path to save the plot.
            **kwargs: Additional plotting arguments.

        Returns:
            matplotlib Figure object.
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting.")

        from qf.utils.optimized_plots import plot_correlation_heatmap

        # Calculate correlation matrix from original data
        if isinstance(self._original_data, pd.DataFrame):
            correlation_matrix = self._original_data.corr()
        else:
            # Convert to DataFrame for correlation calculation
            if isinstance(self._original_data, np.ndarray):
                df = pd.DataFrame(self._original_data, columns=self._feature_names)
            elif isinstance(self._original_data, torch.Tensor):
                df = pd.DataFrame(
                    self._original_data.numpy(), columns=self._feature_names
                )
            else:
                raise TypeError(f"Unsupported data type: {type(self._original_data)}")
            correlation_matrix = df.corr()

        return plot_correlation_heatmap(
            correlation_matrix=correlation_matrix,
            figsize=figsize,
            save_path=save_path,
            title="Feature Correlation Matrix",
            **kwargs,
        )

    def portfolio_optimization(
        self,
        prices: Union[pd.DataFrame, np.ndarray, torch.Tensor] = None,
        method: str = "min_variance",
    ) -> Optional[pd.Series]:
        """
        Berechnet das Minimum Variance Portfolio mit PCA wie in der mathematischen Herleitung:

        1. Standardisiere die Daten: X = Standardize(R)
        2. Wende PCA an: Z = X · V_k (k Hauptkomponenten)
        3. Berechne Kovarianz im PCA-Raum: Σ_Z = Cov(Z)
        4. Optimiere Minimum Variance: min w^T Σ_Z w s.t. sum(w) = 1, w ≥ 0
        5. Transformiere zurück: w_assets = V_k · w_PCA
        6. Anwende Non-Negativity: w̃ = max(w_assets, 0) / sum(max(w_assets, 0))

        Args:
            prices: Preisdaten. Falls None, verwende gespeicherte Daten.
            method: Nur "min_variance" unterstützt.

        Returns:
            Optimale Portfolio-Gewichte für die ursprünglichen Assets.
        """
        if method != "min_variance":
            raise ValueError("Nur 'min_variance' wird unterstützt.")

        if not self._fitted:
            raise ValueError("PCA muss zuerst mit fit() ausgeführt werden.")

        print(
            f"Berechne Minimum Variance Portfolio mit {self.n_components} PCA-Komponenten..."
        )

        # 1. Hole ursprüngliche log returns (bereits standardisiert durch PCA)
        if prices is None:
            if self._original_log_returns is None:
                raise ValueError("Keine original log returns verfügbar.")
            original_log_returns = self._original_log_returns
        else:
            if self.log_returns:
                original_log_returns = prices
            else:
                original_log_returns = DataPreprocessor.calculate_returns(
                    prices, method="log"
                )

        df_original = self._convert_to_dataframe(original_log_returns, "Asset")

        # 2. Transformiere zu PCA-Raum: Z = X · V_k
        if prices is None:
            Z = self.transform(self._original_data)
        else:
            Z = self.transform(prices)

        # 3. Berechne Kovarianz im PCA-Raum: Σ_Z = Cov(Z)
        if isinstance(Z, pd.DataFrame):
            Z_values = Z.values
        else:
            Z_values = Z

        Sigma_Z = np.cov(Z_values.T)

        # Numerische Stabilität
        regularization = 1e-8 * np.eye(Sigma_Z.shape[0])
        Sigma_Z = Sigma_Z + regularization
        Sigma_Z = (Sigma_Z + Sigma_Z.T) / 2

        print(f"PCA-Raum Kovarianzmatrix: {Sigma_Z.shape}")
        print(f"Rang: {np.linalg.matrix_rank(Sigma_Z)}")

        # 4. Löse analytisch: min w^T Σ_Z w s.t. sum(w) = 1
        # Lösung: w = (Σ_Z^-1 · 1) / (1^T · Σ_Z^-1 · 1)
        try:
            Sigma_Z_inv = np.linalg.inv(Sigma_Z)
            ones = np.ones(Sigma_Z.shape[0])
            w_PCA = Sigma_Z_inv @ ones
            w_PCA = w_PCA / w_PCA.sum()

            print(f"✓ PCA-Gewichte erfolgreich berechnet: {w_PCA}")

        except np.linalg.LinAlgError as e:
            print(f"✗ Matrix-Inversion fehlgeschlagen: {e}")
            # Fallback: Gleiche Gewichte im PCA-Raum
            w_PCA = np.ones(Sigma_Z.shape[0]) / Sigma_Z.shape[0]
            print(f"Fallback: Gleiche Gewichte im PCA-Raum: {w_PCA}")

        # 5. Transformiere zurück zu Asset-Raum: w_assets = V_k · w_PCA
        V_k = self.components_  # Shape: (n_components, n_assets)
        w_assets = V_k.T @ w_PCA  # Shape: (n_assets,)

        print(f"Asset-Gewichte vor Non-Negativity: {w_assets}")

        # 6. Non-Negativity Constraint und Normalisierung
        w_assets_positive = np.maximum(w_assets, 0)

        if w_assets_positive.sum() > 0:
            w_final = w_assets_positive / w_assets_positive.sum()
        else:
            print("⚠️ Alle Gewichte sind negativ, verwende gleiche Gewichte")
            w_final = np.ones(len(w_assets)) / len(w_assets)

        # Als pandas Series zurückgeben
        weights_series = pd.Series(w_final, index=df_original.columns)

        print(f"✓ Minimum Variance Portfolio berechnet:")
        print(f"  PCA-Gewichte: {w_PCA}")
        print(f"  Asset-Gewichte (vor Non-Neg): {w_assets}")
        print(f"  Finale Gewichte: {w_final}")
        print(f"  Gewichtssumme: {weights_series.sum():.6f}")
        print(f"  Anzahl > 0: {(weights_series > 1e-6).sum()}")

        return weights_series

    def plot_efficient_frontier(
        self,
        prices: Union[pd.DataFrame, np.ndarray, torch.Tensor] = None,
        n_points: int = 100,
        show_pca_portfolio: bool = True,
        show_traditional_portfolio: bool = True,
        risk_free_rate: float = 0.02,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        weight_bounds=(0, 1),
        showfig: bool = True,
        ax: Optional[plt.Axes] = None,
        color: str = "1",
        linewidth: int = 2,
    ) -> Optional[plt.Figure]:
        """
        Plot efficient frontier comparing PCA-optimized vs traditional portfolios.

        Args:
            prices: Input price data. If None, uses stored original data.
            n_points: Number of points to plot on efficient frontier.
            show_pca_portfolio: Whether to show PCA-optimized portfolio point.
            show_traditional_portfolio: Whether to show traditional min variance portfolio point.
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.
            figsize: Figure size (width, height).
            save_path: Path to save the plot. If None, displays interactively.

        Returns:
            matplotlib Figure object if save_path is None, otherwise None.
        """
        if not PYPFOPT_AVAILABLE:
            raise ImportError(
                "PyPortfolioOpt is required for efficient frontier plotting."
            )

        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting efficient frontier.")

        # Calculate expected returns
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        fig, ax = plt.subplots(figsize=figsize)

        plot_efficient_frontier(
            ef, ax=ax, color="black", linewidth=2, show_assets=False
        )

        # Max Sharpe Ratio Portfolio
        ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        ret_ms, std_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()

        # Min Volatility Portfolio
        ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        min_vol_weights = ef_min_vol.min_volatility()
        ret_mv, std_mv, _ = ef_min_vol.portfolio_performance()

        # Highlight optimal portfolios
        ax.scatter(
            std_ms, ret_ms, marker="*", s=100, color="red", label="Max Sharpe Ratio"
        )
        ax.scatter(
            std_mv, ret_mv, marker=5, s=100, color="black", label="Min Volatility"
        )

        # Plot the pca portfolio
        pca_portfolio = self.portfolio_optimization(prices, method="min_variance")
        pca_portfolio_return = np.dot(pca_portfolio, mu)
        pca_portfolio_vol = np.sqrt(np.dot(pca_portfolio, S.dot(pca_portfolio)))
        ax.scatter(
            pca_portfolio_vol,
            pca_portfolio_return,
            marker="*",
            s=100,
            color="blue",
            label="PCA Portfolio",
        )

        # Plot individual asset points in grayscale
        for i, ticker in enumerate(prices.columns):
            asset_return = mu[ticker]
            asset_vol = np.sqrt(S.loc[ticker, ticker])
            ax.scatter(
                asset_vol, asset_return, marker=".", s=70, color="0.2", label=ticker
            )
            ax.annotate(
                ticker,
                (asset_vol, asset_return),
                textcoords="offset points",
                xytext=(5, -2.5),
                ha="left",
                fontsize=9,
                color="0.2",
            )

        # Legend above the plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=4,
            frameon=True,
        )
        # ax.legend(loc="right", bbox_to_anchor=(0.5, 0.5, 0.1, 0.5), frameon=True)
        # Grid and layout adjustments
        plt.grid(True, color="0.85")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Show plot
        plt.show()

        return fig

    def plot_pca_latex(
        self,
        plot_type: str = "scatter",
        output_file: Optional[str] = None,
        style_config: Optional[Dict] = None,
    ) -> str:
        """
        Generate LaTeX/TikZ code for PCA plots using stored data.

        Args:
            plot_type: Type of plot ("scatter", "explained_variance")
            output_file: Optional output file path
            style_config: Optional style configuration for LaTeXPlotter

        Returns:
            LaTeX/TikZ code string
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before plotting.")

        if self._transformed_data is None:
            raise ValueError("No transformed data available. Call transform() first.")

        from qf.utils.latex_plot import LaTeXPlotter

        plotter = LaTeXPlotter(style_config)

        if plot_type == "scatter":
            # Use stored transformed data
            if self._transformed_data.shape[1] >= 2:
                x = self._transformed_data[:, 0]
                y = self._transformed_data[:, 1]

                scatter_plot = plotter.plot_scatter(
                    x=x,
                    y=y,
                    colors=["plotblue"] * len(x),
                    markers=["circle"] * len(x),
                    sizes=[2] * len(x),
                )

                axis_env = plotter.create_axis_environment(
                    x_label=f"PC1 ({self.explained_variance_ratio_[0]:.1%} variance)",
                    y_label=f"PC2 ({self.explained_variance_ratio_[1]:.1%} variance)",
                    title="PCA Scatter Plot (PC1 vs PC2)",
                    grid=True,
                )

                caption = f"PCA scatter plot showing the first two principal components. PC1 explains {self.explained_variance_ratio_[0]:.1%} and PC2 explains {self.explained_variance_ratio_[1]:.1%} of the total variance."

            elif self._transformed_data.shape[1] == 1:
                x = self._transformed_data[:, 0]
                y = np.zeros_like(x)

                scatter_plot = plotter.plot_scatter(
                    x=x,
                    y=y,
                    colors=["plotblue"] * len(x),
                    markers=["circle"] * len(x),
                    sizes=[2] * len(x),
                )

                axis_env = plotter.create_axis_environment(
                    x_label=f"PC1 ({self.explained_variance_ratio_[0]:.1%} variance)",
                    y_label="Constant",
                    title="PCA Scatter Plot (PC1)",
                    grid=True,
                )

                caption = f"PCA scatter plot showing the first principal component. PC1 explains {self.explained_variance_ratio_[0]:.1%} of the total variance."

            complete_plot = plotter.complete_plot(
                axis_env + scatter_plot + "\\end{axis}",
                caption=caption,
                label="pca_scatter",
            )

        elif plot_type == "explained_variance":
            # Create explained variance plot
            n_components = len(self.explained_variance_ratio_)
            x_values = list(range(1, n_components + 1))

            # Cumulative variance
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)

            # Create line plot for cumulative variance
            line_plot = plotter.plot_line(
                x=x_values,
                y=cumulative_variance * 100,  # Convert to percentage
                label="Cumulative Variance",
                color="plotblue",
                line_style="-",
                marker="o",
            )

            # Add individual variance bars (simplified as points)
            for i, var in enumerate(self.explained_variance_ratio_):
                point_plot = plotter.plot_scatter(
                    x=[i + 1],
                    y=[var * 100],
                    colors=["plotred"],
                    markers=["square"],
                    sizes=[4],
                )
                line_plot += point_plot

            axis_env = plotter.create_axis_environment(
                x_label="Principal Component",
                y_label="Explained Variance (%)",
                title="PCA Explained Variance",
                grid=True,
            )

            caption = f"Explained variance for {n_components} principal components. The first {n_components} components explain {cumulative_variance[-1]:.1%} of the total variance."

            complete_plot = plotter.complete_plot(
                axis_env + line_plot + "\\end{axis}",
                caption=caption,
                label="pca_explained_variance",
            )

        else:
            raise ValueError(
                f"Unknown plot type: {plot_type}. Supported types: 'scatter', 'explained_variance'"
            )

        if output_file:
            plotter.save_plot_to_tikz(complete_plot, output_file)

        return complete_plot

    def get_memory_info(self) -> dict:
        """Get memory usage information for memory-optimized PCA."""
        if not self.memory_optimized:
            return {
                "memory_optimized": False,
                "current_memory_mb": self._get_memory_usage(),
            }

        return {
            "memory_optimized": True,
            "current_memory_mb": self._get_memory_usage(),
            "memory_history": self._memory_usage.copy(),
            "data_shape": (
                self._original_data.shape if self._original_data is not None else None
            ),
            "n_components": self.n_components,
            "chunk_size": self.chunk_size,
            "memory_limit_mb": self.memory_limit_mb,
            "use_incremental": self.use_incremental,
            "dtype": str(self.dtype),
        }

    def clear_memory(self):
        """Clear memory and force garbage collection."""
        import gc

        self._memory_usage = []
        gc.collect()
        if self.memory_optimized:
            print("Memory cleared and garbage collection forced.")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Args:
            data: Transformed data to inverse transform

        Returns:
            Data in original space
        """
        if not self._fitted:
            raise ValueError("PCA must be fitted before inverse transforming data")

        # Inverse transform
        reconstructed = self._pca.inverse_transform(data)

        # Un-standardize if needed
        if self.standardize and self._scaler is not None:
            reconstructed = self._scaler.inverse_transform(reconstructed)

        return reconstructed
