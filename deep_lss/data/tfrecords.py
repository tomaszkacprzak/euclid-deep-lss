"""TFRecord input bridge for PyTorch consumers.

This module is the intentionally isolated TensorFlow boundary for reading
TFRecords.  Keep ``import tensorflow as tf`` here so application code can work
with ``torch.utils.data.Dataset``/``DataLoader`` objects instead of constructing
``tf.data.Dataset`` pipelines directly.
"""

from __future__ import annotations

import glob
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

import tensorflow as tf

TensorTree = Any
Parser = Callable[[Any], TensorTree]


def _to_torch(value: Any) -> Any:
    """Convert a TensorFlow/NumPy tensor tree to a torch tensor tree."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if np.isscalar(value):
        return torch.as_tensor(value)
    if isinstance(value, Mapping):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_torch(item) for item in value)
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    return value


def parse_example_with_schema(
    serialized: Any, feature_schema: Mapping[str, Any], postprocess: Parser | None = None
) -> TensorTree:
    """Parse one serialized Example with an existing TensorFlow feature schema.

    Args:
        serialized: A scalar serialized ``tf.train.Example`` record.
        feature_schema: Existing ``tf.io.FixedLenFeature``/``VarLenFeature`` schema.
        postprocess: Optional callable that reshapes/casts the parsed feature dict
            into the tuple/dict expected by downstream training code.
    """
    parsed = tf.io.parse_single_example(serialized, feature_schema)
    return postprocess(parsed) if postprocess is not None else parsed


class TFRecordTorchIterableDataset(IterableDataset):
    """Read TFRecords with ``tf.data.TFRecordDataset`` and yield torch tensors."""

    def __init__(
        self,
        filenames: str | Sequence[str],
        feature_schema: Mapping[str, Any],
        postprocess: Parser | None = None,
        compression_type: str | None = None,
        buffer_size: int | None = None,
        num_parallel_reads: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(filenames, str):
            expanded = sorted(glob.glob(filenames)) or [filenames]
        else:
            expanded = list(filenames)
        self.filenames = expanded
        self.feature_schema = feature_schema
        self.postprocess = postprocess
        self.compression_type = compression_type
        self.buffer_size = buffer_size
        self.num_parallel_reads = num_parallel_reads

    def _tf_dataset(self):
        dataset = tf.data.TFRecordDataset(
            self.filenames,
            compression_type=self.compression_type,
            buffer_size=self.buffer_size,
            num_parallel_reads=self.num_parallel_reads,
        )
        return dataset.map(lambda record: parse_example_with_schema(record, self.feature_schema, self.postprocess))

    def __iter__(self):
        for parsed in self._tf_dataset().as_numpy_iterator():
            yield _to_torch(parsed)


class TensorTreeDataset(Dataset):
    """Map-style Dataset for one or more aligned NumPy arrays."""

    def __init__(self, *arrays: np.ndarray, transform: Callable[..., TensorTree] | None = None) -> None:
        self.arrays = [np.asarray(array) for array in arrays]
        if not self.arrays:
            raise ValueError("At least one array is required")
        if len({array.shape[0] for array in self.arrays}) != 1:
            raise ValueError("All arrays must share their first dimension")
        self.transform = transform

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, index: int) -> TensorTree:
        items = tuple(torch.from_numpy(array[index]) for array in self.arrays)
        if self.transform is not None:
            return self.transform(*items)
        return items[0] if len(items) == 1 else items


def make_tensor_dataloader(
    *arrays: np.ndarray, batch_size: int, shuffle: bool = False, drop_last: bool = False, **kwargs: Any
) -> DataLoader:
    """Create a PyTorch DataLoader from aligned NumPy arrays."""
    tensors = tuple(torch.as_tensor(array) for array in arrays)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


def torch_tree_to_tf(value: Any) -> Any:
    """Convert torch/NumPy tensor trees back to TensorFlow tensors for legacy models."""
    if isinstance(value, torch.Tensor):
        return tf.convert_to_tensor(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, Mapping):
        return {key: torch_tree_to_tf(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(torch_tree_to_tf(item) for item in value)
    if isinstance(value, list):
        return [torch_tree_to_tf(item) for item in value]
    return value
