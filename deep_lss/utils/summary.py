# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen
"""

import numpy as np


def _to_numpy(value):
    """Convert tensor-like values to NumPy without depending on a specific backend."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _to_scalar(value):
    arr = _to_numpy(value)
    return arr.item() if np.ndim(arr) == 0 else arr


def write_summary(label, value, summary_writer, training=True, summary_type="scalar", print_scalar=False, step=None):
    """Write scalar, histogram, or image summaries without assuming a TensorFlow backend.

    Supports common PyTorch TensorBoard writers (``add_scalar``, ``add_histogram``,
    ``add_image``/``add_images``) and Weights & Biases runs (``log``). If no writer
    is supplied or ``training`` is false, this is a no-op apart from optional scalar
    printing.
    """
    if summary_type not in {"scalar", "histogram", "image"}:
        raise ValueError(f"Invalid summary type {summary_type} was passed")

    if print_scalar and summary_type == "scalar":
        print(f"{label}: {_to_scalar(value)}")

    if summary_writer is None or not training:
        return

    if hasattr(summary_writer, "log"):
        summary_writer.log({label: _to_scalar(value) if summary_type == "scalar" else _to_numpy(value)}, step=step)
        return

    if summary_type == "scalar" and hasattr(summary_writer, "add_scalar"):
        summary_writer.add_scalar(label, _to_scalar(value), global_step=step)
    elif summary_type == "histogram" and hasattr(summary_writer, "add_histogram"):
        summary_writer.add_histogram(label, _to_numpy(value), global_step=step)
    elif summary_type == "image":
        image = _to_numpy(value)
        if hasattr(summary_writer, "add_images") and image.ndim == 4:
            summary_writer.add_images(label, image, global_step=step, dataformats="NHWC")
        elif hasattr(summary_writer, "add_image"):
            if image.ndim == 4:
                image = image[0]
            summary_writer.add_image(label, image, global_step=step, dataformats="HWC")
    elif hasattr(summary_writer, "write_summary"):
        summary_writer.write_summary(label, value, summary_type=summary_type, step=step)
    else:
        raise TypeError(
            "summary_writer does not expose a supported TensorBoard, W&B, or write_summary logging interface"
        )
