"""Checkpoint/restart support placeholder.

Checkpointing is intentionally not implemented yet. This module reserves the
explicit I/O location for future restart support.
"""

from __future__ import annotations


class CheckpointNotImplementedError(NotImplementedError):
	"""Raised by checkpoint placeholders until restart support is implemented."""


def save_checkpoint(*args, **kwargs) -> None:
	raise CheckpointNotImplementedError("KMC checkpoint writing is not implemented yet")


def load_checkpoint(*args, **kwargs):
	raise CheckpointNotImplementedError("KMC checkpoint loading is not implemented yet")


__all__ = ["CheckpointNotImplementedError", "save_checkpoint", "load_checkpoint"]
