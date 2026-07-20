"""Trajectory persistence helpers."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from autokmc.core.constants import TRAJ_DUMP_EVERY


class TrajectoryWriter:
	"""Periodic extended-XYZ dumper."""

	def __init__(
		self,
		output_path: str | Path,
		*,
		dump_every: int = TRAJ_DUMP_EVERY,
		append: bool = False,
	):
		self.output_path = Path(output_path)
		self.dump_every = int(dump_every)
		self.append = bool(append)
		self._n_frames: int = 0
		if self.dump_every > 0:
			self.output_path.parent.mkdir(parents=True, exist_ok=True)
			if not self.append:
				self.output_path.write_text("")

	@property
	def n_frames(self) -> int:
		return self._n_frames

	@property
	def enabled(self) -> bool:
		return self.dump_every > 0

	def maybe_write(self, atoms: Atoms, *, step: int) -> bool:
		if not self.enabled:
			return False
		if step != 0 and (step % self.dump_every) != 0:
			return False
		return self._write(atoms, step=step)

	def write(self, atoms: Atoms, *, step: int = 0) -> bool:
		if not self.enabled:
			return False
		return self._write(atoms, step=step)

	def _write(self, atoms: Atoms, *, step: int) -> bool:
		snap = atoms.copy()
		snap.info["kmc_step"] = int(step)
		ase_write(self.output_path, snap, format="extxyz", append=True)
		self._n_frames += 1
		return True

	def close(self) -> None:
		return

	def __enter__(self):  # pragma: no cover
		return self

	def __exit__(self, *exc):  # pragma: no cover
		self.close()


__all__ = ["TrajectoryWriter"]
