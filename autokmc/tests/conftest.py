"""Shared pytest fixtures for autokmc tests.

The on-the-fly KMC tests need a real ML calculator (per
``dev/PLAN_adsorption_sites.md`` §12).  We load the deployed NequIP
checkpoint from ``autokmc/dev/cpuhcocuau.nequip.pth`` (or the CUDA
``.pt2`` if a GPU is available).  Any failure to import ``nequip`` or
locate the model file makes the ``ml_calc`` fixture skip cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# NequIP availability + deployed model path
# ---------------------------------------------------------------------------

_DEV_DIR = Path(__file__).resolve().parent.parent / "dev"

try:
    import torch  # noqa: F401
    from nequip.ase import NequIPCalculator  # noqa: F401
    _NEQUIP_AVAILABLE = True
except Exception:  # pragma: no cover - missing optional dep
    _NEQUIP_AVAILABLE = False


def _model_path() -> Path:
    import torch
    use_cuda = torch.cuda.is_available()
    fname = "asehcocuau.nequip.pt2" if use_cuda else "cpuhcocuau.nequip.pth"
    return _DEV_DIR / fname


_MODEL_OK = _NEQUIP_AVAILABLE and _model_path().exists() if _NEQUIP_AVAILABLE else False

requires_nequip = pytest.mark.skipif(
    not _MODEL_OK,
    reason=(
        "NequIP calculator unavailable: install `nequip` and ensure the "
        f"deployed model exists at {_DEV_DIR / 'cpuhcocuau.nequip.pth'}."
    ),
)


# ---------------------------------------------------------------------------
# Shared calculator (session-scoped → load once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ml_calc():
    """Session-scoped NequIP ASE calculator built from the deployed model."""
    if not _MODEL_OK:
        pytest.skip("NequIP / deployed model not available")
    import torch
    from nequip.ase import NequIPCalculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NequIPCalculator.from_compiled_model(
        compile_path=os.fspath(_model_path()),
        device=device,
    )


# ---------------------------------------------------------------------------
# Calculator wrapper that counts get_potential_energy / get_forces calls.
# Used by tests that need to assert "no further LBFGS work" after a
# context-cache hit.
# ---------------------------------------------------------------------------

class CountingCalculator:
    """Thin proxy that delegates to *inner* and counts force calls."""

    def __init__(self, inner):
        self._inner = inner
        self.n_force_calls = 0
        self.n_energy_calls = 0

    # ASE looks at these via duck-typing
    @property
    def implemented_properties(self):
        return getattr(self._inner, "implemented_properties", ["energy", "forces"])

    @property
    def parameters(self):
        return getattr(self._inner, "parameters", {})

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.n_force_calls += 1
        return self._inner.calculate(atoms, properties, system_changes)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        self.n_energy_calls += 1
        return self._inner.get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        self.n_force_calls += 1
        return self._inner.get_forces(atoms)

    def __getattr__(self, name):
        return getattr(self._inner, name)

