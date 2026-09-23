"""Import checks for package split helper modules."""

from __future__ import annotations

import logging
import os

from ase import Atoms
import numpy as np
import pytest

from autokmc.utils.logging import get_logger


def test_stability_neb_imports_real_module_exports():
    from autokmc.sites.stability.neb import NEBNotConvergedError, _make_neb_band

    assert issubclass(NEBNotConvergedError, Exception)
    assert callable(_make_neb_band)


def test_logger_uses_autokmc_root():
    logger = get_logger("autokmc.io.persistence")
    assert logger.name == "autokmc.io.persistence"

    logger = get_logger("custom")
    assert logger.name == "autokmc.custom"
    assert logging.getLogger("autokmc").handlers


def test_neb_verbose_logging_uses_stdout():
    from autokmc.sites.stability.bond import _neb_optimizer_logfile as bond_logfile
    from autokmc.sites.stability.diffusion import (
        _neb_optimizer_logfile as diffusion_logfile,
    )

    assert diffusion_logfile(True) == "-"
    assert bond_logfile(True) == "-"
    assert diffusion_logfile(False) == os.devnull
    assert bond_logfile(False) == os.devnull


def test_structure_optimisation_error_retains_last_geometry(monkeypatch):
    from autokmc.structure import StructureOptimisationError
    from autokmc.structure import optimization as optimization_module

    class FailingOptimizer:
        def __init__(self, atoms, *, logfile):
            del logfile
            self.atoms = atoms

        def run(self, *, fmax, steps):
            del fmax, steps
            self.atoms.positions[0, 0] = 1.25
            raise RuntimeError("calculator exploded")

        def get_number_of_steps(self):
            return 3

    monkeypatch.setattr(
        optimization_module,
        "_optimizer_class",
        lambda _name: FailingOptimizer,
    )
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(StructureOptimisationError) as caught:
        optimization_module.optimise_structure(
            atoms,
            calculator=object(),
            verbose=False,
        )

    assert caught.value.steps == 3
    assert caught.value.converged is None
    assert caught.value.atoms.calc is None
    np.testing.assert_allclose(caught.value.atoms.positions, [[1.25, 0.0, 0.0]])
    np.testing.assert_allclose(atoms.positions, [[0.0, 0.0, 0.0]])


def test_structure_fire_forwards_constructor_kwargs(monkeypatch):
    from autokmc.structure import optimization as optimization_module

    captured = []

    class CapturingOptimizer:
        def __init__(self, atoms, *, logfile, **kwargs):
            del logfile
            self.atoms = atoms
            self.kwargs = kwargs
            captured.append(kwargs)

        def run(self, *, fmax, steps):
            del fmax, steps

        def converged(self):
            return True

        def get_number_of_steps(self):
            return 0

    monkeypatch.setattr(
        optimization_module,
        "_optimizer_class",
        lambda _name: CapturingOptimizer,
    )
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    optimization_module.optimise_structure(
        atoms,
        calculator=object(),
        optimizer="fire",
        optimizer_kwargs={
            "dt": 0.01,
            "dtmax": 0.05,
            "maxstep": 0.03,
            "downhill_check": True,
        },
        verbose=False,
    )

    assert captured == [
        {
            "dt": pytest.approx(0.01),
            "dtmax": pytest.approx(0.05),
            "maxstep": pytest.approx(0.03),
            "downhill_check": True,
        }
    ]
