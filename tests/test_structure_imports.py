"""Import checks for package split helper modules."""

from __future__ import annotations

import logging
import os

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
