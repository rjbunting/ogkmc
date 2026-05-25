"""RDKit logging controls."""

from __future__ import annotations


def silence_rdkit_warnings() -> None:
    """Suppress RDKit warning-level C++ log messages.

    RDKit emits repeated warning messages for valid radical/isolated-H
    intermediate species used during bond enumeration.  Error-level RDKit logs
    are left enabled.
    """
    try:
        from rdkit import RDLogger
    except Exception:
        return
    RDLogger.DisableLog("rdApp.warning")


__all__ = ["silence_rdkit_warnings"]
