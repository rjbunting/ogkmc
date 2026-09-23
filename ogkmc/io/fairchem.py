"""Optional FAIR-Chem construction helpers.

The imports stay inside the public helper so OGKMC does not require Torch or
FAIR-Chem unless a configuration explicitly selects this integration.
"""

from __future__ import annotations

from typing import Any

from ogkmc.utils.logging import get_logger


_log = get_logger(__name__)


def get_predict_unit_on_device(
    name_or_path: str,
    *,
    device: str | None = None,
    workers: int = 1,
    **kwargs: Any,
):
    """Build one FAIR-Chem predictor bound to a requested logical device.

    FAIR-Chem's single-worker API accepts ``"cuda"`` but resolves the concrete
    ordinal from PyTorch's current device.  OGKMC configurations can pass
    ``"cuda:N"`` here; this helper selects ``N`` during construction and then
    verifies that the predictor retained the requested device.
    """
    import torch
    from fairchem.core import pretrained_mlip

    n_workers = int(workers)
    if n_workers != 1:
        raise ValueError(
            "get_predict_unit_on_device requires workers=1; use OGKMC "
            "calculator copies for independent per-GPU work"
        )

    if device is None:
        return pretrained_mlip.get_predict_unit(
            name_or_path,
            device=None,
            workers=n_workers,
            **kwargs,
        )

    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"FAIR-Chem predictor device must be cpu or cuda, got {device!r}"
        )

    if requested.type == "cpu" or requested.index is None:
        return pretrained_mlip.get_predict_unit(
            name_or_path,
            device=requested.type,
            workers=n_workers,
            **kwargs,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(f"requested {device!r}, but Torch reports no CUDA device")
    device_count = int(torch.cuda.device_count())
    if requested.index < 0 or requested.index >= device_count:
        raise RuntimeError(
            f"requested {device!r}, but Torch exposes {device_count} CUDA device(s)"
        )

    with torch.cuda.device(requested.index):
        predictor = pretrained_mlip.get_predict_unit(
            name_or_path,
            device="cuda",
            workers=n_workers,
            **kwargs,
        )

    resolved_value = getattr(predictor, "device", None)
    if resolved_value is None:
        raise RuntimeError(
            "FAIR-Chem predictor did not expose its resolved device; cannot "
            f"verify assignment to {device!r}"
        )
    resolved = torch.device(resolved_value)
    if resolved.type != "cuda" or resolved.index != requested.index:
        raise RuntimeError(
            f"FAIR-Chem predictor requested {device!r} but resolved to "
            f"{str(resolved)!r}"
        )

    _log.info(
        "Bound FAIR-Chem predictor %s to %s",
        name_or_path,
        resolved,
    )
    return predictor


__all__ = ["get_predict_unit_on_device"]
