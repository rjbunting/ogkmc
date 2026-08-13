"""Pure builders for persisted per-reaction JSON documents."""

from __future__ import annotations

from typing import Any, Mapping

from autokmc.io.reaction_index import stable_reaction_id
from autokmc.io.schemas import (
    REACTION_DOCUMENT_ARTIFACT_TYPE,
    REACTION_DOCUMENT_SCHEMA_VERSION,
)


def _stats_payload(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": int(stats["count"]),
        "first_step": stats["first_step"],
        "last_step": stats["last_step"],
    }


def _directed_last_event(reaction, step: int, *, fired: bool) -> dict[str, Any] | None:
    if not fired:
        return None
    return {
        "kind": str(reaction.kind),
        "direction": getattr(reaction, "direction", None),
        "delta_e_ev": float(reaction.delta_e),
        "barrier_ev": float(reaction.barrier),
        "rate_hz": float(reaction.rate),
        "step": int(step),
    }


def _adsorption_last_event(reaction, step: int, *, fired: bool) -> dict[str, Any] | None:
    if not fired:
        return None
    return {
        "kind": str(reaction.kind),
        "delta_e_ev": float(reaction.delta_e),
        "barrier_ev": float(reaction.barrier),
        "rate_hz": float(reaction.rate),
        "step": int(step),
    }


def _neb_image_payload(lateral_class) -> dict[str, Any]:
    """Return resolved dynamic-band diagnostics for persisted reactions."""
    return {
        "interior_images": getattr(lateral_class, "neb_n_images", None),
        "total_frames": getattr(lateral_class, "neb_n_frames", None),
        "max_endpoint_displacement_ang": getattr(
            lateral_class,
            "neb_max_endpoint_displacement",
            None,
        ),
        "target_spacing_ang": getattr(
            lateral_class,
            "neb_target_image_spacing",
            None,
        ),
        "estimated_linear_spacing_ang": getattr(
            lateral_class,
            "neb_estimated_image_spacing",
            None,
        ),
        "count_limited_by": getattr(
            lateral_class,
            "neb_image_count_limited_by",
            None,
        ),
        "climb_performed": getattr(
            lateral_class,
            "neb_climb_performed",
            None,
        ),
        "climb_skipped_low_barrier": getattr(
            lateral_class,
            "neb_climb_skipped_low_barrier",
            None,
        ),
        "regular_forward_barrier_ev": getattr(
            lateral_class,
            "neb_regular_forward_barrier",
            None,
        ),
        "regular_reverse_barrier_ev": getattr(
            lateral_class,
            "neb_regular_reverse_barrier",
            None,
        ),
    }


def build_diffusion_payload(
    reaction,
    *,
    iso_class: int,
    lateral_class: int,
    smiles: str,
    description: str,
    step: int,
    fired: bool,
    stats: Mapping[str, Any],
    calculator_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the persisted document for one diffusion lateral class."""
    from autokmc.reactions.rates import EA_MIN

    lc = reaction.lateral_class
    e_a = getattr(lc, "energy_a", None)
    e_b = getattr(lc, "energy_b", None)
    e_ts = getattr(lc, "energy_ts", None)
    e_ts_eff: float | None
    ea_fwd_raw: float | None
    ea_fwd_kmc: float | None
    ea_rev_raw: float | None
    ea_rev_kmc: float | None
    if e_a is not None and e_b is not None and e_ts is not None:
        numeric_a = float(e_a)
        numeric_b = float(e_b)
        numeric_ts = float(e_ts)
        e_ts_eff = max(numeric_ts, max(numeric_a, numeric_b) + EA_MIN)
        ea_fwd_raw = numeric_ts - numeric_a
        ea_rev_raw = numeric_ts - numeric_b
        ea_fwd_kmc = max(EA_MIN, e_ts_eff - numeric_a)
        ea_rev_kmc = max(EA_MIN, e_ts_eff - numeric_b)
    else:
        e_ts_eff = ea_fwd_raw = ea_fwd_kmc = None
        ea_rev_raw = ea_rev_kmc = None

    return {
        "artifact_type": REACTION_DOCUMENT_ARTIFACT_TYPE,
        "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        "kind": "diffusion",
        "valid": True,
        "iso_class": iso_class,
        "lateral_class": lateral_class,
        "reactant_smiles": smiles,
        "kind_directions": ["a_to_b", "b_to_a"],
        "template": {"species": smiles},
        "gas_product": False,
        "description": description,
        "neb_images": _neb_image_payload(lc),
        "energies_ev": {
            "state_a": None if e_a is None else float(e_a),
            "state_b": None if e_b is None else float(e_b),
            "transition_raw": None if e_ts is None else float(e_ts),
            "transition_eff": None if e_ts_eff is None else float(e_ts_eff),
        },
        "free_energies_ev": {
            "g_a": None if getattr(lc, "g_a", None) is None else float(lc.g_a),
            "g_b": None if getattr(lc, "g_b", None) is None else float(lc.g_b),
            "g_ts": None if getattr(lc, "g_ts", None) is None else float(lc.g_ts),
        },
        "vibrations": {
            "state_a": {
                "real_ev": list(getattr(lc, "frequencies_a_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_a_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_a", None),
                "entropy_ev_per_k": getattr(lc, "entropy_a", None),
            },
            "state_b": {
                "real_ev": list(getattr(lc, "frequencies_b_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_b_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_b", None),
                "entropy_ev_per_k": getattr(lc, "entropy_b", None),
            },
            "transition": {
                "real_ev": list(getattr(lc, "frequencies_ts_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_ts_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_ts", None),
                "entropy_ev_per_k": getattr(lc, "entropy_ts", None),
            },
        },
        "barriers_ev": {
            "forward_raw": ea_fwd_raw,
            "forward_kmc": ea_fwd_kmc,
            "reverse_raw": ea_rev_raw,
            "reverse_kmc": ea_rev_kmc,
            "ea_min_floor": EA_MIN,
        },
        "last_event": _directed_last_event(reaction, step, fired=fired),
        "stats": _stats_payload(stats),
        "atoms": {
            "state_a_initial": (
                "state_a_initial.extxyz"
                if getattr(lc, "atoms_a_initial", None) is not None
                else None
            ),
            "state_b_initial": (
                "state_b_initial.extxyz"
                if getattr(lc, "atoms_b_initial", None) is not None
                else None
            ),
            "state_a": "state_a.extxyz",
            "state_b": "state_b.extxyz",
            "transition": "ts.extxyz",
            "neb_path_initial": (
                "neb_path_initial.extxyz"
                if getattr(lc, "atoms_neb_path_initial", None)
                else None
            ),
            "neb_path": ("neb_path.extxyz" if getattr(lc, "atoms_neb_path", None) else None),
        },
        "calculator": dict(calculator_meta),
    }


def build_bond_payload(
    reaction,
    *,
    iso_class: int,
    lateral_class: int,
    smiles: str,
    description: str,
    step: int,
    fired: bool,
    stats: Mapping[str, Any],
    calculator_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the persisted document for one bond lateral class."""
    from autokmc.reactions.rates import EA_MIN

    lc = reaction.lateral_class
    template = getattr(reaction.site, "template", None)
    e_ab = getattr(lc, "energy_ab", None)
    e_c = getattr(lc, "energy_c", None)
    e_c_precursor = getattr(lc, "energy_c_precursor", None)
    gas_reactant = getattr(reaction.site, "gas_reactant", None)
    e_gas_molecule = getattr(gas_reactant, "energy", None)
    e_ts = getattr(lc, "energy_ts", None)
    e_ts_eff: float | None
    ea_couple_raw: float | None
    ea_couple_kmc: float | None
    ea_dissoc_raw: float | None
    ea_dissoc_kmc: float | None
    if e_ab is not None and e_c is not None and e_ts is not None:
        numeric_ab = float(e_ab)
        numeric_c = float(e_c)
        numeric_ts = float(e_ts)
        e_ts_eff = max(numeric_ts, max(numeric_ab, numeric_c) + EA_MIN)
        ea_couple_raw = numeric_ts - numeric_ab
        ea_dissoc_raw = numeric_ts - numeric_c
        ea_couple_kmc = max(EA_MIN, e_ts_eff - numeric_ab)
        ea_dissoc_kmc = max(EA_MIN, e_ts_eff - numeric_c)
    else:
        e_ts_eff = ea_couple_raw = ea_couple_kmc = None
        ea_dissoc_raw = ea_dissoc_kmc = None

    return {
        "artifact_type": REACTION_DOCUMENT_ARTIFACT_TYPE,
        "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        "kind": "bond",
        "valid": True,
        "iso_class": iso_class,
        "lateral_class": lateral_class,
        "reactant_smiles": smiles,
        "template": (
            {
                "smiles_a": template.smiles_a,
                "smiles_b": template.smiles_b,
                "smiles_c": template.smiles_c,
                "bond_type": getattr(template, "bond_type", None),
                "source": getattr(template, "source", None),
            }
            if template is not None
            else None
        ),
        "kind_directions": ["couple", "dissoc"],
        "gas_product": bool(getattr(reaction.site, "gas_product", False)),
        "description": description,
        "neb_images": _neb_image_payload(lc),
        "energies_ev": {
            "state_ab": None if e_ab is None else float(e_ab),
            "state_c": None if e_c is None else float(e_c),
            "state_c_precursor": (
                None if e_c_precursor is None else float(e_c_precursor)
            ),
            "state_c_gas_reference": (
                None
                if getattr(lc, "energy_c_gas_reference", None) is None
                else float(lc.energy_c_gas_reference)
            ),
            "gas_molecule": (
                None if e_gas_molecule is None else float(e_gas_molecule)
            ),
            "transition_raw": None if e_ts is None else float(e_ts),
            "transition_eff": None if e_ts_eff is None else float(e_ts_eff),
        },
        "free_energies_ev": {
            "g_ab": None if getattr(lc, "g_ab", None) is None else float(lc.g_ab),
            "g_c": None if getattr(lc, "g_c", None) is None else float(lc.g_c),
            "g_ts": None if getattr(lc, "g_ts", None) is None else float(lc.g_ts),
        },
        "vibrations": {
            "state_ab": {
                "real_ev": list(getattr(lc, "frequencies_ab_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_ab_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_ab", None),
                "entropy_ev_per_k": getattr(lc, "entropy_ab", None),
            },
            "state_c": {
                "real_ev": list(getattr(lc, "frequencies_c_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_c_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_c", None),
                "entropy_ev_per_k": getattr(lc, "entropy_c", None),
            },
            "transition": {
                "real_ev": list(getattr(lc, "frequencies_ts_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_ts_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_ts", None),
                "entropy_ev_per_k": getattr(lc, "entropy_ts", None),
            },
        },
        "atom_matching": {
            "method": getattr(lc, "atom_matching_method", None),
            "atom_mapping": list(getattr(lc, "atom_mapping", []) or []),
            "diagnostics": dict(getattr(lc, "matching_diagnostics", {}) or {}),
        },
        "barriers_ev": {
            "couple_raw": ea_couple_raw,
            "couple_kmc": ea_couple_kmc,
            "dissoc_raw": ea_dissoc_raw,
            "dissoc_kmc": ea_dissoc_kmc,
            "ea_min_floor": EA_MIN,
        },
        "last_event": _directed_last_event(reaction, step, fired=fired),
        "stats": _stats_payload(stats),
        "atoms": {
            "state_ab_initial": (
                "state_ab_initial.extxyz"
                if getattr(lc, "atoms_ab_initial", None) is not None
                else None
            ),
            "state_c_initial": (
                "state_c_initial.extxyz"
                if getattr(lc, "atoms_c_initial", None) is not None
                else None
            ),
            "state_ab": "state_ab.extxyz",
            "state_c": "state_c.extxyz",
            "state_c_gas_reference": (
                "state_c_gas_reference.extxyz"
                if getattr(lc, "atoms_c_gas_reference", None) is not None
                else None
            ),
            "gas_molecule": (
                "gas_molecule.extxyz"
                if getattr(lc, "atoms_gas_molecule", None) is not None
                else None
            ),
            "transition": "ts.extxyz",
            "neb_path_initial": (
                "neb_path_initial.extxyz"
                if getattr(lc, "atoms_neb_path_initial", None)
                else None
            ),
            "neb_path": ("neb_path.extxyz" if getattr(lc, "atoms_neb_path", None) else None),
        },
        "calculator": dict(calculator_meta),
    }


def build_adsorption_payload(
    reaction,
    *,
    iso_class: int,
    lateral_class: int,
    smiles: str,
    description: str,
    step: int,
    fired: bool,
    stats: Mapping[str, Any],
    calculator_meta: Mapping[str, Any],
    gas_energies: Mapping[str, float] | None,
    gas_free_energies: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Build the persisted document for one adsorption/desorption class."""
    lc = reaction.lateral_class
    e_gas = float((gas_energies or {}).get(smiles, float("nan")))
    g_gas = (
        None
        if gas_free_energies is None or smiles not in gas_free_energies
        else float(gas_free_energies[smiles])
    )
    e_occ = getattr(lc, "energy_occupied", None)
    e_unocc = getattr(lc, "energy_unoccupied", None)
    g_occ = getattr(lc, "g_occupied", None)
    g_unocc = getattr(lc, "g_unoccupied", None)

    return {
        "artifact_type": REACTION_DOCUMENT_ARTIFACT_TYPE,
        "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        "kind": "adsorption",
        "valid": True,
        "iso_class": iso_class,
        "lateral_class": lateral_class,
        "reactant_smiles": smiles,
        "kind_directions": ["adsorption", "desorption"],
        "template": {"species": smiles},
        "gas_product": False,
        "description": description,
        "energies_ev": {
            "occupied": None if e_occ is None else float(e_occ),
            "unoccupied": None if e_unocc is None else float(e_unocc),
            "gas_phase": e_gas,
        },
        "free_energies_ev": {
            "g_occupied": None if g_occ is None else float(g_occ),
            "g_unoccupied": None if g_unocc is None else float(g_unocc),
            "g_gas": g_gas,
        },
        "vibrations": {
            "occupied": {
                "real_ev": list(getattr(lc, "frequencies_occupied_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_occupied_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_occupied", None),
                "entropy_ev_per_k": getattr(lc, "entropy_occupied", None),
            },
            "unoccupied": {
                "real_ev": list(getattr(lc, "frequencies_unoccupied_ev", []) or []),
                "imag_ev": list(getattr(lc, "imaginary_unoccupied_ev", []) or []),
                "zpe_ev": getattr(lc, "zpe_unoccupied", None),
                "entropy_ev_per_k": getattr(lc, "entropy_unoccupied", None),
            },
        },
        "last_event": _adsorption_last_event(reaction, step, fired=fired),
        "stats": _stats_payload(stats),
        "atoms": {
            "occupied_initial": (
                "occupied_initial.extxyz"
                if getattr(lc, "atoms_occupied_initial", None) is not None
                else None
            ),
            "unoccupied_initial": (
                "unoccupied_initial.extxyz"
                if getattr(lc, "atoms_unoccupied_initial", None) is not None
                else None
            ),
            "occupied": "occupied.extxyz",
            "unoccupied": "unoccupied.extxyz",
        },
        "calculator": dict(calculator_meta),
    }


def build_reaction_payload(
    reaction,
    *,
    subdir: str,
    iso_class: int,
    lateral_class: int,
    discovery_step: int,
    smiles: str,
    description: str,
    step: int,
    fired: bool,
    stats: Mapping[str, Any],
    calculator_meta: Mapping[str, Any],
    run_id: str | None,
    gas_energies: Mapping[str, float] | None = None,
    gas_free_energies: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Dispatch to a kind-specific builder and append common run metadata."""
    if subdir == "diffusion":
        payload = build_diffusion_payload(
            reaction,
            iso_class=iso_class,
            lateral_class=lateral_class,
            smiles=smiles,
            description=description,
            step=step,
            fired=fired,
            stats=stats,
            calculator_meta=calculator_meta,
        )
    elif subdir == "bond":
        payload = build_bond_payload(
            reaction,
            iso_class=iso_class,
            lateral_class=lateral_class,
            smiles=smiles,
            description=description,
            step=step,
            fired=fired,
            stats=stats,
            calculator_meta=calculator_meta,
        )
    else:
        payload = build_adsorption_payload(
            reaction,
            iso_class=iso_class,
            lateral_class=lateral_class,
            smiles=smiles,
            description=description,
            step=step,
            fired=fired,
            stats=stats,
            calculator_meta=calculator_meta,
            gas_energies=gas_energies,
            gas_free_energies=gas_free_energies,
        )
    payload["discovery_step"] = int(discovery_step)
    payload["run_id"] = run_id
    payload["rate_energy_bases"] = []
    payload["reaction_id"] = stable_reaction_id(
        str(payload["kind"]),
        smiles,
        iso_class,
        lateral_class,
    )
    return payload


__all__ = [
    "build_adsorption_payload",
    "build_bond_payload",
    "build_diffusion_payload",
    "build_reaction_payload",
]
