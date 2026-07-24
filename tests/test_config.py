"""Tests for autokmc.io.config — load + dynamic calculator instantiation."""

from __future__ import annotations

import textwrap
from dataclasses import fields
from pathlib import Path

import pytest

from autokmc.io.config import (
    ConstantsCfg,
    OptimizationCfg,
    RunConfig,
    ConfigError,
    load_config,
)
from autokmc.io.calculators import (
    CalculatorCfg,
    CalculatorConfigError,
    CalculatorPool,
    build_calculator,
    calculator_meta,
)


def _write(tmp_path: Path, body: str, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


YAML_OK = """\
schema_version: "1"
output:
  dir: ./out
structure:
  kind: surface
  composition: Cu
  miller_index: [1, 1, 1]
reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
kmc:
  temperature_k: 500.0
  n_steps: 10
"""

CALCULATOR_YAML = """\
calculator:
  import_path: ase.calculators.emt.EMT
"""


def test_load_yaml_ok(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK)
    cfg = load_config(p)
    assert isinstance(cfg, RunConfig)
    assert cfg.output.dir == "./out"
    assert cfg.structure.miller_index == (1, 1, 1)
    assert cfg.reactants[0].smiles == "[C-]#[O+]"
    assert cfg.calculator.import_path == "ase.calculators.emt.EMT"
    assert cfg.kmc.n_steps == 10
    assert cfg.diffusion.enabled is False
    assert cfg.diffusion.spring_k == pytest.approx(5.0)
    assert cfg.bond.neb_spring_k == pytest.approx(5.0)
    assert cfg.free_energy.symmetry_tolerance == pytest.approx(0.3)
    assert cfg.adsorbate_sites.anchor_k_max == 4
    assert cfg.output.calculation_cache_lookup_enabled is False
    assert cfg.output.isaac_export_enabled is False
    assert cfg.optimization.optimizer == "lbfgs"
    assert cfg.optimization.neb_optimizer == "bfgs"


def test_loads_optimizer_choices(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        """
schema_version: "1"
optimization:
  optimizer: fire
  neb_optimizer: mdmin
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
""",
    )

    cfg = load_config(path)

    assert cfg.optimization.optimizer == "fire"
    assert cfg.optimization.neb_optimizer == "mdmin"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("optimizer", "not-an-optimizer", "optimization.optimizer"),
        ("neb_optimizer", "lbfgs", "optimization.neb_optimizer"),
    ],
)
def test_rejects_invalid_optimizer_choices(tmp_path, key, value, message):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        f"""
schema_version: "1"
optimization:
  {key}: {value}
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
""",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_loads_every_shared_constant_and_site_geometry_control(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        """
schema_version: "1"
constants:
  neighbor_list_multiplier: 0.91
  co_bond_factor: 0.92
  anchor_bond_factor: 0.83
  anchor_repulsion_weight: 0.15
  site_repulsion_cutoff: null
  adsorbate_contact_factor: 1.08
  adsorbate_standoff_factor: 0.1
  adsorbate_rotational_restarts: 9
  typical_neighbor_distance: 2.7
  adsorbate_bond_tolerance: 0.35
  anchor_hull_tolerance: -0.1
  raycast_coverage_threshold: 0.8
  raycast_disc_samples: 12
  kabsch_max_mappings: 5000
  lateral_shells: 2
structure:
  surface_side: both
  surface_radius_factor: 1.1
  nanoparticle_hull_tolerance_factor: 0.6
adsorbate_sites:
  n_shells_anchor: 2
  pair_n_shells: 3
  max_pair_shells: 12
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
""",
    )

    cfg = load_config(path)

    assert cfg.constants.neighbor_list_multiplier == pytest.approx(0.91)
    assert cfg.constants.co_bond_factor == pytest.approx(0.92)
    assert cfg.constants.anchor_bond_factor == pytest.approx(0.83)
    assert cfg.constants.anchor_repulsion_weight == pytest.approx(0.15)
    assert cfg.constants.site_repulsion_cutoff is None
    assert cfg.constants.adsorbate_contact_factor == pytest.approx(1.08)
    assert cfg.constants.adsorbate_standoff_factor == pytest.approx(0.1)
    assert cfg.constants.adsorbate_rotational_restarts == 9
    assert cfg.constants.typical_neighbor_distance == pytest.approx(2.7)
    assert cfg.constants.adsorbate_bond_tolerance == pytest.approx(0.35)
    assert cfg.constants.anchor_hull_tolerance == pytest.approx(-0.1)
    assert cfg.constants.raycast_coverage_threshold == pytest.approx(0.8)
    assert cfg.constants.raycast_disc_samples == 12
    assert cfg.constants.kabsch_max_mappings == 5000
    assert cfg.constants.lateral_shells == 2
    assert cfg.structure.surface_side == "both"
    assert cfg.structure.surface_radius_factor == pytest.approx(1.1)
    assert cfg.structure.nanoparticle_hull_tolerance_factor == pytest.approx(0.6)
    assert cfg.adsorbate_sites.n_shells_anchor == 2
    assert cfg.adsorbate_sites.pair_n_shells == 3
    assert cfg.adsorbate_sites.max_pair_shells == 12


def test_all_options_template_lists_every_shared_constant():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).parents[1] / "example" / "all_options.yaml"

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(path)

    assert set(raw["constants"]) == {item.name for item in fields(ConstantsCfg)}
    assert set(raw["optimization"]) == {
        item.name for item in fields(OptimizationCfg)
    }
    assert cfg.structure.surface_side == "top"
    assert cfg.adsorbate_sites.max_pair_shells == 10


@pytest.mark.parametrize(
    "filename",
    [
        "co_oxidation_pt111_uma_4gpu.yaml",
        "co_oxidation_ptnano_uma_4gpu.yaml",
    ],
)
def test_uma_four_gpu_examples_delegate_parallelism_to_fairchem(filename):
    pytest.importorskip("yaml")
    path = Path(__file__).parents[1] / "example" / filename

    cfg = load_config(path)

    assert cfg.calculator.factory == (
        "fairchem.core.FAIRChemCalculator.from_model_checkpoint"
    )
    assert cfg.calculator.factory_kwargs["device"] == "cuda"
    assert cfg.calculator.factory_kwargs["workers"] == 4
    assert cfg.calculator.copies == 1
    assert cfg.calculator.max_workers == 1
    assert cfg.calculator.gpu_devices is None


def test_load_toml_ok(tmp_path):
    body = """\
    schema_version = "1"

    [output]
    dir = "./out"

    [structure]
    kind = "surface"
    composition = "Cu"
    miller_index = [1, 1, 1]

    [[reactants]]
    smiles = "[C-]#[O+]"
    add_hydrogens = false

    [calculator]
    import_path = "ase.calculators.emt.EMT"

    [kmc]
    temperature_k = 500.0
    n_steps = 10
    """
    p = _write(tmp_path, body, name="cfg.toml")
    cfg = load_config(p)
    assert cfg.kmc.n_steps == 10


@pytest.mark.parametrize(
    "calculator_yaml",
    ["", "calculator: {}\n", "calculator: null\n"],
)
def test_load_config_requires_an_explicit_calculator(tmp_path, calculator_yaml):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\n"
        "reactants:\n"
        "  - smiles: '[O]'\n"
        + calculator_yaml,
    )

    with pytest.raises(
        ConfigError,
        match=r"calculator\.import_path or calculator\.factory",
    ):
        load_config(path)


def test_load_file_structure_resolves_path_and_coerces_frozen_indices(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        """
schema_version: "1"
structure:
  kind: file
  path: structures/catalyst.extxyz
  format: extxyz
  index: 3
  frozen_indices: [0, 2, 5]
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
""",
    )

    cfg = load_config(path)

    assert cfg.structure.path == str(
        (tmp_path / "structures" / "catalyst.extxyz").resolve()
    )
    assert cfg.structure.format == "extxyz"
    assert cfg.structure.index == 3
    assert cfg.structure.frozen_indices == [0, 2, 5]
    assert isinstance(cfg.structure.frozen_indices, list)


def test_unknown_key_raises(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK + "\nbogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_schema_version_mismatch(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK.replace('"1"', '"99"'))
    with pytest.raises(ConfigError):
        load_config(p)


def test_build_calculator_emt():
    cfg = CalculatorCfg(import_path="ase.calculators.emt.EMT", kwargs={})
    calc = build_calculator(cfg)
    assert isinstance(calc, CalculatorPool)
    with calc.acquire() as concrete:
        # Must look like an ASE calculator.
        assert hasattr(concrete, "get_potential_energy")


def test_build_calculator_requires_construction_path():
    with pytest.raises(
        CalculatorConfigError,
        match=r"calculator\.import_path or calculator\.factory",
    ):
        build_calculator(CalculatorCfg())


def test_calculator_meta_roundtrip():
    cfg = CalculatorCfg(
        import_path="pkg.Foo",
        kwargs={"a": 1},
        copies=2,
        gpu_devices=["cuda:0", "cuda:1"],
    )
    meta = calculator_meta(cfg)
    assert meta["import_path"] == "pkg.Foo"
    assert meta["kwargs"] == {"a": 1}
    assert meta["copies"] == 2
    assert meta["gpu_devices"] == ["cuda:0", "cuda:1"]


def test_load_new_checkpoint_and_parallel_fields(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, """
schema_version: "1"
output:
  dir: ./out
  calculation_cache_enabled: true
  calculation_cache_lookup_enabled: false
  calculation_cache_dir: calc_cache
  isaac_export_enabled: true
  isaac_export_filename: isaac_upload.json
reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
calculator:
  import_path: ase.calculators.emt.EMT
  copies: 2
  gpu_devices: ["cuda:0", "cuda:1"]
  gpu_device_arg: device
  max_workers: 2
checkpoint:
  enabled: true
  path: ./out/checkpoint.pkl
  every_n_steps: 5
structure:
  kind: nanoparticle
  composition: Cu
  n_atoms: 55
  surface_energy_facets: [[1, 1, 1], [1, 0, 0]]
  surface_energy_layers: 4
""")
    cfg = load_config(p)
    assert cfg.calculator.copies == 2
    assert cfg.calculator.gpu_devices == ["cuda:0", "cuda:1"]
    assert cfg.checkpoint.enabled is True
    assert cfg.checkpoint.every_n_steps == 5
    assert cfg.structure.surface_energy_facets == ((1, 1, 1), (1, 0, 0))
    assert cfg.output.calculation_cache_enabled is True
    assert cfg.output.calculation_cache_lookup_enabled is False
    assert cfg.output.calculation_cache_dir == "calc_cache"
    assert cfg.output.isaac_export_enabled is True
    assert cfg.output.isaac_export_filename == "isaac_upload.json"
    assert cfg.output.run_manifest_filename == "run_manifest.json"


def test_load_bond_matching_fields(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, """
schema_version: "1"
reactants:
  - smiles: "[OH]"
    add_hydrogens: false
calculator:
  import_path: ase.calculators.emt.EMT
bond:
  enabled: true
  neb_interpolation: idpp
  atom_matching: hungarian
  matching_trials: 12
  gas_lift_height: 4.5
""")
    cfg = load_config(p)
    assert cfg.bond.enabled is True
    assert cfg.bond.neb_interpolation == "idpp"
    assert cfg.bond.atom_matching == "hungarian"
    assert cfg.bond.matching_trials == 12
    assert cfg.bond.gas_lift_height == 4.5


def test_unknown_extension(tmp_path):
    p = tmp_path / "cfg.ini"
    p.write_text("[x]\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


@pytest.mark.parametrize(
    "fragment",
    [
        "output:\n  calculation_cache_lookup_enabled: 'false'\n",
        "diffusion:\n  enabled: 'false'\n",
        "bond:\n  neb_climb: 'true'\n",
        "kmc:\n  temperature_k: 0\n",
        "free_energy:\n  vibration_nfree: 3\n",
        "free_energy:\n  symmetry_tolerance: 0\n",
        "free_energy:\n  pressure_bar: -0.1\n",
        "adsorbate_sites:\n  anchor_k_max: 0\n",
        "adsorbate_sites:\n  anchor_k_max: true\n",
        "adsorbate_sites:\n  n_shells_anchor: -1\n",
        "adsorbate_sites:\n  pair_n_shells: -1\n",
        "adsorbate_sites:\n  max_pair_shells: -1\n",
        "structure:\n  miller_index: [1, 1, 1.0]\n",
        "structure:\n  surface_side: sideways\n",
        "structure:\n  surface_radius_factor: 0\n",
        "structure:\n  nanoparticle_hull_tolerance_factor: -0.1\n",
        "constants:\n  neighbor_list_multiplier: 0\n",
        "constants:\n  co_bond_factor: -0.1\n",
        "constants:\n  anchor_bond_factor: 0\n",
        "constants:\n  anchor_repulsion_weight: -0.1\n",
        "constants:\n  site_repulsion_cutoff: 0\n",
        "constants:\n  adsorbate_contact_factor: 0\n",
        "constants:\n  adsorbate_standoff_factor: -0.1\n",
        "constants:\n  adsorbate_rotational_restarts: 0\n",
        "constants:\n  typical_neighbor_distance: 0\n",
        "constants:\n  adsorbate_bond_tolerance: -0.1\n",
        "constants:\n  anchor_hull_tolerance: .nan\n",
        "constants:\n  raycast_coverage_threshold: 1.1\n",
        "constants:\n  raycast_disc_samples: 0\n",
        "constants:\n  kabsch_max_mappings: 0\n",
        "constants:\n  lateral_shells: -1\n",
        "kmc:\n  random_seed: -1\n",
        "bond:\n  bond_types: [SINGLE, QUADRUPLE]\n",
        (
            "calculator:\n"
            "  import_path: ase.calculators.emt.EMT\n"
            "  factory: ase.calculators.emt.EMT\n"
        ),
        (
            "calculator:\n"
            "  import_path: ase.calculators.emt.EMT\n"
            "  kwargs: []\n"
        ),
        (
            "calculator:\n"
            "  import_path: ase.calculators.emt.EMT\n"
            "  copies: 1\n"
            "  max_workers: 2\n"
        ),
        (
            "calculator:\n"
            "  import_path: ase.calculators.emt.EMT\n"
            "  copies: 2\n"
            "  gpu_devices: [cuda:0]\n"
        ),
        "output:\n  isaac_export_enabled: 'false'\n",
        "output:\n  reactions_filename: ../events.jsonl\n",
        (
            "output:\n"
            "  reactions_filename: results.json\n"
            "  summary_filename: results.json\n"
        ),
    ],
)
def test_strict_validation_rejects_coercible_types_and_invalid_ranges(tmp_path, fragment):
    pytest.importorskip("yaml")
    calculator = "" if fragment.startswith("calculator:") else CALCULATOR_YAML
    path = _write(
        tmp_path,
        "schema_version: '1'\nreactants:\n  - smiles: '[O]'\n"
        + calculator
        + fragment,
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_empty_reactants_are_rejected(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\nreactants: []\n" + CALCULATOR_YAML,
    )

    with pytest.raises(ConfigError, match="at least one species"):
        load_config(path)


def test_invalid_smiles_is_rejected_during_config_validation(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\nreactants:\n  - smiles: 'not valid ('\n"
        + CALCULATOR_YAML,
    )

    with pytest.raises(ConfigError, match="could not be parsed by RDKit"):
        load_config(path)


def test_explicit_null_anchor_clique_cap_restores_unbounded_mode(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\n"
        "reactants:\n"
        "  - smiles: '[O]'\n"
        + CALCULATOR_YAML
        + "adsorbate_sites:\n"
        "  anchor_k_max: null\n",
    )

    cfg = load_config(path)

    assert cfg.adsorbate_sites.anchor_k_max is None


def test_zero_default_partial_pressure_is_allowed(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        """
schema_version: "1"
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
free_energy:
  pressure_bar: 0.0
""",
    )

    cfg = load_config(path)

    assert cfg.free_energy.pressure_bar == 0.0
    assert cfg.reactants[0].partial_pressure_bar is None


def test_duplicate_canonical_reactants_are_rejected(tmp_path):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        """
schema_version: "1"
reactants:
  - smiles: "C(O)"
  - smiles: "OC"
calculator:
  import_path: ase.calculators.emt.EMT
""",
    )

    with pytest.raises(ConfigError, match="duplicates reactants\\[0\\]"):
        load_config(path)


@pytest.mark.parametrize(
    ("structure", "message"),
    [
        ("kind: file", "structure.path must be a non-empty string"),
        (
            "kind: file\n  path: catalyst.xyz\n  format: '   '",
            "structure.format must be a non-empty string",
        ),
        (
            "kind: file\n  path: catalyst.xyz\n  index: true",
            "structure.index must be an integer",
        ),
        (
            "kind: file\n  path: catalyst.xyz\n  frozen_indices: [0, -1]",
            r"structure\.frozen_indices\[1\] must be >= 0",
        ),
        (
            "kind: file\n  path: catalyst.xyz\n  frozen_indices: [1, 1]",
            "structure.frozen_indices entries must be unique",
        ),
        (
            "kind: file\n  path: catalyst.xyz\n  frozen_indices: [false]",
            r"structure\.frozen_indices\[0\] must be an integer",
        ),
        (
            "kind: surface\n  path: catalyst.xyz",
            "structure.path is only valid",
        ),
        (
            "kind: nanoparticle\n  format: xyz",
            "structure.format is only valid",
        ),
        (
            "kind: surface\n  frozen_indices: []",
            "structure.frozen_indices is only valid",
        ),
        (
            "kind: surface\n  index: 0",
            "structure.index is only valid",
        ),
    ],
)
def test_file_structure_validation_rejects_invalid_or_unused_fields(
    tmp_path,
    structure,
    message,
):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\n"
        "reactants:\n"
        "  - smiles: '[O]'\n"
        + CALCULATOR_YAML
        + "structure:\n"
        f"  {structure}\n",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)
