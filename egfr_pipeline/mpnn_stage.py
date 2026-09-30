"""ProteinMPNN redesign helpers with explicit positional controls.

ProteinMPNN supports positional control in several ways. This module exposes
only controls that can be audited in a saved JSON configuration:

* fixed_positions: preserve the residue already present in the backbone;
* allowed_aas: restrict a position to a permitted amino-acid set;
* bias_by_res: add per-amino-acid logits at a position without forcing it.

Positions are 1-based and chain-local. They are never PDB residue numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PositionConstraint:
    """A ProteinMPNN constraint at one chain-local, 1-based position."""

    position: int
    chain: str = "B"
    fixed: bool = False
    allowed_aas: str | None = None
    bias: Mapping[str, float] = field(default_factory=dict)
    label: str = ""

    def validate(self, binder_length: int) -> None:
        if self.chain != "B":
            raise ValueError("this binder redesign helper expects binder chain B")
        if not 1 <= self.position <= binder_length:
            raise ValueError(f"position {self.position} is outside binder length {binder_length}")
        if self.fixed and self.allowed_aas:
            raise ValueError("fixed and allowed_aas are mutually exclusive")
        if self.allowed_aas and (not set(self.allowed_aas) or not set(self.allowed_aas) <= AMINO_ACIDS):
            raise ValueError(f"allowed_aas contains invalid symbols: {self.allowed_aas}")
        if any(aa not in AMINO_ACIDS for aa in self.bias):
            raise ValueError("bias contains an invalid amino-acid symbol")


def validate_sequence(sequence: str, expected_length: int | None = None) -> str:
    cleaned = "".join(sequence.split()).upper()
    if expected_length is not None and len(cleaned) != expected_length:
        raise ValueError(f"sequence length {len(cleaned)} does not match expected length {expected_length}")
    if not cleaned or not set(cleaned) <= AMINO_ACIDS:
        raise ValueError("sequence must contain standard amino-acid symbols only")
    return cleaned


def merge_constraints(constraints: Iterable[PositionConstraint], binder_length: int) -> list[PositionConstraint]:
    """Validate and reject contradictory duplicate position specifications."""
    by_position: dict[int, PositionConstraint] = {}
    for constraint in constraints:
        constraint.validate(binder_length)
        previous = by_position.get(constraint.position)
        if previous is not None and previous != constraint:
            raise ValueError(f"conflicting constraints at binder position {constraint.position}")
        by_position[constraint.position] = constraint
    return [by_position[position] for position in sorted(by_position)]


def build_colabdesign_mpnn_options(
    binder_length: int,
    constraints: Iterable[PositionConstraint] = (),
    omit_aas: str = "",
) -> dict[str, Any]:
    """Build arguments for ``mk_mpnn_model().prep_inputs``/sampling.

    ColabDesign's ``fix_pos`` uses a comma-separated chain/position string,
    e.g. ``B,3,17``. ``bias_by_res`` is a position x amino-acid matrix and is
    intentionally returned separately so callers can inspect it before use.
    """
    merged = merge_constraints(constraints, binder_length)
    if any(aa not in AMINO_ACIDS for aa in omit_aas):
        raise ValueError("omit_aas contains an invalid amino-acid symbol")
    fixed = [item.position for item in merged if item.fixed]
    allowed = {str(item.position): item.allowed_aas for item in merged if item.allowed_aas}
    bias_by_res = {str(item.position): dict(item.bias) for item in merged if item.bias}
    return {
        "binder_chain": "B",
        "binder_length": binder_length,
        "fix_pos": ",".join(["B", *[str(position) for position in fixed]]) if fixed else "",
        "fixed_positions": fixed,
        "allowed_aas": allowed,
        "bias_by_res": bias_by_res,
        "omit_aas": omit_aas,
    }


def build_proteinmpnn_jsonl_record(
    sequence: str,
    constraints: Iterable[PositionConstraint] = (),
    chain: str = "B",
) -> dict[str, Any]:
    """Create an auditable ProteinMPNN-style chain constraint record.

    This record is a portable description for ProteinMPNN wrappers. Native
    ProteinMPNN installations commonly split fixed positions into separate
    JSONL files; the exact file names are version-specific, so this helper
    does not pretend one universal CLI schema exists.
    """
    sequence = validate_sequence(sequence)
    merged = merge_constraints(constraints, len(sequence))
    return {
        "chain": chain,
        "sequence": sequence,
        "fixed_positions": [item.position for item in merged if item.fixed],
        "allowed_aas": {str(item.position): item.allowed_aas for item in merged if item.allowed_aas},
        "bias_by_res": {str(item.position): dict(item.bias) for item in merged if item.bias},
        "constraints": [
            {
                "position": item.position,
                "chain": item.chain,
                "fixed": item.fixed,
                "allowed_aas": item.allowed_aas,
                "bias": dict(item.bias),
                "label": item.label,
            }
            for item in merged
        ],
    }


def write_constraint_record(record: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(record), indent=2) + "\n")
    return destination


def make_histidine_bias(position: int, strength: float = 1.5, label: str = "pH_candidate") -> PositionConstraint:
    """Bias one position toward histidine without forcing histidine.

    Positive ProteinMPNN bias increases sampling probability; it is not an
    affinity or pKa guarantee. Use fixed=False for exploratory pH campaigns.
    """
    return PositionConstraint(position=position, bias={"H": strength}, label=label)
