"""One-row-per-binder metric table construction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from .plip_stage import parse_plip_targets
from .structural_checks import aligned_binder_rmsd, aligned_complex_rmsd


def _prefixed(destination: dict[str, Any], values: Mapping[str, Any] | None, prefix: str) -> None:
    if not values:
        return
    for key, value in values.items():
        if key not in {"sequence", "design_id"}:
            destination[f"{prefix}{key}"] = value


def build_binder_metric_row(
    design: Mapping[str, Any],
    bindcraft_metrics: Mapping[str, Any] | None = None,
    structural_metrics: Mapping[str, Any] | None = None,
    mpnn_metrics: Mapping[str, Any] | None = None,
    monomer_metrics: Mapping[str, Any] | None = None,
    complex_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    rmsd_metrics: Mapping[str, Any] | None = None,
    plip_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    colabfold_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Flatten available evidence into one binder row."""
    design_id = design.get("design_id", design.get("Design", ""))
    sequence = design.get("sequence", design.get("Sequence", ""))
    row: dict[str, Any] = {
        "design_id": design_id,
        "sequence": sequence,
        "sequence_length": len(str(sequence)) if sequence else None,
    }
    for key, value in design.items():
        if key not in {"design_id", "Design", "sequence", "Sequence"}:
            row[f"design_{key}"] = value
    _prefixed(row, bindcraft_metrics, "bindcraft_")
    _prefixed(row, structural_metrics, "structural_")
    _prefixed(row, mpnn_metrics, "mpnn_")
    _prefixed(row, monomer_metrics, "esmfold_monomer_")
    for target, metrics in (complex_metrics or {}).items():
        _prefixed(row, metrics, f"esmfold_{target}_")
    for target, value in (rmsd_metrics or {}).items():
        if isinstance(value, Mapping):
            row[f"esmfold_{target}_aligned_binder_rmsd"] = value.get("binder")
            row[f"esmfold_{target}_aligned_complex_rmsd"] = value.get("complex")
        else:
            row[f"esmfold_{target}_aligned_binder_rmsd"] = value
            row[f"esmfold_{target}_aligned_complex_rmsd"] = None
    for target, metrics in (plip_metrics or {}).items():
        _prefixed(row, metrics, f"{target}_")
    for target, metrics in (colabfold_metrics or {}).items():
        _prefixed(row, metrics, f"colabfold_{target}_")
    return row


def write_binder_metrics_csv(rows: list[Mapping[str, Any]], output_csv: str | Path) -> Path:
    """Write a clear one-row-per-binder CSV with a stable column union."""
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = [
        "design_id",
        "sequence",
        "sequence_length",
        "esmfold_human_aligned_binder_rmsd",
        "esmfold_human_aligned_complex_rmsd",
        "esmfold_mouse_aligned_binder_rmsd",
        "esmfold_mouse_aligned_complex_rmsd",
    ]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["design_id", "sequence", "sequence_length"]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def assemble_binder_metric_rows(
    designs: list[Mapping[str, Any]],
    bindcraft_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    structural_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    mpnn_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    monomer_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    complex_results: list[Mapping[str, Any]] | None = None,
    reference_complexes: Mapping[str, Mapping[str, str | Path]] | None = None,
    plip_xmls: Mapping[str, Mapping[str, str | Path]] | None = None,
    colabfold_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    target_chain: str = "A",
    binder_chain: str = "B",
) -> list[dict[str, Any]]:
    """Join stage outputs into one row per design.

    ``reference_complexes[design_id][target]`` points to the initial
    ColabFold/BindCraft complex. ``complex_results`` contains ESMFold rows
    returned by ``screen_complexes_by_target`` and includes ``target``.
    ``plip_xmls`` contains one XML path per design and target.
    """
    complexes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for result in complex_results or []:
        design_id = str(result.get("design_id", ""))
        target = str(result.get("target", ""))
        complexes.setdefault(design_id, {})[target] = result

    rows: list[dict[str, Any]] = []
    for design in designs:
        design_id = str(design.get("design_id", design.get("Design", "")))
        rmsd: dict[str, dict[str, float | None]] = {}
        for target, reference_path in (reference_complexes or {}).get(design_id, {}).items():
            predicted = complexes.get(design_id, {}).get(target, {})
            predicted_path = predicted.get("pdb")
            if predicted_path and Path(reference_path).exists() and Path(predicted_path).exists():
                rmsd[target] = {
                    "binder": aligned_binder_rmsd(reference_path, predicted_path, target_chain, binder_chain),
                    "complex": aligned_complex_rmsd(reference_path, predicted_path, target_chain, binder_chain),
                }
            else:
                rmsd[target] = {"binder": None, "complex": None}
        plip_metrics = (
            parse_plip_targets((plip_xmls or {}).get(design_id, {}), target_chain, binder_chain)
            if design_id in (plip_xmls or {}) else {}
        )
        rows.append(build_binder_metric_row(
            design,
            bindcraft_metrics=(bindcraft_metrics or {}).get(design_id),
            structural_metrics=(structural_metrics or {}).get(design_id),
            mpnn_metrics=(mpnn_metrics or {}).get(design_id),
            monomer_metrics=(monomer_metrics or {}).get(design_id),
            complex_metrics=complexes.get(design_id),
            rmsd_metrics=rmsd,
            plip_metrics=plip_metrics,
            colabfold_metrics=(colabfold_metrics or {}).get(design_id),
        ))
    return rows
