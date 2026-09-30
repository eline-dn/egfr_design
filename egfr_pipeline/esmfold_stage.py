"""Memory-conscious ESMFold monomer and complex screening helpers."""

from __future__ import annotations

import gc
import csv
from pathlib import Path
from typing import Any

from .mpnn_stage import validate_sequence


def clear_esmfold_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def complex_sequence(binder_sequence: str, target_sequence: str, mode: str = "colon") -> str:
    """Format a two-chain ESMFold input.

    ``colon`` is the requested modern chain-break representation. ``glycine``
    is retained as an explicit compatibility fallback for older wrappers that
    reject colons; its output is not a true chain break and must be treated as
    a heuristic screen only.
    """
    binder = validate_sequence(binder_sequence)
    target = validate_sequence(target_sequence)
    if mode == "colon":
        return f"{binder}:{target}"
    if mode == "glycine":
        return f"{binder}{'G' * 25}{target}"
    raise ValueError("mode must be 'colon' or 'glycine'")


def load_esmfold(chunk_size: int | None = 64):
    """Load fair-esm ESMFold in half precision on CUDA when available."""
    import esm

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    try:
        import torch
        if torch.cuda.is_available():
            model = model.cuda().half()
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for ESMFold") from exc
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
    return model


def _plddt_from_pdb(pdb_text: str, binder_length: int | None = None) -> tuple[float, float | None]:
    """Extract mean all-residue and binder-residue pLDDT from PDB B-factors."""
    values: list[float] = []
    binder_values: list[float] = []
    seen_ca: set[tuple[str, int]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        try:
            residue_key = (line[21], int(line[22:26]))
            value = float(line[60:66])
        except (IndexError, ValueError):
            continue
        if residue_key in seen_ca:
            continue
        seen_ca.add(residue_key)
        values.append(value)
        if binder_length is not None and len(binder_values) < binder_length:
            binder_values.append(value)
    if not values:
        raise ValueError("ESMFold PDB contained no readable CA pLDDT values")
    return sum(values) / len(values), (sum(binder_values) / len(binder_values) if binder_values else None)


def predict_sequence(model: Any, sequence: str, output_pdb: str | Path, binder_length: int | None = None) -> dict[str, Any]:
    """Predict one sequence, save its PDB, and return pLDDT metrics."""
    ptm = None
    if hasattr(model, "infer") and hasattr(model, "output_to_pdb"):
        output = model.infer(sequence)
        pdb_text = model.output_to_pdb(output)[0]
        raw_ptm = output.get("ptm") if hasattr(output, "get") else None
        if raw_ptm is not None:
            try:
                ptm = float(raw_ptm.detach().cpu().item())
            except AttributeError:
                try:
                    ptm = float(raw_ptm)
                except (TypeError, ValueError):
                    ptm = None
    else:
        pdb_text = model.infer_pdb(sequence)
    output = Path(output_pdb)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pdb_text)
    mean_plddt, binder_plddt = _plddt_from_pdb(pdb_text, binder_length)
    return {
        "pdb": str(output),
        "mean_plddt": round(mean_plddt, 3),
        "binder_plddt": round(binder_plddt, 3) if binder_plddt is not None else None,
        "ptm": round(ptm, 3) if ptm is not None else None,
        "sequence": sequence,
    }


def screen_monomers(model: Any, designs: list[dict[str, Any]], output_dir: str | Path, minimum_binder_plddt: float = 75.0) -> list[dict[str, Any]]:
    """Run the cheap binder-only screen and retain passing designs."""
    output = Path(output_dir)
    passing: list[dict[str, Any]] = []
    for design in designs:
        sequence = validate_sequence(str(design["sequence"]))
        result = predict_sequence(model, sequence, output / f"{design['design_id']}_monomer.pdb", len(sequence))
        result.update({"design_id": design["design_id"], "passes_monomer": (result["binder_plddt"] or 0) >= minimum_binder_plddt})
        if result["passes_monomer"]:
            passing.append({**design, **result})
    return passing


def screen_complexes(model: Any, designs: list[dict[str, Any]], target_sequence: str, output_dir: str | Path, separator_mode: str = "colon") -> list[dict[str, Any]]:
    """Repredict monomer-passing designs against a trimmed target."""
    output = Path(output_dir)
    results: list[dict[str, Any]] = []
    for design in designs:
        sequence = complex_sequence(str(design["sequence"]), target_sequence, separator_mode)
        result = predict_sequence(model, sequence, output / f"{design['design_id']}_complex.pdb", len(str(design["sequence"])))
        results.append({**design, **result, "separator_mode": separator_mode})
    return results


def screen_complexes_by_target(
    model: Any,
    designs: list[dict[str, Any]],
    targets: dict[str, str],
    output_dir: str | Path,
    separator_mode: str = "colon",
) -> list[dict[str, Any]]:
    """Run complex screening for every target and annotate each result.

    ``targets`` maps a stable target label, such as ``human`` or ``mouse``,
    to the corresponding trimmed amino-acid sequence. Each target receives a
    separate output directory, preventing PDB filename collisions.
    """
    output = Path(output_dir)
    results: list[dict[str, Any]] = []
    for target_name, target_sequence in targets.items():
        target_results = screen_complexes(
            model,
            designs,
            target_sequence,
            output / target_name,
            separator_mode=separator_mode,
        )
        results.extend({**result, "target": target_name} for result in target_results)
    return results


def write_screening_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    """Write resumable screening results with a stable union of columns."""
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["design_id", "sequence", "passes_monomer", "binder_plddt", "ptm"]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination
