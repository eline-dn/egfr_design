"""Colab-facing configuration and smoke checks for the first design stage."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any


def clear_gpu_memory() -> None:
    """Release Python and CUDA allocations between model stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def gpu_report() -> dict[str, Any]:
    """Report the runtime without importing BindCraft or JAX eagerly."""
    report: dict[str, Any] = {"cuda_available": False, "device": "cpu"}
    try:
        import torch
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["device"] = torch.cuda.get_device_name(0)
            report["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except ImportError:
        report["torch_installed"] = False
    return report


def build_bindcraft_settings(
    output_dir: str | Path,
    human_pdb: str | Path,
    mouse_pdb: str | Path,
    binder_name: str = "egfr_human_mouse",
    binder_lengths: tuple[int, int] = (65, 75),
    final_designs: int = 30,
) -> dict[str, Any]:
    """Build a multi-template target configuration for BindCraft.

    This configuration intentionally contains no relaxation instruction. The
    actual BindCraft engine remains an optional later execution step because
    its standard post-processing uses PyRosetta relaxation.
    """
    if not 40 <= binder_lengths[0] <= binder_lengths[1] <= 90:
        raise ValueError("binder lengths must remain within the contest's 40-90 aa range")
    if binder_lengths[0] < 65 or binder_lengths[1] > 75:
        raise ValueError("this first-stage campaign is restricted to 65-75 aa")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    default_filters = {
        "Average_pLDDT": {"threshold": 0.70, "higher": True},
        "Average_i_pTM": {"threshold": 0.20, "higher": True},
        "Average_Relaxed_Clashes": {"threshold": 0, "higher": False},
        "Average_n_InterfaceResidues": {"threshold": 7, "higher": True},
    }
    settings = {
        "general_information": {
            "design_path": str(output),
            "binder_name": binder_name,
            "chains": "A",
            "binder_chain": "B",
            "lengths": list(binder_lengths),
            "number_of_final_designs": final_designs,
        },
        "template_1_information": {
            "template_name": "human_egfr",
            "template_pdb": str(Path(human_pdb)),
            "template_hostspot_residues": "",
            "template_targeting_strategy": "target",
            "template_mpnn_filters": default_filters,
            "loss_weighting": "1",
        },
        "template_2_information": {
            "template_name": "mouse_egfr",
            "template_pdb": str(Path(mouse_pdb)),
            "template_hostspot_residues": "",
            "template_targeting_strategy": "target",
            "template_mpnn_filters": default_filters,
            "loss_weighting": "1",
        },
        "pipeline_notes": {
            "relaxation": "disabled_in_this_stage",
            "validation": "early_structure_checks_only",
            "pH_optimization": "deferred",
        },
    }
    return settings


def write_json_settings(settings: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(settings, indent=2) + "\n")
    return destination


def require_gpu() -> None:
    report = gpu_report()
    if not report.get("cuda_available"):
        raise RuntimeError("A CUDA GPU is required for the BindCraft/ColabDesign stage; select a Colab GPU runtime.")


def bindcraft_import_check() -> dict[str, str]:
    """Check optional imports and return versions without starting a design."""
    versions: dict[str, str] = {}
    for module_name in ("jax", "colabdesign", "Bio"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "available"))
        except ImportError:
            versions[module_name] = "missing"
    return versions
