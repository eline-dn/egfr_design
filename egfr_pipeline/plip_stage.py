"""PLIP XML parsing for protein-protein interface interaction metrics."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

INTERACTION_CATEGORIES = {
    "hydrogen_bonds": "hbond",
    "hydrophobic_interactions": "hydrophobic",
    "salt_bridges": "salt_bridge",
    "pi_stacks": "pi_stack",
    "pi_cation_interactions": "pi_cation",
    "halogen_bonds": "halogen_bond",
    "water_bridges": "water_bridge",
    "metal_complexes": "metal_complex",
}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _chains_in_subtree(element: ET.Element) -> set[str]:
    chains: set[str] = set()
    for node in element.iter():
        for key, value in node.attrib.items():
            if "chain" in key.lower() and value.strip():
                chains.add(value.strip())
        if "chain" in _normalise(node.tag):
            value = (node.text or "").strip()
            if value:
                chains.add(value)
    return chains


def parse_plip_xml(
    xml_path: str | Path,
    target_chain: str = "A",
    binder_chain: str = "B",
) -> dict[str, Any]:
    """Count PLIP interaction records spanning target and binder chains.

    PLIP XML layouts vary slightly by version. The parser therefore identifies
    the documented interaction category containers and inspects chain labels
    anywhere in each interaction record. Records without explicit chain labels
    are reported as unassigned rather than silently attributed to the interface.
    """
    root = ET.parse(xml_path).getroot()
    counts = Counter()
    unassigned = Counter()
    category_nodes = {
        _normalise(element.tag): element
        for element in root.iter()
        if _normalise(element.tag) in {_normalise(key) for key in INTERACTION_CATEGORIES}
    }
    for category_tag, category_element in category_nodes.items():
        interaction_type = next(
            value for key, value in INTERACTION_CATEGORIES.items() if _normalise(key) == category_tag
        )
        records = list(category_element)
        for record in records:
            if not list(record) and not (record.text or "").strip():
                continue
            chains = _chains_in_subtree(record)
            if target_chain in chains and binder_chain in chains:
                counts[interaction_type] += 1
            elif not chains:
                unassigned[interaction_type] += 1
    result: dict[str, Any] = {
        "plip_xml": str(xml_path),
        "plip_total_interactions": sum(counts.values()),
        "plip_unassigned_records": sum(unassigned.values()),
    }
    result.update({f"plip_{name}_count": counts.get(name, 0) for name in INTERACTION_CATEGORIES.values()})
    result.update({f"plip_unassigned_{name}_count": unassigned.get(name, 0) for name in INTERACTION_CATEGORIES.values()})
    return result


def parse_plip_targets(
    xml_paths: dict[str, str | Path],
    target_chain: str = "A",
    binder_chain: str = "B",
) -> dict[str, dict[str, Any]]:
    """Parse one PLIP XML file per target, keyed by target label."""
    return {
        target: parse_plip_xml(path, target_chain=target_chain, binder_chain=binder_chain)
        for target, path in xml_paths.items()
    }


def run_plip_xml(
    pdb_path: str | Path,
    output_dir: str | Path,
    plip_executable: str = "plip",
) -> Path:
    """Run PLIP with XML output and return the generated XML path."""
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [plip_executable, "-f", str(pdb_path), "-o", str(output_dir), "-x"],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = sorted(output_dir.rglob("*.xml"))
    if not candidates:
        raise FileNotFoundError(f"PLIP produced no XML file in {output_dir}")
    return candidates[0]
