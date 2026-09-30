"""Read ColabFold complex score files, including ipSAE metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_METRIC_KEYS = (
    "ptm",
    "iptm",
    "ipsae",
    "pdockq",
    "pdockq2",
    "plddt",
    "max_pae",
)


def _find_metric(payload: Mapping[str, Any], name: str) -> Any:
    """Read a metric with tolerant casing and common nested score layouts."""
    wanted = name.lower()
    for key, value in payload.items():
        if str(key).lower() == wanted:
            return value
    for key in ("scores", "summary", "confidence", "ranking_scores"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            value = _find_metric(nested, name)
            if value is not None:
                return value
    return None


def _interface_value(value: Any, chain_pair: str) -> Any:
    """Select one chain-pair value while preserving the full mapping if needed."""
    if not isinstance(value, Mapping):
        return value
    if chain_pair in value:
        return value[chain_pair]
    reverse = "-".join(reversed(chain_pair.split("-")))
    if reverse in value:
        return value[reverse]
    return value


def parse_colabfold_scores(
    scores_json: str | Path,
    target: str,
    chain_pair: str = "A-B",
) -> dict[str, Any]:
    """Extract target-specific ColabFold metrics from a saved score JSON.

    Current ColabFold complex outputs can contain ``iptm`` plus interface
    scores ``ipsae``, ``pdockq``, and ``pdockq2``. Values are copied rather
    than recomputed. Pair dictionaries are reduced to the requested target-
    binder pair and the complete dictionaries are retained for auditability.
    """
    path = Path(scores_json)
    payload = json.loads(path.read_text())
    result: dict[str, Any] = {
        "colabfold_scores_json": str(path),
        "colabfold_target": target,
        "colabfold_chain_pair": chain_pair,
    }
    for metric in _METRIC_KEYS:
        value = _find_metric(payload, metric)
        if value is None:
            continue
        selected = _interface_value(value, chain_pair)
        result[f"{metric}"] = selected
        if isinstance(value, Mapping):
            result[f"{metric}_all_interfaces"] = dict(value)
    return result


def parse_colabfold_targets(
    score_jsons: Mapping[str, str | Path],
    chain_pair: str = "A-B",
) -> dict[str, dict[str, Any]]:
    """Parse one ColabFold score JSON per target label."""
    return {
        target: parse_colabfold_scores(path, target, chain_pair=chain_pair)
        for target, path in score_jsons.items()
    }
