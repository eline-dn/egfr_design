"""Direct ColabDesign binder hallucination for paired human/mouse EGFR.

This module intentionally does not import or call BindCraft. It uses the
ColabDesign public model constructor plus the same small internal optimizer
hooks needed to combine gradients from multiple target batches.
"""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HallucinationConfig:
    binder_length: int = 70
    iterations_logits: int = 40
    iterations_soft: int = 30
    iterations_hard: int = 10
    learning_rate: float = 0.1
    num_recycles: int = 1
    num_models: int = 1
    sample_models: bool = True
    multi_grad_rms_norm: bool = True
    inter_contact_distance: float = 20.0
    inter_contact_number: int = 2
    intra_contact_distance: float = 14.0
    intra_contact_number: int = 2
    weights_plddt: float = 0.1
    weights_pae_intra: float = 0.4
    weights_pae_inter: float = 0.1
    weights_con_intra: float = 1.0
    weights_con_inter: float = 1.5
    weights_iptm: float = 0.1
    weights_rg: float = 0.3

    def validate(self) -> None:
        if not 65 <= self.binder_length <= 75:
            raise ValueError("binder_length must be between 65 and 75")
        if min(self.iterations_logits, self.iterations_soft, self.iterations_hard) < 0:
            raise ValueError("iteration counts cannot be negative")


def clear_colabdesign_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _make_model(af_params_dir: str | Path, config: HallucinationConfig) -> Any:
    try:
        from colabdesign import mk_af_model
    except ImportError as exc:
        raise RuntimeError("Install ColabDesign before running direct hallucination") from exc
    return mk_af_model(
        protocol="binder",
        debug=False,
        data_dir=str(af_params_dir),
        use_multimer=True,
        num_recycles=config.num_recycles,
        best_metric="loss",
    )


def _configure_losses(model: Any, config: HallucinationConfig) -> None:
    model.opt["weights"].update({
        "plddt": config.weights_plddt,
        "pae": config.weights_pae_intra,
        "i_pae": config.weights_pae_inter,
        "con": config.weights_con_intra,
        "i_con": config.weights_con_inter,
        "i_ptm": config.weights_iptm,
    })
    model.opt["con"].update({
        "num": config.intra_contact_number,
        "cutoff": config.intra_contact_distance,
        "binary": False,
        "seqsep": 9,
    })
    model.opt["i_con"].update({
        "num": config.inter_contact_number,
        "cutoff": config.inter_contact_distance,
        "binary": False,
    })
    model.opt["learning_rate"] = config.learning_rate
    model.opt["num_recycles"] = config.num_recycles
    model.opt["sample_models"] = config.sample_models
    model.opt["num_models"] = config.num_models


def _prepare_templates(
    model: Any,
    template_pdbs: Mapping[str, str | Path],
    binder_length: int,
    chain: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for name, pdb_path in template_pdbs.items():
        model.prep_inputs(
            pdb_filename=str(pdb_path),
            chain=chain,
            binder_len=binder_length,
            seed=seed,
            rm_aa="C",
        )
        states[name] = {
            "batch": copy.deepcopy(model._inputs["batch"]),
            "hotspot": copy.deepcopy(model.opt.get("hotspot")),
            "target_length": int(model._target_len),
        }
    if any(state["target_length"] > 150 for state in states.values()):
        raise ValueError("direct design templates must be trimmed to at most 150 target residues")
    target_lengths = {state["target_length"] for state in states.values()}
    if len(target_lengths) != 1:
        raise ValueError("all paired ColabDesign templates must have equal target lengths")
    return states


def _restore_template(model: Any, state: Mapping[str, Any]) -> None:
    model._inputs["batch"] = copy.deepcopy(state["batch"])
    if state.get("hotspot") is None:
        model.opt.pop("hotspot", None)
    else:
        model.opt["hotspot"] = copy.deepcopy(state["hotspot"])


def _aggregate_step(
    model: Any,
    states: Mapping[str, Mapping[str, Any]],
    config: HallucinationConfig,
    soft: float,
    hard: float,
    temperature: float,
    verbose: bool,
) -> dict[str, float]:
    import jax.numpy as jnp
    import numpy as np

    gradients = []
    losses = []
    for state in states.values():
        _restore_template(model, state)
        model.set_opt(soft=soft, hard=hard, temp=temperature, dropout=False)
        model.run(
            num_recycles=config.num_recycles,
            num_models=config.num_models,
            sample_models=config.sample_models,
            backprop=True,
        )
        gradients.append(copy.deepcopy(model.aux["grad"]["seq"]))
        losses.append(float(model.aux["log"]["loss"]))

    if config.multi_grad_rms_norm:
        normalized = []
        for gradient in gradients:
            rms = jnp.sqrt(jnp.mean(jnp.square(gradient))) + 1e-8
            normalized.append(gradient / rms)
        gradients = normalized

    running_gradient = gradients[0]
    for gradient in gradients[1:]:
        running_gradient = running_gradient + gradient
    model.aux["grad"]["seq"] = running_gradient / len(gradients)
    if model.opt.get("norm_seq_grad"):
        model._norm_seq_grad()
    model._state, model.aux["grad"] = model._optimizer(model._state, model.aux["grad"], model._params)
    learning_rate = model.opt["learning_rate"]
    model._params = jax_tree_map(lambda value, gradient: value - learning_rate * gradient, model._params, model.aux["grad"])
    model._k += 1
    if verbose:
        print({"step": model._k, "mean_loss": float(np.mean(losses)), "template_losses": losses})
    return {"mean_loss": float(np.mean(losses)), **{f"loss_{i}": loss for i, loss in enumerate(losses)}}


def jax_tree_map(function: Any, tree: Any, *trees: Any) -> Any:
    import jax
    return jax.tree_util.tree_map(function, tree, *trees)


def _run_stage(
    model: Any,
    states: Mapping[str, Mapping[str, Any]],
    config: HallucinationConfig,
    iterations: int,
    soft: float,
    hard: float,
    start_temperature: float,
    end_temperature: float,
    verbose: bool,
) -> None:
    if iterations == 0:
        return
    for index in range(iterations):
        fraction = (index + 1) / iterations
        temperature = start_temperature + (end_temperature - start_temperature) * fraction
        _aggregate_step(model, states, config, soft, hard, temperature, verbose)


def generate_shared_binder(
    template_pdbs: Mapping[str, str | Path],
    output_dir: str | Path,
    af_params_dir: str | Path,
    config: HallucinationConfig = HallucinationConfig(),
    chain: str = "A",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Hallucinate one binder jointly against all supplied target templates.

    Returns the shared sequence, per-target trajectory PDB paths, and final
    ColabDesign metrics. The generated PDBs are the direct input to MPNN.
    """
    config.validate()
    if set(template_pdbs) != {"human", "mouse"}:
        raise ValueError("the paired campaign requires exactly human and mouse templates")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clear_colabdesign_memory()
    model = _make_model(af_params_dir, config)
    states = _prepare_templates(model, template_pdbs, config.binder_length, chain, seed)
    first_state = next(iter(states.values()))
    _restore_template(model, first_state)
    _configure_losses(model, config)
    model.restart(seed=seed, reset_opt=False)

    _run_stage(model, states, config, config.iterations_logits, 0.0, 0.0, 1.0, 1.0, verbose)
    _run_stage(model, states, config, config.iterations_soft, 1.0, 0.0, 1.0, 0.01, verbose)
    _run_stage(model, states, config, config.iterations_hard, 1.0, 1.0, 0.01, 0.01, verbose)

    sequence = model.get_seq(get_best=True)[0]
    metrics: dict[str, Any] = {"sequence": sequence, "binder_length": len(sequence), "seed": seed}
    trajectory_paths: dict[str, str] = {}
    for name, state in states.items():
        _restore_template(model, state)
        model.set_seq(sequence)
        model.predict(seq=sequence, models=[0], num_recycles=config.num_recycles, verbose=False)
        path = output / f"shared_binder_{name}.pdb"
        model.save_pdb(str(path))
        trajectory_paths[name] = str(path)
        metrics[f"{name}_plddt"] = float(model.aux["log"].get("plddt", 0.0))
        metrics[f"{name}_iptm"] = float(model.aux["log"].get("i_ptm", 0.0))
    metrics["trajectory_pdbs"] = trajectory_paths
    metrics["colabdesign_model"] = "mk_af_model(protocol='binder', use_multimer=True)"
    del model
    clear_colabdesign_memory()
    return metrics
