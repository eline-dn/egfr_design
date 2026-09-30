# EGFR Human/Mouse Minibinder Pipeline

## Purpose

This project prepares a low-cost Google Colab workflow for the EGFR contest. The immediate campaign is a shared human/mouse EGFR minibinder:

- target: EGFR extracellular domain III;
- binder length: 65-75 residues in the first campaign;
- human and mouse templates: both treated as positive targets;
- pH optimization: intentionally deferred;
- structure relaxation: intentionally disabled.

The contest itself ranks pH selectivity first, mouse cross-reactivity second, and human affinity third. The current implementation is therefore an infrastructure and cross-reactivity scaffold, not yet a pH-selective design system.

## Files

- [egfr_human_mouse_pipeline.ipynb](egfr_human_mouse_pipeline.ipynb): interactive Colab notebook.
- [egfr_pipeline/target_prep.py](egfr_pipeline/target_prep.py): target definitions, downloads, sequence windows, and PDB trimming.
- [egfr_pipeline/bindcraft_stage.py](egfr_pipeline/bindcraft_stage.py): GPU checks, memory cleanup, and multi-template settings generation.
- [egfr_pipeline/mpnn_stage.py](egfr_pipeline/mpnn_stage.py): auditable ProteinMPNN positional constraints.
- [egfr_pipeline/esmfold_stage.py](egfr_pipeline/esmfold_stage.py): monomer-first ESMFold screening and human/mouse complex screening.
- [egfr_pipeline/structural_checks.py](egfr_pipeline/structural_checks.py): contacts, clashes, and target-aligned binder RMSD.
- [egfr_pipeline/plip_stage.py](egfr_pipeline/plip_stage.py): PLIP XML execution and interaction parsing.
- [egfr_pipeline/colabfold_stage.py](egfr_pipeline/colabfold_stage.py): ColabFold score JSON parsing, including ipTM, ipSAE, pDockQ, and pDockQ2.
- [egfr_pipeline/results.py](egfr_pipeline/results.py): one-row-per-binder metric CSV construction.

## Step-by-step workflow

### 1. Start a Colab GPU runtime

The notebook installs lightweight dependencies and checks CUDA. BindCraft, ColabDesign, ESMFold, and their weights are not all installed by the local helper package. The first Colab run should be a smoke test with only a few trajectories.

The repository code must be available under `/content/egfr_design/week1`, or `REPO_ROOT` in the notebook must be changed to the uploaded repository location.

### 2. Prepare the EGFR targets

The target helper stores the supplied human and mouse ectodomain sequences and downloads:

- human EGFR: PDB 6ARU;
- mouse EGFR: AlphaFold DB model AF-Q01279-F1.

The design window is intended to be 150 residues around domain III. The current notebook uses explicit PDB ranges `345-494` for both structures. These ranges must be inspected in the downloaded files before a production run. AlphaFold DB author numbering may not match UniProt numbering, and the current code does not perform an automatic sequence-to-structure alignment.

### 3. Generate the multi-template design configuration

`build_bindcraft_settings()` creates one human template and one mouse template, both with `target` strategy. Conceptually, this asks a BindCraft/ColabDesign multi-template run to optimize one binder sequence against both structures.

The intended binder range is 65-75 residues, within the contest's allowed 40-90 residue range.

### 4. Run backbone/interface generation

The intended design engine is the referenced BindCraft multi-template notebook. It uses ColabDesign/AlphaFold-style differentiable hallucination and ProteinMPNN, rather than RFdiffusion.

The current repository writes configuration but does not launch the BindCraft engine. A Colab adapter still needs to connect the generated settings to the installed BindCraft notebook or its Python entry point. Relaxation should remain disabled for this campaign.

The first run should generate 2-5 trajectories and verify:

- both templates load;
- target chain A exists;
- binder chain B is produced;
- output PDBs and CSVs are written;
- memory is released between model stages.

### 5. ProteinMPNN redesign

ProteinMPNN positional controls are represented by `PositionConstraint`:

```python
from egfr_pipeline.mpnn_stage import PositionConstraint, make_histidine_bias

constraints = [
    make_histidine_bias(position=12, strength=1.5),
    PositionConstraint(position=28, allowed_aas="DEH"),
    PositionConstraint(position=41, fixed=True),
]
```

Positions are **1-based and chain-local on binder chain B**. They are not PDB residue numbers and not zero-based Python indices.

The controls have different strengths:

- `fixed=True`: preserve the backbone residue; strongest constraint.
- `allowed_aas="DEH"`: permit only the listed residues.
- `bias={"H": 1.5}`: increase histidine sampling probability without forcing histidine.

Soft histidine bias is the appropriate starting point for later pH exploration. It is not a pKa model and does not establish pH-dependent affinity.

The current helper creates validated option/JSON records. It does not yet invoke every ProteinMPNN wrapper variant directly because ColabDesign and standalone ProteinMPNN versions expose different argument/file formats.

### 6. Monomer ESMFold filter

After MPNN sequences are exported, provide records of the form:

```python
[
    {"design_id": "d001", "sequence": "..."},
]
```

`screen_monomers()` predicts each binder alone and extracts pLDDT from CA-atom B-factors. Designs with binder pLDDT below 75 are removed before complex screening.

The model is loaded with `esmfold_v1`, half precision on CUDA, and an optional chunk size. GPU cleanup is available through `clear_esmfold_memory()`.

### 7. Human and mouse complex ESMFold screening

Every monomer-passing sequence is predicted against both trimmed targets:

```python
trimmed_targets = {
    "human": human_trimmed_sequence,
    "mouse": mouse_trimmed_sequence,
}
complex_results = screen_complexes_by_target(
    esm_model,
    monomer_pass,
    trimmed_targets,
    output_dir,
    separator_mode="colon",
)
```

The primary format is `binder:target`. A 25-glycine fallback is available for wrappers that reject colons, but it is not a true chain break and should be treated as a weaker heuristic.

The cross-reactivity assessment must consider both target-specific results. A binder that performs well against human but poorly against mouse is not a successful cross-reactive design.

### 8. PLIP interaction analysis

For every human and mouse complex PDB, run PLIP with XML output:

```python
from egfr_pipeline.plip_stage import run_plip_xml, parse_plip_targets

human_xml = run_plip_xml(human_complex_pdb, output_dir / "human")
mouse_xml = run_plip_xml(mouse_complex_pdb, output_dir / "mouse")
plip = parse_plip_targets({"human": human_xml, "mouse": mouse_xml})
```

The parser reports counts for:

- hydrogen bonds;
- hydrophobic interactions;
- salt bridges / ionic interactions;
- pi stacking;
- pi-cation interactions;
- halogen bonds;
- water bridges;
- metal complexes.

Only records whose XML explicitly identifies both target chain A and binder chain B are counted as interface interactions. Records without usable chain labels are retained as `plip_unassigned_*` diagnostics.

PLIP interaction counts describe a predicted contact pattern. They are not affinity measurements and should not be used as a direct replacement for experimental binding data.

### 9. ESMFold complex RMSD

The initial ColabFold/BindCraft complex is the reference. For each target, `aligned_binder_rmsd()`:

1. matches shared CA residues on target chain A;
2. fits the ESMFold complex to the reference using those target coordinates;
3. calculates CA RMSD over shared binder-chain B residues.

The reference and predicted structures must use compatible residue numbering, chain IDs, and binder residue numbering. Missing or inconsistent residues produce a missing RMSD rather than a fabricated score.

### 10. Final metric CSV

`assemble_binder_metric_rows()` and `write_binder_metrics_csv()` produce one row per binder. The row contains at least:

- `design_id`;
- `sequence` and sequence length;
- BindCraft pLDDT, pTM, ipTM, PAE, interface metrics when present;
- early contacts, clashes, and pass/fail flags;
- ProteinMPNN score/recovery or constraint metadata when supplied;
- monomer ESMFold pLDDT/pTM;
- human and mouse complex ESMFold metrics;
- human and mouse target-aligned binder RMSD and whole-complex RMSD;
- human and mouse PLIP interaction counts.
- human and mouse ColabFold `ipTM`, `ipSAE`, `pDockQ`, and `pDockQ2` when score JSON files are supplied.

Human and mouse fields are prefixed separately, for example:

```text
human_plip_hbond_count
mouse_plip_hbond_count
esmfold_human_binder_plddt
esmfold_mouse_binder_plddt
esmfold_human_aligned_binder_rmsd
esmfold_mouse_aligned_binder_rmsd
```

The notebook writes the intended result to:

```text
outputs/egfr_human_mouse_stage1/binder_metrics.csv
```

## Comparison with the original goals

| Original goal | Current status | Assessment |
|---|---|---|
| Design human EGFR binders | Partial | Human template/configuration and validation plumbing exist; no production campaign has run. |
| Design human/mouse cross-reactive binders | Partial | Multi-template configuration and dual-target ESMFold/PLIP analysis exist; the BindCraft engine invocation is not connected. |
| Target functional domain III | Partial | Domain-III-centered sequence window is planned, but structure numbering and epitope geometry require manual verification. |
| Binder length 40-90 aa | Implemented for first campaign | The first settings restrict designs to 65-75 aa. |
| RFdiffusion backbone generation | Not implemented | The code uses BindCraft/ColabDesign-style configuration; no RFdiffusion model or runner is included. |
| ProteinMPNN redesign | Partial | Constraint representation and audit records exist; actual wrapper execution is not yet unified across installations. |
| Monomer ESMFold filtering | Implemented as helper/notebook hook | Requires Colab GPU, installed ESMFold, and MPNN output files. |
| Human and mouse complex validation | Implemented as helper/notebook hook | Runs both targets, but has not been executed in this workspace. |
| Binder pLDDT filter below 75 | Implemented | Applied in `screen_monomers()`. |
| pTM extraction | Partial | Captured only when the installed fair-esm API exposes it. |
| PLIP interaction metrics | Implemented as parser/runner | Requires valid PLIP XML and correct chain annotations. |
| Relaxation-free workflow | Implemented by design | PyRosetta relaxation is not used in the current stage. |
| pH-selective binding | Not implemented | Only future histidine bias controls exist; no pH-conditioned structural or affinity objective exists. |
| Final AlphaFold Server validation | Not implemented | The pipeline can prepare candidates but does not automate AlphaFold Server submission. |

## Limitations and criticism

### The most important limitation: this is not yet a complete generator

The repository currently provides configuration, adapters, metrics, and notebook hooks. It has not produced a validated set of EGFR binders. The notebook summary shows that no cells have been executed. A successful Python compile or mock test only proves plumbing, not biological usefulness.

### RFdiffusion requirement is unmet

The original tools document explicitly asks for RFdiffusion backbone generation. The current approach instead assumes BindCraft/ColabDesign hallucination. That is a reasonable low-cost alternative, but it is a different algorithm and should be reported as such. No RFdiffusion checkpoint, inference call, or backbone sampling code is present.

### BindCraft integration is incomplete

The generated settings are not sufficient evidence that the original `Multi_BC_Original.ipynb` will run unchanged. BindCraft expects particular settings keys, weights, binaries, AlphaFold parameters, and PyRosetta-related components. The current notebook does not yet launch its engine or prove compatibility with the current Colab environment.

### Target trimming is fragile

The current `345-494` range is explicit but not automatically derived from the PDB structure. The human PDB and mouse AlphaFold model may use different author numbering. A wrong range could trim the wrong region while still producing a syntactically valid PDB. Sequence-to-structure alignment and a structural domain boundary check are needed before production design.

### ESMFold complex prediction is not a reliable affinity predictor

ESMFold can provide a fast structural plausibility screen, but its complex interface confidence is not equivalent to AlphaFold3 ipTM/PAE or experimental affinity. The glycine-linker fallback further weakens interpretation because it changes the physical problem. Complex ESMFold results should be used to reduce the candidate set, not to claim human or mouse binding.

### pLDDT extraction has assumptions

The current parser assumes the binder residues occur first in the returned structure when calculating binder pLDDT. This must be checked against the actual fair-esm output and chain IDs. A production implementation should identify binder residues by chain and residue mapping, not only by the first `N` CA records.

### PLIP depends on chain-aware input and schema details

PLIP XML formats vary by version and by whether the input is treated as a protein-protein complex or ligand interaction. If chain labels are absent or rewritten, the parser reports unassigned interactions. The parser is intentionally conservative, but a zero count can mean “no interaction” or “chain labels were not recoverable.” XML fixtures from the exact Colab PLIP version should be added.

### RMSD is only meaningful under consistent numbering

Target-aligned RMSD requires shared residue numbering and chain IDs. It can be misleading if the two predictions have missing loops, shifted numbering, different chain assignments, or a linker representation that changes residue indexing. The CSV should retain the number of atoms used in each fit in a future refinement.

### No real batching yet

The code performs screening in Python loops. It releases CUDA memory, but it does not yet batch ESMFold sequences, resume individual completed predictions, or cap memory based on measured sequence length. This will be slow for hundreds of trajectories and may still exceed free Colab limits.

### Metrics can be mistaken for evidence

A high pLDDT, many PLIP contacts, favorable ipTM, or low RMSD does not establish affinity, specificity, pH selectivity, expression, or experimental stability. The final CSV is a triage table, not a ranking oracle.

### pH selectivity remains the largest scientific gap

The contest's primary ranking criterion is binding at pH 6.5 but not pH 7.4. Histidine biasing is only a design heuristic. The current pipeline has no protonation-aware energy model, constant-pH simulation, pH-conditioned predictor, or negative-design objective at pH 7.4. Experimental measurements remain essential, and pH candidates should be diversified rather than selected from one computational score.

## Recommended next work

1. Validate human and mouse PDB numbering and automate sequence-to-structure mapping.
2. Run a 2-5 trajectory BindCraft smoke test in Colab.
3. Connect actual ProteinMPNN sampling and save per-sequence score/recovery fields.
4. Execute monomer and both complex ESMFold screens on a small batch.
5. Generate PLIP XML fixtures and verify every interaction category against the installed PLIP version.
6. Produce the final one-row-per-binder CSV and inspect missing metrics.
7. Add independent full-ectodomain validation before selecting candidates.
8. Develop a separate pH-candidate campaign with multiple histidine placements and experimental prioritization.
9. Send approximately the top 30 diverse candidates to AlphaFold Server for ipTM/PAE review and experimental testing.
