# Practical Google Colab Guide

This guide runs the current EGFR human/mouse minibinder workflow with Google Colab and Google Drive persistence.

The guide is intentionally honest about the current state: the repository contains the target preparation, ProteinMPNN constraint, ESMFold, PLIP, RMSD, and CSV helpers, but the upstream BindCraft design engine is not yet launched automatically by the notebook. The BindCraft smoke test is therefore a required handoff step.

## 0. What this run is trying to produce

The first campaign is designed for the human/mouse cross-reactivity objective:

- human EGFR extracellular domain III target;
- mouse EGFR extracellular domain III target;
- one shared binder sequence optimized against both templates;
- binder length 65-75 aa;
- no PyRosetta relaxation;
- pH optimization deferred for the baseline campaign;
- monomer ESMFold pLDDT filter at 75;
- ESMFold complex prediction against both human and mouse trimmed targets;
- PLIP interaction counts for both complexes;
- one final row per binder in `binder_metrics.csv`.

This is not yet a proof of affinity or pH selectivity. Those require full ectodomain and experimental testing.

## 1. Create a Colab notebook and select a GPU

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new Python notebook.
3. Select `Runtime -> Change runtime type`.
4. Choose `T4 GPU` or another available CUDA GPU.
5. Run:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

If this prints `False`, stop and change the runtime before loading ESMFold or BindCraft.

## 2. Mount Google Drive

Drive prevents the campaign outputs from disappearing when Colab restarts.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Create a persistent project directory:

```python
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/egfr_design')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

for name in [
    'raw_targets',
    'trimmed_targets',
    'outputs',
    'settings',
    'esmfold',
    'plip',
    'logs',
]:
    (DRIVE_ROOT / name).mkdir(parents=True, exist_ok=True)

print(DRIVE_ROOT)
```

Keep model weights and large outputs on Drive only if performance is acceptable. Drive I/O is slower than local Colab storage. A practical compromise is to keep the repository and active model files in `/content`, then copy final PDBs/CSVs to Drive.

## 3. Clone the repository

Replace `REPO_URL` with the GitHub URL of your repository. The notebook currently expects the local checkout at `/content/egfr_design/week1`.

```python
REPO_URL = 'https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git'
REPO_DIR = Path('/content/egfr_design/week1')

if not REPO_DIR.exists():
    !git clone "$REPO_URL" "$REPO_DIR"

import sys
sys.path.insert(0, str(REPO_DIR))

print(REPO_DIR)
```

If the repository is private, authenticate with GitHub before cloning. Do not put a personal access token directly into a notebook cell that will be shared.

If you are testing the current local files without a GitHub remote, upload the `week1` folder to Drive and use:

```python
REPO_DIR = Path('/content/drive/MyDrive/egfr_design/week1')
sys.path.insert(0, str(REPO_DIR))
```

The package should contain:

```text
egfr_pipeline/
    target_prep.py
    bindcraft_stage.py
    mpnn_stage.py
    esmfold_stage.py
    structural_checks.py
    plip_stage.py
    results.py
egfr_human_mouse_pipeline.ipynb
problem_description.md
tools_restrictions.md
```

## 4. Install the lightweight Python packages

Run this in a notebook code cell:

```python
%pip -q install numpy biopython pandas tqdm 'fair-esm[esmfold]' plip
```

Restart the Colab runtime if the installation reports that PyTorch, OpenFold, or JAX was already imported with incompatible versions. After restarting, remount Drive and rerun the repository setup cell.

Check the project imports:

```python
from egfr_pipeline.target_prep import TARGETS, validate_target_specs
from egfr_pipeline.bindcraft_stage import gpu_report, bindcraft_import_check
from egfr_pipeline.mpnn_stage import PositionConstraint, make_histidine_bias
from egfr_pipeline.esmfold_stage import load_esmfold
from egfr_pipeline.plip_stage import parse_plip_xml, run_plip_xml
from egfr_pipeline.results import assemble_binder_metric_rows, write_binder_metrics_csv

validate_target_specs()
print(gpu_report())
print(bindcraft_import_check())
```

`colabdesign` may still show as missing at this point. That is expected until BindCraft/ColabDesign is installed.

## 5. Install BindCraft and its large dependencies

The current code writes a BindCraft configuration but does not replace the upstream BindCraft engine. Clone BindCraft separately:

```python
BINDCRAFT_DIR = Path('/content/BindCraft')
if not BINDCRAFT_DIR.exists():
    !git clone --depth 1 https://github.com/martinpacesa/BindCraft.git "$BINDCRAFT_DIR"
```

Follow the upstream installation instructions for the current Colab-compatible revision. The exact CUDA, conda, AlphaFold parameter, DSSP, DAlphaBall, and PyRosetta requirements can change. Do not assume that a local installation command from an older BindCraft commit still works unchanged.

The upstream workflow requires, at minimum:

- CUDA-compatible GPU;
- ColabDesign and JAX;
- AlphaFold parameter files, which are several GB;
- ProteinMPNN weights;
- DSSP and DAlphaBall binaries;
- PyRosetta for the standard BindCraft relaxation/interface-scoring path.

For this project, leave relaxation disabled where possible. PyRosetta is not required by the helper-only stages, but the original BindCraft pipeline may still import or require it during setup. Check its licensing terms before use.

After installation, verify the environment before a long run:

```python
import jax
print(jax.devices())
from colabdesign.mpnn import mk_mpnn_model
print('ColabDesign ProteinMPNN import succeeded')
```

If `jax.devices()` lists only CPU devices, do not start the design campaign.

## 6. Run the project notebook in order

Open:

```text
egfr_human_mouse_pipeline.ipynb
```

Run the notebook sections in this order:

1. Install dependencies.
2. Import project helpers.
3. Define the paired human/mouse campaign.
4. Download and trim targets.
5. Write BindCraft multi-template settings.
6. Run early structural checks on generated PDBs.
7. Define ProteinMPNN constraints if needed.
8. Run monomer-first ESMFold.
9. Run human and mouse complex ESMFold.
10. Run PLIP and write the final metrics CSV.

Do not run the final metric cell before the design and validation files exist.

## 7. Download and inspect the target structures

The helper downloads:

- human EGFR: `6ARU.pdb`;
- mouse EGFR: `AF-Q01279-F1-model_v4.pdb`.

Run the target preparation section:

```python
from egfr_pipeline.target_prep import download_targets, trim_pdb_by_residue_range

raw_paths = download_targets(DRIVE_ROOT / 'raw_targets')
print(raw_paths)
```

The current notebook proposes PDB ranges `345-494` on chain `A` for both species. Inspect them before using them:

```python
for name, path in raw_paths.items():
    print(name, path)
    lines = path.read_text().splitlines()
    chains = sorted({line[21] for line in lines if line.startswith('ATOM  ')})
    residues = sorted({int(line[22:26]) for line in lines if line.startswith('ATOM  ')})
    print('chains:', chains, 'residue range:', (min(residues), max(residues)))
```

This check is essential. AlphaFold DB residue numbering may not match the UniProt numbering in the contest description. If the mouse model does not contain residues 345-494 in chain A, determine the correct domain-III range before trimming.

Then create the inputs:

```python
from egfr_pipeline.target_prep import TARGETS

trimmed_paths = {}
for name, target in TARGETS.items():
    trimmed_paths[name] = trim_pdb_by_residue_range(
        raw_paths[target.name],
        DRIVE_ROOT / 'trimmed_targets' / f'{name}_egfr_domain3.pdb',
        start_residue=345,
        end_residue=494,
        chain_id='A',
    )
    print(name, trimmed_paths[name])
```

Open the two trimmed PDBs in a viewer or inspect them in PyMOL/ChimeraX. Confirm that they show the same functional domain and that the target chain remains `A`.

## 8. Write the shared human/mouse BindCraft settings

```python
from egfr_pipeline.bindcraft_stage import build_bindcraft_settings, write_json_settings

settings = build_bindcraft_settings(
    output_dir=DRIVE_ROOT / 'outputs' / 'egfr_human_mouse_stage1',
    human_pdb=trimmed_paths['human'],
    mouse_pdb=trimmed_paths['mouse'],
    binder_name='egfr_human_mouse_stage1',
    binder_lengths=(65, 75),
    final_designs=30,
)

settings_path = write_json_settings(
    settings,
    DRIVE_ROOT / 'settings' / 'egfr_human_mouse_stage1.json',
)
print(settings_path)
```

The two templates are both marked `target`. This means the intended result is one sequence that works against both human and mouse EGFR, not separate species-specific sequences.

## 9. Run a BindCraft smoke test before production

Do not immediately request 30 accepted designs. Start with 2-5 trajectories.

Use the upstream BindCraft multi-template notebook or the corresponding Python entry point. Point it to:

- the generated settings JSON;
- the trimmed human PDB;
- the trimmed mouse PDB;
- the AlphaFold parameter directory;
- the ProteinMPNN weights;
- the output directory on Drive or local Colab storage.

Confirm that the smoke test creates:

```text
Trajectory/
Trajectory/Relaxed/       # may remain empty when relaxation is disabled
MPNN/
MPNN/Sequences/
Accepted/
Rejected/
*.csv
```

Inspect at least one generated complex:

```python
from egfr_pipeline.structural_checks import early_structure_report

report = early_structure_report(
    '/content/egfr_design/outputs/egfr_human_mouse_stage1/Trajectory/example.pdb',
    target_chain='A',
    binder_chain='B',
)
print(report)
```

Reject obvious structures with zero interface contacts or heavy-atom clashes. These are triage checks, not affinity predictions.

## 10. Export ProteinMPNN sequences

Collect the MPNN sequences into a Python list or CSV with at least:

```python
designs = [
    {'design_id': 'egfr_human_mouse_stage1_l70_s123', 'sequence': 'ACDEFG...'},
]
```

If using the positional controls, remember that positions are 1-based and local to binder chain B:

```python
from egfr_pipeline.mpnn_stage import PositionConstraint, make_histidine_bias

constraints = [
    make_histidine_bias(position=12, strength=1.5),
    PositionConstraint(position=28, allowed_aas='DEH'),
]
```

For the baseline cross-reactivity campaign, do not add pH constraints yet. Save the generated sequence and ProteinMPNN score/recovery for every candidate so they can be included in the final CSV.

## 11. Run monomer ESMFold filtering

Load ESMFold once:

```python
from egfr_pipeline.esmfold_stage import load_esmfold, screen_monomers, clear_esmfold_memory

esm_model = load_esmfold(chunk_size=64)
monomer_pass = screen_monomers(
    esm_model,
    designs,
    DRIVE_ROOT / 'esmfold' / 'monomers',
    minimum_binder_plddt=75.0,
)
print('monomer-passing designs:', len(monomer_pass))
clear_esmfold_memory()
```

Only monomer-passing candidates continue. A high monomer pLDDT means the binder is structurally plausible in isolation; it does not mean that it binds EGFR.

## 12. Run ESMFold complexes against both species

Use the same binder sequence for both targets:

```python
from egfr_pipeline.target_prep import domain_iii_window
from egfr_pipeline.esmfold_stage import screen_complexes_by_target

trimmed_target_sequences = {
    name: domain_iii_window(target, 345, 150)[2]
    for name, target in TARGETS.items()
}

complex_results = screen_complexes_by_target(
    esm_model,
    monomer_pass,
    trimmed_target_sequences,
    DRIVE_ROOT / 'esmfold' / 'complexes',
    separator_mode='colon',
)

from egfr_pipeline.esmfold_stage import write_screening_csv
write_screening_csv(
    complex_results,
    DRIVE_ROOT / 'esmfold' / 'complex_screening.csv',
)
clear_esmfold_memory()
```

There should be two complex rows per monomer-passing binder: one `human` and one `mouse`.

If the installed ESMFold wrapper rejects colon-separated chains, try `separator_mode='glycine'`. Treat the glycine-linker result as a weaker compatibility screen, not as an equivalent multimer prediction.

## 13. Generate PLIP XML for both complexes

For each binder and species, run PLIP in XML mode. The helper invokes the PLIP command line:

```python
from egfr_pipeline.plip_stage import run_plip_xml, parse_plip_targets

plip_xmls = {}
for design in monomer_pass:
    design_id = design['design_id']
    plip_xmls[design_id] = {}
    for target in ['human', 'mouse']:
        complex_pdb = DRIVE_ROOT / 'esmfold' / 'complexes' / target / f'{design_id}_complex.pdb'
        xml_path = run_plip_xml(
            complex_pdb,
            DRIVE_ROOT / 'plip' / design_id / target,
        )
        plip_xmls[design_id][target] = xml_path

print('PLIP XML generated')
```

PLIP interaction counts are separated by species and include hydrogen bonds, hydrophobic contacts, salt bridges, pi interactions, halogen bonds, water bridges, and metal complexes where chain labels are available.

## 14. Compare ESMFold complexes to initial BindCraft complexes

Create a map to the initial complex PDBs:

```python
reference_complexes = {
    'egfr_human_mouse_stage1_l70_s123': {
        'human': DRIVE_ROOT / 'outputs' / 'egfr_human_mouse_stage1' / 'initial_human.pdb',
        'mouse': DRIVE_ROOT / 'outputs' / 'egfr_human_mouse_stage1' / 'initial_mouse.pdb',
    },
}
```

The structures must use target chain A and binder chain B with compatible residue numbering. The RMSD calculation first aligns the predicted complex to the initial complex using target-chain CA atoms, then records both binder-chain CA RMSD and whole-complex CA RMSD.

## 15. Retrieve ColabFold interface scores

Use ColabFold for the more rigorous complex-validation branch. Retain the score JSON produced alongside each human and mouse prediction. Current ColabFold complex outputs can include `iptm`, `ipsae`, `pdockq`, and `pdockq2`. These are retrieved from the saved JSON; they are not calculated by ESMFold.

```python
from egfr_pipeline.colabfold_stage import parse_colabfold_targets

colabfold_score_jsons = {
    'egfr_human_mouse_stage1_l70_s123': {
        'human': DRIVE_ROOT / 'colabfold' / 'd001' / 'human_scores.json',
        'mouse': DRIVE_ROOT / 'colabfold' / 'd001' / 'mouse_scores.json',
    },
}

colabfold_metrics = {
    design_id: parse_colabfold_targets(score_paths, chain_pair='A-B')
    for design_id, score_paths in colabfold_score_jsons.items()
}
print(colabfold_metrics)
```

The parser selects the `A-B` target-binder pair when interface metrics are stored as chain-pair dictionaries. It also preserves the complete interface mapping under `*_all_interfaces` for auditability. Missing ipSAE remains blank; do not substitute ESMFold pTM for ipSAE.

## 16. Write the final one-row-per-binder CSV

Provide optional metric maps from BindCraft, structural checks, MPNN, and monomer ESMFold. The aggregator preserves missing values rather than deleting candidates:

```python
from egfr_pipeline.results import assemble_binder_metric_rows, write_binder_metrics_csv

final_rows = assemble_binder_metric_rows(
    designs=designs,
    bindcraft_metrics=bindcraft_metrics,
    structural_metrics=structural_metrics,
    mpnn_metrics=mpnn_metrics,
    monomer_metrics=monomer_metrics,
    complex_results=complex_results,
    reference_complexes=reference_complexes,
    plip_xmls=plip_xmls,
    colabfold_metrics=colabfold_metrics,
    target_chain='A',
    binder_chain='B',
)

final_csv = write_binder_metrics_csv(
    final_rows,
    DRIVE_ROOT / 'outputs' / 'egfr_human_mouse_stage1' / 'binder_metrics.csv',
)
print(final_csv)
```

Important columns include:

```text
design_id
sequence
sequence_length
esmfold_monomer_binder_plddt
esmfold_human_binder_plddt
esmfold_mouse_binder_plddt
esmfold_human_aligned_binder_rmsd
esmfold_mouse_aligned_binder_rmsd
esmfold_human_aligned_complex_rmsd
esmfold_mouse_aligned_complex_rmsd
human_plip_hbond_count
mouse_plip_hbond_count
human_plip_hydrophobic_count
mouse_plip_hydrophobic_count
colabfold_human_iptm
colabfold_human_ipsae
colabfold_human_pdockq
colabfold_human_pdockq2
colabfold_mouse_iptm
colabfold_mouse_ipsae
colabfold_mouse_pdockq
colabfold_mouse_pdockq2
```

Open the CSV with pandas:

```python
import pandas as pd

metrics = pd.read_csv(final_csv)
metrics.head()
```

For cross-reactivity, inspect the weaker human/mouse result rather than ranking by the human result alone.

## 17. Candidate selection and next validation

Use the CSV to select a diverse shortlist, not simply the top numerical rows. Prefer candidates with:

- binder pLDDT at least 75;
- plausible human and mouse complex predictions;
- no obvious clashes;
- meaningful contacts to both targets;
- reasonable aligned binder RMSD;
- non-pathological surface hydrophobicity;
- diverse sequences and interface geometries.

The computational pipeline cannot establish the contest's primary pH-selectivity objective. For the eventual pH campaign, create separate sequence families with controlled histidine placements and compare them experimentally at pH 6.5 and 7.4.

For final candidates:

1. Re-predict against the full human and mouse extracellular regions.
2. Submit approximately the top 30 diverse candidates to the official AlphaFold Server.
3. Inspect ipTM and PAE.
4. Order experimentally testable candidates only after checking synthesis, expression, and aggregation risks.
5. Measure human affinity, mouse cross-reactivity, and pH selectivity experimentally.

## Troubleshooting

### `ModuleNotFoundError: egfr_pipeline`

The repository path is not on `sys.path`.

```python
import sys
sys.path.insert(0, '/content/egfr_design/week1')
```

### CUDA is unavailable

Reconnect or change the runtime to GPU. Do not proceed with ESMFold or BindCraft on CPU.

### Out-of-memory errors

- use 65 aa before 75 aa;
- lower ESMFold chunk size;
- process one sequence at a time;
- call `clear_gpu_memory()` and `clear_esmfold_memory()` between stages;
- keep the target window at or below 150 residues;
- restart the runtime after a failed large model load.

### No PLIP interactions are counted

Check that:

- the XML was generated for the intended complex;
- the complex contains target chain A and binder chain B;
- the PLIP version writes chain labels into the XML;
- the XML path is passed under the correct design ID and target name.

A zero count may mean either no predicted interaction or missing chain annotations. Inspect `plip_unassigned_*` fields.

### RMSD is missing

Check that both reference and ESMFold PDBs contain:

- at least three shared target CA residues;
- shared binder CA residues;
- chain IDs A and B;
- compatible residue numbering.

### BindCraft installation fails

Use the upstream BindCraft notebook and installation instructions for a compatible commit. Colab CUDA/JAX/PyRosetta compatibility changes over time. Start with a fresh runtime rather than repeatedly mixing package versions.
