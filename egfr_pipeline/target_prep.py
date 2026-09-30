"""Target preparation helpers for paired human/mouse EGFR design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

HUMAN_EGFR_ECTODOMAIN = (
    "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS"
)

MOUSE_EGFR_ECTODOMAIN = (
    "LEEKKVCQGTSNRLTQLGTFEDHFLSLQRMYNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNALYENTYALAILSNYGTNRTGLRELPMRNLQEILIGAVRFSNNPILCNMDTIQWRDIVQNVFMSNMSMDLQSHPSSCPKCDPSCPNGSCWGGGEENCQKLTKIICAQQCSHRCRGRSPSDCCHNQCAAGCTGPRESDCLVCQKFQDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGPDYYEVEEDGIRKCKKCDGPCRKVCNGIGIGEFKDTLSINATNIKHFKYCTAISGDLHILPVAFKGDSFTRTPPLDPRELEILKTVKEITGFLLIQAWPDNWTDLHAFENLEIIRGRTKQHGQFSLAVVGLNITSLGLRSLKEISDGDVIISGNRNLCYANTINWKKLFGTPNQKTKIMNNRAEKDCKAVNHVCNPLCSSEGCWGPEPRDCVSCQNVSRGRECVEKCNILEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGIMGENNTLVWKYADANNVCHLCHANCTYGCAGPGLQGCEVWPSGPKIPS"
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    accession: str
    sequence: str
    pdb_url: str

    @property
    def residue_count(self) -> int:
        return len(self.sequence)


TARGETS = {
    "human": TargetSpec(
        "human_egfr", "P00533", HUMAN_EGFR_ECTODOMAIN,
        "https://files.rcsb.org/download/6ARU.pdb",
    ),
    "mouse": TargetSpec(
        "mouse_egfr", "Q01279", MOUSE_EGFR_ECTODOMAIN,
        "https://alphafold.ebi.ac.uk/files/AF-Q01279-F1-model_v4.pdb",
    ),
}


def validate_target_specs(targets: Iterable[TargetSpec] = TARGETS.values()) -> None:
    """Validate the supplied sequences before spending GPU time."""
    for target in targets:
        sequence = target.sequence
        if not sequence or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence):
            raise ValueError(f"{target.name} contains invalid amino-acid symbols")
        if target.residue_count < 100:
            raise ValueError(f"{target.name} is unexpectedly short")


def domain_iii_window(
    target: TargetSpec,
    start_residue: int = 345,
    max_length: int = 150,
) -> tuple[int, int, str]:
    """Return a bounded domain-III-centered sequence window.

    The ectodomain sequences begin at UniProt residue 25. The defaults select
    a 150-residue window inside the recommended domain-III region. Structural
    PDB residue numbering remains the source of truth for final hotspot choice.
    """
    if max_length > 150:
        raise ValueError("trimmed targets must be at most 150 residues")
    offset = 25
    start = max(0, start_residue - offset)
    end = min(len(target.sequence), start + max_length)
    if end - start < max_length:
        start = max(0, end - max_length)
    return start + offset, end + offset - 1, target.sequence[start:end]


def download_targets(output_dir: str | Path, targets: Iterable[TargetSpec] = TARGETS.values()) -> dict[str, Path]:
    """Download target structures, returning paths keyed by target name."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for target in targets:
        path = output / f"{target.name}.pdb"
        if not path.exists():
            urlretrieve(target.pdb_url, path)
        paths[target.name] = path
    return paths


def trim_pdb_by_residue_range(
    input_pdb: str | Path,
    output_pdb: str | Path,
    start_residue: int,
    end_residue: int,
    chain_id: str = "A",
) -> Path:
    """Write a simple ATOM-only PDB slice while preserving author numbering."""
    source = Path(input_pdb)
    destination = Path(output_pdb)
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for line in source.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[21] == chain_id:
            try:
                residue_number = int(line[22:26])
            except ValueError:
                continue
            if start_residue <= residue_number <= end_residue:
                kept.append(line)
        elif line.startswith(("TER", "END")):
            kept.append(line)
    if not any(line.startswith("ATOM  ") for line in kept):
        raise ValueError(f"No atoms found for chain {chain_id} residues {start_residue}-{end_residue}")
    destination.write_text("\n".join(kept) + "\n")
    return destination
