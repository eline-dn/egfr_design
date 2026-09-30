target: EGFR - Epidermal growth factor receptor

# Goal: Design a conditional EGFR binder

EGFR is one of the most clinically validated targets in oncology, and also one of the clearest illustrations of why affinity alone is not enough. This challenge asks for three properties that current EGFR therapeutics achieve only partially: binding human EGFR, binding the orthologs used in preclinical studies, and binding selectively in acidic tissue.

# Target information

Design against human EGFR’s extracellular region. We recommend targeting a functional epitope such as domain III.
## Human EGFR - Extracellular region

UniProt reference
    P00533-1
Target residues  UniProt
    25–645 / 621 aa
Recommended epitope
    Domain III
Structure reference
    6ARU  Chain A 

sequence: 
LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS


## Mouse EGFR
Extracellular region

UniProt reference
    Q01279
Target residues · UniProt
    25–647 / 623 aa
Recommended epitope
    Domain III
Predicted structure
    AF-Q01279-F1

Amino acid sequence: LEEKKVCQGTSNRLTQLGTFEDHFLSLQRMYNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNALYENTYALAILSNYGTNRTGLRELPMRNLQEILIGAVRFSNNPILCNMDTIQWRDIVQNVFMSNMSMDLQSHPSSCPKCDPSCPNGSCWGGGEENCQKLTKIICAQQCSHRCRGRSPSDCCHNQCAAGCTGPRESDCLVCQKFQDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGPDYYEVEEDGIRKCKKCDGPCRKVCNGIGIGEFKDTLSINATNIKHFKYCTAISGDLHILPVAFKGDSFTRTPPLDPRELEILKTVKEITGFLLIQAWPDNWTDLHAFENLEIIRGRTKQHGQFSLAVVGLNITSLGLRSLKEISDGDVIISGNRNLCYANTINWKKLFGTPNQKTKIMNNRAEKDCKAVNHVCNPLCSSEGCWGPEPRDCVSCQNVSRGRECVEKCNILEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGIMGENNTLVWKYADANNVCHLCHANCTYGCAGPGLQGCEVWPSGPKIPS



# Three objectives

This challenge consists of 3 main objectives, and each will be accounted for when selecting the best designs.

    01
    Binding affinity

    Design a binder against human EGFR. We recommend targeting a functional epitope such as domain III, which is also targeted by the therapeutic antibodies cetuximab and panitumumab.

    We’ll measure its binding affinity using the full extracellular region of the human receptor.
    human EGFR

    02
    Mouse cross-reactivity

    In the next objective, you should design a binder that recognises both human and mouse EGFR. On top of the validation from Objective 1, we will test binding against the mouse EGFR.

    Binding both species would make it easier to study the same design in mouse models before moving to human studies, without developing a separate mouse-specific binder.
    humanmouse

    03
    pH-selective binding

    Finally, you must design a binder that binds human EGFR at pH 6.5 and shows no detectable binding at pH 7.4. pH 6.5 replicates the tumour microenvironment conditions, ensuring that your binder would be therapeutically-relevant by only targeting tumours.

    For this, we will measure binding against human EGFR in both in vitro conditions (pH 6.5 and 7.4).


## How designs are ranked

1
pH-selective binding

Does the design bind EGFR at pH 6.5 but show no detectable binding at pH 7.4?
2
Mouse cross-reactivity

Does the same sequence bind mouse EGFR as well as human?
3 Affinity against human EGFR.


Focus on Protein minibinders: Proteins consisting of between 40 and 90 amino acids (inclusive)



What criteria will be used to determine the winners of the competition?

The challenges in this protein design competition are multivariable, making it difficult to define a single metric for success. We expect to announce winners across categories within challenges. For example, consider the challenge of designing a mouse cross-reactive, pH-sensitive binder against a human target. We could announce winners for each of the following categories:

    Highest affinity human binder against a functional epitope
    Most mouse cross-reactive binder, measured for example using mouse affinity (with human affinity above a threshold)
    Most pH-sensitive binder, measured for example using the ratio of affinities at different pH values (with affinity above a threshold)
    Most pH-sensitive, mouse cross-reactive binder measured using a combination of the above criteria

The primary goal of the competition is to show frontier protein design capabilities and this will be taken into account for assessing winning designs. For example, taking the examples above, pH-sensitive binder design is generally much harder than designing high-affinity binders, so a weak, but clearly pH-sensitive binder may be considered more impactful than a high-affinity binder that is not pH-sensitive. We will be mindful when selecting the winners to account for these challenge-specific difficulties.

Another goal of the competition is to demonstrate therapeutic relevance so we recommend participants target functional epitopes.