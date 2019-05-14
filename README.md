# teamRNN

## Installation

### Conda

```shell
# OSX
conda env create -f extras/team-py3-osx.yaml

# Linux
conda env create -f extras/team-py3-linux.yaml

# Activate Environment
conda activate team-py3
```

## Testing

```shell
# Activate environment
conda activate team-py3
python setup.py test
```

> We recommend installing first since some the dependencies are rather large

## Input specification

Batch Input = [batch x sequence_length x input_size]

| Index | Data type | Description |
|-------|-----------|-------------|
| 0 | uint-4 | Numerical mapping from reference base |
| 1 | float-32 | Fraction of chromosome (first character would be 1/len) |
| 2 | float-32 | CG methylation ratio |
| 3 | uint-8 | Number of CG reads |
| 4 | float-32 | CHG methylation ratio |
| 5 | uint-8 | Number of CHG reads |
| 6 | float-32 | CHH methylation ratio |
| 7 | uint-8 | Number of CHH reads |
| 8 | uint-4 | Ploidy |
| 9 | uint-4 | Assembly quality |

### Numerical Mapping Key - [0,16]

| Reference | Numerical Mapping |
|-----------|-------------------|
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |
| U | 4 |
| R | 5 |
| Y | 6 |
| K | 7 |
| M | 8 |
| S | 9 |
| W | 10 |
| B | 11 |
| D | 12 |
| H | 13 |
| V | 14 |
| N | 15 |
| - | 16 |

Valid characters taken from [FASTA specification](https://en.wikipedia.org/wiki/FASTA_format#Sequence_representation)

### Fraction of chromosome - (0, 1]


Besides conveying a relative position to the model, genes are known to cluster on the chromosome arms and repetative elements are easily found in the heterochromatin.

For a 100 basepair chromosome, we would see the following values.

| 0-index | Fraction | Explanation |
|---------|----------|-------------|
| 0 | 1/100 | The first value in the sequence |
| 1 | 2/100 | The second value in the sequence |
| 99 | 100/100 | The last value in the sequence |

### Methylation ratio - [0, 1]

The sample probability that a site is methylated. This should range [0,1], where 0 is not methylated and 1 is always methylation.

### Number of reads - [0, inf)

Since small quantities of reads can skew our methylation ratio, we included the number of reads to provide a confidence measure.

### Ploidy - [1, 16]

The number of sets of chromosomes for the input species, which needs to be specified at input time.

### Assembly Quality - [0, 3]

This is detected automatically by looking for Ensembl's reference name format

```
>C1 dna:chromosome chromosome:BOL:C1:1:43764888:1 REF
```

where the second field is used for quality.

0. Unknown
1. Contigs
  - `dna:contig`
  - `dna:supercontig`
2. Scaffolds
  - `dna:scaffold`
3. Whole chromosomes
  - `dna:chromosome`

## Output specification

Batch Output = [batch x sequence_length x input_size]

| +Index | -Index | Data type | Description |
|--------|--------|-----------|-------------|
| 0  | 28 | bool | [CDS](https://www.vectorbase.org/glossary/cds-coding-sequence) - Coding sequence |
| 1  | 29 | bool | [RNase_MRP_RNA](https://www.vectorbase.org/glossary/rnasemrprna) - The RNA molecule essential for the catalytic activity of RNase MRP |
| 2  | 30 | bool | [SRP_RNA](https://www.vectorbase.org/glossary/srprna) - Signal recognition particle |
| 3  | 33 | bool | antisense\_RNA |
| 4  | 33 | bool | antisense\_lncRNA |
| 5  | 31 | bool | [biological_region](http://www.sequenceontology.org/browser/current_svn/term/SO:0001411) - This is a parental feature spanning all other feature annotation on each functional element RefSeq |
| 6  | 32 | bool | chromosome - Signifies that sequence originates from a whole chromosome |
| 7  | 33 | bool | [contig](https://www.vectorbase.org/glossary/contig) - Signifies that the sequence originates from a contiguous region |
| 6  | 34 | bool | [exon](https://www.vectorbase.org/glossary/exon) - Genomic sequences that remains in the mRNA after introns have been spliced out |
| 7  | 35 | bool | [five_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 5' end of the first codon |
| 8  | 36 | bool | [gene](https://en.wikipedia.org/wiki/Gene) - A sequence of DNA that codes for a molecule that has a function |
| 9  | 37 | bool | [lnc_RNA](https://www.vectorbase.org/glossary/lncrna) - Encodes a long non-coding RNA |
| 10 | 38 | bool | [mRNA](https://en.wikipedia.org/wiki/Messenger_RNA) - Messenger RNA |
| 11 | 39 | bool | [miRNA](https://www.vectorbase.org/glossary/mirna) - MicroRNA |
| 12 | 40 | bool | [ncRNA](https://www.vectorbase.org/glossary/ncrna-non-coding-rna) - Non-coding RNA |
| 13 | 41 | bool | [ncRNA_gene](http://www.sequenceontology.org/miso/current_svn/term/SO:0001263) - Genes that do not encode proteins |
| 14 | 42 | bool | [pre_miRNA](https://www.vectorbase.org/glossary/premirna) - Region that remains after Drosha processing |
| 15 | 43 | bool | [pseudogene](https://www.vectorbase.org/glossary#Pseudogene) - A non-coding sequence similar to an active protein |
| 18 | 44 | bool | pseudogenic\_exon |
| 19 | 44 | bool | pseudogenic\_tRNA |
| 16 | 44 | bool | [pseudogenic_transcript](http://www.sequenceontology.org/so_wiki/index.php/Category:SO:0000516_!_pseudogenic_transcript) - A non-functional descendant of a transcript |
| 17 | 45 | bool | [rRNA](https://www.vectorbase.org/glossary/rrna) - Ribosomal RNA |
| 18 | 46 | bool | region - Genomic region |
| 19 | 47 | bool | [snRNA](https://www.vectorbase.org/glossary/snrna) - Small nuclear RNA molecule involved in pre-mRNA splicing and processing |
| 20 | 48 | bool | [snoRNA](https://www.vectorbase.org/glossary/snorna) - Small nucleolar RNA |
| 21 | 49 | bool | [supercontig](https://www.vectorbase.org/glossary/supercontigs) - Several sequence contigs combined into scaffolds |
| 22 | 50 | bool | [tRNA](https://www.vectorbase.org/glossary/trna) - Transfer RNA |
| 23 | 51 | bool | [three_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 3' end of the last codon |
| 24 | 52 | bool | [tmRNA](https://en.wikipedia.org/wiki/Transfer-messenger_RNA) - Transfer messenger RNA |
| 25 | 53 | bool | [transposable_element](https://en.wikipedia.org/wiki/Transposable_element) - A DNA sequence that can change its position in a genome |
| 26 | 54 | bool | [transposable_element_gene](https://rgd.mcw.edu/rgdweb/ontology/view.html?acc_id=SO:0000111&offset=230) - A gene encoded within a transposable element |
| 27 | 55 | bool | [transposon_fragment](http://www.sequenceontology.org/browser/release_2.5/term/SO:0001054) - A portion of a transposon, interrupted by the insertion of another element |

| Index | Data type | Description |
|-------|-----------|-------------|
| 56 | uint-8 | transposable_element Order |
| 57 | uint-8 | transposable_element Superfamily |

Mapping rules for *A. thaliana*:

- DNA? -> DNA
- LINE? -> LINE
- RathE{1,2,3}\_cons -> SINE/RathE{1,2,3}

Mapping rules for *Z. mays*:

- solo_LTR -> LTR/solo
- RC/Helitron? -> RC/Helitron

### TE Order

| Value | Name | Description |
|-------|------|-------------|
| 0	| Unassigned | Order was not specified or is unknown |
| 1	| [DNA](https://en.wikipedia.org/wiki/Transposable_element#DNA_transposons) | Dna transposon |
| 2	| [LINE](https://en.wikipedia.org/wiki/Long_interspersed_nuclear_element) | Long interspersed nuclear repeat |
| 3	| [LTR](https://en.wikipedia.org/wiki/LTR_retrotransposon) | Long terminal repeat |
| 4	| [Low_complexity](http://www.repeatmasker.org/webrepeatmaskerhelp.html) | Tandem repeats, polypurine, and AT-rich regions |
| 5	| [RC](https://en.wikipedia.org/wiki/Rolling_circle_replication) | Rolling circle replication |
| 6	| [Retroposon](https://en.wikipedia.org/wiki/Retroposon) | Repetative DNA fragments that were reverse transcribed from RNA |
| 7	| [rRNA](https://en.wikipedia.org/wiki/Ribosomal_RNA) | Ribosomal RNA |
| 8	| [Satellite](https://en.wikipedia.org/wiki/Satellite_DNA) | Large arrays of tandemly repeating, non-coding DNA |
| 9	| [Simple_repeat](http://www.repeatmasker.org/webrepeatmaskerhelp.html) | Micro-satellites |
| 10	| [SINE](https://en.wikipedia.org/wiki/Short_interspersed_nuclear_element) | Short interspersed nuclear elements |
| 11	| [snRNA](https://en.wikipedia.org/wiki/Small_nuclear_RNA) | Small nuclear RNA |
| 12	| [TIR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3562082/) | Terminal inverse repeats |
| 13	| [tRNA](https://www.nature.com/scitable/definition/trna-transfer-rna-256) | Transfer RNA |

### TE Superfamilies

| Value | Name | Description |
|-------|------|-------------|
| 0	| Unassigned | Superfamily was not specified or is unknown |
| 1	| Cassandra | |
| 2	| Caulimovirus | |
| 3	| centr | |
| 4	| CMC-EnSpm | |
| 5	| Copia | |
| 6	| En-Spm | |
| 7	| Gypsy | |
| 8	| HAT | |
| 9	| hAT-Ac | |
| 10	| hAT-Charlie | |
| 11	| hAT-Tag1 | |
| 12	| hAT-Tip100 | |
| 13	| Harbinger | |
| 14	| Helitron | |
| 15	| L1 | | |
| 16	| L1-dep | |
| 17	| Mariner | |
| 18	| MuDR | |
| 19	| MULE-MuDR | |
| 20	| PIF-Harbinger | |
| 21	| Pogo | |
| 22	| RathE1_cons | |
| 23	| RathE2_cons | |
| 24	| RathE3_cons | |
| 25	| Tc1 | |
| 26	| TcMar-Mariner | |
| 27	| TcMar-Pogo | |
| 28	| TcMar-Stowaway | |
| 29	| tRNA | |
| 30	| [solo](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC479113/#__sec3title) | A relatively intact LTR flanked by TSDs | |
