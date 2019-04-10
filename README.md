# teamRNN

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
| 0  | 27 | bool | [CDS](https://www.vectorbase.org/glossary/cds-coding-sequence) - Coding sequence |
| 1  | 28 | bool | [RNase_MRP_RNA](https://www.vectorbase.org/glossary/rnasemrprna) - The RNA molecule essential for the catalytic activity of RNase MRP |
| 2  | 29 | bool | [SRP_RNA](https://www.vectorbase.org/glossary/srprna) - Signal recognition particle |
| 3  | 30 | bool | [biological_region](http://www.sequenceontology.org/browser/current_svn/term/SO:0001411) - This is a parental feature spanning all other feature annotation on each functional element RefSeq |
| 4  | 31 | bool | chromosome - Signifies that sequence originates from a whole chromosome |
| 5  | 32 | bool | [contig](https://www.vectorbase.org/glossary/contig) - Signifies that the sequence originates from a contiguous region |
| 6  | 33 | bool | [exon](https://www.vectorbase.org/glossary/exon) - Genomic sequences that remains in the mRNA after introns have been spliced out |
| 7  | 34 | bool | [five_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 5' end of the first codon |
| 8  | 35 | bool | [gene](https://en.wikipedia.org/wiki/Gene) - A sequence of DNA that codes for a molecule that has a function |
| 9  | 36 | bool | [lnc_RNA](https://www.vectorbase.org/glossary/lncrna) - Encodes a long non-coding RNA |
| 10 | 37 | bool | [mRNA](https://en.wikipedia.org/wiki/Messenger_RNA) - Messenger RNA |
| 11 | 38 | bool | [miRNA](https://www.vectorbase.org/glossary/mirna) - MicroRNA |
| 12 | 39 | bool | [ncRNA](https://www.vectorbase.org/glossary/ncrna-non-coding-rna) - Non-coding RNA |
| 13 | 40 | bool | [ncRNA_gene](http://www.sequenceontology.org/miso/current_svn/term/SO:0001263) - Genes that do not encode proteins |
| 14 | 41 | bool | [pre_miRNA](https://www.vectorbase.org/glossary/premirna) - Region that remains after Drosha processing |
| 15 | 42 | bool | [pseudogene](https://www.vectorbase.org/glossary#Pseudogene) - A non-coding sequence similar to an active protein |
| 16 | 43 | bool | [pseudogenic_transcript](http://www.sequenceontology.org/so_wiki/index.php/Category:SO:0000516_!_pseudogenic_transcript) - A non-functional descendant of a transcript |
| 17 | 44 | bool | [rRNA](https://www.vectorbase.org/glossary/rrna) - Ribosomal RNA |
| 18 | 45 | bool | region - Genomic region |
| 19 | 46 | bool | [snRNA](https://www.vectorbase.org/glossary/snrna) - Small nuclear RNA molecule involved in pre-mRNA splicing and processing |
| 20 | 47 | bool | [snoRNA](https://www.vectorbase.org/glossary/snorna) - Small nucleolar RNA |
| 21 | 48 | bool | [supercontig](https://www.vectorbase.org/glossary/supercontigs) - Several sequence contigs combined into scaffolds |
| 22 | 49 | bool | [tRNA](https://www.vectorbase.org/glossary/trna) - Transfer RNA |
| 23 | 50 | bool | [three_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 3' end of the last codon |
| 24 | 51 | bool | [tmRNA](https://en.wikipedia.org/wiki/Transfer-messenger_RNA) - Transfer messenger RNA |
| 25 | 52 | bool | [transposable_element](https://en.wikipedia.org/wiki/Transposable_element) - A DNA sequence that can change its position in a genome |
| 26 | 53 | bool | [transposable_element_gene](https://rgd.mcw.edu/rgdweb/ontology/view.html?acc_id=SO:0000111&offset=230) - A gene encoded within a transposable element |

| Index | Data type | Description |
|-------|-----------|-------------|
| 54 | uint-8 | transposable_element class/family | 

### TE class/family

Any class/family combinations not recorded in this table will default to `Unassigned`

| Value | Identity |
|-------|----------|
| 0  | Unassigned |
| 1  | DNA |
| 2  | DNA? |
| 3  | DNA/CMC-EnSpm |
| 4  | DNA/En-Spm |
| 5  | DNA/HAT |
| 6  | DNA/hAT-Ac |
| 7  | DNA/hAT-Charlie |
| 8  | DNA/hAT-Tag1 |
| 9  | DNA/hAT-Tip100 |
| 10 | DNA/Harbinger |
| 11 | DNA/Mariner |
| 12 | DNA/MuDR |
| 13 | DNA/MULE-MuDR |
| 14 | DNA/PIF-Harbinger |
| 15 | DNA/Pogo |
| 16 | DNA/Tc1 |
| 17 | DNA/TcMar-Mariner |
| 18 | DNA/TcMar-Pogo |
| 19 | DNA/TcMar-Stowaway |
| 20 | LINE/L1 |
| 21 | LINE? |
| 22 | Low_complexity |
| 23 | LTR/Cassandra |
| 24 | LTR/Caulimovirus |
| 25 | LTR/Copia |
| 26 | LTR/Gypsy |
| 27 | RC/Helitron |
| 28 | RC/Helitron? |
| 29 | RathE1_cons |
| 30 | RathE2_cons |
| 31 | RathE3_cons |
| 32 | Retroposon |
| 33 | Retroposon/L1-dep |
| 34 | rRNA |
| 35 | Satellite |
| 36 | Satellite/centr |
| 37 | Simple_repeat |
| 38 | SINE |
| 39 | SINE/tRNA |
| 40 | SINE/tRNA? |
| 41 | snRNA |
| 42 | tRNA |
