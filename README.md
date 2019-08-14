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

## Usage

teamRNN has three different argument sections

- **Top-level** parameters that specify inputs and the location for storing the model
- **Training** parameters that define the model structure and how training should be run
- **Classification** parameters for controlling classification output

### Top-level data and location parameters

| Parameter | Argument | Default | Description |
|-----------|----------|---------|-------------|
| `-R/--reference` | FASTA | | The fasta reference file for the organism |
| `-D/--directory` | DIR | `./model` | The directory for all model files |
| `-N/--name` | STR | default | The name of the model, which allows for the creation of multiple models of the same structure in the same directory without overwriting each other |
| `-M/--methratio` | FILE | | Methratio file used as input (generated with BSMAP) |
| `-o/--offset` | INT | 1 | Number of based to slide between windows.<br>*NOTE: This number should not exceed the sequence size, but should be larger than 1 for performance* |
| `-Q/--quality` | INT | -1 | Input assembly quality: <ol start="-1"><li>auto detect</li><li>unknown</li><li>contig</li><li>scaffold</li><li>chromosome</li></ol> |
| `-P/--ploidy` | INT | 2 | Input genome ploidy (cannot be determined automatically) |

```bash
usage: teamRNN [-h] -R FASTA [-D DIR] [-N STR] -M FILE [-o INT]
               [-Q INT] [-P INT] [-v] {train,classify} ...
```

### Training / Model specification

```bash
usage: teamRNN train [-h] -A GFF3 [-E INT] [-B INT] [-L INT] [-n INT]
         [-l INT] [-r FLOAT] [-d FLOAT] [-C STR] [-b] [-m STR] [-S]
		 [--reg_kernel] [--reg_bias] [--reg_activity] [--l1 FLOAT]
		 [--l2 FLOAT] [-H STR] [-f] [--train STR] [--test STR]
```

| Parameter | Argument | Default | Description |
|-----------|----------|---------|-------------|
| `-A/--annotation` | GFF3 | | GFF3 reference annotation used for training |
| `-B/--batch_size` | INT | 100 | This functions differently with distributed execution<br><dl><dt>Independent Batches</dt><dd>Each MPI rank will process `B` batches</dd><dt>Stateful Batches</dt><dd>To keep stateful sequence lengths consistent at different scales, each MPI rank from a pool of `N` ranks will process `int(B/N)` batches</dd></dl> |
| `-L/--sequence_length` | INT | 500 | Genome will be classified in sequences of this length. Small values will make the detection of long features difficult, while large values will require more memory and near-zero gradients |
| `-n/--neurons` | INT | 100 | The number of neurons in each RNN/LSTM cell |
| `-l/--layers` | INT | 1 | The number of RNN/LSTM layers |
| `-d/--dropout` | FLOAT | 0.0 | Fraction of the units to drop for the linear transformation of the inputs at each recurrent layer |
| `-c/--cell_type` | STR | lstm | The recurrent layers consist of either {lstm, rnn} cells |
| `-b/--bidirectional` | | False | Recurrent layers are bidirectional(incompatible with stateful) |
| `-m/--merge` | STR | concat | Bidirectional layers can be merged with the following operations:<br><ul><li>concat - concatenation</li><li>sum - summation</li><li>mul - multiplication</li><li>ave - average</li><li>none</li></ul> |
| `-S/--stateful` | | False | The recurrent model is executed statefully (incompatible with bidirectional) |
| `--reg_kernel` | | False | Apply a regularizer to the kernel weights matrix |
| `--reg_bias` | | False | Apply a regularizer to the bias vector |
| `--reg_activity` | | False | Apply a regularizer to the activation layer |
| `--l1` | FLOAT | 0.01 | L1 regularizer lambda (Lasso) |
| `--l2` | FLOAT | 0.0 | L2 regularizer lambda (Ridge)<br><sub>*NOTE: Setting either L1 or L2 to 0 will exclude that regularization function, allowing for different combinations*</sub> |
| `-H/--hidden_list` | STR | | Comma separated list of time distributed hidden layers to add after recurrent layers<br><br><sub>The argument `20,50` would add 2 time-distributed hidden layers. The first would have 20 neurons, and the second would have 50.</sub> |
| `-r/--learning_rate` | FLOAT | 0.001 | The learning rate of the optimizer |
| `-E/--epochs` | INT | 100 | Number of training epochs |
| `--train` | STR | all | Comma separated list of chromosomes to train on |
| `--test` | STR | none | Comma separated list of chromosomes to test on |
| `-f/--force` | | False | Overwrite a previously saved model |

### Classification

```
usage: teamRNN classify [-h] [-O GFF3] [-T FLOAT]
```

| Parameter | Argument | Default | Description |
|-----------|----------|---------|-------------|
| `-O/--output` | FILE | output.gff3 | Output GFF3 file with predicted annotation |
| `-T/--threshold` | FLOAT | 0.5 | This functions differently with statefulness<br><dl><dt>Independent Batches</dt><dd>Overlapping predictions will vote on the final output, and the final prediction will need at least `-T` of the votes.</dd><dt>Stateful Batches</dt><dd>Since stateful sequences may take a batch or two to correctly predict their state, voting is not used. Instead, later predictions overwrite later predictions they overlap with.</dd></dl>

### Example usage

```bash
# Train
teamRNN train

# Classify
teamRNN classify
```

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


Besides conveying a relative position to the model, genes are known to cluster on the chromosome arms and repetitive elements are easily found in the heterochromatin.

For a 100 base pair chromosome, we would see the following values.

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
| 0  | 32 | bool | [CDS](https://www.vectorbase.org/glossary/cds-coding-sequence) - Coding sequence |
| 1  | 33 | bool | [RNase_MRP_RNA](https://www.vectorbase.org/glossary/rnasemrprna) - The RNA molecule essential for the catalytic activity of RNase MRP |
| 2  | 34 | bool | [SRP_RNA](https://www.vectorbase.org/glossary/srprna) - Signal recognition particle |
| 3  | 35 | bool | antisense\_RNA |
| 4  | 36 | bool | antisense\_lncRNA |
| 5  | 37 | bool | [biological_region](http://www.sequenceontology.org/browser/current_svn/term/SO:0001411) - This is a parental feature spanning all other feature annotation on each functional element RefSeq |
| 6  | 38 | bool | chromosome - Signifies that sequence originates from a whole chromosome |
| 7  | 39 | bool | [contig](https://www.vectorbase.org/glossary/contig) - Signifies that the sequence originates from a contiguous region |
| 8  | 40 | bool | [exon](https://www.vectorbase.org/glossary/exon) - Genomic sequences that remains in the mRNA after introns have been spliced out |
| 9  | 41 | bool | [five_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 5' end of the first codon |
| 10 | 42 | bool | [gene](https://en.wikipedia.org/wiki/Gene) - A sequence of DNA that codes for a molecule that has a function |
| 11 | 43 | bool | [lnc_RNA](https://www.vectorbase.org/glossary/lncrna) - Encodes a long non-coding RNA |
| 12 | 44 | bool | [mRNA](https://en.wikipedia.org/wiki/Messenger_RNA) - Messenger RNA |
| 13 | 45 | bool | [miRNA](https://www.vectorbase.org/glossary/mirna) - MicroRNA |
| 14 | 46 | bool | [ncRNA](https://www.vectorbase.org/glossary/ncrna-non-coding-rna) - Non-coding RNA |
| 15 | 47 | bool | [ncRNA_gene](http://www.sequenceontology.org/miso/current_svn/term/SO:0001263) - Genes that do not encode proteins |
| 16 | 48 | bool | [pre_miRNA](https://www.vectorbase.org/glossary/premirna) - Region that remains after Drosha processing |
| 17 | 49 | bool | [pseudogene](https://www.vectorbase.org/glossary#Pseudogene) - A non-coding sequence similar to an active protein |
| 18 | 50 | bool | pseudogenic\_exon |
| 19 | 51 | bool | pseudogenic\_tRNA |
| 20 | 52 | bool | [pseudogenic_transcript](http://www.sequenceontology.org/so_wiki/index.php/Category:SO:0000516_!_pseudogenic_transcript) - A non-functional descendant of a transcript |
| 21 | 53 | bool | [rRNA](https://www.vectorbase.org/glossary/rrna) - Ribosomal RNA |
| 22 | 54 | bool | region - Genomic region |
| 23 | 55 | bool | [snRNA](https://www.vectorbase.org/glossary/snrna) - Small nuclear RNA molecule involved in pre-mRNA splicing and processing |
| 24 | 56 | bool | [snoRNA](https://www.vectorbase.org/glossary/snorna) - Small nucleolar RNA |
| 25 | 57 | bool | [supercontig](https://www.vectorbase.org/glossary/supercontigs) - Several sequence contigs combined into scaffolds |
| 26 | 58 | bool | [tRNA](https://www.vectorbase.org/glossary/trna) - Transfer RNA |
| 27 | 59 | bool | [three_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) - Untranslated region from the 3' end of the last codon |
| 28 | 60 | bool | [tmRNA](https://en.wikipedia.org/wiki/Transfer-messenger_RNA) - Transfer messenger RNA |
| 29 | 61 | bool | [transposable_element](https://en.wikipedia.org/wiki/Transposable_element) - A DNA sequence that can change its position in a genome |
| 30 | 62 | bool | [transposable_element_gene](https://rgd.mcw.edu/rgdweb/ontology/view.html?acc_id=SO:0000111&offset=230) - A gene encoded within a transposable element |
| 31 | 63 | bool | [transposon_fragment](http://www.sequenceontology.org/browser/release_2.5/term/SO:0001054) - A portion of a transposon, interrupted by the insertion of another element |

| Index | Data type | Description |
|-------|-----------|-------------|
| 64 | uint-8 | transposable_element Order |
| 65 | uint-8 | transposable_element Superfamily |

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
| 6	| [Retroposon](https://en.wikipedia.org/wiki/Retroposon) | Repetitive DNA fragments that were reverse transcribed from RNA |
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
