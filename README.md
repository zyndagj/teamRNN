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

### Numerical Mapping Key - [0,16]

<table>
  <tr>
    <td><b>Reference</b></td>
    <td>A</td>
    <td>C</td>
    <td>G</td>
    <td>T</td>
    <td>U</td>
    <td>R</td>
    <td>Y</td>
    <td>K</td>
    <td>M</td>
    <td>S</td>
    <td>W</td>
    <td>B</td>
    <td>D</td>
    <td>H</td>
    <td>V</td>
    <td>N</td>
    <td>-</td>
  </tr><tr>
    <td><b>Numerical Mapping</b></td>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
    <td>12</td>
    <td>13</td>
    <td>14</td>
    <td>15</td>
    <td>0</td>
    <td>16</td>
  </tr>
</table>

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

## Output specification

Batch Output = [batch x sequence_length x input_size]

| Index | Data type | Description |
|-------|-----------|-------------|
| 0  | [CDS](https://www.vectorbase.org/glossary/cds-coding-sequence) | Coding sequence |
| 1  | [RNase_MRP_RNA](https://www.vectorbase.org/glossary/rnasemrprna) | The RNA molecule essential for the catalytic activity of RNase MRP |
| 2  | [SRP_RNA](https://www.vectorbase.org/glossary/srprna) | Signal recognition particle |
| 3  | [biological_region](http://www.sequenceontology.org/browser/current_svn/term/SO:0001411) | This is a parental feature spanning all other feature annotation on each functional element RefSeq |
| 4  | chromosome | Signifies that sequence originates from a whole chromosome |
| 5  | [contig](https://www.vectorbase.org/glossary/contig) | Signifies that the sequence originates from a contiguous region |
| 6  | [exon](https://www.vectorbase.org/glossary/exon) | Genomic sequences that remains in the mRNA after introns have been spliced out |
| 7  | [five_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) | Untranslated region from the 5' end of the first codon |
| 8  | [gene](https://en.wikipedia.org/wiki/Gene) | A sequence of DNA that codes for a molecule that has a function |
| 9  | [lnc_RNA](https://www.vectorbase.org/glossary/lncrna) | Encodes a long non-coding RNA |
| 10 | [mRNA](https://en.wikipedia.org/wiki/Messenger_RNA) | Messenger RNA |
| 11 | [miRNA](https://www.vectorbase.org/glossary/mirna) | MicroRNA |
| 12 | [ncRNA](https://www.vectorbase.org/glossary/ncrna-non-coding-rna) | Non-coding RNA |
| 13 | [ncRNA_gene](http://www.sequenceontology.org/miso/current_svn/term/SO:0001263) | Genes that do not encode proteins |
| 14 | [pre_miRNA](https://www.vectorbase.org/glossary/premirna) | Region that remains after Drosha processing |
| 15 | [pseudogene](https://www.vectorbase.org/glossary#Pseudogene) | A non-coding sequence similar to an active protein |
| 16 | [pseudogenic_transcript](http://www.sequenceontology.org/so_wiki/index.php/Category:SO:0000516_!_pseudogenic_transcript) | A non-functional descendant of a transcript |
| 17 | [rRNA](https://www.vectorbase.org/glossary/rrna) | Ribosomal RNA |
| 18 | region | Genomic region |
| 19 | [snRNA](https://www.vectorbase.org/glossary/snrna) | Small nuclear RNA molecule involved in pre-mRNA splicing and processing |
| 20 | [snoRNA](https://www.vectorbase.org/glossary/snorna) | Small nucleolar RNA |
| 21 | [supercontig](https://www.vectorbase.org/glossary/supercontigs) | Several sequence contigs combined into scaffolds |
| 22 | [tRNA](https://www.vectorbase.org/glossary/trna) | Transfer RNA |
| 23 | [three_prime_UTR](https://www.vectorbase.org/glossary/utr-untranslated-region) | Untranslated region from the 3' end of the last codon |
| 24 | [tmRNA](https://en.wikipedia.org/wiki/Transfer-messenger_RNA) | Transfer messenger RNA |
| 25 | [transposable_element](https://en.wikipedia.org/wiki/Transposable_element) | A DNA sequence that can change its position in a genome |
| 26 | [transposable_element_gene](https://rgd.mcw.edu/rgdweb/ontology/view.html?acc_id=SO:0000111&offset=230) | A gene encoded within a transposable element |
