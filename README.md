# teamRNN

## Input specification

Batch Input = [batch x sequence_length x input_size]

### Per-base input

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
