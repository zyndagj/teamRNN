NT=64

echo "Downloading plant annotations"
curl -s --list-only ftp://ftp.ensemblgenomes.org/pub/plants/current/gff3/ | xargs -n 1 -P $NT -I {} bash -c 'N={}; wget -q -np -nd -r ftp://ftp.ensemblgenomes.org/pub/plants/current/gff3/${N} -A gff3.gz -R *chromosome*.gff3.gz,*chr*.gff3.gz -P plants/${N} && echo Downloaded ${N} || echo ERROR downloading ${N}'
echo "Downloading plant references"
curl -s --list-only ftp://ftp.ensemblgenomes.org/pub/plants/current/fasta/ | xargs -n 1 -P $NT -I {} bash -c 'N={}; wget -q -np -nd -r ftp://ftp.ensemblgenomes.org/pub/plants/current/fasta/${N}/dna -A dna.toplevel.fa.gz -P plants/${N}/ && echo Downloaded ${N} || echo ERROR downloading ${N}'
