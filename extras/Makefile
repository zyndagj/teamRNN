TEST_AT_CHROM := 3
TRAIN_AT_CHROM := 1 2 4 5 Mt Pt
TEST_ZM_CHROM := 10

##########################################
# Check software
##########################################
.PHONY: TAIR_te_families.py samtools ZM_te_families.py
.SILENT: TAIR_te_families.py samtools ZM_te_families.py
samtools:
	$@ help 1> /dev/null 2> /dev/null; \
	if [ ! $$? -eq 0 ]; then \
		echo "\n[ERROR] $@ does not seem to be on your path\n"; \
		exit 1; \
	fi
ZM_te_families.py TAIR_te_families.py:
	$@ -h &> /dev/null; \
	if [ ! $$? -eq 0 ]; then \
		echo "\n[ERROR] $@ does not seem to be on your path. Please install teamRNN\n"; \
		exit 1; \
	fi
##########################################
# Download and transform files
##########################################
# TAIR10
# https://onlinelibrary.wiley.com/doi/full/10.1111/tpj.13415
TAIR10_Transposable_Elements.txt:
	curl -sSL https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_transposable_elements/TAIR10_Transposable_Elements.txt > $@.tmp && mv $@.tmp $@
Araport11_GFF3_genes_transposons.201606.gff: | TAIR10_Transposable_Elements.txt TAIR_te_families.py
	curl -sSL https://www.arabidopsis.org/download_files/Genes/Araport11_genome_release/Araport11_GFF3_genes_transposons.201606.gff.gz | zcat -d | TAIR_te_families.py -i - -t TAIR10_Transposable_Elements.txt > $@.tmp \
	&& mv $@.tmp $@
Arabidopsis_thaliana.TAIR10.dna.toplevel.fa:
	curl -sSL ftp://ftp.ensemblgenomes.org/pub/plants/current/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz | zcat -d > $@.tmp && mv $@.tmp $@
Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai: Arabidopsis_thaliana.TAIR10.dna.toplevel.fa | samtools
	samtools faidx $<
#############
# https://www.nature.com/articles/nature22971
Zea_mays.B73_RefGen_v4.42.chr.gff3:
	curl -sSL ftp://ftp.gramene.org/pub/gramene/CURRENT_RELEASE/gff3/zea_mays/Zea_mays.B73_RefGen_v4.42.chr.gff3.gz | zcat -d > $@.tmp && mv $@.tmp $@
B73v4.TE.filtered.gff3: | ZM_te_families.py
	curl -sSL ftp://ftp.gramene.org/pub/gramene/CURRENT_RELEASE/gff3/zea_mays/repeat_annotation/B73v4.TE.filtered.gff3.gz | zcat -d | ZM_te_families.py -i - > $@.tmp && mv $@.tmp $@
B73_combined.gff3: Zea_mays.B73_RefGen_v4.42.chr.gff3 B73v4.TE.filtered.gff3
	cat $^ | LC_ALL=C sort -k1,1 -k4,5n > $@.tmp && mv $@.tmp $@
Zea_mays.B73_RefGen_v4.dna.toplevel.fa:
	curl -sSL ftp://ftp.ensemblgenomes.org/pub/plants/current/fasta/zea_mays/dna/Zea_mays.B73_RefGen_v4.dna.toplevel.fa.gz | zcat -d > $@.tmp && mv $@.tmp $@
Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai: Zea_mays.B73_RefGen_v4.dna.toplevel.fa | samtools
	samtools faidx $<

DOWNLOADS := TAIR10_Transposable_Elements.txt Araport11_GFF3_genes_transposons.201606.gff \
	Arabidopsis_thaliana.TAIR10.dna.toplevel.fa Zea_mays.B73_RefGen_v4.42.chr.gff3 \
	Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai \
	B73v4.TE.filtered.gff3 Zea_mays.B73_RefGen_v4.dna.toplevel.fa \
	Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai \
	B73_combined.gff3

clean-downloads:
	rm -f $(DOWNLOADS) *tmp

##########################################
# Create training and test files
##########################################

arabidopsis_thaliana_train.gff3: | Araport11_GFF3_genes_transposons.201606.gff
	TF=train_at.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TRAIN_AT_CHROM); do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
arabidopsis_thaliana_test.gff3: | Araport11_GFF3_genes_transposons.201606.gff
	TF=test_at.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_AT_CHROM); do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
arabidopsis_thaliana_train.fasta: Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai | samtools
	TF=train_at.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TRAIN_AT_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
arabidopsis_thaliana_test.fasta: Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai | samtools
	TF=test_at.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_AT_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
ATHALIANA := $(shell echo arabidopsis_thaliana_{train,test}.{gff3,fasta})
#############
zea_mays_train.gff3: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai | B73_combined.gff3
	all_chroms=$$(cut -f 1 Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai) \
	; test_chroms=($(TEST_ZM_CHROM)); TF=train_zm.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $${all_chroms[@]/$$test_chroms}; do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
zea_mays_test.gff3: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai | B73_combined.gff3
	TF=test_zm.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_ZM_CHROM); do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
zea_mays_train.fasta: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai | samtools
	all_chroms=$$(cut -f 1 Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai) \
	; test_chroms=($(TEST_ZM_CHROM)); TF=train_zm.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $${all_chroms[@]/$$test_chroms}; do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
zea_mays_test.fasta: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai | samtools
	TF=test_zm.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_ZM_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
ZMAYS := $(shell echo zea_mays_{train,test}.{gff3,fasta})
	
clean-data:
	rm -f $(ATHALIANA) arabidopsis_thaliana_{train,test}.fasta.fai \
		$(ZMAYS) zea_mays_{train,test}.fasta.fai

##########################################
# Top level targets
##########################################

all: $(ATHALIANA) $(DOWNLOADS) $(ZMAYS)

clean: clean-downloads clean-data
