TEST_AT_CHROM := 3
TRAIN_AT_CHROM := 1 2 4 5 Mt Pt
TEST_ZM_CHROM := 10

PREF_FLAG := -X 100G -q
DUMP_FLAG := -X 5 -W
fq-dump2 = export SRA=$(1) NAME=$(2); export NAME=${NAME/_[12].fastq/} \
	&& prefetch $(PREF_FLAG) $${SRA} && fastq-dump $(DUMP_FLAG) --split-files $${SRA} \
	&& mv $${SRA}_1.fastq $${NAME}_1.fastq && mv $${SRA}_2.fastq $${NAME}_2.fastq
fq-dump1 = export SRA=$(1) NAME=$(2) \
	&& prefetch $(PREF_FLAG) $${SRA} && fastq-dump $(DUMP_FLAG) $${SRA} \
	&& mv $${SRA}.fastq $${NAME}.fastq

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
# reads
arabidopsis_thaliana/reads: arabidopsis_thaliana
	[ ! -e $@ ] && mkdir $@
# https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004785
# https://www.ebi.ac.uk/ena/data/view/PRJEB6701
# Based on Table S2 th_M*-C_*t
arabidopsis_thaliana/reads/th_M1-C_Rt_1.fastq:
	# ERR575727       Arabidopsis thaliana    th_M1-C_Rt root control
	$(call fq-dump2,ERR575727,$@)
arabidopsis_thaliana/reads/th_M1-C_St_1.fastq:
	# ERR575728       Arabidopsis thaliana    th_M1-C_St stem control
	$(call fq-dump2,ERR575728,$@)
arabidopsis_thaliana/reads/th_M2-C_Rt_1.fastq:
	# ERR575731       Arabidopsis thaliana    th_M2-C_Rt root control
	$(call fq-dump2,ERR575731,$@)
arabidopsis_thaliana/reads/th_M2-C_St_1.fastq:
	# ERR575732       Arabidopsis thaliana    th_M2-C_St stem control
	$(call fq-dump2,ERR575732,$@)
AT_READS := $(shell echo arabidopsis_thaliana/reads/th_M{1,2}-C_{S,R}t_1.fastq)
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
# reads
zea_mays/reads: zea_mays
	[ ! -e $@ ] && mkdir $@
# https://www.ebi.ac.uk/ena/data/view/PRJNA391551
zea_mays/reads/ZmMC_BS_1.3_1.fastq:
	# SRR5748715	Zea mays	ZmMC_BS_1.3	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748715,$@)
zea_mays/reads/ZmMC_BS_1.2_1.fastq:
	# SRR5748716	Zea mays	ZmMC_BS_1.2	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748716,$@)
zea_mays/reads/ZmMC_BS_1.1_1.fastq:
	# SRR5748717	Zea mays	ZmMC_BS_1.1	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748717,$@)
zea_mays/reads/ZmMC_BS_2.3_1.fastq:
	# SRR5748719	Zea mays	ZmMC_BS_2.3	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748719,$@)
zea_mays/reads/ZmMC_BS_2.2_1.fastq:
	# SRR5748720	Zea mays	ZmMC_BS_2.2	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748720,$@)
zea_mays/reads/ZmMC_BS_2.1_1.fastq:
	# SRR5748721	Zea mays	ZmMC_BS_2.1	PAIRED	Bisulfite-Seq
	$(call  fq-dump2,SRR5748721,$@)
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4311211/
# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA202919
zea_mays/reads/B73xMo17_1.fastq:
	#PRJNA202919 	SRR850328	Zea mays	B73xMo17	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR850328,B73xMo17,$@)
ZM_READS := $(shell echo zea_mays/reads/{B73xMo17,ZmMC_BS_{1,2}.{1,2,3}}_1.fastq)

DOWNLOADS := TAIR10_Transposable_Elements.txt Araport11_GFF3_genes_transposons.201606.gff \
	Arabidopsis_thaliana.TAIR10.dna.toplevel.fa Zea_mays.B73_RefGen_v4.42.chr.gff3 \
	Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai \
	B73v4.TE.filtered.gff3 Zea_mays.B73_RefGen_v4.dna.toplevel.fa \
	Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai \
	B73_combined.gff3 $(AT_READS) $(ZM_READS)

clean-downloads:
	rm -f $(DOWNLOADS) *tmp

##########################################
# Create training and test files
##########################################

arabidopsis_thaliana:
	[ ! -e $@ ] && mkdir $@
arabidopsis_thaliana/arabidopsis_thaliana_train.gff3: Araport11_GFF3_genes_transposons.201606.gff | arabidopsis_thaliana
	TF=train_at.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TRAIN_AT_CHROM); do \
		grep "^$$chrom\s" $<; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
arabidopsis_thaliana/arabidopsis_thaliana_test.gff3: Araport11_GFF3_genes_transposons.201606.gff | arabidopsis_thaliana
	TF=test_at.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_AT_CHROM); do \
		grep "^$$chrom\s" $<; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
arabidopsis_thaliana/arabidopsis_thaliana_train.fasta: Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai | samtools arabidopsis_thaliana
	TF=train_at.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TRAIN_AT_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
arabidopsis_thaliana/arabidopsis_thaliana_test.fasta: Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai | samtools arabidopsis_thaliana
	TF=test_at.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_AT_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
ATHALIANA := $(shell echo arabidopsis_thaliana/arabidopsis_thaliana_{train,test}.{gff3,fasta})
#############
zea_mays:
	[ ! -e $@ ] && mkdir $@
zea_mays/zea_mays_train.gff3: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai zea_mays | B73_combined.gff3
	all_chroms=$$(cut -f 1 $<) \
	; test_chroms=($(TEST_ZM_CHROM)); TF=train_zm.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $${all_chroms[@]/$$test_chroms}; do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
zea_mays/zea_mays_test.gff3: zea_mays | B73_combined.gff3
	TF=test_zm.gff3; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_ZM_CHROM); do \
		grep "^$$chrom\s" $|; \
	done | LC_ALL=C sort -k1,1 -k4,5n > $$TF && mv $$TF $@
zea_mays/zea_mays_train.fasta: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai zea_mays | samtools
	all_chroms=$$(cut -f 1 $<) \
	; test_chroms=($(TEST_ZM_CHROM)); TF=train_zm.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $${all_chroms[@]/$$test_chroms}; do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
zea_mays/zea_mays_test.fasta: Zea_mays.B73_RefGen_v4.dna.toplevel.fa.fai zea_mays | samtools
	TF=test_zm.fasta; [ -e $$TF ] && rm $$TF \
	; for chrom in $(TEST_ZM_CHROM); do \
		samtools faidx $(basename $<) $$chrom >> $$TF; \
	done && mv $$TF $@
	samtools faidx $@
ZMAYS := $(shell echo zea_mays/zea_mays_{train,test}.{gff3,fasta})
	
clean-data:
	rm -f $(ATHALIANA) arabidopsis_thaliana $(ZMAYS) zea_mays

##########################################
# Top level targets
##########################################

all: $(ATHALIANA) $(ZMAYS) $(DOWNLOADS)

clean: clean-downloads clean-data
