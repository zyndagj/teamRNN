DUMP_FLAG := -X 5

ANNO_PREFIX := ftp://ftp.ensemblgenomes.org/pub/plants/current/gff3
REF_PREFIX := ftp://ftp.ensemblgenomes.org/pub/plants/current/fasta

pull-ref = wget -q -np -nd -r $(REF_PREFIX)/$(notdir $(1))/dna -A dna.toplevel.fa.gz -P $(1)/ \
	&& wget -q -np -nd -r $(ANNO_PREFIX)/$(notdir $(1)) -A gff3.gz -R *chromosome*.gff3.gz,*chr*.gff3.gz -P $(1)/ \
	&& ls $(1)/*gz | xargs -n 1 -P 2 gzip -d

fq-dump2 = export SRA=$(1) NAME=$(2) \
	&& prefetch -q $${SRA} && fastq-dump $(DUMP_FLAG) --split-files $${SRA} \
	&& mv $${SRA}_1.fastq $(3)/$${NAME}_1.fastq && mv $${SRA}_2.fastq $(3)/$${NAME}_2.fastq

fq-dump1 = export SRA=$(1) NAME=$(2) \
	&& prefetch -q $${SRA} && fastq-dump $(DUMP_FLAG) $${SRA} \
	&& mv $${SRA}.fastq $(3)/$${NAME}.fastq

#.PHONY: plants/aegilops_tauschii
plants/aegilops_tauschii:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4674939/
	# https://www.ebi.ac.uk/ena/data/view/PRJEB8762
	#
	# ERR789187 - Wheat_12B_ACAGTG_L007_R1_001.fastq
	$(call fq-dump2,ERR789187,Wheat_12B_ACAGTG_L007,$@)
	# ERR789188 - Wheat_12C_CAGATC_L007_R1_001.fastq
	$(call fq-dump2,ERR789188,Wheat_12C_CAGATC_L007,$@)

plants/amborella_trichopoda:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5410703/
	# https://www.ncbi.nlm.nih.gov/sra/SRX470016[accn]
	#
	# SRR1168334
	$(call fq-dump1,SRR1168334,SRR1168334,$@)

plants/arabidopsis_lyrata:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004785
	# https://www.ebi.ac.uk/ena/data/view/PRJEB6701
	# Based on Table S2 ly_M*-C_*t
	# ERR575711       Arabidopsis lyrata      ly_M1-C_Rt root control
	$(call fq-dump2,ERR575711,ly_M1-C_Rt,$@)
	# ERR575712       Arabidopsis lyrata      ly_M1-C_St stem control
	$(call fq-dump2,ERR575712,ly_M1-C_St,$@)
	# ERR575715       Arabidopsis lyrata      ly_M2-C_Rt root control
	$(call fq-dump2,ERR575715,ly_M2-C_Rt,$@)
	# ERR575716       Arabidopsis lyrata      ly_M2-C_St stem control
	$(call fq-dump2,ERR575716,ly_M2-C_St,$@)

plants/arabidopsis_thaliana:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004785
	# https://www.ebi.ac.uk/ena/data/view/PRJEB6701
	# Based on Table S2 th_M*-C_*t
	# ERR575727       Arabidopsis thaliana    th_M1-C_Rt root control
	$(call fq-dump2,ERR575727,th_M1-C_Rt,$@)
	# ERR575728       Arabidopsis thaliana    th_M1-C_St stem control
	$(call fq-dump2,ERR575728,th_M1-C_St,$@)
	# ERR575731       Arabidopsis thaliana    th_M2-C_Rt root control
	$(call fq-dump2,ERR575731,th_M2-C_Rt,$@)
	# ERR575732       Arabidopsis thaliana    th_M2-C_St stem control
	$(call fq-dump2,ERR575732,th_M2-C_St,$@)

plants/beta_vulgaris:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5037628/
	# https://www.ncbi.nlm.nih.gov/sra/SRX1656913[accn]
	# SRR3286253      161934  Beta vulgaris   SINGLE
	$(call fq-dump1,SRR3286253,SRR3286253,$@)
	# SRR3286254      161934  Beta vulgaris   SINGLE
	$(call fq-dump1,SRR3286254,SRR3286254,$@)

plants/brachypodium_distachyon:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5037628/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA316109
	# SRR3286247	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r1	
	$(call fq-dump1,SRR3286247,SRR3286247,$@)
	# SRR3286248	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r2	
	$(call fq-dump1,SRR3286248,SRR3286248,$@)
	# SRR3286249	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r3	
	$(call fq-dump1,SRR3286249,SRR3286249,$@)
	# SRR3286250	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r4	
	$(call fq-dump1,SRR3286250,SRR3286250,$@)
	# SRR3286251	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r5	
	$(call fq-dump1,SRR3286251,SRR3286251,$@)
	# SRR3286252	15368	Brachypodium distachyon	SINGLE	Bisulfite-Seq	GSM2096945_r6	
	$(call fq-dump1,SRR3286252,SRR3286252,$@)

plants/brassica_napus:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6165103/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA488317
	# PRJNA488317	SRR7763971	Brassica napus	WM4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7763971,WM4,$@)
	# PRJNA488317	SRR7763972	Brassica napus	WM3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7763972,WM3,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6289790/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA470566
	# PRJNA470566	SRR7135722	Brassica napus	q44-9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135722,q44-9,$@)
	# PRJNA470566	SRR7135723	Brassica napus	q44-12	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135723,q44-12,$@)
	# PRJNA470566	SRR7135726	Brassica napus	ZJ34-9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135726,ZJ34-9,$@)
	# PRJNA470566	SRR7135727	Brassica napus	ZJ34-12	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135727,ZJ34-12,$@)
	# PRJNA470566	SRR7135728	Brassica napus	FJ43-9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135728,FJ43-9,$@)
	# PRJNA470566	SRR7135729	Brassica napus	FJ43-12	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135729,FJ43-12,$@)
	# PRJNA470566	SRR7135730	Brassica napus	q33-9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135730,q33-9,$@)
	# PRJNA470566	SRR7135731	Brassica napus	q33-12	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR7135731,q33-12,$@)

plants/brassica_oleracea:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA428942
	# PRJNA428942	SRR6516714	Brassica oleracea var. capitata	180125004	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6516714,180125004,$@)
	# PRJNA428942	SRR6516715	Brassica oleracea var. capitata	180125003	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6516715,180125003,$@)
	# PRJNA428942	SRR6516716	Brassica oleracea var. capitata	180125002	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6516716,180125002,$@)
	# PRJNA428942	SRR6516717	Brassica oleracea var. capitata	180125001	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6516717,180125001,$@)

plants/brassica_rapa:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4987809/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA302602
	# PRJNA302602	SRR2922637	Brassica rapa	SINGLE	Bisulfite-Seq	GSM1942115_r1
	$(call fq-dump1,SRR2922637,GSM1942115_r1,$@)
	# PRJNA302602	SRR2922638	Brassica rapa	SINGLE	Bisulfite-Seq	GSM1942115_r2
	$(call fq-dump1,SRR2922638,GSM1942115_r2,$@)

plants/chlamydomonas_reinhardtii:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://academic.oup.com/mbe/article/34/9/2285/3848952
	# https://www.ebi.ac.uk/ena/data/view/PRJNA384135
	# PRJNA384135	SRR5487349	Chlamydomonas reinhardtii	P1B3_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5487349,P1B3_WGBS,$@)
	# PRJNA384135	SRR5487336	Chlamydomonas reinhardtii	Anc_CC2937_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5487336,Anc_CC2937_WGBS,$@)
	
plants/cucumis_sativus:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA437233
	# This contains pacbio, bisulfite, hic, and 10x data
	# PRJNA437233	SRR6976811	Cucumis sativus	Illumina HiSeq 2000	9930_44_64_70_cucumber_methylation	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6976811,9930_44_64_70_cucumber_methylation,$@)

plants/cyanidioschyzon_merolae:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3969382/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA201680
	# PRJNA201680	SRR847294	Cyanidioschyzon merolae	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR847294,SRR847294,$@)

plants/glycine_max:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6130073/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA432760
	# PRJNA432760	SRR6796748	Glycine max	Illumina HiSeq 2500	ACC-100_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796748,ACC-100_WGBS,$@)
	# PRJNA432760	SRR6796749	Glycine max	Illumina HiSeq 2500	ACC-089_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796749,ACC-089_WGBS,$@)
	# PRJNA432760	SRR6796754	Glycine max	Illumina HiSeq 2500	ACC-082_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796754,ACC-082_WGBS,$@)
	# PRJNA432760	SRR6796755	Glycine max	Illumina HiSeq 2500	ACC-076_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796755,ACC-076_WGBS,$@)
	# PRJNA432760	SRR6796756	Glycine max	Illumina HiSeq 2500	ACC-051_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796756,ACC-051_WGBS,$@)
	# PRJNA432760	SRR6796757	Glycine max	Illumina HiSeq 2500	ACC-001_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796757,ACC-001_WGBS,$@)
	# PRJNA432760	SRR6796826	Glycine max	Illumina HiSeq 2500	ACC-616_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796826,ACC-616_WGBS,$@)
	# PRJNA432760	SRR6796827	Glycine max	Illumina HiSeq 2500	ACC-546_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796827,ACC-546_WGBS,$@)
	# PRJNA432760	SRR6796828	Glycine max	Illumina HiSeq 2500	ACC-504_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796828,ACC-504_WGBS,$@)
	# PRJNA432760	SRR6796829	Glycine max	Illumina HiSeq 2500	ACC-438_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796829,ACC-438_WGBS,$@)
	# PRJNA432760	SRR6796830	Glycine max	Illumina HiSeq 2500	ACC-433_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796830,ACC-433_WGBS,$@)
	# PRJNA432760	SRR6796831	Glycine max	HiSeq X Ten	ACC-319_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796831,ACC-319_WGBS,$@)
	# PRJNA432760	SRR6796832	Glycine max	Illumina HiSeq 2500	ACC-315_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796832,ACC-315_WGBS,$@)
	# PRJNA432760	SRR6796833	Glycine max	Illumina HiSeq 2500	ACC-281_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796833,ACC-281_WGBS,$@)
	# PRJNA432760	SRR6796834	Glycine max	Illumina HiSeq 2500	ACC-262_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796834,ACC-262_WGBS,$@)
	# PRJNA432760	SRR6796845	Glycine max	HiSeq X Ten	ACC-1412_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796845,ACC-1412_WGBS,$@)
	# PRJNA432760	SRR6796846	Glycine max	HiSeq X Ten	ACC-1433_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796846,ACC-1433_WGBS,$@)
	# PRJNA432760	SRR6796847	Glycine max	HiSeq X Ten	ACC-1279_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796847,ACC-1279_WGBS,$@)
	# PRJNA432760	SRR6796848	Glycine max	HiSeq X Ten	ACC-128_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796848,ACC-128_WGBS,$@)
	# PRJNA432760	SRR6796849	Glycine max	Illumina HiSeq 2500	ACC-103_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796849,ACC-103_WGBS,$@)
	# PRJNA432760	SRR6796850	Glycine max	Illumina HiSeq 2500	ACC-108_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796850,ACC-108_WGBS,$@)
	# PRJNA432760	SRR6796851	Glycine max	HiSeq X Ten	ACC-1379_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796851,ACC-1379_WGBS,$@)
	# PRJNA432760	SRR6796852	Glycine max	HiSeq X Ten	ACC-1399_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796852,ACC-1399_WGBS,$@)
	# PRJNA432760	SRR6796853	Glycine max	HiSeq X Ten	ACC-1297_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796853,ACC-1297_WGBS,$@)
	# PRJNA432760	SRR6796854	Glycine max	Illumina HiSeq 2500	ACC-130_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796854,ACC-130_WGBS,$@)
	# PRJNA432760	SRR6796876	Glycine max	HiSeq X Ten	ACC-2219_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796876,ACC-2219_WGBS,$@)
	# PRJNA432760	SRR6796877	Glycine max	HiSeq X Ten	ACC-2218_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796877,ACC-2218_WGBS,$@)
	# PRJNA432760	SRR6796878	Glycine max	Illumina HiSeq 2500	ACC-172_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796878,ACC-172_WGBS,$@)
	# PRJNA432760	SRR6796879	Glycine max	Illumina HiSeq 2500	ACC-171_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796879,ACC-171_WGBS,$@)
	# PRJNA432760	SRR6796880	Glycine max	HiSeq X Ten	ACC-1843_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796880,ACC-1843_WGBS,$@)
	# PRJNA432760	SRR6796881	Glycine max	Illumina HiSeq 2500	ACC-176_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796881,ACC-176_WGBS,$@)
	# PRJNA432760	SRR6796882	Glycine max	Illumina HiSeq 2500	ACC-212_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796882,ACC-212_WGBS,$@)
	# PRJNA432760	SRR6796883	Glycine max	HiSeq X Ten	ACC-1942_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796883,ACC-1942_WGBS,$@)
	# PRJNA432760	SRR6796884	Glycine max	HiSeq X Ten	ACC-2210_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796884,ACC-2210_WGBS,$@)
	# PRJNA432760	SRR6796885	Glycine max	HiSeq X Ten	ACC-215_WGBS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR6796885,ACC-215_WGBS,$@)

plants/gossypium_raimondii:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pubmed/28558752
	# https://www.ebi.ac.uk/ena/data/view/PRJNA314138
	# PRJNA314138	SRR3219088	Gossypium raimondii	Gossypium raimondii_Rep1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR3219088,G_raimondii_Rep1,$@)
	# PRJNA314138	SRR3219089	Gossypium raimondii	Gossypium raimondii_Rep2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR3219089,G_raimondii_Rep2,$@)

plants/hordeum_vulgare:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://mobilednajournal.biomedcentral.com/articles/10.1186/s13100-017-0102-3
	# https://www.ebi.ac.uk/ena/data/view/PRJNA358450
	# PRJNA358450	SRR5124893	Hordeum vulgare	PAIRED	Bisulfite-Seq	barley_leaf_methylome
	$(call fq-dump2,SRR5124893,barley_leaf_methylome,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA340292
	# PRJNA340292	SRR4099340	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099340,Hvulgare_bisulfite_repA.1,$@)
	# PRJNA340292	SRR4099341	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099341,Hvulgare_bisulfite_repB.1,$@)
	# PRJNA340292	SRR4099342	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099342,Hvulgare_bisulfite_repB.4,$@)
	# PRJNA340292	SRR4099353	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099353,Hvulgare_bisulfite_repC.4,$@)
	# PRJNA340292	SRR4099364	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099364,Hvulgare_bisulfite_repA.5,$@)
	# PRJNA340292	SRR4099395	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099395,Hvulgare_bisulfite_repB.5,$@)
	# PRJNA340292	SRR4099406	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099406,Hvulgare_bisulfite_repC.5,$@)
	# PRJNA340292	SRR4099417	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099417,Hvulgare_bisulfite_repA.6,$@)
	# PRJNA340292	SRR4099418	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099418,Hvulgare_bisulfite_repB.6,$@)
	# PRJNA340292	SRR4099419	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099419,Hvulgare_bisulfite_repC.6,$@)
	# PRJNA340292	SRR4099420	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099420,Hvulgare_bisulfite_repA.7,$@)
	# PRJNA340292	SRR4099421	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099421,Hvulgare_bisulfite_repB.7,$@)
	# PRJNA340292	SRR4099422	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099422,Hvulgare_bisulfite_repC.1,$@)
	# PRJNA340292	SRR4099423	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099423,Hvulgare_bisulfite_repC.7,$@)
	# PRJNA340292	SRR4099424	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099424,Hvulgare_bisulfite_repA.8,$@)
	# PRJNA340292	SRR4099425	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099425,Hvulgare_bisulfite_repB.8,$@)
	# PRJNA340292	SRR4099426	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099426,Hvulgare_bisulfite_repC.8,$@)
	# PRJNA340292	SRR4099427	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099427,Hvulgare_bisulfite_repA.9,$@)
	# PRJNA340292	SRR4099428	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099428,Hvulgare_bisulfite_repB.9,$@)
	# PRJNA340292	SRR4099429	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099429,Hvulgare_bisulfite_repC.9,$@)
	# PRJNA340292	SRR4099430	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099430,Hvulgare_bisulfite_repA.10,$@)
	# PRJNA340292	SRR4099431	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099431,Hvulgare_bisulfite_repB.10,$@)
	# PRJNA340292	SRR4099432	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099432,Hvulgare_bisulfite_repC.10,$@)
	# PRJNA340292	SRR4099433	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099433,Hvulgare_bisulfite_repA.2,$@)
	# PRJNA340292	SRR4099444	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099444,Hvulgare_bisulfite_repB.2,$@)
	# PRJNA340292	SRR4099475	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099475,Hvulgare_bisulfite_repC.2,$@)
	# PRJNA340292	SRR4099486	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099486,Hvulgare_bisulfite_repA.3,$@)
	# PRJNA340292	SRR4099497	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repB.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099497,Hvulgare_bisulfite_repB.3,$@)
	# PRJNA340292	SRR4099508	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repC.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099508,Hvulgare_bisulfite_repC.3,$@)
	# PRJNA340292	SRR4099519	Hordeum vulgare subsp. vulgare	Hvulgare_bisulfite_repA.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099519,Hvulgare_bisulfite_repA.4,$@)

all: plants/aegilops_tauschii plants/amborella_trichopoda plants/arabidopsis_lyrata \
	plants/arabidopsis_thaliana plants/beta_vulgaris plants/brachypodium_distachyon \
	plants/brassica_napus plants/brassica_oleracea plants/brassica_rapa \
	plants/chlamydomonas_reinhardtii plants/cucumis_sativus plants/cyanidioschyzon_merolae \
	plants/glycine_max plants/gossypium_raimondii plants/hordeum_vulgare \

# No methylation
#arabidopsis_halleri
#chondrus_crispus:
#corchorus_capsularis
#daucus_carota
#dioscorea_rotundata
#galdieria_sulphuraria
#helianthus_annuus

#leersia_perrieri
#lupinus_angustifolius
#manihot_esculenta
#medicago_truncatula
#musa_acuminata
#nicotiana_attenuata
#oryza_barthii
#oryza_brachyantha
#oryza_glaberrima
#oryza_glumipatula
#oryza_indica
#oryza_longistaminata
#oryza_meridionalis
#oryza_nivara
#oryza_punctata
#oryza_rufipogon
#oryza_sativa
#ostreococcus_lucimarinus
#phaseolus_vulgaris
#physcomitrella_patens
#populus_trichocarpa
#prunus_persica
#selaginella_moellendorffii
#setaria_italica
#solanum_lycopersicum
#solanum_tuberosum
#sorghum_bicolor
#theobroma_cacao
#trifolium_pratense
#triticum_aestivum
#triticum_dicoccoides
#triticum_urartu
#vigna_angularis
#vigna_radiata
#vitis_vinifera
#zea_mays
