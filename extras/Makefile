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

plants/lupinus_angustifolius:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1195-1
	# https://www.ebi.ac.uk/ena/data/view/PRJNA322330
	# PRJNA322330     SRR3560754      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560754,SRR3560754,$@)
	# PRJNA322330     SRR3560755      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560755,SRR3560755,$@)
	# PRJNA322330     SRR3560756      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560756,SRR3560756,$@)
	# PRJNA322330     SRR3560757      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560757,SRR3560757,$@)
	# PRJNA322330     SRR3560758      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560758,SRR3560758,$@)
	# PRJNA322330     SRR3560759      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560759,SRR3560759,$@)
	# PRJNA322330     SRR3560760      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560760,SRR3560760,$@)
	# PRJNA322330     SRR3560761      Lupinus angustifolius   SINGLE  Bisulfite-Seq
	$(call fq-dump1,SRR3560761,SRR3560761,$@)

plants/manihot_esculenta:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4640745/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA297491
	# PRJNA297491     SRR2542438      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899953_r1
	$(call fq-dump1,SRR2542438,GSM1899953_r1,$@)
	# PRJNA297491     SRR2542439      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899953_r2
	$(call fq-dump1,SRR2542439,GSM1899953_r2,$@)
	# PRJNA297491     SRR2542440      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899953_r3
	$(call fq-dump1,SRR2542440,GSM1899953_r3,$@)
	# PRJNA297491     SRR2542441      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899953_r4
	$(call fq-dump1,SRR2542441,GSM1899953_r4,$@)
	# PRJNA297491     SRR2542442      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899954_r1
	$(call fq-dump1,SRR2542442,GSM1899954_r1,$@)
	# PRJNA297491     SRR2542443      Manihot esculenta       SINGLE  Bisulfite-Seq   GSM1899955_r1
	$(call fq-dump1,SRR2542443,GSM1899955_r1,$@)

plants/medicago_truncatula:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5037628/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA316109
	# SRR3286295	3880	Medicago truncatula	SINGLE	Bisulfite-Seq	GSM2096978_r1
	$(call fq-dump1,SRR3286295,GSM2096978_r1,$@)
	# SRR3286296	3880	Medicago truncatula	SINGLE	Bisulfite-Seq	GSM2096978_r2
	$(call fq-dump1,SRR3286296,GSM2096978_r2,$@)
	# SRR3286297	3880	Medicago truncatula	SINGLE	Bisulfite-Seq	GSM2096978_r3
	$(call fq-dump1,SRR3286297,GSM2096978_r3,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA472400 (just a subset)
	# PRJNA472400     SRR7192448      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148697_r1
	$(call fq-dump1,SRR7192448,GSM3148697_r1,$@)
	# PRJNA472400     SRR7192449      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148698_r1
	$(call fq-dump1,SRR7192449,GSM3148698_r1,$@)
	# PRJNA472400     SRR7192450      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148699_r1
	$(call fq-dump1,SRR7192450,GSM3148699_r1,$@)
	# PRJNA472400     SRR7192451      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148700_r1
	$(call fq-dump1,SRR7192451,GSM3148700_r1,$@)
	# PRJNA472400     SRR7192452      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148701_r1
	$(call fq-dump1,SRR7192452,GSM3148701_r1,$@)
	# PRJNA472400     SRR7192453      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148702_r1
	$(call fq-dump1,SRR7192453,GSM3148702_r1,$@)
	# PRJNA472400     SRR7192454      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148703_r1
	$(call fq-dump1,SRR7192454,GSM3148703_r1,$@)
	# PRJNA472400     SRR7192455      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148704_r1
	$(call fq-dump1,SRR7192455,GSM3148704_r1,$@)
	# PRJNA472400     SRR7192456      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148705_r1
	$(call fq-dump1,SRR7192456,GSM3148705_r1,$@)
	# PRJNA472400     SRR7192457      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148706_r1
	$(call fq-dump1,SRR7192457,GSM3148706_r1,$@)
	# PRJNA472400     SRR7192458      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148707_r1
	$(call fq-dump1,SRR7192458,GSM3148707_r1,$@)
	# PRJNA472400     SRR7192459      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148708_r1
	$(call fq-dump1,SRR7192459,GSM3148708_r1,$@)
	# PRJNA472400     SRR7192460      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148709_r1
	$(call fq-dump1,SRR7192460,GSM3148709_r1,$@)
	# PRJNA472400     SRR7192461      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148710_r1
	$(call fq-dump1,SRR7192461,GSM3148710_r1,$@)
	# PRJNA472400     SRR7192462      Medicago truncatula     SINGLE  Bisulfite-Seq   GSM3148711_r1
	$(call fq-dump1,SRR7192462,GSM3148711_r1,$@)

plants/musa_acuminata:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.nature.com/articles/s41477-018-0249-z
	# https://www.ebi.ac.uk/ena/data/view/PRJNA381300
	# PRJNA381300	SRR6328785	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH013_Ma_pulb_S4_BS_R1.fq.gz
	$(call fq-dump2,SRR6328785,Banana.PH013_Ma_pulb_S4_BS,$@)
	# PRJNA381300	SRR6328788	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH005_Ma_pulb_S1_BS_R2.fq.gz
	$(call fq-dump2,SRR6328788,Banana.PH005_Ma_pulb_S1_BS,$@)
	# PRJNA381300	SRR6328789	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH013_Ma_leaf_BS4_R1.fq.gz
	$(call fq-dump2,SRR6328789,Banana.PH013_Ma_leaf_BS4,$@)
	# PRJNA381300	SRR6328790	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH005_Ma_pulb_S4_BS_R1.fq.gz
	$(call fq-dump2,SRR6328790,Banana.PH005_Ma_pulb_S4_BS,$@)
	# PRJNA381300	SRR6328791	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH013_Ma_pulb_S1_BS_R1.fq.gz
	$(call fq-dump2,SRR6328791,Banana.PH013_Ma_pulb_S1_BS,$@)
	# PRJNA381300	SRR6328804	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH013_Ma_leaf_BS3_R2.fq.gz
	$(call fq-dump2,SRR6328804Banana.PH013_Ma_leaf_BS3,$@)
	# PRJNA381300	SRR6328805	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH005_Ma_leaf_BS4_R2.fq.gz
	$(call fq-dump2,SRR6328805,Banana.PH005_Ma_leaf_BS4,$@)
	# PRJNA381300	SRR6328806	Musa acuminata AAA Group	PAIRED	Bisulfite-Seq	Banana.PH005_Ma_leaf_BS3_R1.fq.gz
	$(call fq-dump2,SRR6328806,Banana.PH005_Ma_leaf_BS3,$@)

plants/oryza_sativa:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA496509
	# PRJNA496509     SRR8060862      Oryza sativa    PAIRED  Bisulfite-Seq   GSM3430700_r1
	$(call fq-dump2,SRR8060862,SRR8060862,$@)
	# PRJNA496509     SRR8060863      Oryza sativa    PAIRED  Bisulfite-Seq   GSM3430701_r1
	$(call fq-dump2,SRR8060863,SRR8060863,$@)
	# https://www.ncbi.nlm.nih.gov/sra?linkname=bioproject_sra_all&from_uid=292165
	$(call fq-dump2,DRX013571,WT_500_7pM,$@)
	$(call fq-dump2,DRX013570,WT_350_7pM,$@)
	$(call fq-dump2,DRX013569,WT_500_5pM,$@)
	$(call fq-dump2,DRX013568,WT_350_5pM,$@)

plants/ostreococcus_lucimarinus:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3969382/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA201680
	# PRJNA201680	SRR847306	Ostreococcus sp. 'lucimarinus'	Illumina HiSeq 2000	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR847306,SRR847306,$@)

plants/phaseolus_vulgaris:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA471093 (subset)
	#PRJNA471093	SRR7153430	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153430,SRR7153430,$@)
	#PRJNA471093	SRR7153431	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153431,SRR7153431,$@)
	#PRJNA471093	SRR7153432	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153432,SRR7153432,$@)
	#PRJNA471093	SRR7153433	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153433,SRR7153433,$@)
	#PRJNA471093	SRR7153434	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153434,SRR7153434,$@)
	#PRJNA471093	SRR7153435	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153435,SRR7153435,$@)
	#PRJNA471093	SRR7153436	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153436,SRR7153436,$@)
	#PRJNA471093	SRR7153437	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153437,SRR7153437,$@)
	#PRJNA471093	SRR7153438	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153438,SRR7153438,$@)
	#PRJNA471093	SRR7153439	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153439,SRR7153439,$@)
	#PRJNA471093	SRR7153440	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153440,SRR7153440,$@)
	#PRJNA471093	SRR7153441	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153441,SRR7153441,$@)
	#PRJNA471093	SRR7153442	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153442,SRR7153442,$@)
	#PRJNA471093	SRR7153443	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153443,SRR7153443,$@)
	#PRJNA471093	SRR7153444	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153444,SRR7153444,$@)
	#PRJNA471093	SRR7153445	Phaseolus vulgaris	SINGLE	Bisulfite-Seq
	$(call fq-dump1,SRR7153445,SRR7153445,$@)

plants/physcomitrella_patens:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118153
	# PRJNA484724	SRR7647623	Physcomitrella patens	SINGLE	Bisulfite-Seq	WT
	$(call fq-dump1,SRR7647623,SRR7647623_WT,$@)
	# http://science.sciencemag.org/content/328/5980/916.long
	# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA122153
	# PRJNA122153	SRR042644	Physcomitrella patens	PAIRED	Bisulfite-Seq	Physcomitrella_pattens_BS-Seq
	$(call fq-dump2,SRR042644,Physcomitrella_pattens_BS-Seq,$@)

plants/populus_trichocarpa:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4118614/
	# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA238502
	# PRJNA238502	SRR1171318	Populus trichocarpa	PAIRED	Bisulfite-Seq	well_water_01
	$(call fq-dump2,SRR1171318,well_water_01,$@)
	# PRJNA238502	SRR1171319	Populus trichocarpa	PAIRED 	Bisulfite-Seq	well_water_02
	$(call fq-dump2,SRR1171319,well_water_02,$@)
	# PRJNA238502	SRR1171316	Populus trichocarpa	PAIRED 	Bisulfite-Seq	water_starved_01
	$(call fq-dump2,SRR1171316,water_starved_01,$@)
	# PRJNA238502	SRR1171317	Populus trichocarpa	PAIRED 	Bisulfite-Seq	water_starved_02
	$(call fq-dump2,SRR1171317,water_starved_02,$@)

plants/prunus_persica:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.nature.com/articles/s41477-018-0249-z
	# https://www.ebi.ac.uk/ena/data/view/PRJNA381300
	#PRJNA381300	SRR6328741	Prunus persica	PAIRED	Bisulfite-Seq	Peach.PH097_Peach-19D_R1.fq.gz
	$(call fq-dump1,SRR6328741,Peach.PH097_Peach-19D,$@)
	#PRJNA381300	SRR6328742	Prunus persica	PAIRED	Bisulfite-Seq	Peach.PH061_Pp-L-BS_R1.fq.gz
	$(call fq-dump1,SRR6328742,Peach.PH061_Pp-L-BS,$@)
	#PRJNA381300	SRR6328749	Prunus persica	PAIRED	Bisulfite-Seq	Peach.PH013_Pp_L_BS_R1.fq.gz
	$(call fq-dump1,SRR6328749,Peach.PH013_Pp_L_BS,$@)
	#PRJNA381300	SRR6328830	Prunus persica	PAIRED	Bisulfite-Seq	Peach.PH097_Peach-127D_R1.fq.gz
	$(call fq-dump1,SRR6328830,Peach.PH097_Peach-127D,$@)
	#PRJNA381300	SRR6328831	Prunus persica	PAIRED	Bisulfite-Seq	Peach.4_NegCtrl_21_R1.fq.gz
	$(call fq-dump1,SRR6328831,Peach.4_NegCtrl_21,$@)
	#PRJNA381300	SRR6328832	Prunus persica	PAIRED	Bisulfite-Seq	Peach.4_NegCtrl_105_R1.fq.gz
	$(call fq-dump1,SRR6328832,Peach.4_NegCtrl_105,$@)

plants/selaginella_moellendorffii:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# http://science.sciencemag.org/content/328/5980/916.long
	# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA122153
	#PRJNA122153	SRR042650	Selaginella moellendorffii	SINGLE	Bisulfite-Seq	GSM497268: Selaginella_moellendorffii_BS-Seq
	$(call fq-dump1,SRR042650,SRR042650,$@)
	#PRJNA122153	SRR042651	Selaginella moellendorffii	SINGLE	Bisulfite-Seq	GSM497268: Selaginella_moellendorffii_BS-Seq
	$(call fq-dump1,SRR042651,SRR042651,$@)
	
plants/setaria_italica:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA340292
	#PRJNA340292	SRR4099343	Setaria italica	Sitalica_bisulfite_repB.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099343,Sitalica_bisulfite_repB.4,$@)
	#PRJNA340292	SRR4099344	Setaria italica	Sitalica_bisulfite_repC.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099344,Sitalica_bisulfite_repC.4,$@)
	#PRJNA340292	SRR4099345	Setaria italica	Sitalica_bisulfite_repA.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099345,Sitalica_bisulfite_repA.5,$@)
	#PRJNA340292	SRR4099346	Setaria italica	Sitalica_bisulfite_repB.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099346,Sitalica_bisulfite_repB.5,$@)
	#PRJNA340292	SRR4099347	Setaria italica	Sitalica_bisulfite_repC.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099347,Sitalica_bisulfite_repC.5,$@)
	#PRJNA340292	SRR4099348	Setaria italica	Sitalica_bisulfite_repA.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099348,Sitalica_bisulfite_repA.6,$@)
	#PRJNA340292	SRR4099349	Setaria italica	Sitalica_bisulfite_repB.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099349,Sitalica_bisulfite_repB.6,$@)
	#PRJNA340292	SRR4099350	Setaria italica	Sitalica_bisulfite_repC.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099350,Sitalica_bisulfite_repC.6,$@)
	#PRJNA340292	SRR4099351	Setaria italica	Sitalica_bisulfite_repA.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099351,Sitalica_bisulfite_repA.7,$@)
	#PRJNA340292	SRR4099352	Setaria italica	Sitalica_bisulfite_repB.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099352,Sitalica_bisulfite_repB.7,$@)
	#PRJNA340292	SRR4099354	Setaria italica	Sitalica_bisulfite_repC.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099354,Sitalica_bisulfite_repC.7,$@)
	#PRJNA340292	SRR4099355	Setaria italica	Sitalica_bisulfite_repA.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099355,Sitalica_bisulfite_repA.8,$@)
	#PRJNA340292	SRR4099356	Setaria italica	Sitalica_bisulfite_repB.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099356,Sitalica_bisulfite_repB.8,$@)
	#PRJNA340292	SRR4099357	Setaria italica	Sitalica_bisulfite_repC.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099357,Sitalica_bisulfite_repC.8,$@)
	#PRJNA340292	SRR4099358	Setaria italica	Sitalica_bisulfite_repA.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099358,Sitalica_bisulfite_repA.9,$@)
	#PRJNA340292	SRR4099359	Setaria italica	Sitalica_bisulfite_repB.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099359,Sitalica_bisulfite_repB.9,$@)
	#PRJNA340292	SRR4099360	Setaria italica	Sitalica_bisulfite_repC.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099360,Sitalica_bisulfite_repC.9,$@)
	#PRJNA340292	SRR4099361	Setaria italica	Sitalica_bisulfite_repA.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099361,Sitalica_bisulfite_repA.10,$@)
	#PRJNA340292	SRR4099362	Setaria italica	Sitalica_bisulfite_repB.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099362,Sitalica_bisulfite_repB.10,$@)
	#PRJNA340292	SRR4099363	Setaria italica	Sitalica_bisulfite_repC.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099363,Sitalica_bisulfite_repC.10,$@)
	#PRJNA340292	SRR4099520	Setaria italica	Sitalica_bisulfite_repA.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099520,Sitalica_bisulfite_repA.1,$@)
	#PRJNA340292	SRR4099521	Setaria italica	Sitalica_bisulfite_repB.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099521,Sitalica_bisulfite_repB.1,$@)
	#PRJNA340292	SRR4099522	Setaria italica	Sitalica_bisulfite_repC.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099522,Sitalica_bisulfite_repC.1,$@)
	#PRJNA340292	SRR4099523	Setaria italica	Sitalica_bisulfite_repA.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099523,Sitalica_bisulfite_repA.2,$@)
	#PRJNA340292	SRR4099524	Setaria italica	Sitalica_bisulfite_repB.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099524,Sitalica_bisulfite_repB.2,$@)
	#PRJNA340292	SRR4099525	Setaria italica	Sitalica_bisulfite_repC.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099525,Sitalica_bisulfite_repC.2,$@)
	#PRJNA340292	SRR4099526	Setaria italica	Sitalica_bisulfite_repA.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099526,Sitalica_bisulfite_repA.3,$@)
	#PRJNA340292	SRR4099547	Setaria italica	Sitalica_bisulfite_repB.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099547,Sitalica_bisulfite_repB.3,$@)
	#PRJNA340292	SRR4099548	Setaria italica	Sitalica_bisulfite_repC.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099548,Sitalica_bisulfite_repC.3,$@)
	#PRJNA340292	SRR4099549	Setaria italica	Sitalica_bisulfite_repA.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099549,Sitalica_bisulfite_repA.4,$@)

plants/solanum_lycopersicum:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://onlinelibrary.wiley.com/doi/abs/10.1111/jipb.12778
	# https://www.ebi.ac.uk/ena/data/view/PRJNA397191
	# PRJNA397191	SRX3062711	wtR1_BS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX3062711,wtR1_BS,$@)
	# PRJNA397191	SRX3062710	wtR0_BS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX3062710,wtR0_BS,$@)
	# PRJNA397191	SRX3062709	wtO8_BS	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX3062709,wtO8_BS,$@)

plants/solanum_tuberosum:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://onlinelibrary.wiley.com/doi/full/10.1111/tpj.13790
	# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA398269
	#PRJNA398269	SRR5932770	Solanum tuberosum	Potato_D113-J03_good	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5932770,Potato_D113-J03_good,$@)
	#PRJNA398269	SRR5932769	Solanum tuberosum	Potato_D113-J02_good	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5932769,Potato_D113-J02_good,$@)
	#PRJNA398269	SRR5932768	Solanum tuberosum	Potato_D113-J01_good	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5932768,Potato_D113-J01_good,$@)

plants/sorghum_bicolor:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://linkinghub.elsevier.com/retrieve/pii/S1674-2052(17)30339-8
	# https://www.ebi.ac.uk/ena/data/view/PRJNA391551
	#PRJNA391551	SRR5748817	Sorghum bicolor	SbMC_BS_1.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5748817,SbMC_BS_1.3,$@)
	#PRJNA391551	SRR5748818	Sorghum bicolor	SbMC_BS_2.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5748818,SbMC_BS_2.1,$@)
	#PRJNA391551	SRR5748819	Sorghum bicolor	SbMC_BS_1.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5748819,SbMC_BS_1.1,$@)
	#PRJNA391551	SRR5748820	Sorghum bicolor	SbMC_BS_1.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR5748820,SbMC_BS_1.2,$@)
	#PRJNA391551	SRR5748825	Sorghum bicolor	SbMC_BS_2.2	PAIRED	Bisulfite-Seq

plants/theobroma_cacao:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5037628/
	# https://www.ebi.ac.uk/ena/data/view/PRJNA316109
	#PRJNA316109	SRR3286313	Theobroma cacao	SINGLE	Bisulfite-Seq	GSM2096986_r1	
	$(call fq-dump1,SRR3286313,SRR3286313,$@)
	#PRJNA316109	SRR3286314	Theobroma cacao	SINGLE	Bisulfite-Seq	GSM2096986_r2	
	$(call fq-dump1,SRR3286314,SRR3286314,$@)

plants/triticum_aestivum:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/28449092/
	# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84253
	$(call fq-dump2,SRR3825299,adult,$@)
	
plants/triticum_urartu:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ebi.ac.uk/ena/data/view/PRJNA340292
	#PRJNA340292	SRR4099365	Triticum urartu	Turartu_bisulfite_repA.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099365,Turartu_bisulfite_repA.1,$@)
	#PRJNA340292	SRR4099366	Triticum urartu	Turartu_bisulfite_repB.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099366,Turartu_bisulfite_repB.1,$@)
	#PRJNA340292	SRR4099367	Triticum urartu	Turartu_bisulfite_repC.1	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099367,Turartu_bisulfite_repC.1,$@)
	#PRJNA340292	SRR4099368	Triticum urartu	Turartu_bisulfite_repA.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099368,Turartu_bisulfite_repA.2,$@)
	#PRJNA340292	SRR4099369	Triticum urartu	Turartu_bisulfite_repB.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099369,Turartu_bisulfite_repB.2,$@)
	#PRJNA340292	SRR4099379	Triticum urartu	Turartu_bisulfite_repC.2	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099379,Turartu_bisulfite_repC.2,$@)
	#PRJNA340292	SRR4099391	Triticum urartu	Turartu_bisulfite_repA.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099391,Turartu_bisulfite_repA.3,$@)
	#PRJNA340292	SRR4099392	Triticum urartu	Turartu_bisulfite_repB.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099392,Turartu_bisulfite_repB.3,$@)
	#PRJNA340292	SRR4099393	Triticum urartu	Turartu_bisulfite_repC.3	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099393,Turartu_bisulfite_repC.3,$@)
	#PRJNA340292	SRR4099394	Triticum urartu	Turartu_bisulfite_repA.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099394,Turartu_bisulfite_repA.4,$@)
	#PRJNA340292	SRR4099396	Triticum urartu	Turartu_bisulfite_repB.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099396,Turartu_bisulfite_repB.4,$@)
	#PRJNA340292	SRR4099397	Triticum urartu	Turartu_bisulfite_repC.4	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099397,Turartu_bisulfite_repC.4,$@)
	#PRJNA340292	SRR4099398	Triticum urartu	Turartu_bisulfite_repA.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099398,Turartu_bisulfite_repA.5,$@)
	#PRJNA340292	SRR4099399	Triticum urartu	Turartu_bisulfite_repB.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099399,Turartu_bisulfite_repB.5,$@)
	#PRJNA340292	SRR4099400	Triticum urartu	Turartu_bisulfite_repC.5	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099400,Turartu_bisulfite_repC.5,$@)
	#PRJNA340292	SRR4099401	Triticum urartu	Turartu_bisulfite_repA.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099401,Turartu_bisulfite_repA.6,$@)
	#PRJNA340292	SRR4099402	Triticum urartu	Turartu_bisulfite_repB.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099402,Turartu_bisulfite_repB.6,$@)
	#PRJNA340292	SRR4099403	Triticum urartu	Turartu_bisulfite_repC.6	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099403,Turartu_bisulfite_repC.6,$@)
	#PRJNA340292	SRR4099404	Triticum urartu	Turartu_bisulfite_repA.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099404,Turartu_bisulfite_repA.7,$@)
	#PRJNA340292	SRR4099405	Triticum urartu	Turartu_bisulfite_repB.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099405,Turartu_bisulfite_repB.7,$@)
	#PRJNA340292	SRR4099407	Triticum urartu	Turartu_bisulfite_repC.7	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099407,Turartu_bisulfite_repC.7,$@)
	#PRJNA340292	SRR4099408	Triticum urartu	Turartu_bisulfite_repA.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099408,Turartu_bisulfite_repA.8,$@)
	#PRJNA340292	SRR4099409	Triticum urartu	Turartu_bisulfite_repB.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099409,Turartu_bisulfite_repB.8,$@)
	#PRJNA340292	SRR4099410	Triticum urartu	Turartu_bisulfite_repC.8	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099410,Turartu_bisulfite_repC.8,$@)
	#PRJNA340292	SRR4099411	Triticum urartu	Turartu_bisulfite_repA.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099411,Turartu_bisulfite_repA.9,$@)
	#PRJNA340292	SRR4099412	Triticum urartu	Turartu_bisulfite_repB.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099412,Turartu_bisulfite_repB.9,$@)
	#PRJNA340292	SRR4099413	Triticum urartu	Turartu_bisulfite_repC.9	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099413,Turartu_bisulfite_repC.9,$@)
	#PRJNA340292	SRR4099414	Triticum urartu	Turartu_bisulfite_repA.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099414,Turartu_bisulfite_repA.10,$@)
	#PRJNA340292	SRR4099415	Triticum urartu	Turartu_bisulfite_repB.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099415,Turartu_bisulfite_repB.10,$@)
	#PRJNA340292	SRR4099416	Triticum urartu	Turartu_bisulfite_repC.10	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR4099416,Turartu_bisulfite_repC.10,$@)

plants/vitis_vinifera:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE77066
	#PRJNA309385	SRX1539286	Vitis vinifera	X	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX1539286,X,$@)
	#PRJNA309385	SRX1539287	Vitis vinifera	Q	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX1539287,Q,$@)
	#PRJNA309385	SRX1539288	Vitis vinifera	H	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRX1539288,H,$@)

plants/zea_mays:
	# Pull reference and annotation
	$(call pull-ref,$@)
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4311211/
	# https://www.ncbi.nlm.nih.gov/bioproject/PRJNA202919
	#PRJNA202919 	SRR850328	Zea mays	B73xMo17	PAIRED	Bisulfite-Seq
	$(call fq-dump2,SRR850328,B73xMo17,$@)

all: plants/aegilops_tauschii plants/amborella_trichopoda plants/arabidopsis_lyrata \
	plants/arabidopsis_thaliana plants/beta_vulgaris plants/brachypodium_distachyon \
	plants/brassica_napus plants/brassica_oleracea plants/brassica_rapa \
	plants/chlamydomonas_reinhardtii plants/cucumis_sativus plants/cyanidioschyzon_merolae \
	plants/glycine_max plants/gossypium_raimondii plants/hordeum_vulgare \
	plants/lupinus_angustifolius plants/manihot_esculenta plants/medicago_truncatula \
	plants/musa_acuminata plants/oryza_sativa plants/ostreococcus_lucimarinus \
	plants/phaseolus_vulgaris plants/physcomitrella_patens plants/populus_trichocarpa \
	plants/prunus_persica plants/selaginella_moellendorffii plants/setaria_italica \
	plants/solanum_lycopersicum plants/solanum_tuberosum plants/sorghum_bicolor \
	plants/theobroma_cacao plants/triticum_aestivum plants/triticum_urartu \
	plants/vitis_vinifera plants/zea_mays

# No methylation
#arabidopsis_halleri
#chondrus_crispus:
#corchorus_capsularis
#daucus_carota
#dioscorea_rotundata
#galdieria_sulphuraria
#helianthus_annuus
#leersia_perrieri
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
#trifolium_pratense
#triticum_dicoccoides
#vigna_angularis
#vigna_radiata
