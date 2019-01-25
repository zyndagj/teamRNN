#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
import subprocess as sp
import numpy as np
from quicksect import IntervalTree
try:
	# Python 2
	pass
except:
	# Python 3
	pass

class refcache:
	def __init__(self, FA, chrom, cacheSize=50000):
		self.FA = FA
		self.chrom = chrom
		self.chromLen = self.FA.get_reference_length(chrom)
		self.start = 0
		self.cacheSize = cacheSize
		self.end = min(cacheSize, self.chromLen)
		self.seq = self.FA.fetch(self.chrom, 0, self.end)
	def fetch(self, pos, pos2):
		assert(pos >= self.start)
		if pos2 > self.end:
			assert(pos2 <= self.chromLen)
			self.start = pos
			self.end = pos+self.cacheSize
			self.seq = self.FA.fetch(self.chrom, self.start, self.end)
		sI = pos-self.start
		eI = pos2-self.start
		retseq = self.seq[sI:eI]
		return retseq

gff3_dict = {v:i for i,v in enumerate([s+e for s in ('+','-') for e in['CDS', 'RNase_MRP_RNA', 'SRP_RNA', 'biological_region', 'chromosome', 'contig', 'exon', 'five_prime_UTR', 'gene', 'lnc_RNA', 'mRNA', 'miRNA', 'ncRNA', 'ncRNA_gene', 'pre_miRNA', 'pseudogene', 'pseudogenic_transcript', 'rRNA', 'region', 'snRNA', 'snoRNA', 'supercontig', 'tRNA', 'three_prime_UTR', 'tmRNA', 'transposable_element', 'transposable_element_gene']])}

def gff2interval(gff3, chrom_list):
	#Chr1    TAIR10  transposable_element_gene       433031  433819  .       -       .       ID=AT1G02228;Note=transposable_element_gene;Name=AT1G02228;Derives_from=AT1TE01405
	itd = {c:IntervalTree() for c in chrom_list}
	with open(gff3,'r') as IF:
		for line in filter(lambda x: x[0] != "#", IF):
			tmp = line.split('\t')
			chrom = tmp[0]
			strand = tmp[6]
			element = tmp[2]
			element_id = gff3_dict[strand+element]
			start, end = map(int, tmp[3:5])
			itd[chrom].add(start-1, end, element_id)
	return itd

def intervals2features(itd, chrom, start, end):
	outA = np.zeros((end-start, len(gff3_dict)))
	#print("Fetching %s:%i-%i"%(chrom, start, end))
	for interval in itd[chrom].search(start,end):
		s = max(interval.start, start)-start
		e = min(interval.end, end)-start
		i = interval.data
		#print("Detected %s at %i-%i"%(i,s,e))
		outA[s:e,i] = 1
	return outA

def input_gen(fasta, meth_file, gff3='', seq_len=5):
	# https://github.com/zyndagj/teamRNN#numerical-mapping-key---016
	base_dict = {b:i for i,b in enumerate('ACGTURYKMSWBDHVN-')}
	contexts = ('CG','CHG','CHH')
	strands = ('+', '-')
	FA = FastaFile(fasta)
	M5 = Meth5py(meth_file, fasta)
	if gff3:
		ITD = gff2interval(gff3, FA.references)
	for cur_chrom in sorted(FA.references):
		cur_len = FA.get_reference_length(cur_chrom)
		cur_rc = refcache(FA, cur_chrom)
		for cur in range(cur_len-seq_len):
			seq = cur_rc.fetch(cur, cur+seq_len)
			# [[context_I, strand_I, c, ct, g, ga], ...]
			meth = M5.fetch(cur_chrom, cur+1, cur+seq_len)
			assert(len(seq) == len(meth))
			# Transform output
			out_slice = []
			for i in range(len(seq)):
				# get base index
				base = base_dict[seq[i]]
				# get location
				frac = float(cur+1+i)/cur_len
				out_row = [base, frac, 0,0, 0,0, 0,0]
				# get methylation info
				cI, sI, c, ct, g, ga = meth[i]
				if cI != -1:
					meth_index = 2+cI*2
					out_row[meth_index:meth_index+2] = [float(c)/ct, ct]
				out_slice.append(out_row)
			if gff3:
				y_array = intervals2features(ITD, cur_chrom, cur, cur+seq_len)
				yield (out_slice, y_array)
			else:
				yield out_slice
