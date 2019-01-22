#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
import subprocess as sp
import numpy as np
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

def input_gen(fasta, meth_file, gff3='', seq_len=5):
	# https://github.com/zyndagj/teamRNN#numerical-mapping-key---016
	base_dict = {b:i for i,b in enumerate('ACGTURYKMSWBDHVN-')}
	contexts = ('CG','CHG','CHH')
	strands = ('+', '-')
	FA = FastaFile(fasta)
	M5 = Meth5py(meth_file, fasta)
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
				base = base_dict[seq[i]]
				frac = float(cur+1+i)/cur_len
				out_row = [base, frac, 0,0, 0,0, 0,0]
				cI, sI, c, ct, g, ga = meth[i]
				if cI != -1:
					print(cur_chrom, cur, seq[i], i, cur+1+i, meth[i])
					ratio = float(c)/ct
					reads = ct
					meth_index = 2+cI*2
					out_row[meth_index:meth_index+2] = [ratio, reads]
				out_slice.append(out_row)
			yield out_slice
