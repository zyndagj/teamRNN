#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
import subprocess as sp
import numpy as np
from quicksect import IntervalTree
from teamRNN.constants import gff3_f2i, gff3_i2f, contexts, strands, base_dict
from teamRNN.util import irange, iterdict
from collections import defaultdict as dd

class refcache:
	def __init__(self, fasta_file, cacheSize=5000000):
		self.fasta_file = fasta_file
		self.FA = FastaFile(fasta_file)
		self.chroms = self.FA.references
		self.chrom_lens = {c:self.FA.get_reference_length(c) for c in self.chroms}
		self.cacheSize = cacheSize
		self.start = {c:0 for c in self.chroms}
		self.end = {c:min(cacheSize, self.chrom_lens[c]) for c in self.chroms}
		self.chrom_caches = {c:self.FA.fetch(c,0,self.end[c]) for c in self.chroms}
	def __del__(self):
		self.FA.close()
	def fetch(self, chrom, pos, pos2):
		assert(pos >= self.start[chrom])
		if pos2 > self.end[chrom]:
			assert(pos2 <= self.chrom_lens[chrom])
			self.start[chrom] = pos
			self.end[chrom] = pos+self.cacheSize
			self.chrom_caches[chrom] = self.FA.fetch(chrom, self.start, self.end)
		sI = pos-self.start[chrom]
		eI = pos2-self.start[chrom]
		return self.chrom_caches[chrom][sI:eI]

class gff3_interval:
	def __init__(self, gff3):
		self.gff3 = gff3
		self._2tree()
		# creates self.interval_tree
	def _2tree(self):
		#Chr1    TAIR10  transposable_element_gene       433031  433819  .       -       .       ID=AT1G02228;Note=transposable_element_gene;Name=AT1G02228;Derives_from=AT1TE01405
		self.interval_tree = dd(IntervalTree)
		with open(self.gff3,'r') as IF:
			for line in filter(lambda x: x[0] != "#", IF):
				tmp = line.split('\t')
				chrom, strand, element = tmp[0], tmp[6], tmp[2]
				element_id = gff3_f2i[strand+element]
				start, end = map(int, tmp[3:5])
				self.interval_tree[chrom].add(start-1, end, element_id)
	def fetch(self, chrom, start, end):
		outA = np.zeros((end-start, len(gff3_f2i)), dtype=bool)
		#print("Fetching %s:%i-%i"%(chrom, start, end))
		for interval in self.interval_tree[chrom].search(start,end):
			s = max(interval.start, start)-start
			e = min(interval.end, end)-start
			i = interval.data
			#print("Detected %s at %i-%i"%(i,s,e))
			outA[s:e,i] = 1
		return outA

class input_slicer:
	def __init__(self, fasta_file, meth_file, gff3_file=''):
		self.FA = FastaFile(fasta_file)
		self.M5 = Meth5py(meth_file, fasta_file)
		self.gff3_file = gff3_file
		if gff3_file:
			self.GI = gff3_interval(gff3_file)
		self.RC = refcache(fasta_file)
	def __del__(self):
		self.FA.close()
		self.M5.close()
	def chrom_iter(self, chrom, seq_len=5):
		chrom_len = self.FA.get_reference_length(chrom)
		for cur in irange(chrom_len-seq_len):
			coord = (chrom, cur, cur+seq_len)
			seq = self.RC.fetch(chrom, cur, cur+seq_len)
			# [[context_I, strand_I, c, ct, g, ga], ...]
			meth = self.M5.fetch(chrom, cur+1, cur+seq_len)
			assert(len(seq) == len(meth))
			# Transform output
			out_slice = []
			for i in range(len(seq)):
				# get base index
				base = base_dict[seq[i]]
				# get location
				frac = float(cur+1+i)/chrom_len
				out_row = [base, frac, 0,0, 0,0, 0,0]
				# get methylation info
				cI, sI, c, ct, g, ga = meth[i]
				if cI != -1:
					meth_index = 2+cI*2
					out_row[meth_index:meth_index+2] = [float(c)/ct, ct]
				out_slice.append(out_row)
			if self.gff3_file:
				y_array = self.GI.fetch(chrom, cur, cur+seq_len)
				yield (coord, out_slice, y_array)
			else:
				yield (coord, out_slice)
		# TODO output location with slice for voting
	def genome_iter(self, seq_len=5):
		for chrom in sorted(self.FA.references):
			for out in self.chrom_iter(chrom, seq_len):
				yield out
