#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
import subprocess as sp
import numpy as np
from quicksect import IntervalTree
from teamRNN.constants import gff3_f2i, gff3_i2f, contexts, strands, base2index, te_feature_names
from teamRNN.constants import te_order_f2i, te_order_i2f, te_sufam_f2i, te_sufam_i2f
from teamRNN.util import irange, iterdict
from collections import defaultdict as dd
import re


known_qualities = dd(int)
known_qualities.update({'dna:chromosome':3, 'dna:contig':1, 'dna:scaffold':2, 'dna:supercontig':1})
def _split2quality(split_name):
	if len(split_name) > 1:
		lower_name = split_name[1].lower()
		return known_qualities[lower_name]
	else:
		return 0

class refcache:
	def __init__(self, fasta_file, cacheSize=5000000):
		self.fasta_file = fasta_file
		self.FA = FastaFile(fasta_file)
		self.chroms = self.FA.references
		self._get_offsets()
		self.chrom_qualities = {chrom:self.detect_quality(chrom) for chrom in self.chroms}
		self.chrom_lens = {c:self.FA.get_reference_length(c) for c in self.chroms}
		self.cacheSize = cacheSize
		self.start = {c:0 for c in self.chroms}
		self.end = {c:min(cacheSize, self.chrom_lens[c]) for c in self.chroms}
		self.chrom_caches = {c:self.FA.fetch(c,0,self.end[c]) for c in self.chroms}
	def __del__(self):
		self.FA.close()
	def _get_offsets(self):
		self.chrom_offsets = {}
		fai = '%s.fai'%(self.fasta_file)
		with open(fai, 'r') as FAI:
			for split_line in map(lambda x: x.rstrip('\n').split('\t'), FAI):
				self.chrom_offsets[split_line[0]] = int(split_line[2])
	def detect_quality(self, chrom):
		fasta_name = '>%s'%(chrom)
		with open(self.fasta_file, 'r') as FA:
			FA.seek(max(0, self.chrom_offsets[chrom]-200))
			for line in filter(lambda x: x[0] == '>', FA):
				split_line = line.rstrip('\n').split(' ')
				if split_line[0] == fasta_name:
					return _split2quality(split_line)
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
	def __init__(self, gff3, include_chrom=False):
		self.gff3 = gff3
		self._order_re = re.compile('Order=(?P<order>[^;/]+)(/(?P<sufam>[^;]+))?')
		self._sufam_re = re.compile('Superfamily=(?P<sufam>[^;]+)')
		# creates self.interval_tree
		self._2tree(include_chrom)
	def _2tree(self, include_chrom=False):
		#Chr1    TAIR10  transposable_element_gene       433031  433819  .       -       .       ID=AT1G02228;Note=transposable_element_gene;Name=AT1G02228;Derives_from=AT1TE01405
		exclude = set(['chromosome','contig','supercontig']) if include_chrom else set([])
		self.interval_tree = dd(IntervalTree)
		with open(self.gff3,'r') as IF:
			for line in filter(lambda x: x[0] != "#", IF):
				tmp = line.rstrip('\n').split('\t')
				chrom, strand, element, attributes = tmp[0], tmp[6], tmp[2], tmp[8]
				if element not in exclude:
					element_id = gff3_f2i[strand+element]
					te_order_id = 0
					te_sufam_id = 0
					if element in te_feature_names:
						m1 = self._order.search(attributes)
						if m1:
							# Order
							te_order = m1.group('order').lower()
							if te_order in te_order_f2i:
								te_order_id = te_order_f2i[te_order]
							m2 = self._family_re.search(attributes)
							if m2:
								# Super family
								if m1.group('sufam'):
									te_sufam = m1.group('sufam')
								elif m2.group('sufam'):
									te_sufam = m2.group('sufam')
								if te_sufam and te_sufam in te_sufam_f2i:
									te_sufam_id = te_sufam_f2i[te_sufam]
					start, end = map(int, tmp[3:5])
					self.interval_tree[chrom].add(start-1, end, (element_id, te_order_id, te_sufam_id))
	def fetch(self, chrom, start, end):
		outA = np.zeros((end-start, len(gff3_f2i)+2), dtype=np.uint8)
		#print("Fetching %s:%i-%i"%(chrom, start, end))
		for interval in self.interval_tree[chrom].search(start,end):
			s = max(interval.start, start)-start
			e = min(interval.end, end)-start
			element_id, te_order_id, te_sufam_id = interval.data
			#print("Detected %s at %i-%i"%(i,s,e))
			outA[s:e,element_id] = 1
			outA[s:e,-2] = te_order_id
			outA[s:e,-1] = te_sufam_id
		return outA

class input_slicer:
	def __init__(self, fasta_file, meth_file, gff3_file='', quality=-1, ploidy=2):
		self.FA = FastaFile(fasta_file)
		self.M5 = Meth5py(meth_file, fasta_file)
		self.gff3_file = gff3_file
		if gff3_file:
			self.GI = gff3_interval(gff3_file)
		self.RC = refcache(fasta_file)
		self.quality = quality
		self.ploidy = ploidy
	def __del__(self):
		self.FA.close()
		self.M5.close()
#>C1 dna:chromosome chromosome:BOL:C1:1:43764888:1 REF
	def chrom_iter(self, chrom, seq_len=5):
		chrom_len = self.FA.get_reference_length(chrom)
		chrom_quality = self.RC.chrom_qualities[chrom] if self.quality == -1 else self.quality
		for cur in irange(chrom_len-seq_len+1):
			coord = (chrom, cur, cur+seq_len)
			seq = self.RC.fetch(chrom, cur, cur+seq_len)
			# [[context_I, strand_I, c, ct, g, ga], ...]
			meth = self.M5.fetch(chrom, cur+1, cur+seq_len)
			assert(len(seq) == len(meth))
			# Transform output
			out_slice = []
			for i in range(len(seq)):
				# get base index
				base = base2index[seq[i]]
				# get location
				frac = float(cur+1+i)/chrom_len
				#out_row = [base, frac, CGr, nCG, CHGr, nCGH, CHHr, nCHH, Ploidy, Quality]
				out_row = [base, frac, 0,0, 0,0, 0,0, self.ploidy, chrom_quality]
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
	def genome_iter(self, seq_len=5):
		for chrom in sorted(self.FA.references):
			for out in self.chrom_iter(chrom, seq_len):
				yield out
	def batch_iter(self, seq_len=5, batch_size=50):
		c_batch = []
		x_batch = []
		y_batch = []
		for out in self.genome_iter(seq_len):
			if self.gff3_file:
				c, x, y = out
				y_batch.append(y)
			else:
				c, x = out
			c_batch.append(c)
			x_batch.append(x)
			if len(c_batch) == batch_size:
				if self.gff3_file:
					yield (c_batch, x_batch, y_batch)
					y_batch = []
				else:
					yield(c_batch, x_batch)
				c_batch, x_batch = [], []
		if c_batch:
			if self.gff3_file:
				yield (c_batch, x_batch, y_batch)
			else:
				yield (c_batch, x_batch)
