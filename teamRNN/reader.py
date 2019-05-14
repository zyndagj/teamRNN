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
import re, logging, os
logger = logging.getLogger(__name__)
try:
	import cPickle as pickle
except:
	import pickle


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
	def __init__(self, gff3, include_chrom=False, force=False):
		self.gff3 = gff3
		self.pkl = "%s.pkl"%(gff3)
		self.force = force
		self._order_re = re.compile('Order=(?P<order>[^;/]+)')
		self._sufam_re = re.compile('Superfamily=(?P<sufam>[^;]+)')
		# creates self.interval_tree
		self._2tree(include_chrom)
	def _2tree(self, include_chrom=False):
		#Chr1    TAIR10  transposable_element_gene       433031  433819  .       -       .       ID=AT1G02228;Note=transposable_element_gene;Name=AT1G02228;Derives_from=AT1TE01405
		exclude = set(['chromosome','contig','supercontig']) if include_chrom else set([])
		self.interval_tree = dd(IntervalTree)
		if os.path.exists(self.pkl) and not self.force:
			with open(self.pkl,'rb') as P:
				chrom_file_dict = pickle.load(P)
			for chrom, pkl_file in iterdict(chrom_file_dict):
				self.interval_tree[chrom].load(pkl_file)
			return
		with open(self.gff3,'r') as IF:
			for line in filter(lambda x: x[0] != "#", IF):
				tmp = line.rstrip('\n').split('\t')
				chrom, strand, element, attributes = tmp[0], tmp[6], tmp[2], tmp[8]
				if element not in exclude and strand+element in gff3_f2i:
					element_id = gff3_f2i[strand+element]
					te_order_id = 0
					te_sufam_id = 0
					if element in te_feature_names:
						te_order, te_sufam = self._extract_order_sufam(attributes)
						try:
							te_order_id = te_order_f2i[te_order.lower()]
							te_sufam_id = te_sufam_f2i[te_sufam.lower()]
						except:
							te_order_id, te_sufam_id = 0,0
					start, end = map(int, tmp[3:5])
					self.interval_tree[chrom].add(start-1, end, (element_id, te_order_id, te_sufam_id))
		logger.debug("Finished creating interval trees")
		chrom_file_dict = {chrom:'%s.%s.pkl'%(self.gff3, chrom) for chrom in self.interval_tree}
		for chrom, pkl_file in iterdict(chrom_file_dict):
			self.interval_tree[chrom].dump(pkl_file)
		with open(self.pkl, 'wb') as P:
			pickle.dump(chrom_file_dict, P)
		logger.debug("Finished creating pickle files")
	def _extract_order_sufam(self, attribute_string):
		order_match = self._order_re.search(attribute_string)
		sufam_match = self._sufam_re.search(attribute_string)
		order_str = order_match.group('order') if order_match else ''
		sufam_str = sufam_match.group('sufam') if sufam_match else ''
		return (order_str, sufam_str)
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
	def _get_region(self, chrom, cur, chrom_len, chrom_quality, seq_len):
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
			return (coord, out_slice, y_array)
		else:
			return (coord, out_slice)
	def chrom_iter(self, chrom, seq_len=5, offset=1, batch_size=False, hvd_rank=0, hvd_size=1):
		chrom_len = self.FA.get_reference_length(chrom)
		chrom_quality = self.RC.chrom_qualities[chrom] if self.quality == -1 else self.quality
		for cur in irange(hvd_rank*(offset-1), chrom_len-seq_len+1, offset*hvd_size):
			yield self._get_region(chrom, cur, chrom_len, chrom_quality, seq_len)
		if batch_size and cur+seq_len < chrom_len:
			cur = cur+batch_size+offset*hvd_size
			#print "New cursor at %i"%(cur)
			if cur < chrom_len:
				yield self._get_region(chrom, cur, chrom_len, chrom_quality, seq_len=chrom_len-cur)
	def genome_iter(self, seq_len=5, offset=1, batch_size=False, hvd_rank=0, hvd_size=1):
		for chrom in sorted(self.FA.references):
			logger.debug("Starting %s"%(chrom))
			for out in self.chrom_iter(chrom, seq_len, offset, batch_size, hvd_rank, hvd_size):
				yield out
			logger.debug("Finished %s"%(chrom))
	def _list2batch_num(self, input_list, seq_len, batch_size):
		#print "original"
		#for i in input_list: print i
		npa = np.array(input_list)
		#print npa.shape, npa.strides
		s0, s1 = npa.strides
		n_inputs = npa.shape[1]
		#print "seq_len = %i   batch_size = %i  strides = %s   inputs = %i"%(seq_len, batch_size, str(npa.strides), n_inputs)
		ret = np.lib.stride_tricks.as_strided(npa, (batch_size, seq_len, n_inputs), (s0,s0,s1))
		#for i in range(ret.shape[0]):
		#	print "Batch",i
		#	for j in ret[i,:,:]: print j
		return ret
	def _list2batch_str(self, input_list, seq_len, batch_size):
		return [input_list[i:i+seq_len] for i in range(batch_size)]
	def _coord2batch(self, coord, seq_len, batch_size):
		c,s,e = coord
		return [(c,s+i,s+i+seq_len) for i in range(batch_size)]
	def new_batch_iter(self, seq_len=5, batch_size=50, skip=0, rank=0, size=1):
		for out in self.genome_iter(seq_len+batch_size-1, offset=batch_size, batch_size=batch_size):
			if self.gff3_file:
				c,x,y = out
				#print c
				if len(x) == seq_len+batch_size-1:
					yb = self._list2batch_num(y, seq_len, batch_size)
				elif len(x) >= seq_len:
					yb = self._list2batch_num(y, seq_len, len(x)-seq_len+1)
				else:
					continue
			else:
				c,x = out
			if len(x) == seq_len+batch_size-1:
				cb = self._coord2batch(c, seq_len, batch_size)
				xb = self._list2batch_num(x, seq_len, batch_size)
			elif len(x) >= seq_len:
				cb = self._coord2batch(c, seq_len, len(x)-seq_len+1)
				xb = self._list2batch_num(x, seq_len, len(x)-seq_len+1)
			else:
				continue
			if self.gff3_file:
				yield (cb, xb, yb)
			else:
				yield (cb, xb)
	def batch_iter(self, seq_len=5, batch_size=50):
		c_batch = []
		x_batch = []
		y_batch = []
		for out in self.genome_iter(seq_len, offset=1):
			if self.gff3_file:
				c, x, y = out
				if len(y) == seq_len:
					y_batch.append(y)
			else:
				c, x = out
			if len(x) == seq_len:
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
