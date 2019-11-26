#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
#import subprocess as sp
import numpy as np
import multiprocessing as mp
from functools import partial
from quicksect import IntervalTree
from teamRNN.constants import gff3_f2i, gff3_i2f, contexts, strands, base2index, te_feature_names
from teamRNN.constants import te_order_f2i, te_order_i2f, te_sufam_f2i, te_sufam_i2f
from teamRNN.util import irange, iterdict
from collections import defaultdict as dd
import re, logging, os
from time import time
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
		assert(pos2 <= self.chrom_lens[chrom])
		if pos2-pos+1 >= self.cacheSize:
			logger.debug("Region was too large for refcache, you should consider increasing the cache size to %i"%((pos2-pos1+1)*10))
			return self.FA.fetch(chrom, pos, pos2)
		if pos < self.start[chrom] or pos2 > self.end[chrom]:
			self.start[chrom] = pos
			self.end[chrom] = min(pos+self.cacheSize, self.chrom_lens[chrom])
			self.chrom_caches[chrom] = self.FA.fetch(chrom, self.start[chrom], self.end[chrom])
		assert(pos >= self.start[chrom])
		sI = pos-self.start[chrom]
		eI = pos2-self.start[chrom]
		return self.chrom_caches[chrom][sI:eI]

class gff3_interval:
	def __init__(self, gff3, out_dim=len(gff3_f2i)+2, include_chrom=False, force=False):
		self.gff3 = gff3
		self.pkl = "%s.pkl"%(gff3)
		self.out_dim = out_dim
		self.noTEMD = out_dim == len(gff3_f2i)
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
		outA = np.zeros((end-start, self.out_dim), dtype=np.uint8)
		#print("Fetching %s:%i-%i"%(chrom, start, end))
		for interval in self.interval_tree[chrom].search(start,end):
			s = max(interval.start, start)-start
			e = min(interval.end, end)-start
			element_id, te_order_id, te_sufam_id = interval.data
			#print("Detected %s at %i-%i"%(i,s,e))
			outA[s:e,element_id] = 1
			if not self.noTEMD:
				outA[s:e,-2] = te_order_id
				outA[s:e,-1] = te_sufam_id
		return outA

class input_slicer:
	def __init__(self, fasta_file, meth_file, gff3_file='', quality=-1, ploidy=2, out_dim=len(gff3_f2i)+2, stateful=False):
		self.fasta_file = fasta_file
		self.FA = FastaFile(fasta_file)
		self.meth_file = meth_file
		self.M5 = False if stateful else Meth5py(meth_file, fasta_file)
		self.gff3_file = gff3_file
		if gff3_file:
			self.GI = gff3_interval(gff3_file, out_dim=out_dim)
		self.RC = refcache(fasta_file)
		self.quality = quality
		self.ploidy = ploidy
		self.out_dim = out_dim
		if stateful:
			self.pool = mp.Pool(4, slicer_init, (self.fasta_file, self.meth_file, \
						self.gff3_file, self.quality, self.ploidy, self.out_dim))
		else:
			self.pool = False
	def __del__(self):
		for f in (self.FA, self.M5):
			if f: f.close()
		if self.pool:
			self.pool.close()
			self.pool.join()
#>C1 dna:chromosome chromosome:BOL:C1:1:43764888:1 REF
	def _get_region(self, chrom, cur, chrom_len, chrom_quality, seq_len, print_region=False):
		def log_time(st,et,c):
			out_str = 'get (s) - '
			out_str += ' '.join(['%s:%i'%(i,int(et[i]-st[i])) for i in c])
			logger.debug(out_str)
		time_categories = ('reference', 'methylation', 'transform', 'gff3')
		#startTimes = {c:0 for c in time_categories}
		#endTimes = {c:0 for c in time_categories}
		if print_region: logger.debug("Fetching %s:%i-%i"%(chrom, cur, cur+seq_len))
		#print "Fetching %s:%i-%i"%(chrom, cur, cur+seq_len)
		coord = (chrom, cur, cur+seq_len)
		#startTimes['reference'] = time()
		seq = self.RC.fetch(chrom, cur, cur+seq_len)
		#endTimes['reference'] = time()
		# [[context_I, strand_I, c, ct, g, ga], ...]
		#startTimes['methylation'] = time()
		meth = self.M5.fetch(chrom, cur+1, cur+seq_len)
		#endTimes['methylation'] = time()
		assert(len(seq) == len(meth))
		# Transform output
		out_slice = np.zeros((len(seq), 10), dtype=np.float32)
		#startTimes['transform'] = time()
		out_slice[:,0] = [base2index[b] for b in seq]
		out_slice[:,1] = np.arange(cur+1,cur+len(seq)+1)/float(chrom_len)
		out_slice[:,8] = self.ploidy
		out_slice[:,9] = chrom_quality
		### Methylation
		not_n1 = meth[:,0] != -1
		new_index = meth[not_n1, 0]*2+2
		out_slice[not_n1, new_index] = np.true_divide(meth[not_n1, 2], meth[not_n1, 3])
		out_slice[not_n1, new_index+1] = meth[not_n1, 3]
		#endTimes['transform'] = time()
		if self.gff3_file:
			#startTimes['gff3'] = time()
			y_array = self.GI.fetch(chrom, cur, cur+seq_len)
			#endTimes['gff3'] = time()
			#log_time(startTimes, endTimes, time_categories)
			return (coord, out_slice, y_array)
		else:
			#log_time(startTimes, endTimes, time_categories)
			return (coord, out_slice)
	def chrom_iter(self, chrom, seq_len=5, offset=1, batch_size=False, hvd_rank=0, hvd_size=1):
		if hvd_size > 1:
			my_batches = self.chrom_iter_len(chrom, seq_len, offset, batch_size, hvd_rank, hvd_size)
			n_batches_list = [self.chrom_iter_len(chrom, seq_len, offset, batch_size, i, hvd_size) for i in range(hvd_size)]
			max_batches = int(max(n_batches_list))
			if hvd_rank == 0:
				logger.debug("All work loads %s. Using %i for all ranks"%(str(n_batches_list), max_batches))
		# TODO make this the same as genome_iter
		chrom_len = self.FA.get_reference_length(chrom)
		chrom_quality = self.RC.chrom_qualities[chrom] if self.quality == -1 else self.quality
		full_len = seq_len+(batch_size-1)*offset
		start_range = offset * batch_size * hvd_rank
		#stop_range = chrom_len - seq_len + 1
		stop_range = chrom_len - full_len + 1
		step_size = offset * batch_size * hvd_size
		for cur in irange(start_range, stop_range, step_size):
			cur_len = min(full_len, chrom_len-cur)
			if self.gff3_file:
				c,x,y = self._get_region(chrom, cur, chrom_len, chrom_quality, cur_len)
				assert(len(y) == seq_len+(batch_size-1)*offset)
				yb = self._list2batch_num(y, seq_len, batch_size, offset)
			else:
				c,x = self._get_region(chrom, cur, chrom_len, chrom_quality, cur_len)
			#print c
			assert(len(x) == seq_len+(batch_size-1)*offset)
			cb = self._coord2batch(c, seq_len, batch_size, offset)
			xb = self._list2batch_num(x, seq_len, batch_size, offset)
			if self.gff3_file:
				yield (cb, xb, yb)
			else:
				yield (cb, xb)
		if hvd_size > 1:
			for i,cur in enumerate(irange(start_range, stop_range, step_size)):
				if i < max_batches-my_batches:
					cur_len = min(full_len, chrom_len-cur)
					if self.gff3_file:
						c,x,y = self._get_region(chrom, cur, chrom_len, chrom_quality, cur_len)
						assert(len(y) == seq_len+(batch_size-1)*offset)
						yb = self._list2batch_num(y, seq_len, batch_size, offset)
					else:
						c,x = self._get_region(chrom, cur, chrom_len, chrom_quality, cur_len)
					#print c
					assert(len(x) == seq_len+(batch_size-1)*offset)
					cb = self._coord2batch(c, seq_len, batch_size, offset)
					xb = self._list2batch_num(x, seq_len, batch_size, offset)
					if self.gff3_file:
						yield (cb, xb, yb)
					else:
						yield (cb, xb)
				else:
					break
	def chrom_iter_len(self, chrom, seq_len=5, offset=1, batch_size=False, hvd_rank=0, hvd_size=1):
		chrom_len = self.FA.get_reference_length(chrom)
		full_len = seq_len+(batch_size-1)*offset
		start_range = offset * batch_size * hvd_rank
		stop_range = chrom_len - full_len + 1
		step_size = offset * batch_size * hvd_size
		return (stop_range - start_range - 1) / step_size + 1
	def genome_iter(self, seq_len=5, offset=1, batch_size=1, hvd_rank=0, hvd_size=1):
		for chrom in sorted(self.FA.references):
			logger.debug("Starting %s"%(chrom))
			for out in self.chrom_iter(chrom, seq_len, offset, batch_size, hvd_rank, hvd_size):
				yield out
			logger.debug("Finished %s"%(chrom))
	def _list2batch_num(self, input_list, seq_len, batch_size, offset=1):
		#print "original"
		#for i in input_list: print i
		npa = np.array(input_list)
		#print npa.shape, npa.strides
		s0, s1 = npa.strides
		n_inputs = npa.shape[1]
		#print "seq_len = %i   batch_size = %i  strides = %s   inputs = %i"%(seq_len, batch_size, str(npa.strides), n_inputs)
		ret = np.lib.stride_tricks.as_strided(npa, (batch_size, seq_len, n_inputs), (s0*offset,s0,s1))
		#for i in range(ret.shape[0]):
		#	print "Batch",i
		#	for j in ret[i,:,:]: print j
		return ret
	def _list2batch_str(self, input_list, seq_len, batch_size, offset=1):
		return [input_list[i:i+seq_len] for i in range(0, offset*batch_size, offset)]
	def _coord2batch(self, coord, seq_len, batch_size, offset=1):
		c,s,e = coord
		ret = [(c,s+i,s+i+seq_len) for i in range(0,offset*batch_size,offset)]
		#print "args: %s %i %i"%(str(coord), seq_len, batch_size)
		#print "original: %s  return: %s"%(str(coord), str(ret))
		return ret
	def stateful_chrom_iter(self, chrom, seq_len=5, offset=1, batch_size=5, hvd_rank=0, hvd_size=1):
		#print "seq_len: %i   batch_size: %i   hvd_size: %i   hvd_rank: %i"%(seq_len, batch_size, hvd_size, hvd_rank)
		chrom_len = self.FA.get_reference_length(chrom)
		chrom_quality = self.RC.chrom_qualities[chrom] if self.quality == -1 else self.quality
		# Calculate the number of contiguous sequences
		contigs_per_rank = int(batch_size/hvd_size)
		if not contigs_per_rank and hvd_size > 1:
			logger.warn("Using %i contiguous sequence because of batch size and worker pool size. All ranks will have the same sequence"%(batch_size))
			hvd_rank, hvd_size = 0, 1
			contigs_per_rank = batch_size
		# Calculate the number of batches for each contiguous sequence
		max_contig_len = (2*chrom_len)/(batch_size+1)
		n_batches = max_contig_len/seq_len
		logger.debug("Generating %i batches of input data"%(n_batches))
		#print "contigs_per_rank: %i   max_contig_len: %.1f   n_batches: %i"%(contigs_per_rank, max_contig_len, n_batches)
		# Calculate the start and end values for looping
		starts = np.arange(batch_size)*(max_contig_len/2)
		ends = starts+max_contig_len
		#print "starts: [%s]   ends: [%s]"%(', '.join(map(str, starts)),', '.join(map(str, ends)))
		partial_wgr = partial(worker_get_region, chrom=chrom, chrom_len=chrom_len, chrom_quality=chrom_quality, seq_len=seq_len)
		for iB in irange(n_batches):
			rank_start_inds = range(contigs_per_rank*hvd_rank, contigs_per_rank*(hvd_rank+1))
			rank_region_starts = starts[rank_start_inds]+iB*seq_len
			if self.gff3_file:
				c, x, y = zip(*self.pool.imap(partial_wgr, rank_region_starts, chunksize=25))
				yield (list(c), np.array(x), np.array(y))
			else:
				c, x = zip(*self.pool.imap(partial_wgr, rank_region_starts, chunksize=25))
				yield (list(c), np.array(x))
	def _get_region_map(self, cur, chrom, chrom_len, chrom_quality, seq_len):
		return self._get_region(chrom, cur, chrom_len, chrom_quality, seq_len, print_region=False)

def slicer_init(fasta_file, meth_file, gff3_file, quality, ploidy, out_dim):
	import os
	global wIS
	wIS = input_slicer(fasta_file, meth_file, gff3_file, quality, ploidy, out_dim)
	logger.debug("%i Finished initializing worker input slicer"%(os.getpid()))
def worker_get_region(region_start, chrom, chrom_len, chrom_quality, seq_len):
	#global wIS
	global wIS
	return wIS._get_region(chrom, region_start, chrom_len, chrom_quality, seq_len)
def worker_close(pid):
	global wIS
	del wIS
	logger.debug("P%i - closed slicer"%(pid))
	return 0
