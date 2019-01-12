#!/usr/bin/env python

from pysam import FastaFile
from Meth5py import Meth5py
from string import maketrans
import subprocess as sp
import numpy as np

class refcache:
	def __init__(self, FA, chrom, cacheSize=50000):
		self.FA = FA
		self.chrom = chrom
		self.chromLen = self.FA.get_reference_length(chrom)
		self.start = 0
		self.cacheSize = cacheSize
		self.end = min(cacheSize, chromLen)
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

class input_iter:
	def __init__(self, fasta, meth_file, gff3='', seq_len=5):
		self.fasta_file = fasta
		self.FA = FastaFile(self.fasta_file)
		self.chroms = sorted(self.FA.references)
		self.cur_chrom = self.chroms.pop()
		self.cur_len = self.FA.get_reference_length(self.cur_chrom)
		# 0-indexed
		self.cur_rc = refcache(FA, self.cur_chrom)
		self.cur = 0
		# 1-indexed
		self.M5 = Meth5py(meth_file, fasta)
		if not isinstance(meth_files, basestring):
			self.meth_files = [meth_files]
		else:
			self.math_files = meth_files
		if gff3: self.gff3 = gff3
	def __iter__(self):
		return self
	def __next__(self):
		seq = self.cur_rc.fetch(self.cur, self.cur+seq_len)
		meth = self.M5.fetch(self.cur_chrom, self.cur+1, self.cur+seq_len)
		assert(len(seq) == len(meth))
		if self.cur+1+seq_len <= self.cur_len:
			self.cur += 1
		else:
			self.cur_chrom = self.chroms.pop()
			self.cur_len = self.FA.get_reference_length(self.cur_chrom)
			self.cur_rc = refcache(FA, self.cur_chrom)
			self.cur = 0
		print(seq, meth)
