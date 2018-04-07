#!/usr/bin/env python

from pysam import FastaFile
from multiprocessing import Pool, RawArray, cpu_count
from string import maketrans
from array import array

class refcache:
	def __init__(self, FA, chrom, chromLen, cacheSize=50000):
		self.FA = FA
		self.chrom = chrom
		self.chromLen = chromLen
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

def isOdd(num):
	return num % 2 != 0
def isEven(num):
	return num % 2 == 0

#class methParser:
#	def __init__(self, fastaFile, methFile, windowSize, slideSize):
#	def getArray(self, chrom):
#	def get(self, chrom, index):

b2i = maketrans('AGCT','0123')
def kmer2index(kmer):
	'''
	Transforms a kmer into its index. While hashing may
	be more efficient it removes the possibility of collisions.

	Returns -1 if N is present.

	Keyword arguments:
	kmer -- string of dna bases

	Output:
	int -- base10 index of kmer (base 4)
	'''
	if 'N' in kmer: return -1
	return int(kmer.translate(b2i),4)

def kmer_worker(args):
	global baseArray, kmerArray
	k, pid, Nprocs = args
	down = (k-1)/2
	up = down+1
	for i in xrange(pid, len(kmerArray), Nprocs):
		kmerArray[i] = kmer2index(baseArray[i:i+up+down])
	return 0

class refParser:
	def __init__(self, fastaFile, slideSize, k):
		if isEven(k): raise ValueError("k should be odd")
		self.k = k
		self.FA = FastaFile(fastaFile)
		self.chromosomes = sorted(self.FA.references)
		self.slideSize = slideSize
		self.down = (k-1)/2
		self.up = self.down
	def wrapBases(self, chrom):
		'''
		Keyword arguments:
		chrom -- chromosome name (should exist in fasta file)
		
		Output: tuple containing
		int -- original length of chromosome
		int -- length of wrapped chromosome
		str -- wrapped chromosome sequence
		'''
		chromLen = self.FA.get_reference_length(chrom)
		reflectEnd = self.FA.fetch(chrom, chromLen-self.up-1, chromLen-1)[::-1]
		reflectStart = self.FA.fetch(chrom, 1, self.down+1)[::-1]
		wrapLen = chromLen + self.down + self.up
		return (chromLen, wrapLen, ''.join([reflectStart, self.FA.fetch(chrom), reflectEnd]))
	def getArray(self, chrom, nProcs=0):
		'''
		Keyword arguments:
		chrom -- chromosome name (should exist in fasta file)
		p -- numper of processors to use [all]
		
		Output:
		array(long) -- array of kmer indicies
		'''
		chromLen, wrapLen, chromStr = self.wrapBases(chrom)
		reflectStart = self.FA.fetch(chrom, 1, self.down+1)[::-1]
		global baseArray
		baseArray = RawArray('c', chromStr)
		del chromStr
		global kmerArray
		kmerArray = RawArray('L', [0]*chromLen)
		N = nProcs if nProcs else cpu_count()
		p = Pool(N)
		ret = p.map(kmer_worker, [(self.k, i, N) for i in xrange(N)])
		p.close()
		p.join()
		del baseArray
		A = array('L', kmerArray)
		del kmerArray
		return A
