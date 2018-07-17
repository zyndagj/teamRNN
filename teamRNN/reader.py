#!/usr/bin/env python

from pysam import FastaFile
from multiprocessing import Pool, RawArray, cpu_count
from string import maketrans
from array import array
import subprocess as sp
import numpy as np

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

#contextDict = {'+':{'CG':6, 'CHG':8, 'CHH':10}, '-':{'CG':7, 'CHG':9, 'CHH':11}}
def methHelper(line):
	tmp = line.split('\t')
	index = int(tmp[1])-1
	C, CT = map(int, tmp[6:8])
	sign, context = tmp[2:4]
	mDict[sign][context][index] = C
	tDict[sign][context][index] = CT
class methParser:
	def __init__(self, fastaFile, methFile, windowSize, slideSize=1):
		# Load fasta
		self.FA = FastaFile(fastaFile)
		self.chromosomes = sorted(self.FA.references)
		self.methFile = methFile
		self.windowSize = windowSize
		# Always use a slide size of 1 to match kmer data
		self.slideSize = slideSize
	def genOutput(self, chrom):
		chromLen = self.FA.get_reference_length(chrom)
		down = (self.windowSize-1)/2
		for i in xrange(chromLen):
			s = i-down
			e = i+down+1
			if s < 0:
				I = range(abs(s), 0, -1)+range(0, e)
			elif e >= chromLen:
				diff = e-chromLen
				I = range(s, chromLen)+range(chromLen-diff-1, chromLen-1)
			else:
				I = range(s,e)
			# Print CHH
			print [t+1 for t in I]
			for context in ('CG', 'CHG', 'CHH'):
				for strand in ('+','-'):
					# Calc these
			mp = sum([mCHHp[t] for t in I])
			tp = sum([tCHHp[t] for t in I])
			mn = sum([mCHHn[t] for t in I])
			tn = sum([tCHHn[t] for t in I])
			print chrom, i+1, "%i/%i+\t%i/%i-"%(mp,tp,mn,tn)
	def getArray(self, chrom, nProcs=0):
		# Make sure chromosome is valid
		if chrom not in self.chromosomes:
			raise ValueError("%s not in %s"%(chrom, self.FA.filename))
		chromLen = self.FA.get_reference_length(chrom)
		# mC*p for positive strand
		global mCGp, tCGp, mCHGp, tCHGp, mCHHp, tCHHp
		mCGp = RawArray('h', [0]*chromLen)
		tCGp = RawArray('h', [0]*chromLen)
		mCHGp = RawArray('h', [0]*chromLen)
		tCHGp = RawArray('h', [0]*chromLen)
		mCHHp = RawArray('h', [0]*chromLen)
		tCHHp = RawArray('h', [0]*chromLen)
		# mC*n for negative string
		global mCGn, tCGn, mCHGn, tCHGn, mCHHn, tCHHn
		mCGn = RawArray('h', [0]*chromLen)
		tCGn = RawArray('h', [0]*chromLen)
		mCHGn = RawArray('h', [0]*chromLen)
		tCHGn = RawArray('h', [0]*chromLen)
		mCHHn = RawArray('h', [0]*chromLen)
		tCHHn = RawArray('h', [0]*chromLen)
		global mDict, tDict
		mDict = {'+':{'CG':mCGp, 'CHG':mCHGp, 'CHH':mCHHp}, '-':{'CG':mCGn, 'CHG':mCHGn, 'CHH':mCHHn}}
		tDict = {'+':{'CG':tCGp, 'CHG':tCHGp, 'CHH':tCHHp}, '-':{'CG':tCGn, 'CHG':tCHGn, 'CHH':tCHHn}}
		# Initialize workers
		N = nProcs if nProcs else cpu_count()
		p = Pool(N)
		# Filter file and process
		chromLines = sp.Popen('LC_ALL=C grep "^%s\s" %s'%(chrom, self.methFile), shell=True, stdout=sp.PIPE).stdout
		ret = p.map_async(methHelper, chromLines, 100)
		# Close pool
		p.close()
		p.join()
		# Stack arrays and return
		self.genOutput(chrom)
		#outArray = np.hstack((mCG, tCG, mCHG, tCHG, mCHH, tCHH))
		# Delete shared arrays
		del mCGp, tCGp, mCHGp, tCHGp, mCHHp, tCHHp
		del mCGn, tCGn, mCHGn, tCHGn, mCHHn, tCHHn
		del mDict, tDict
		return outArray

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
