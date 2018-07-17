import unittest, sys
from teamRNN import reader
#try:
#	from unittest.mock import patch
#except:
#	from mock import patch
#from StringIO import StringIO
from os import path

class TestReader(unittest.TestCase):
	def setUp(self):
		self.fasta = path.join(path.dirname(__file__),'test.fa')
		self.RP = reader.refParser(self.fasta, 1, 5)
		self.methFile = path.join(path.dirname(__file__),'test1_methratio.txt')
	def test_odd(self):
		self.assertTrue(reader.isOdd(5))
		self.assertFalse(reader.isOdd(4))
		self.assertFalse(reader.isOdd(0))
		self.assertTrue(reader.isOdd(-1))
	def test_even(self):
		self.assertTrue(reader.isEven(4))
		self.assertFalse(reader.isEven(5))
		self.assertTrue(reader.isEven(0))
		self.assertFalse(reader.isEven(-1))
	def test_kmer2index(self):
		# AGAG
		# 0101
		self.assertEqual(reader.kmer2index('AGAG'), 17)
		self.assertEqual(reader.kmer2index('AGAN'), -1)
		self.assertEqual(reader.kmer2index('N'), -1)
	def test_up(self):
		self.assertEqual(self.RP.up, 2)
	def test_down(self):
		self.assertEqual(self.RP.down, 2)
	def test_array_Chr1(self):
		chr1A = 'AAAAAAGGGGCCCCTTTTTT'
		self.assertEqual(self.RP.wrapBases('Chr1'), (16, 20, chr1A))
	def test_kmer_worker(self):
		cLen, wrapLen, wrapChr1A = self.RP.wrapBases('Chr1')
		#chr1A = 'AAAAAAGGGGCCCCTTTTTT'
		chr1A = [0,0,1,5,21,85,342,346,362,426,683,687,703,767,1023,1023]
		reader.baseArray = wrapChr1A
		reader.kmerArray = [0]*cLen
		reader.kmer_worker((self.RP.k, 0, 1))
		self.assertEqual(reader.kmerArray, chr1A)
		del reader.baseArray, reader.kmerArray
	def test_kmer_Chr1(self):
		#chr1A = 'AAAAAAGGGGCCCCTTTTTT'
		#chr1A = '00000011112222333333'
		chr1A = [0,0,1,5,21,85,342,346,362,426,683,687,703,767,1023,1023]
		self.assertEqual(list(self.RP.getArray('Chr1',4)), chr1A)
		self.assertEqual(self.RP.FA.get_reference_length('Chr1'), len(chr1A))
	def test_array_Chr2(self):
		chr2A = 'ATCTAGCTAGCTAGCTAGAT'
		self.assertEqual(self.RP.wrapBases('Chr2'), (16, 20, chr2A))
	def test_meth_array_Chr1(self):
		self.MF = reader.methParser(self.fasta, self.methFile, 5)
		self.MF.getArray('Chr1')
		
#		chr     pos     strand  context ratio   eff_CT  C       CT
#		Chr1    5       -       CHH     0.0     0.0     5       7
#		Chr1    6       -       CHH     0.0     0.0     0       0
#		Chr1    7       -       CHH     0.0     0.0     3       7
#		Chr1    8       -       CHH     0.0     0.0     2       7
#		Chr1    9       +       CHH     0.0     0.0     5       5
#		Chr1    10      +       CHH     0.0     0.0     4       7
#		Chr1    11      +       CHH     0.0     0.0     2       2
#		Chr1    12      +       CHH     0.0     0.0     0       7
#		Chr2    1       +       CG      0.0     0.0     1       9
#		Chr2    2       -       CG      0.0     0.0     1       2
#		Chr2    5       +       CG      0.0     0.0     6       7
#		Chr2    6       -       CG      0.0     0.0     8       8
#		Chr2    9       +       CG      0.0     0.0     7       8
#		Chr2    10      -       CG      0.0     0.0     2       3
#		Chr2    13      +       CG      0.0     0.0     1       1
#		Chr2    14      -       CG      0.0     0.0     2       2
#		Chr2    15      -       CHG     0.0     0.0     1       5
		


if __name__ == "__main__":
	unittest.main()
