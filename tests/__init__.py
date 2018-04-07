import unittest, sys
from teamRNN import reader
#try:
#	from unittest.mock import patch
#except:
#	from mock import patch
from os import path
#from StringIO import StringIO

class TestReader(unittest.TestCase):
	def setUp(self):
		self.fasta = path.join(path.dirname(__file__),'test.fa')
		self.RP = reader.refParser(self.fasta, 1, 5)
	def test_up(self):
		self.assertEqual(self.RP.up, 2)
	def test_down(self):
		self.assertEqual(self.RP.down, 2)
	def test_array_Chr1(self):
		chr1A = 'AAAAAAGGGGCCCCTTTTTT'
		self.assertEqual(self.RP.wrapBases('Chr1'), (16, 20, chr1A))
	def test_kmer_Chr1(self):
		#chr1A = 'AAAAAAGGGGCCCCTTTTTT'
		#chr1A = '00000011112222333333'
		chr1A = [0,0,1,5,21,85,342,346,362,426,683,687,703,767,1023,1023]
		self.assertEqual(list(self.RP.getArray('Chr1',4)), chr1A)
		self.assertEqual(self.RP.FA.get_reference_length('Chr1'), len(chr1A))
	def test_array_Chr2(self):
		chr2A = 'ATCTAGCTAGCTAGCTAGAT'
		self.assertEqual(self.RP.wrapBases('Chr2'), (16, 20, chr2A))

if __name__ == "__main__":
	unittest.main()
