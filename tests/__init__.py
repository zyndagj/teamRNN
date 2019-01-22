import unittest, sys, os
try:
	from StringIO import StringIO
except:
	from io import StringIO
# buffer for capturing log info
logStream = StringIO()
# Need to start logger BEFORE importing any pyPlateCalibrate code
import logging
FORMAT = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)15s] %(message)s"
logging.basicConfig(stream=logStream, level=logging.DEBUG, format=FORMAT)
from teamRNN import reader
from pysam import FastaFile
#try:
#	from unittest.mock import patch
#except:
#	from mock import patch

class TestReader(unittest.TestCase):
	def setUp(self):
		tpath = os.path.dirname(__file__)
		self.fa = os.path.join(tpath, 'test.fa')
		self.fai = os.path.join(tpath, 'test.fa.fai')
		self.gff3 = os.path.join(tpath, 'test.gff')
		self.mr1 = os.path.join(tpath, 'test_meth.txt')
	def tearDown(self):
		## Runs after every test function ##
		# Wipe log
		logStream.truncate(0)
		## Runs after every test function ##
	def test_refcache(self):
		FA = FastaFile(self.fa)
		for chrom in FA.references:
			chrom_len = FA.get_reference_length(chrom)
			chrom_seq = FA.fetch(chrom)
			rc = reader.refcache(FA, chrom)
			self.assertEqual(chrom_seq, rc.fetch(0, chrom_len))
			for i in range(chrom_len-3):
				self.assertEqual(rc.fetch(i, i+3), FA.fetch(chrom, i, i+3))
		# clean up
		FA.close()
		del rc
	def test_input_iter(self):
		IG = reader.input_gen(self.fa, self.mr1, seq_len=5)
		IL = list(IG)
		self.assertEqual(IL[0][0], [0, 1.0/20, 0,0,0,0,0,0])
		self.assertEqual(IL[9][0], [1, 10.0/20, 0,0,0,0,10.0/20,20])
		self.assertEqual(len(IL), 15+15)
	def test_input_iter_gff3(self):
		IG = reader.input_gen(self.fa, self.mr1, gff3=self.gff3, seq_len=5)
		IL = list(IG)
		self.assertEqual(IL[0][0], [0, 1.0/20, 0,0,0,0,0,0])
		self.assertEqual(IL[9][0], [1, 10.0/20, 0,0,0,0,10.0/20,20])
		self.assertEqual(len(IL), 15+15)

if __name__ == "__main__":
	unittest.main()
