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
import numpy as np
#try:
#	from unittest.mock import patch
#except:
#	from mock import patch

class TestReader(unittest.TestCase):
	def setUp(self):
		tpath = os.path.dirname(__file__)
		self.fa = os.path.join(tpath, 'test.fa')
		self.fai = os.path.join(tpath, 'test.fa.fai')
		self.gff3 = os.path.join(tpath, 'test.gff3')
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
	def test_gff2interval(self):
		FA = FastaFile(self.fa)
		itd = reader.gff2interval(self.gff3, FA.references)
		FA.close()
		res1 = itd['Chr1'].search(0,2)
		self.assertEqual(len(res1), 2)
		res2 = itd['Chr1'].search(0,3)
		self.assertEqual(len(res2), 3)
	def test_gff2array(self):
		##gff-version   3
		#Chr1	test	CDS	3	10	.	+	.	ID
		#Chr1	test	gene	3	10	.	+	.	ID
		#Chr1	test	exon	4	7	.	+	.	ID
		#Chr2	test	CDS	2	15	.	-	.	ID
		#Chr2	test	gene	2	15	.	-	.	ID
		#Chr2	test	exon	3	7	.	-	.	ID
		#Chr2	test	exon	9	14	.	-	.	ID
		FA = FastaFile(self.fa)
		itd = reader.gff2interval(self.gff3, FA.references)
		FA.close()
		res1 = reader.intervals2features(itd, 'Chr1', 0, 5)
		tmp = np.zeros((5,len(reader.gff3_dict)))
		tmp[2:5,reader.gff3_dict['+CDS']] = 1
		tmp[2:5,reader.gff3_dict['+gene']] = 1
		tmp[3:5,reader.gff3_dict['+exon']] = 1
		self.assertTrue(np.array_equal(res1, tmp))
		res2 = reader.intervals2features(itd, 'Chr2', 0, 18)
		tmp = np.zeros((18,len(reader.gff3_dict)))
		tmp[1:15,reader.gff3_dict['-CDS']] = 1
		tmp[1:15,reader.gff3_dict['-gene']] = 1
		tmp[2:7,reader.gff3_dict['-exon']] = 1
		tmp[8:14,reader.gff3_dict['-exon']] = 1
		self.assertTrue(np.array_equal(res2, tmp))
	def test_input_iter_gff3(self):
		XY = reader.input_gen(self.fa, self.mr1, gff3=self.gff3, seq_len=5)
		XYL = list(XY)
		self.assertEqual(XYL[0][0][0], [0, 1.0/20, 0,0,0,0,0,0])
		self.assertEqual(XYL[9][0][0], [1, 10.0/20, 0,0,0,0,10.0/20,20])
		self.assertEqual(XYL[9][1][0][reader.gff3_dict['+CDS']], 1)
		self.assertEqual(sum(XYL[0][1][0]), 0)
		self.assertEqual(sum(XYL[1][1][0]), 0)
		self.assertEqual(sum(XYL[10][1][0]), 0)
		self.assertEqual(sum(XYL[11][1][0]), 0)
		self.assertEqual(sum(XYL[8][1][2]), 0)
		self.assertEqual(sum(XYL[8][1][1]), 2)
		self.assertEqual(len(XYL), 15+15)

if __name__ == "__main__":
	unittest.main()
