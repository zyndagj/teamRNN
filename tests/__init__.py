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
from teamRNN import reader, constants, writer
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
		RC = reader.refcache(self.fa)
		FA = FastaFile(self.fa)
		for chrom in FA.references:
			chrom_len = FA.get_reference_length(chrom)
			chrom_seq = FA.fetch(chrom)
			self.assertEqual(chrom_seq, RC.fetch(chrom, 0, chrom_len))
			for i in range(chrom_len-3):
				self.assertEqual(RC.fetch(chrom, i, i+3), FA.fetch(chrom, i, i+3))
		# clean up
		FA.close()
	def test_input_iter(self):
		I = reader.input_slicer(self.fa, self.mr1)
		IL = list(I.genome_iter())
		self.assertEqual(IL[0][1][0], [0, 1.0/20, 0,0,0,0,0,0])
		self.assertEqual(IL[9][1][0], [1, 10.0/20, 0,0,0,0,10.0/20,20])
		self.assertEqual(len(IL), 15+15)
	def test_gff2interval(self):
		GI = reader.gff3_interval(self.gff3)
		res1 = GI.interval_tree['Chr1'].search(0,2)
		self.assertEqual(len(res1), 2)
		res2 = GI.interval_tree['Chr1'].search(0,3)
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
		GI = reader.gff3_interval(self.gff3)
		res1 = GI.fetch('Chr1', 0, 5)
		tmp = np.zeros((5,len(constants.gff3_f2i)))
		tmp[2:5,constants.gff3_f2i['+CDS']] = 1
		tmp[2:5,constants.gff3_f2i['+gene']] = 1
		tmp[3:5,constants.gff3_f2i['+exon']] = 1
		self.assertTrue(np.array_equal(res1, tmp))
		res2 = GI.fetch('Chr2', 0, 18)
		tmp = np.zeros((18,len(constants.gff3_f2i)))
		tmp[1:15,constants.gff3_f2i['-CDS']] = 1
		tmp[1:15,constants.gff3_f2i['-gene']] = 1
		tmp[2:7,constants.gff3_f2i['-exon']] = 1
		tmp[8:14,constants.gff3_f2i['-exon']] = 1
		self.assertTrue(np.array_equal(res2, tmp))
	def test_input_iter_gff3(self):
		I = reader.input_slicer(self.fa, self.mr1, self.gff3)
		XYL = list(I.genome_iter())
		self.assertEqual(XYL[0][1][0], [0, 1.0/20, 0,0,0,0,0,0])
		self.assertEqual(XYL[9][1][0], [1, 10.0/20, 0,0,0,0,10.0/20,20])
		self.assertEqual(XYL[9][2][0][constants.gff3_f2i['+CDS']], 1)
		self.assertEqual(sum(XYL[0][2][0]), 0)
		self.assertEqual(sum(XYL[1][2][0]), 0)
		self.assertEqual(sum(XYL[10][2][0]), 0)
		self.assertEqual(sum(XYL[11][2][0]), 0)
		self.assertEqual(sum(XYL[8][2][2]), 0)
		self.assertEqual(sum(XYL[8][2][1]), 2)
		self.assertEqual(len(XYL), 15+15)
	def test_vote(self):
		from functools import reduce
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		OA = writer.output_aggregator(self.fa)
		for c,x,y in IS.genome_iter():
			#iSet = reduce(set.union, [set(np.where(b == 1)[0]) for b in y])
			#fList = sorted([constants.gff3_i2f[i] for i in iSet])
			#print("%s:%i-%i [%s]"%(*c, ', '.join(fList)))
			OA.vote(*c, array=y)
		out_lines = OA.write_gff3()
		with open(self.gff3,'r') as GFF3:
			for ol, fl in zip(out_lines, GFF3.readlines()):
				if ol[0] != '#':
					ols = ol.split('\t')[:7]
					ols[1] = 'test'
					fls = fl.split('\t')[:7]
					self.assertEqual(ols, fls)
		#print('\n'.join(OA.write_gff3()))

if __name__ == "__main__":
	unittest.main()
