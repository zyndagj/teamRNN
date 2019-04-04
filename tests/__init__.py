import unittest, sys, os
from glob import glob
from time import time
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
from teamRNN import reader, constants, writer, model
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
	def test_split2quality(self):
		self.assertEqual(reader._split2quality('> dna:chromosome bears'.split(' ')), 3)
		self.assertEqual(reader._split2quality('> dna:scaffold'.split(' ')), 2)
		self.assertEqual(reader._split2quality('> dna:supercontig'.split(' ')), 1)
		self.assertEqual(reader._split2quality('> dna:contig hello there'.split(' ')), 1)
		self.assertEqual(reader._split2quality('>cats'.split(' ')), 0)
		self.assertEqual(reader._split2quality('>cats and dogs'.split(' ')), 0)
	def test_refcache_quality(self):
		RC = reader.refcache(self.fa)
		FA = FastaFile(self.fa)
		for chrom in FA.references:
			self.assertEqual(RC.chrom_qualities[chrom], 3)
	def test_input_iter(self):
		I = reader.input_slicer(self.fa, self.mr1)
		IL = list(I.genome_iter())
		#for c, x in IL:
		#	print(''.join(map(lambda i: constants.index2base[i[0]], x)))
		self.assertEqual(IL[0][1][0], [0, 1.0/20, 0,0,0,0,0,0, 2,3])
		self.assertEqual(IL[9][1][0], [1, 10.0/20, 0,0,0,0,10.0/20,20, 2,3])
		self.assertEqual(len(IL), 16+16)
	def test_input_iter_10(self):
		I = reader.input_slicer(self.fa, self.mr1)
		IL = list(I.genome_iter(seq_len=10))
		self.assertEqual(len(IL), 11+11)
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
		self.assertEqual(len(XYL), (20-5+1)*2)
		self.assertEqual(XYL[0][1][0], [0, 1.0/20, 0,0,0,0,0,0, 2,3])
		self.assertEqual(XYL[9][1][0], [1, 10.0/20, 0,0,0,0,10.0/20,20, 2,3])
		self.assertEqual(XYL[9][2][0][constants.gff3_f2i['+CDS']], 1)
		self.assertEqual(sum(XYL[0][2][0]), 0)
		self.assertEqual(sum(XYL[1][2][0]), 0)
		self.assertEqual(sum(XYL[10][2][0]), 0)
		self.assertEqual(sum(XYL[11][2][0]), 0)
		self.assertEqual(sum(XYL[8][2][2]), 0)
		self.assertEqual(sum(XYL[8][2][1]), 2)
		self.assertEqual(len(XYL), 16+16)
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
	def test_batch_gff3(self):
		#12345678901234567890
		#    1234567890123456
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		BL = list(IS.batch_iter(batch_size=4))
		self.assertEqual(len(BL), 4+4)
		for out in BL:
			self.assertEqual(len(out), 3 if IS.gff3_file else 2)
			self.assertEqual(np.array(out[1]).shape, (4, 5, 10))
			if IS.gff3_file:
				self.assertEqual(np.array(out[2]).shape, (4, 5, len(constants.gff3_f2i)))
	def test_batch(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.batch_iter(batch_size=4))
		self.assertEqual(len(BL), 4+4)
		for out in BL:
			self.assertEqual(len(out), 3 if IS.gff3_file else 2)
			self.assertEqual(np.array(out[1]).shape, (4, 5, 10))
			if IS.gff3_file:
				self.assertEqual(np.array(out[2]).shape, (4, 5, len(constants.gff3_f2i)))
	def test_batch_10(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.batch_iter(seq_len=10, batch_size=11))
		self.assertEqual(len(BL), np.round(22/11))
		for out in BL:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (11, 10, 10))
	def test_batch_uneven(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.batch_iter(batch_size=5))
		self.assertEqual(len(BL), 7)
		for out in BL[:-1]:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (5, 5, 10))
		out = BL[-1]
		self.assertEqual(np.array(out[1]).shape, (2, 5, 10))
	def test_same_model(self):
		seq_len, n_inputs, n_outputs = 5, 10, len(constants.gff3_f2i)
		model.reset_graph()
		# create models
		models = []
		for i in range(2):
			name = 'model_%i'%(i)
			m = model.sleight_model(name, n_inputs, seq_len, n_outputs, \
				n_neurons=20, n_layers=1, learning_rate=0.001, training_keep=0.95, \
				dropout=False, cell_type='rnn', peep=False, stacked=False, \
				bidirectional=False, reg_losses=False, hidden_list=[])
			models.append(m)
		# train models
		for epoch in range(3):
			IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
			for cb, xb, yb in IS.batch_iter(seq_len, batch_size=4):
				mse = [m.train(xb, yb) for m in models]
				self.assertEqual(mse[0], mse[1])
	def test_model_effect(self):
		seq_len, n_inputs, n_outputs = 5, 10, len(constants.gff3_f2i)
		model.reset_graph()
		# create models
		models = [model.sleight_model('d%i'%(i), n_inputs, seq_len, n_outputs) for i in range(2)]
		models.append(model.sleight_model('neurons', n_inputs, seq_len, n_outputs, n_neurons=50))
		models.append(model.sleight_model('layers', n_inputs, seq_len, n_outputs, n_layers=2))
		models.append(model.sleight_model('learning', n_inputs, seq_len, n_outputs, learning_rate=0.01))
		models.append(model.sleight_model('dropout', n_inputs, seq_len, n_outputs, training_keep=0.9, dropout=True))
		models.append(model.sleight_model('lstm', n_inputs, seq_len, n_outputs, cell_type='lstm'))
		models.append(model.sleight_model('lstm_peep', n_inputs, seq_len, n_outputs, cell_type='lstm', peep=True))
		models.append(model.sleight_model('peep', n_inputs, seq_len, n_outputs, peep=True))
		models.append(model.sleight_model('stacked', n_inputs, seq_len, n_outputs, stacked=True))
		models.append(model.sleight_model('bidirectional', n_inputs, seq_len, n_outputs, bidirectional=True))
		models.append(model.sleight_model('reg_losses', n_inputs, seq_len, n_outputs, reg_losses=True))
		models.append(model.sleight_model('reg_losses_stacked', n_inputs, seq_len, n_outputs, stacked=True, reg_losses=True))
		models.append(model.sleight_model('hidden', n_inputs, seq_len, n_outputs, hidden_list=[10]))
		# train models
		first = True
		for epoch in range(3):
			IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
			for cb, xb, yb in IS.batch_iter(seq_len, batch_size=4):
				mse = [m.train(xb, yb) for m in models]
				# Should be qual
				self.assertEqual(mse[0], mse[1])
				self.assertEqual(mse[0], mse[8])
				self.assertEqual(mse[0], mse[11])
				if first:
					self.assertEqual(mse[0], mse[4])
				else:
				# Should be different
					self.assertNotEqual(mse[0], mse[4])
				self.assertNotEqual(mse[0], mse[2])
				self.assertNotEqual(mse[0], mse[3])
				self.assertNotEqual(mse[0], mse[5])
				self.assertNotEqual(mse[0], mse[6])
				self.assertNotEqual(mse[0], mse[7])
				self.assertNotEqual(mse[0], mse[9])
				self.assertNotEqual(mse[0], mse[10])
				self.assertNotEqual(mse[0], mse[12])
				self.assertNotEqual(mse[11], mse[12])
				self.assertNotEqual(mse[0], mse[13])
				first = False
	def test_train_01(self):
		from random import shuffle, random
		seq_len = 15
		batch_size = 6
		n_inputs, n_outputs = 10, len(constants.gff3_f2i)
		model.reset_graph()
		# create models
		M = model.sleight_model('default', n_inputs, seq_len, n_outputs, bidirectional=True, save_dir='test_model')
		# train models
		ts = time()
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		ISBL = list(IS.batch_iter(seq_len, batch_size))
		for epoch in range(1,1001):
			for cb, xb, yb in ISBL:
				# shuffle indices
				self.assertEqual(len(xb), batch_size)
				randInds = list(range(len(xb)))
				shuffle(randInds)
				xbs = [xb[i] for i in randInds]
				ybs = [yb[i] for i in randInds]
				if random() > 0.2:
					M.train(xbs, ybs)
			if epoch % 500 == 0:
				#print("Trained epoch %i in %.2f seconds"%(epoch, time()-ts))
				ts = time()
		M.save()
		self.assertTrue(len(glob('%s*'%(M.save_file))) > 0)
		# Vote
		OA = writer.output_aggregator(self.fa)
		for c, xb, yb in IS.batch_iter(seq_len, batch_size=1):
			y = yb[0]
			y_pred = M.predict(xb, yb)[0]
			for feature_index in range(len(y[0])):
				yl, ypl = [], []
				for base_index in range(len(y)):
					yl.append('%i'%(y[base_index][feature_index]))
					ypl.append('%i'%(y_pred[base_index][feature_index]))
				ys = ', '.join(yl)
				yps = ', '.join(ypl)
				if ys != yps:
					print("%s:%i-%i FI:%2i Y=[%s]  Y_PRED=[%s]"%(*c[0], feature_index, ys, yps))
			self.assertTrue(np.array_equal(y, y_pred))
			OA.vote(*c[0], array=y_pred)
		# Compare
		out_lines = OA.write_gff3()
		with open(self.gff3,'r') as GFF3:
			for ol, fl in zip(out_lines, GFF3.readlines()):
				if ol[0] != '#':
					ols = ol.split('\t')[:7]
					ols[1] = 'test'
					fls = fl.split('\t')[:7]
					self.assertEqual(ols, fls)
	def test_train_02(self):
		from random import shuffle, random
		seq_len = 15
		batch_size = 6
		n_inputs, n_outputs = 10, len(constants.gff3_f2i)
		model.reset_graph()
		# create models
		M = model.sleight_model('default', n_inputs, seq_len, n_outputs, bidirectional=True, save_dir='test_model')
		self.assertTrue(len(glob('%s*'%(M.save_file))) > 0)
		M.restore()
		# Vote
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		OA = writer.output_aggregator(self.fa)
		for c, xb, yb in IS.batch_iter(seq_len, batch_size=1):
			y = yb[0]
			y_pred = M.predict(xb, yb)[0]
			for feature_index in range(len(y[0])):
				yl, ypl = [], []
				for base_index in range(len(y)):
					yl.append('%i'%(y[base_index][feature_index]))
					ypl.append('%i'%(y_pred[base_index][feature_index]))
				ys = ', '.join(yl)
				yps = ', '.join(ypl)
				if ys != yps:
					print("%s:%i-%i FI:%2i Y=[%s]  Y_PRED=[%s]"%(*c[0], feature_index, ys, yps))
			self.assertTrue(np.array_equal(y, y_pred))
			OA.vote(*c[0], array=y_pred)
		# Compare
		out_lines = OA.write_gff3()
		with open(self.gff3,'r') as GFF3:
			for ol, fl in zip(out_lines, GFF3.readlines()):
				if ol[0] != '#':
					ols = ol.split('\t')[:7]
					ols[1] = 'test'
					fls = fl.split('\t')[:7]
					self.assertEqual(ols, fls)
		if os.path.exists(M.save_dir):
			from shutil import rmtree
			rmtree(M.save_dir)

if __name__ == "__main__":
	unittest.main()
