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
import teamRNN
from teamRNN import reader, constants, writer, model
from pysam import FastaFile
import numpy as np
try:
	from unittest.mock import patch
except:
	from mock import patch

class TestReader(unittest.TestCase):
	def setUp(self):
		tpath = os.path.dirname(__file__)
		self.fa = os.path.join(tpath, 'test.fa')
		self.fai = os.path.join(tpath, 'test.fa.fai')
		self.gff3 = os.path.join(tpath, 'test.gff3')
		self.mr1 = os.path.join(tpath, 'test_meth.txt')
		self.n_inputs = 10
		self.n_outputs = len(constants.gff3_f2i)+2
		self.test_model = False
		self.n_epoch = 500
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
	def test_chrom_iter(self):
		I = reader.input_slicer(self.fa, self.mr1)
		CL = [c for c, x in I.chrom_iter('Chr1', seq_len=5, offset=1, batch_size=False, hvd_rank=0, hvd_size=1)]
		print CL
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
		GI = reader.gff3_interval(self.gff3, force=True)
		res1 = GI.interval_tree['Chr1'].search(0,2)
		self.assertEqual(len(res1), 2)
		res2 = GI.interval_tree['Chr1'].search(0,3)
		self.assertEqual(len(res2), 3)
	def test_gff2array(self):
		###gff-version   3
		#Chr1    test    CDS     3       10      .       +       .       ID=team_0
		#Chr1    test    gene    3       10      .       +       .       ID=team_1
		#Chr1    test    exon    4       7       .       +       .       ID=team_2
		#Chr1    test    transposable_element    11      15      .       -       .       ID=team_3;Order=LTR;Superfamily=Gypsy
		#Chr2    test    CDS     2       15      .       -       .       ID=team_4
		#Chr2    test    gene    2       15      .       -       .       ID=team_5
		#Chr2    test    exon    3       7       .       -       .       ID=team_6
		#Chr2    test    exon    9       14      .       -       .       ID=team_7
		GI = reader.gff3_interval(self.gff3, force=True)
		res1 = GI.fetch('Chr1', 0, 15)
		tmp = np.zeros((15, self.n_outputs), dtype=np.uint8)
		tmp[2:10,constants.gff3_f2i['+CDS']] = 1
		tmp[2:10,constants.gff3_f2i['+gene']] = 1
		tmp[3:7,constants.gff3_f2i['+exon']] = 1
		tmp[10:15,constants.gff3_f2i['-transposable_element']] = 1
		tmp[10:15,len(constants.gff3_f2i)] = constants.te_order_f2i['ltr']
		tmp[10:15,len(constants.gff3_f2i)+1] = constants.te_sufam_f2i['gypsy']
		self.assertEqual(res1.shape, tmp.shape)
		for i in range(15):
			if not np.array_equal(res1[i], tmp[i]):
				print("At index %i"%(i))
				print("Code:",res1[i])
				print("Test:",tmp[i])
		self.assertTrue(np.array_equal(res1, tmp))
		res2 = GI.fetch('Chr2', 0, 18)
		tmp = np.zeros((18,self.n_outputs))
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
		for i in range(10, 15):
			self.assertEqual(XYL[i][2][0][constants.gff3_f2i['-transposable_element']], 1)
			self.assertEqual(XYL[i][2][0][len(constants.gff3_f2i)], 3)
			self.assertEqual(XYL[i][2][0][len(constants.gff3_f2i)+1], 7)
		self.assertEqual(sum(XYL[0][2][0]), 0)
		self.assertEqual(sum(XYL[1][2][0]), 0)
		self.assertEqual(sum(XYL[16][2][0]), 0)
		self.assertEqual(sum(XYL[8][2][2]), 11)
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
					ols = ol.split('\t')[:9]
					ols[1] = 'test'
					fls = fl.rstrip('\n').split('\t')[:9]
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
				self.assertEqual(np.array(out[2]).shape, (4, 5, self.n_outputs))
	def test_batch(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.batch_iter(seq_len=5, batch_size=4))
		#for c,x in BL: print c
		self.assertEqual(len(BL), 4+4)
		for out in BL:
			self.assertEqual(len(out), 3 if IS.gff3_file else 2)
			self.assertEqual(np.array(out[1]).shape, (4, 5, 10))
			if IS.gff3_file:
				self.assertEqual(np.array(out[2]).shape, (4, 5, len(constants.gff3_f2i)))
	def test_batch_new(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.new_batch_iter(seq_len=5, batch_size=4))
		#for c,x in BL: print c
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
	def test_batch_new_10(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.new_batch_iter(seq_len=10, batch_size=11))
		self.assertEqual(len(BL), np.round(22/11))
		for out in BL:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (11, 10, 10))
	def test_batch_uneven(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.batch_iter(batch_size=5))
		#for c,x in BL: print c
		self.assertEqual(len(BL), 7)
		for out in BL[:-1]:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (5, 5, 10))
		out = BL[-1]
		self.assertEqual(np.array(out[1]).shape, (2, 5, 10))
	def test_batch_new_uneven(self):
		IS = reader.input_slicer(self.fa, self.mr1)
		BL = list(IS.new_batch_iter(batch_size=5))
		#for c,x in BL: print c
		self.assertEqual(len(BL), 8)
		for out in BL[:3]:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (5, 5, 10))
		for out in BL[4:7]:
			self.assertEqual(len(out), 2)
			self.assertEqual(np.array(out[1]).shape, (5, 5, 10))
		for index in (3,7):
			self.assertEqual(len(BL[index][0]), 1)
			self.assertEqual(BL[index][1].shape, (1,5,10))
	def test_batch_new_vs_old(self):
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		c_old, x_old, y_old = [], [], []
		c_new, x_new, y_new = [], [], []
		for cb, xb, yb in IS.new_batch_iter(seq_len=3, batch_size=5):
			c_new += cb
			x_new += list(xb)
			y_new += list(yb)
		for cb, xb, yb in IS.batch_iter(seq_len=3, batch_size=5):
			c_old += cb
			x_old += xb
			y_old += list(yb)
		#print "old"
		#for i in c_old: print i
		#print "new"
		#for i in c_new: print i
		self.assertEqual(c_old, c_new)
		self.assertEqual(np.array(x_old).shape, np.array(x_new).shape)
#		for i in range(len(x_old)):
#			print c_old[i]==c_new[i], c_old[i], c_new[i]
#			for j in range(len(x_old[i])):
#				print np.array_equal(x_old[i][j],x_new[i][j]), x_old[i][j], x_new[i][j]
		self.assertTrue(np.array_equal(np.array(x_old), np.array(x_new)))
		self.assertTrue(np.array_equal(np.array(y_old), np.array(y_new)))
	def test_same_model(self):
		if not self.test_model: return
		seq_len = 5
		model.reset_graph()
		# create models
		models = []
		for i in range(2):
			name = 'model_%i'%(i)
			m = model.sleight_model(name, self.n_inputs, seq_len, self.n_outputs, \
				n_neurons=20, n_layers=1, learning_rate=0.001, training_keep=0.95, \
				dropout=False, cell_type='rnn', peep=False, stacked=False, \
				bidirectional=False, reg_losses=False, hidden_list=[])
			models.append(m)
		# train models
		for epoch in range(3):
			IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
			for cb, xb, yb in IS.new_batch_iter(seq_len, batch_size=20-seq_len+1):
				mse = [m.train(xb, yb) for m in models]
				self.assertEqual(mse[0], mse[1])
	def test_model_effect(self):
		if not self.test_model: return
		seq_len = 5
		model.reset_graph()
		# create models
		models = [model.sleight_model('d%i'%(i), self.n_inputs, seq_len, self.n_outputs) for i in range(2)]
		models.append(model.sleight_model('neurons', self.n_inputs, seq_len, self.n_outputs, n_neurons=50))
		models.append(model.sleight_model('layers', self.n_inputs, seq_len, self.n_outputs, n_layers=2))
		models.append(model.sleight_model('learning', self.n_inputs, seq_len, self.n_outputs, learning_rate=0.01))
		models.append(model.sleight_model('dropout', self.n_inputs, seq_len, self.n_outputs, training_keep=0.9, dropout=True))
		models.append(model.sleight_model('lstm', self.n_inputs, seq_len, self.n_outputs, cell_type='lstm'))
		models.append(model.sleight_model('lstm_peep', self.n_inputs, seq_len, self.n_outputs, cell_type='lstm', peep=True))
		models.append(model.sleight_model('peep', self.n_inputs, seq_len, self.n_outputs, peep=True))
		models.append(model.sleight_model('stacked', self.n_inputs, seq_len, self.n_outputs, stacked=True))
		models.append(model.sleight_model('bidirectional', self.n_inputs, seq_len, self.n_outputs, bidirectional=True))
		models.append(model.sleight_model('reg_losses', self.n_inputs, seq_len, self.n_outputs, reg_losses=True))
		models.append(model.sleight_model('reg_losses_stacked', self.n_inputs, seq_len, self.n_outputs, stacked=True, reg_losses=True))
		models.append(model.sleight_model('hidden', self.n_inputs, seq_len, self.n_outputs, hidden_list=[10]))
		# train models
		first = True
		for epoch in range(3):
			IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
			for cb, xb, yb in IS.new_batch_iter(seq_len, batch_size=20-seq_len+1):
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
		if not self.test_model: return
		from random import shuffle, random
		seq_len = 15
		batch_size = 20-seq_len+1
		model.reset_graph()
		# create models
		M = model.sleight_model('default', self.n_inputs, seq_len, self.n_outputs, bidirectional=True, save_dir='test_model')
		# train models
		ts = time()
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		ISBL = list(IS.new_batch_iter(seq_len, batch_size))
		for epoch in range(1,self.n_epoch+1):
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
		for c, xb, yb in IS.new_batch_iter(seq_len, batch_size=1):
			y = yb[0]
			y_pred = M.predict(xb)[0]
			for feature_index in range(len(y[0])):
				yl, ypl = [], []
				for base_index in range(len(y)):
					yl.append('%i'%(y[base_index][feature_index]))
					ypl.append('%i'%(y_pred[base_index][feature_index]))
				ys = ', '.join(yl)
				yps = ', '.join(ypl)
				if ys != yps:
					print("%s:%i-%i FI:%2i Y=[%s]  Y_PRED=[%s]"%(c[0][0], c[0][1], c[0][2], feature_index, ys, yps))
			self.assertTrue(np.array_equal(y, y_pred))
			OA.vote(*c[0], array=y_pred)
		# Compare
		out_lines = OA.write_gff3()
		with open(self.gff3,'r') as GFF3:
			for ol, fl in zip(out_lines, GFF3.readlines()):
				if ol[0] != '#':
					ols = ol.split('\t')[:9]
					ols[1] = 'test'
					fls = fl.rstrip('\n').split('\t')[:9]
					self.assertEqual(ols, fls)
	def test_train_02_restore(self):
		if not self.test_model: return
		from random import shuffle, random
		seq_len = 15
		batch_size = 20-seq_len+1
		model.reset_graph()
		# create models
		M = model.sleight_model('default', self.n_inputs, seq_len, self.n_outputs, bidirectional=True, save_dir='test_model')
		self.assertTrue(len(glob('%s*'%(M.save_file))) > 0)
		M.restore()
		# Vote
		IS = reader.input_slicer(self.fa, self.mr1, self.gff3)
		OA = writer.output_aggregator(self.fa)
		for c, xb, yb in IS.new_batch_iter(seq_len, batch_size=1):
			y = yb[0]
			y_pred = M.predict(xb)[0]
			for feature_index in range(len(y[0])):
				yl, ypl = [], []
				for base_index in range(len(y)):
					yl.append('%i'%(y[base_index][feature_index]))
					ypl.append('%i'%(y_pred[base_index][feature_index]))
				ys = ', '.join(yl)
				yps = ', '.join(ypl)
				if ys != yps:
					print("%s:%i-%i FI:%2i Y=[%s]  Y_PRED=[%s]"%(c[0][0], c[0][1], c[0][2], feature_index, ys, yps))
			self.assertTrue(np.array_equal(y, y_pred))
			OA.vote(*c[0], array=y_pred)
		# Compare
		out_lines = OA.write_gff3()
		with open(self.gff3,'r') as GFF3:
			for ol, fl in zip(out_lines, GFF3.readlines()):
				if ol[0] != '#':
					ols = ol.split('\t')[:9]
					ols[1] = 'test'
					fls = fl.rstrip('\n').split('\t')[:9]
					self.assertEqual(ols, fls)
		if os.path.exists(M.save_dir):
			from shutil import rmtree
			rmtree(M.save_dir)
	def test_train_cli_01(self):
		if not self.test_model: return
		testArgs = ['teamRNN', \
			'-R', self.fa, \
			'-D', 'test_cli', \
			'-N', 'plain', \
			'-M', self.mr1, \
			'-B', '6', \
			'train', \
			'-A', self.gff3, \
			'-E', str(self.n_epoch), \
			'-L', '15', \
			'-b', '-f', \
			'-C', 'rnn']
		with patch('sys.argv', testArgs):
			teamRNN.main()
		output = logStream.getvalue()
		splitOut = output.split('\n')
		self.assertTrue('Done' in splitOut[-2])
		self.assertTrue(os.path.exists('test_cli/plain_i15x10_birnn1x100_learn0.001_pF_sF_d0.95_rF_h0.ckpt.index'))
		self.assertTrue(os.path.exists('test_cli/config.pkl'))
	def test_train_cli_02(self):
		if not self.test_model: return
		testArgs = ['teamRNN', \
			'-R', self.fa, \
			'-D', 'test_cli', \
			'-N', 'plain', \
			'-M', self.mr1, \
			'-B', '6', \
			'classify', \
			'-O', 'test_cli/out.gff3']
		with patch('sys.argv', testArgs):
			teamRNN.main()
		output = logStream.getvalue()
		splitOut = output.split('\n')
		self.assertTrue('Done' in splitOut[-2])
		self.assertTrue(os.path.exists('test_cli/out.gff3'))
		F1 = open(self.gff3,'r')
		F2 = open('test_cli/out.gff3','r')
		for test_line, cli_line in zip(F1.readlines(), F2.readlines()):
			if cli_line[0] != '#':
				test_split = test_line.rstrip('\n').split('\t')
				test_split[1] = 'teamRNN'
				cli_split = cli_line.rstrip('\n').split('\t')
				self.assertEqual(test_split, cli_split)
		if os.path.exists('test_cli'):
			from shutil import rmtree
			rmtree('test_cli')

if __name__ == "__main__":
	unittest.main()
