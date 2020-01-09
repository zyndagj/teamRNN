#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 09/05/2019
###############################################################################
# BSD 3-Clause License
# 
# Copyright (c) 2019, Texas Advanced Computing Center
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import sys, os, argparse, logging
FORMAT = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
from teamRNN.constants import tacc_nodes
node_name = os.getenv('TACC_NODE_TYPE', False)
if not node_name: node_name = os.getenv('TACC_SYSTEM', False)
if node_name in tacc_nodes:
        intra, inter = tacc_nodes[node_name]
        #logger.debug("Using config for TACC %s node (%i, %i)"%(node_name, intra, inter))
        os.environ['KMP_BLOCKTIME'] = '0'
	os.environ['TF_DISABLE_MKL'] = '1'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        os.environ['OMP_NUM_THREADS'] = str(intra)
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' # Little to no effect
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2' # Little to no effect
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
from glob import glob
from time import time
import pickle
try:
	import horovod.tensorflow.keras as hvd
	#hvd.init() # Can't init until after mp forking
except:
	hvd = False
from teamRNN import reader, constants, writer, model
from teamRNN.util import irange, fivenum
from pysam import FastaFile
import numpy as np

def main():
	# Main parser and description
	parser = argparse.ArgumentParser(description="A tool for generating reference annotations with bisulfite sequencing")
	subparsers = parser.add_subparsers()
	fC = fileCheck()
	##############################################
	# Arguments that apply to both
	##############################################
	parser.add_argument('-R', '--reference', metavar="FASTA", help='Reference file', type=fC.fasta, required=True)
	parser.add_argument('-D', '--directory', metavar="DIR", help='Model directory [%(default)s]', default='model', type=str)
	parser.add_argument('-N', '--name', metavar="STR", help='Name of model to use [%(default)s]', default='default', type=str)
	parser.add_argument('-M', '--methratio', metavar='FILE', type=fC.methratio, help='Methratio file used as input', required=True)
	parser.add_argument('-o', '--offset', metavar='INT', help='Number of bases to slide between windows [%(default)s]', default=1, type=int)
	parser.add_argument('-Q', '--quality', metavar='INT', help='Input assembly quality [%(default)s]', default=-1, type=int)
	parser.add_argument('-P', '--ploidy', metavar='INT', help='Input chromosome ploidy [%(default)s]', default=2, type=int)
	parser.add_argument('--max_fill', metavar='INT', help='Maximum gap size to be filled [%(default)s]', default=50, type=int)
	parser.add_argument('--min_feat', metavar='INT', help='Minimum feature size to be kept [%(default)s]', default=75, type=int)
	parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
	##############################################
	# Training
	##############################################
	parser_train = subparsers.add_parser("train", help="Train the model on data")
	parser_train.add_argument('-A', '--annotation', metavar="GFF3", help='Reference annotation used for training', type=fC.gff, required=True)
	parser_train.add_argument('-E', '--epochs', metavar="INT", help='Number of training epochs [%(default)s]', default=100, type=int)
	parser_train.add_argument('-B', '--batch_size', metavar='INT', help='Batch size for each rank[%(default)s]', default=100, type=int)
	parser_train.add_argument('-L', '--sequence_length', metavar="INT", help='Length of sequence used for classification [%(default)s]', default=500, type=int)
	parser_train.add_argument('-n', '--neurons', metavar="INT", help='Number of neurons in each RNN/LSTM cell [%(default)s]', default=100, type=int)
	parser_train.add_argument('-l', '--layers', metavar="INT", help='Number of layers of RNN/LSTM cells [%(default)s]', default=1, type=int)
	parser_train.add_argument('-r','--learning_rate', metavar="FLOAT", help='Learning rate of the optimizer [%(default)s]', default=0.001, type=float)
	parser_train.add_argument('-d', '--dropout', metavar="FLOAT", help='Dropout rate of the model [%(default)s]', default=0, type=float)
	cell_types = ('lstm', 'rnn', 'gru')
	parser_train.add_argument('-C', '--cell_type', metavar="STR", help='The recurrent cell type of the model ([lstm], rnn, gru)', default='lstm', type=_argChecker(cell_types, 'cell type').check)
	parser_train.add_argument('-b', '--bidirectional', action='store_true', help='Reccurent layers are bidirectional')
	merge_modes = ('sum', 'mul', 'concat', 'ave', 'none')
	parser_train.add_argument('-m', '--merge', metavar="STR", help='Bidirectional layer merge modes ([concat], sum, mul, ave, none)', default='concat', type=_argChecker(merge_modes, 'cell type').check)
	parser_train.add_argument('-S', '--stateful', action='store_true', help='The reccurent model is stateful (cannot be used with bidirectional)')
	parser_train.add_argument('--reg_kernel', action='store_true', help='Apply a regularizer to the kernel weights matrix')
	parser_train.add_argument('--reg_bias', action='store_true', help='Apply a regularizer to the bias vector')
	parser_train.add_argument('--reg_activity', action='store_true', help='Apply a regularizer to the activation layer')
	parser_train.add_argument('--l1', metavar="FLOAT", help="L1 regularizer lambda [%(default)s]", default=0.01, type=float)
	parser_train.add_argument('--l2', metavar="FLOAT", help="L2 regularizer lambda [None]", default=0, type=float)
	parser_train.add_argument('--conv', metavar="INT", help="Width of 1D convolution [None]", default=0, type=int)
	parser_train.add_argument('--batch_norm', action='store_true', help='Apply batch normalization between layers')
	parser_train.add_argument('-H', '--hidden_list', metavar="STR", help='Comma separated list of hidden layer widths used after recurrent layers', type=str)
	parser_train.add_argument('--noTEMD', action='store_true', help='Disable the collection and prediction of TE order and superfamiles')
	parser_train.add_argument('-f', '--force', action='store_true', help='Overwrite a previously saved model')
	parser_train.add_argument('--train', metavar="STR", help='Comma separated list of chromosomes to train on [all]', type=str)
	parser_train.add_argument('--test', metavar="STR", help='Comma separated list of chromosomes to test on [none]', type=str)
	parser_train.add_argument('--every', metavar="INT", help='Collect MSE values for the first training and test chromosomes every [5] epochs', type=int, default=5)
	#
	parser_train.set_defaults(target_function=train)
	#parser_train.add_argument('-', '--', action='store_true', help='')
	#parser_train.add_argument('-', '--', metavar="", help=' [%(default)s]', default='', type=)
	##############################################
	# Classify
	##############################################
	parser_classify = subparsers.add_parser("classify", help="Classify data using model")
	parser_classify.add_argument('-O', '--output', metavar="GFF3", help='Output gff3 [%(default)s]', default='output.gff3', type=str)
	parser_classify.add_argument('-T', '--threshold', metavar="FLOAT", help='[%(default)s] of all votes needed for output classification', default=0.5, type=float)
	#
	parser_classify.set_defaults(target_function=classify)
	#parser_classify.add_argument('-', '--', action='store_true', help='')
	#parser_classify.add_argument('-', '--', metavar="", help=' [%(default)s]', default='', type=)
	##############################################
	# Parse args
	##############################################
	args = parser.parse_args()
	args.config = os.path.join(args.directory, 'config.pkl')
	################################
	# Configure logging
	################################
	if args.verbose:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.INFO)
	if args.verbose: logger.debug("DEBUG logging enabled")
	##############################################
	# RUN target function
	##############################################
	args.target_function(args)

def _target_checker(targets, IS, default):
	if targets:
		split_chroms = targets.split(',')
		if set(split_chroms).issubset(set(IS.FA.references)):
			return split_chroms
		else:
			diff = set(split_chroms).difference(set(IS.FA.refernces))
			sys.exit("[%s] do not exist in %s"%(', '.join(diff), IS.fasta_file))
	else:
		return default

def train(args):
	#if args.bidirectional and args.stateful:
	#	sys.exit("Cannot use stateful and bidirectional")
	logger.debug("Model will be trained on the given input")
	# Attempt to load old parameters
	if os.path.exists(args.config) and not args.force:
		logger.info("Loading model parameters from %s"%(args.config))
		with open(args.config, 'rb') as CF:
			cached_args = pickle.load(CF)
		# Open the input
		out_dim = calc_n_outputs(args, cached_args)
		IS = reader.input_slicer(args.reference, args.methratio, args.annotation, args.quality, args.ploidy, out_dim, stateful=bool(cached_args.stateful))
		init_hvd(args)
	else:
		# Open the input
		out_dim = calc_n_outputs(args, args)
		IS = reader.input_slicer(args.reference, args.methratio, args.annotation, args.quality, args.ploidy, out_dim, stateful=bool(args.stateful))
		init_hvd(args)
		# Save the parameters
		if not hvd or (hvd and hvd.rank() == 0):
			logger.info("Saving model parameters to %s"%(args.config))
			if not os.path.exists(args.directory): os.makedirs(args.directory)
			with open(args.config, 'wb') as CF:
				pickle.dump(args, CF)
		cached_args = args
	# Check dropout
	assert(cached_args.dropout >= 0 and cached_args.dropout <= 1)
	# Parse hidden list
	hidden_list = map(int, cached_args.hidden_list.split(',')) if cached_args.hidden_list else []
	model_batch = int(cached_args.batch_size/args.hvd_size) if hvd and cached_args.stateful else cached_args.batch_size
	# Load model
	M = model.sleight_model(args.name, \
		n_inputs = 10, \
		n_steps = cached_args.sequence_length, \
		n_outputs = calc_n_outputs(args, cached_args), \
		n_neurons = cached_args.neurons, \
		n_layers = cached_args.layers, \
		learning_rate = cached_args.learning_rate, \
		dropout = cached_args.dropout, \
		cell_type = cached_args.cell_type, \
		reg_kernel = cached_args.reg_kernel, \
		reg_bias = cached_args.reg_bias, \
		reg_activity = cached_args.reg_activity, \
		l1 = cached_args.l1, \
		l2 = cached_args.l2, \
		bidirectional = cached_args.bidirectional, \
		merge_mode = cached_args.merge, \
		stateful = model_batch if cached_args.stateful else False, \
		hidden_list=hidden_list, \
		noTEMD = 'noTEMD' in cached_args and cached_args.noTEMD, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if glob(M.save_file+"*") and not args.force:
		M.restore()
	# Check target chromosomes
	train_chroms = _target_checker(cached_args.train, IS, IS.FA.references)
	test_chroms = _target_checker(cached_args.test, IS, [])

	# Define the iterfunction
	iter_func = IS.stateful_chrom_iter if cached_args.stateful else IS.chrom_iter
	
	train_x = {}
	train_y = {}
	logger.info("Caching training data")
	for chrom in sorted(train_chroms):
		cb, xb, yb = zip(*iter_func(chrom, seq_len=args.sequence_length, offset=args.offset, batch_size=cached_args.batch_size, hvd_rank=args.hvd_rank, hvd_size=args.hvd_size))
		#print cb
		train_x[chrom] = np.vstack(xb)
		train_y[chrom] = np.vstack(yb)
		assert(train_x[chrom].shape == (len(xb)*model_batch, args.sequence_length, M.n_inputs))
		assert(train_y[chrom].shape == (len(xb)*model_batch, args.sequence_length, M.n_outputs))
		del cb, xb, yb
		logger.info("Finished caching %s"%(chrom))
	# Run
	for E in irange(args.epochs):
		#### Train #################################################
		train_start = time()
		for chrom in sorted(train_chroms):
			history = LossHistory()
			start = time()
			if cached_args.stateful: M.model.reset_states()
			M.model.fit(train_x[chrom], train_y[chrom], \
				batch_size=model_batch, epochs=1, shuffle=False, \
				callbacks=[history], verbose=0)
			L5 = np.round(fivenum(history.losses),3)
			A5 = np.round(fivenum(history.acc),3)
			rate = train_x[chrom].shape[0]/float(time()-start)
			logger.debug("E%i %s LOSS%s ACC%s %i seq/s"%(E, chrom, str(L5), str(A5), int(rate)))
			#logger.debug("E-%i %s ACC  %s"%(E, chrom, str(fivenum(history.acc))))
			if cached_args.stateful: M.model.reset_states()
		logger.info("Epoch-%04i - Finished training in %i seconds"%(E, int(time()-train_start)))
		#### Calculate MSE #########################################
		#if (E+1)%5 == 0: # Every 5th epoch [4, 9, ...]
		#if (E+1)%1 == 0:
		if (E+1)%cached_args.every == 0:
			MI = writer.MSE_interval(args.reference, args.directory, args.hvd_rank)
			test_start = time()
			logger.debug("Epoch-%04i - Collecting Training MSE values"%(E))
			for chrom in [train_chroms[0]]+([test_chroms[0]] if test_chroms else []):
				chrom_time, start_time = time(), time()
				if cached_args.stateful: M.model.reset_states()
				for count, batch in enumerate(iter_func(chrom, \
						seq_len=args.sequence_length, offset=args.offset, \
						batch_size=cached_args.batch_size, \
						hvd_rank=args.hvd_rank, hvd_size=args.hvd_size)):
					cb, xb, yb = batch
					cc, cs, ce = cb[0][0], cb[0][1], cb[-1][2]
					assert(len(cb) == model_batch)
					y_pred_batch, predict_time = M.predict(xb, return_time=True)
					MI.add_predict_batch(cb, yb, y_pred_batch)
					#logger.debug("TEST: Batch-%03i %s:%i-%i TRAIN=%.1fs TOTAL=%.1fs RATE=%.1f seq/s"%(count, cc, cs, ce, predict_time, time()-start_time, len(xb)/predict_time))
					start_time = time()
				if cached_args.stateful: M.model.reset_states()
				logger.debug("Finished testing %s in %i seconds"%(chrom, int(time()-chrom_time)))
			logger.info("Epoch-%04i - Finished testing in %i seconds"%(E, int(time()-test_start)))
			# Write output values
			if train_chroms:
				MI.write(hvd, [train_chroms[0]], 'TRAIN', E, 10000, 'mean', 'midpoint')
			if test_chroms:
				MI.write(hvd, [test_chroms[0]], 'TEST', E, 10000, 'mean', 'midpoint')
			MI.close()
			del MI
		if not hvd or (hvd and args.hvd_rank == 0):
			# Save between epochs
			if (E+1)%cached_args.every == 0:
				M.save(epoch=E)
			else:
				M.save()
	#### Write output #############################################
	args.threshold = 0.5
	args.output = os.path.join(args.directory, 'training_output.gff3')
	args.raw_output = os.path.join(args.directory, 'training_output_raw.gff3')
	#### Classify #################################################
	OA = make_predictions(IS, M, args, cached_args, model_batch)
	#### Write #####################################################
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Features need %.2f of the votes to be output"%(args.threshold))
		logger.info("Writing %s"%(args.output))
	OA.write_gff3(out_file=args.raw_output, threshold=args.threshold, min_size=0, max_fill_size=0)
	OA.write_gff3(out_file=args.output, threshold=args.threshold, min_size=args.min_feat, max_fill_size=args.max_fill)
	#### Shut Down #################################################
	del M
	if hvd:
		logger.debug("Waiting on other processes")
		print hvd.allgather([hvd.rank()], name="Barrier")
		logger.debug("Exiting")
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Done")
	#logger.debug("Closing the tensorflow session")
	#M.sess.close()

def classify(args):
	logger.debug("Model will classify the given input")
	# Attempt to load old parameters
	if not os.path.exists(args.config):
		logger.error("Could not find previous model configuration")
		sys.exit()
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Loading model parameters from %s"%(args.config))
	with open(args.config, 'rb') as CF:
		cached_args = pickle.load(CF)
	# Open the input
	out_dim = calc_n_outputs(args, cached_args)
	IS = reader.input_slicer(args.reference, args.methratio, quality=args.quality, ploidy=args.ploidy, \
		out_dim=out_dim, stateful=bool(cached_args.stateful))
	init_hvd(args)
	hidden_list = map(int, cached_args.hidden_list.split(',')) if cached_args.hidden_list else []
	model_batch = int(cached_args.batch_size/args.hvd_size) if hvd and cached_args.stateful else cached_args.batch_size
	# Load model
	M = model.sleight_model(args.name, \
		n_inputs = 10, \
		n_steps = cached_args.sequence_length, \
		n_outputs = calc_n_outputs(args, cached_args), \
		n_neurons = cached_args.neurons, \
		n_layers = cached_args.layers, \
		learning_rate = cached_args.learning_rate, \
		dropout = cached_args.dropout, \
		cell_type = cached_args.cell_type, \
		reg_kernel = cached_args.reg_kernel, \
		reg_bias = cached_args.reg_bias, \
		reg_activity = cached_args.reg_activity, \
		l1 = cached_args.l1, \
		l2 = cached_args.l2, \
		bidirectional = cached_args.bidirectional, \
		merge_mode = cached_args.merge, \
		stateful = model_batch if cached_args.stateful else False, \
		hidden_list=hidden_list, \
		noTEMD = 'noTEMD' in cached_args and cached_args.noTEMD, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if not glob(M.save_file+"*"):
		logger.error("Could not find match for %s. No model to restore from."%(M.save_file))
		sys.exit()
	M.restore()
	#print M.model.summary()
	#### Classify #################################################
	OA = make_predictions(IS, M, args, cached_args, model_batch)
	#### Write #####################################################
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Features need %.2f of the votes to be output"%(args.threshold))
		logger.info("Writing %s"%(args.output))
	OA.write_gff3(out_file=args.output, threshold=args.threshold, min_size=args.min_feat, max_fill_size=args.max_fill)
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Done")

def calc_n_outputs(args, cached_args):
	if 'noTEMD' in args and args.noTEMD:
		logger.info("Not including TE metadata in output")
		return len(constants.gff3_f2i)
	elif 'noTEMD' in cached_args and cached_args.noTEMD:
		logger.info("Not including TE metadata in output")
		return len(constants.gff3_f2i)
	return len(constants.gff3_f2i)+2

def make_predictions(IS, M, args, cached_args, model_batch):
	# Open the output
	noTEMD = 'noTEMD' in cached_args and cached_args.noTEMD
	OA = writer.output_aggregator(args.reference, noTEMD=noTEMD, h5_file=os.path.join(args.directory, 'tmp_vote.h5'))
	# Store iteration method
	iter_func = IS.stateful_chrom_iter if cached_args.stateful else IS.chrom_iter
	#### Classify #################################################
	for chrom in sorted(IS.FA.references):
		seqs = 0
		start_time = time()
		if cached_args.stateful: M.model.reset_states()
		for count, batch in enumerate(iter_func(chrom, seq_len=cached_args.sequence_length, \
				offset=args.offset, batch_size=cached_args.batch_size, \
				hvd_rank=args.hvd_rank, hvd_size=args.hvd_size)):
			if len(batch) == 2:
				cb, xb = batch
			elif len(batch) == 3:
				cb, xb, yb = batch
			else:
				raise ValueError(len(batch) in (2,3))
			cc, cs, ce = cb[0][0], cb[0][1], cb[-1][2]
			assert(len(cb) == model_batch)
			seqs += model_batch
			y_pred_batch, predict_time = M.predict(xb, return_time=True)
			#logger.debug("PREDICT: Batch-%03i %s:%i-%i TRAIN=%.1fs TOTAL=%.1fs RATE=%.1f seq/s"%(count, cc, cs, ce, predict_time, time()-start_time, len(xb)/predict_time))
			if not y_pred_batch.sum(): logger.warn("No predictions in Batch-%03i %s:%i-%i"%(count, cc, cs, ce))
			for c, x, yp in zip(cb, xb, y_pred_batch):
				chrom,s,e = c
				if cached_args.stateful:
					OA.vote(*c, array=yp, overwrite=True)
				else:
					OA.vote(*c, array=yp)
		if cached_args.stateful: M.model.reset_states()
		rate = seqs/float(time()-start_time)
		logger.debug("Finished predictions for %s at a rate of %.1f seq/s"%(chrom, rate))
	return OA

def test_barrier(msg):
	if hvd.rank() == 0: logger.debug(msg)
	print hvd.allgather([hvd.rank()], name="Barrier")
	if hvd.rank() == 0: logger.debug("OK")
def init_hvd(args):
	if hvd:
		hvd.init()
		FORMAT = "[%%(levelname)s - P%i/%i - %%(filename)s:%%(lineno)s - %%(funcName)s] %%(message)s"%(hvd.rank(), hvd.size())
		# Remove all handlers associated with the root logger object.
		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)
		logging.basicConfig(level=logging.INFO, format=FORMAT)
		if args.verbose:
			logger.setLevel(logging.DEBUG)
		else:
			logger.setLevel(logging.INFO)
		logger.debug("Updated logger to print process")
	args.hvd_rank = hvd.rank() if hvd else 0
	args.hvd_size = hvd.size() if hvd else 1

class fileCheck:
	def check(self, file, exts):
		ext = os.path.splitext(file)[1][1:]
		fName = os.path.split(file)[1]
		if not ext in exts:
			raise argparse.ArgumentTypeError("%s not a %s"%(fName, exts[0]))
		if not os.path.exists(file):
			raise argparse.ArgumentTypeError("%s does not exist"%(file))
	def gff(self, file):
		self.check(file, ['gff','gff3','gtf'])
		return file
	def fastq(self, file):
		self.check(file, ['fastq','fq'])
		return file
	def fasta(self, file):
		self.check(file, ['fasta','fa'])
		return file
	def methratio(self, file):
		self.check(file, ['txt','mr','methratio'])
		return file
class _argChecker:
	def __init__(self, options, name):
		self.options = options
		self.name = name
	def check(self, x):
		if x in self.options:
			return x
		else:
			raise argparse.ArgumentTypeError("%s not a valid %s"%(x, self.name))

import tensorflow
class LossHistory(tensorflow.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.acc = []
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.acc.append(logs.get('acc'))

if __name__ == "__main__":
	main()
