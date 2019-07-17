#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 07/15/2019
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

import sys, os, argparse
from glob import glob
from time import time
import pickle
import logging
try:
	#import horovod.keras as hvd
	import horovod.tensorflow.keras as hvd
	hvd.init()
	FORMAT = "[%%(levelname)s - P%i/%i - %%(filename)s:%%(lineno)s - %%(funcName)s] %%(message)s"%(hvd.rank(), hvd.size())
except:
	hvd = False
	FORMAT = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)s] %(message)s"
import tensorflow as tf
logger = logging.getLogger(__name__)
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
	cell_types = ('lstm', 'rnn')
	parser_train.add_argument('-C', '--cell_type', metavar="STR", help='The recurrent cell type of the model ([lstm], rnn)', default='lstm', type=_argChecker(cell_types, 'cell type').check)
	parser_train.add_argument('-b', '--bidirectional', action='store_true', help='Reccurent layers are bidirectional')
	merge_modes = ('sum', 'mul', 'concat', 'ave', 'none')
	parser_train.add_argument('-m', '--merge', metavar="STR", help='Bidirectional layer merge modes ([concat], sum, mul, ave, none)', default='concat', type=_argChecker(merge_modes, 'cell type').check)
	parser_train.add_argument('-S', '--stateful', action='store_true', help='The reccurent model is stateful (cannot be used with bidirectional)')
	parser_train.add_argument('--reg_kernel', action='store_true', help='Apply a regularizer to the kernel weights matrix')
	parser_train.add_argument('--reg_bias', action='store_true', help='Apply a regularizer to the bias vector')
	parser_train.add_argument('--reg_activity', action='store_true', help='Apply a regularizer to the activation layer')
	parser_train.add_argument('--l1', metavar="FLOAT", help="L1 regularizer lambda [%(default)s]", default=0.01, type=float)
	parser_train.add_argument('--l2', metavar="FLOAT", help="L2 regularizer lambda [None]", default=0, type=float)
	parser_train.add_argument('-H', '--hidden_list', metavar="STR", help='Comma separated list of hidden layer widths used after recurrent layers', type=str)
	parser_train.add_argument('-f', '--force', action='store_true', help='Overwrite a previously saved model')
	parser_train.add_argument('--train', metavar="STR", help='Comma separated list of chromosomes to train on [all]', type=str)
	parser_train.add_argument('--test', metavar="STR", help='Comma separated list of chromosomes to test on [none]', type=str)
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
	##############################################
	# Check for rank and size
	##############################################
	args.hvd_rank = hvd.rank() if hvd else 0
	args.hvd_size = hvd.size() if hvd else 1
	################################
	# Configure logging
	################################
	if args.verbose:
		logging.basicConfig(level=logging.DEBUG, format=FORMAT)
	else:
		logging.basicConfig(level=logging.INFO, format=FORMAT)
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
	if args.bidirectional and args.stateful:
		sys.exit("Cannot use stateful and bidirectional")
	logger.debug("Model will be trained on the given input")
	# Attempt to load old parameters
	if os.path.exists(args.config) and not args.force:
		logger.info("Loading model parameters from %s"%(args.config))
		with open(args.config, 'rb') as CF:
			cached_args = pickle.load(CF)
	else:
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
	# Load model
	M = model.sleight_model(args.name, \
		n_inputs = 10, \
		n_steps = cached_args.sequence_length, \
		n_outputs = len(constants.gff3_f2i)+2, \
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
		stateful = cached_args.batch_size if cached_args.stateful else False, \
		hidden_list=hidden_list, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if glob(M.save_file+"*") and not args.force:
		M.restore()
	# Open the input
	IS = reader.input_slicer(args.reference, args.methratio, args.annotation, args.quality, args.ploidy, stateful=bool(cached_args.stateful))
	# Check target chromosomes
	train_chroms = _target_checker(cached_args.train, IS, IS.FA.references)
	test_chroms = _target_checker(cached_args.test, IS, [])
	# Run
	for E in irange(args.epochs):
		iter_func = IS.stateful_chrom_iter if cached_args.stateful else IS.chrom_iter
		#### Train #################################################
		train_start = time()
		for chrom in sorted(train_chroms):
			start_time = time()
			if cached_args.stateful: M.model.reset_states()
			for count, batch in enumerate(iter_func(chrom, seq_len=args.sequence_length, \
					offset=args.offset, batch_size=cached_args.batch_size, \
					hvd_rank=args.hvd_rank, hvd_size=args.hvd_size)):
				cb, xb, yb = batch
				cc, cs, ce = cb[0][0], cb[0][1], cb[-1][2]
				assert(len(cb) == cached_args.batch_size)
				MSE, acc, train_time = M.train(xb, yb)
				if hvd: assert(len(cb) == cached_args.batch_size)
				logger.debug("TRAIN: Batch-%03i MSE=%.6f %s:%i-%i TRAIN=%.1fs TOTAL=%.1fs"%(count, MSE, cc, cs, ce, train_time, time()-start_time))
				start_time = time()
			if cached_args.stateful: M.model.reset_states()
		logger.info("Epoch-%04i - Finished training in %i seconds"%(E, int(time()-train_start)))
		#### Calculate MSE #########################################
		if (E+1)%5 == 0: # Every 5th epoch [4, 9, ...]
			MI = writer.MSE_interval(args.reference, args.directory, args.hvd_rank)
			test_start = time()
			logger.debug("Epoch-%04i - Collecting Training MSE values"%(E))
			for chrom in sorted(train_chroms)+sorted(test_chroms):
				chrom_time, start_time = time(), time()
				if cached_args.stateful: M.model.reset_states()
				for count, batch in enumerate(iter_func(chrom, \
						seq_len=args.sequence_length, offset=args.offset, \
						batch_size=cached_args.batch_size, \
						hvd_rank=args.hvd_rank, hvd_size=args.hvd_size)):
					cb, xb, yb = batch
					if hvd: assert(len(cb) == cached_args.batch_size)
					cc, cs, ce = cb[0][0], cb[0][1], cb[-1][2]
					y_pred_batch, predict_time = M.predict(xb, return_time=True)
					MI.add_predict_batch(cb, yb, y_pred_batch)
					logger.debug("TEST: Batch-%03i %s:%i-%i TRAIN=%.1fs TOTAL=%.1fs"%(count, cc, cs, ce, predict_time, time()-start_time))
					start_time = time()
				if cached_args.stateful: M.model.reset_states()
				logger.debug("Finished testing %s in %i seconds"%(chrom, int(time()-chrom_time)))
			logger.info("Epoch-%04i - Finished testing in %i seconds"%(E, int(time()-test_start)))
			# Write output values
			if train_chroms:
				MI.write(hvd, train_chroms, 'TRAIN', E, 10000, 'mean', 'midpoint')
			if test_chroms:
				MI.write(hvd, test_chroms, 'TEST', E, 10000, 'mean', 'midpoint')
			MI.close()
		if not hvd or (hvd and args.hvd_rank == 0):
			# Save between epochs
			if (E+1)%5 == 0: # Every 5th epoch [4, 9, ...]
				M.save(epoch=E)
			else:
				M.save()
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
	hidden_list = map(int, cached_args.hidden_list.split(',')) if cached_args.hidden_list else []
	# Load model
	M = model.sleight_model(args.name, \
		n_inputs = 10, \
		n_steps = cached_args.sequence_length, \
		n_outputs = len(constants.gff3_f2i)+2, \
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
		stateful = cached_args.batch_size if cached_args.stateful else False, \
		hidden_list=hidden_list, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if not glob(M.save_file+"*"):
		logger.error("No model to restore from")
		sys.exit()
	M.restore()
	#print M.model.summary()
	# Open the input
	IS = reader.input_slicer(args.reference, args.methratio, quality=args.quality, ploidy=args.ploidy)
	# Open the output
	OA = writer.output_aggregator(args.reference)
	# Classify the input
	def non_zero(a):
	        return ' '.join(["%i:%i"%(i,a[i]) for i in np.nonzero(a)[0]])
	if cached_args.stateful:
		for chrom in sorted(IS.FA.references):
			M.model.reset_states()
			for cb, xb in IS.stateful_chrom_iter(chrom, seq_len=cached_args.sequence_length, offset=args.offset, batch_size=cached_args.batch_size, hvd_rank=args.hvd_rank, hvd_size=args.hvd_size):
				y_pred_batch = M.predict(xb)
				for c, x, yp in zip(cb, xb, y_pred_batch):
					chrom,s,e = c
					#for i in range(s,e): print (chrom,i), non_zero(yp[i-s])
					OA.vote(*c, array=yp, overwrite=True)
			M.model.reset_states()
	else:
		for cb, xb in IS.genome_iter(seq_len=cached_args.sequence_length, offset=args.offset, batch_size=cached_args.batch_size, hvd_rank=args.hvd_rank, hvd_size=args.hvd_size):
			y_pred_batch = M.predict(xb)
			for c, x, yp in zip(cb, xb, y_pred_batch):
				OA.vote(*c, array=yp)
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Features need %.2f of the votes to be output"%(args.threshold))
		logger.info("Writing %s"%(args.output))
	OA.write_gff3(out_file=args.output, threshold=args.threshold)
	if not hvd or (hvd and hvd.rank() == 0):
		logger.info("Done")

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

if __name__ == "__main__":
	main()
