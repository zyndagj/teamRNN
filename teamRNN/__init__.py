#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 05/15/2019
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
import pickle
import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)15s] %(message)s"
from teamRNN import reader, constants, writer, model
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
	parser.add_argument('-B', '--batch_size', metavar='INT', help='Batch size [%(default)s]', default=100, type=int)
	parser.add_argument('-Q', '--quality', metavar='INT', help='Input assembly quality [%(default)s]', default=-1, type=int)
	parser.add_argument('-P', '--ploidy', metavar='INT', help='Input chromosome ploidy [%(default)s]', default=2, type=int)
	parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
	##############################################
	# Training
	##############################################
	parser_train = subparsers.add_parser("train", help="Train the model on data")
	parser_train.add_argument('-A', '--annotation', metavar="GFF3", help='Reference annotation used for training', type=fC.gff, required=True)
	parser_train.add_argument('-E', '--epochs', metavar="INT", help='Number of training epochs [%(default)s]', default=100, type=int)
	parser_train.add_argument('-L', '--sequence_length', metavar="INT", help='Length of sequence used for classification [%(default)s]', default=10000, type=int)
	parser_train.add_argument('-n', '--neurons', metavar="INT", help='Number of neurons in each RNN/LSTM cell [%(default)s]', default=100, type=int)
	parser_train.add_argument('-l', '--layers', metavar="INT", help='Number of layers of RNN/LSTM cells [%(default)s]', default=1, type=int)
	parser_train.add_argument('-r','--learning_rate', metavar="FLOAT", help='Learning rate of the optimizer [%(default)s]', default=0.001, type=float)
	parser_train.add_argument('-d', '--dropout', metavar="FLOAT", help='Dropout rate of the model [%(default)s]', default=0.05, type=float)
	cell_types = ('lstm', 'rnn')
	parser_train.add_argument('-C', '--cell_type', metavar="STR", help='The recurrent cell type of the model ([lstm], rnn)', default='lstm', type=_argChecker(cell_types, 'cell type').check)
	parser_train.add_argument('-P', '--peep', action='store_true', help='Use peep-hole connections on LSTM cells')
	parser_train.add_argument('-S', '--stacked', action='store_true', help='Use stacked outputs')
	parser_train.add_argument('-b', '--bidirectional', action='store_true', help='Reccurent layers are bidirectional')
	parser_train.add_argument('-w', '--regularize', action='store_true', help='Use a weight regularizer to penalize huge weights')
	parser_train.add_argument('-H', '--hidden_list', metavar="STR", help='Comma separated list of hidden layer widths used after recurrent layers', type=str)
	parser_train.add_argument('-f', '--force', action='store_true', help='Overwrite a previously saved model')
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
		logging.basicConfig(level=logging.DEBUG, format=FORMAT)
	else:
		logging.basicConfig(level=logging.INFO, format=FORMAT)
	if args.verbose: logger.debug("DEBUG logging enabled")
	##############################################
	# RUN target function
	##############################################
	args.target_function(args)

def train(args):
	logger.debug("Model will be trained on the given input")
	# Attempt to load old parameters
	if os.path.exists(args.config) and not args.force:
		logger.info("Loading model parameters from %s"%(args.config))
		with open(args.config, 'rb') as CF:
			cached_args = pickle.load(CF)
	else:
		# Save the parameters
		if not os.path.exists(args.directory): os.makedirs(args.directory)
		with open(args.config, 'wb') as CF:
			pickle.dump(args, CF)
		cached_args = args
	# Check dropout
	assert(cached_args.dropout >= 0)
	assert(cached_args.dropout <= 1)
	logger.debug("Using a dropout of %.4f"%(cached_args.dropout))
	# Parse hidden list
	hidden_list = map(int, cached_args.hidden_list.split(',')) if cached_args.hidden_list else []
	logger.debug("%i hidden layers of sizes %s will be added onto the model"%(len(hidden_list), args.hidden_list))
	# Load model
	model.reset_graph()
	M = model.sleight_model(args.name, n_inputs=10, n_outputs=len(constants.gff3_f2i)+2, \
		n_steps=cached_args.sequence_length, \
		n_neurons=cached_args.neurons, \
		n_layers=cached_args.layers, \
		learning_rate=cached_args.learning_rate, \
		training_keep=1-cached_args.dropout, \
		dropout=(cached_args.dropout>1), \
		cell_type=cached_args.cell_type, \
		peep=cached_args.peep, \
		stacked=cached_args.stacked, \
		bidirectional=cached_args.bidirectional, \
		reg_losses=cached_args.regularize, \
		hidden_list=hidden_list, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if glob(M.save_file+"*") and not args.force:
		M.restore()
	# Open the input
	IS = reader.input_slicer(args.reference, args.methratio, args.annotation, args.quality, args.ploidy)
	for E in range(args.epochs):
		for cb, xb, yb in IS.new_batch_iter(seq_len=args.sequence_length, offset=args.offset, batch_size=args.batch_size):
			MSE, train_time = M.train(xb, yb)
			logger.debug("Finished batch %s:%i-%i   MSE = %.2f"%(cb[0][0], cb[0][1], cb[-1][2], MSE))
		# Print MSE
		logger.info("Finished epoch %3i - MSE = %.6f"%(E+1, MSE))
		# Save between epochs
		M.save()
	logger.info("Done")

def classify(args):
	logger.debug("Model will classify the given input")
	# Attempt to load old parameters
	if not os.path.exists(args.config):
		logger.error("Could not find previous model configuration")
		sys.exit()
	logger.info("Loading model parameters from %s"%(args.config))
	with open(args.config, 'rb') as CF:
		cached_args = pickle.load(CF)
	hidden_list = map(int, cached_args.hidden_list.split(',')) if cached_args.hidden_list else []
	# Load model
	model.reset_graph()
	M = model.sleight_model(args.name, n_inputs=10, n_outputs=len(constants.gff3_f2i)+2, \
		n_steps=cached_args.sequence_length, \
		n_neurons=cached_args.neurons, \
		n_layers=cached_args.layers, \
		learning_rate=cached_args.learning_rate, \
		training_keep=1-cached_args.dropout, \
		dropout=(cached_args.dropout>1), \
		cell_type=cached_args.cell_type, \
		peep=cached_args.peep, \
		stacked=cached_args.stacked, \
		bidirectional=cached_args.bidirectional, \
		reg_losses=cached_args.regularize, \
		hidden_list=hidden_list, \
		save_dir=args.directory)
	# See if there is a checkpoint to restore from
	if not glob(M.save_file+"*"):
		logger.error("No model to restore from")
		sys.exit()
	M.restore()
	# Open the input
	IS = reader.input_slicer(args.reference, args.methratio, quality=args.quality, ploidy=args.ploidy)
	# Open the output
	OA = writer.output_aggregator(args.reference)
	# Classify the input
	for cb, xb in IS.new_batch_iter(seq_len=cached_args.sequence_length, offset=args.offset, batch_size=args.batch_size):
		y_pred_batch = M.predict(xb)
		for c, x, yp in zip(cb, xb, y_pred_batch):
			OA.vote(*c, array=yp)
	logger.info("Features need %.2f of the votes to be output"%(args.threshold))
	logger.info("Writing %s"%(args.output))
	OA.write_gff3(out_file=args.output, threshold=args.threshold)
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
