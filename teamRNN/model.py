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

#!pip install hmmlearn &> /dev/null
import numpy as np
import os, psutil, random
#from hmmlearn import hmm
os.putenv('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
from tensorflow.keras.backend import set_session, clear_session
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, SimpleRNN, Dense, CuDNNLSTM, Dropout, TimeDistributed
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
try:
	import horovod.tensorflow.keras as hvd
	#hvd.init() # Only need to do this once
except:
	hvd = False
from time import time
import logging
logger = logging.getLogger(__name__)

class sleight_model:
	# https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/autoencoder/autoencoder_models/DenoisingAutoencoder.py
	def __init__(self, name, n_inputs=1, n_steps=50, n_outputs=1, n_neurons=20, n_layers=1, \
		 learning_rate=0.001, dropout=0, cell_type='rnn', reg_kernel=False, reg_bias=False, \
		 reg_activity=False, l1=0, l2=0, bidirectional=False, merge_mode='concat', \
		 stateful=False, hidden_list=[], save_dir='.'):
		self.name = name # Name of the model
		self.n_inputs = n_inputs # Number of input features
		self.n_outputs = n_outputs # Number of outputs
		self.n_steps = n_steps # Size of input sequence
		self.n_neurons = n_neurons # Number of neurons per RNN cell
		self.n_layers = n_layers # Number of RNN layers
		self.learning_rate = learning_rate # Learning rate of the optimization algorithm
		self.dropout = dropout # Dropout rate
		self.cell_type = cell_type # Type of RNN cells to use
		self.gpu = self._detect_gpu()
		# https://keras.io/regularizers/
		self.reg_kernel = reg_kernel # Use kernel regularization
		self.reg_bias = reg_bias # Use bias regularization
		self.reg_activity = reg_activity # Use activity regularization
		self.l1, self.l2 = l1, l2 # Store the l1 and l2 rates
		# Recurrent properties
		self.bidirectional = bidirectional # each RNN layer is bidirectional
		self.merge_mode = None if merge_mode == 'none' else merge_mode
		self.stateful = stateful # Whether the batches are stateful
		#if self.stateful: assert(not self.bidirectional)
		# supported cell types
		self.cell_options = {'rnn':SimpleRNN, 'lstm':CuDNNLSTM if self.gpu else LSTM}
		# Additional layers
		self.hidden_list = hidden_list # List of hidden layer sizes
		# Holder variable for stateful model test
		self.test = False
		# TODO remake this name
		self.param_name = self._gen_name()
		if save_dir[0] == '/':
			self.save_dir = save_dir
		else:
			self.save_dir = os.path.join(os.getcwd(), save_dir)
		self.save_file = os.path.join(self.save_dir, '%s.h5'%(self.param_name))
		#self.graph = tf.Graph() # Init graph
		#logger.debug("Created graph")
		######################################
		# Set the RNG seeds
		######################################
		clear_session()
		tf.reset_default_graph()
		np.random.seed(42)
		random.seed(42)
		tf.set_random_seed(42)
		logger.debug("Cleared session and reset random seeds")
		######################################
		# Configure the session
		######################################
		tacc_nodes = {'knl':(136,2), 'skx':(48,2), 'hikari':(24,2)}
		node_name = os.getenv('TACC_NODE_TYPE', False)
		if not node_name: node_name = os.getenv('TACC_SYSTEM', False)
		if node_name in tacc_nodes:
			intra, inter = tacc_nodes[node_name]
			logger.debug("Using config for TACC %s node (%i, %i)"%(node_name, intra, inter))
			os.putenv('KMP_BLOCKTIME', '0')
			os.putenv('TF_DISABLE_MKL', '1')
			os.putenv('KMP_AFFINITY', 'granularity=fine,noverbose,compact,1,0')
			os.putenv('OMP_NUM_THREADS', str(intra))
			config = tf.ConfigProto(intra_op_parallelism_threads=intra, \
					inter_op_parallelism_threads=inter, \
					log_device_placement=False, allow_soft_placement=True)
					#allow_soft_placement=True, device_count = {'CPU': intra})
			off = rewriter_config_pb2.RewriterConfig.OFF
			config.graph_options.rewrite_options.arithmetic_optimization = off
			sess = tf.Session(config=config)
			set_session(sess)  # set this TensorFlow session as the default session for Keras
		elif self.gpu:
			config = tf.ConfigProto()
			logger.debug("Allowing memory growth on GPU")
			config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
			#config.log_device_placement = True  # to log device placement (on which device the operation ran)
		else:
			config = tf.ConfigProto()
			logger.debug("Using default config")
		sess = tf.Session(graph=tf.get_default_graph(), config=config)
		set_session(sess)  # set this TensorFlow session as the default session for Keras
		######################################
		# Build graph
		######################################
		self.model = self._build_graph()
		######################################
		# Define optimizer and compile
		######################################
		self._compile_graph(self.model, 'mse', 'adam')
		# Done
	def _build_graph(self, test=False):
		model = Sequential()
		logger.debug("Creating %i %slayers with %s%% dropout after each layer"%(self.n_layers, "bidirectional " if self.bidirectional else "", self.dropout))
		if self.bidirectional:
			logger.debug("Bidirectional layers will be merged with %s"%(self.merge_mode))
		for i in range(self.n_layers):
			model.add(self._gen_rnn_layer(i, test))
			# Dropout is added at the cell level
		# Handel hidden layers
		if self.hidden_list and self.hidden_list[0] > 0:
			logger.debug("Appending %s TimeDistributed hidden layers"%(str(self.hidden_list)))
		for hidden_neurons in self.hidden_list:
			if hidden_neurons > 0:
				model.add(TimeDistributed(Dense(hidden_neurons, activation='relu')))
		# Final
		model.add(TimeDistributed(Dense(self.n_outputs, activation='linear')))
		return model
	def _compile_graph(self, model, loss_func='mse', opt_func='adam'):
		loss_functions = {'mse':'mean_squared_error', \
				'msle':'mean_squared_logarithmic_error', \
				'cc':'categorical_crossentropy'}
				#'scc':'sparse_categorical_crossentropy'} - wants a single output
		opt_functions = {'adam':Adam, 'sgd':SGD, 'rms':RMSprop}
		logger.debug("Using the %s optimizer with a learning rate of %s and the %s loss function"%(opt_func, str(self.learning_rate), loss_func))
		opt = opt_functions[opt_func](lr=self.learning_rate)
		if hvd:
			if hvd.rank() == 0: logger.debug("Compiling distributed optimizer")
			opt = hvd.DistributedOptimizer(opt)
			self.callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
		# compile
		model.compile(loss=loss_functions[loss_func], optimizer=opt, metrics=['accuracy'])
	def _gen_name(self):
		out_name = "%s_s%ix%i_o%i"%(self.name, self.n_steps, self.n_inputs, self.n_outputs)
		cell_prefix = 'bi' if self.bidirectional else ''
		out_name += "_%ix%s%s%i"%(self.n_layers, cell_prefix, self.cell_type, self.n_neurons)
		if cell_prefix:
			out_name += '_merge-%s'%(str(self.merge_mode))
		out_name += "_stateful%s"%(str(self.stateful) if self.stateful else 'F')
		out_name += "_learn%s_drop%s"%(str(self.learning_rate), str(self.dropout))
		if (self.reg_kernel or self.reg_bias or self.reg_activity) and (self.l1 or self.l2):
			reg_str = "reg"
			reg_str += 'K' if self.reg_kernel else ''
			reg_str += 'B' if self.reg_bias else ''
			reg_str += 'A' if self.reg_activity else ''
			if self.l1:
				if self.l2:
					reg_str += '-l1_l2(%s)'%(str(self.l1))
				else:
					reg_str += '-l1(%s)'%(str(self.l1))
			elif self.l2:
				reg_str += '-l2(%s)'%(str(self.l2))
		
			out_name += reg_str
		if self.hidden_list:
			out_name += '_'+'h'.join(map(str, self.hidden_list))
		return out_name
	def _gen_rnn_layer(self, num=0, test=False):
		# I may need to modify this for "use_bias"
		if num == 0:
			input_shape = (self.n_steps, self.n_inputs)
			if self.bidirectional:
				if self.stateful:
					#sys.exit("Should never get here")
					bis = (self.stateful, self.n_steps, self.n_inputs)
					return Bidirectional(self._gen_cell_layer(num), \
						merge_mode=self.merge_mode, batch_input_shape=bis)
				return Bidirectional(self._gen_cell_layer(num+1), \
					merge_mode=self.merge_mode, input_shape=input_shape)
			else:
				return self._gen_cell_layer(num, test)
		else:
			if self.bidirectional:
				return Bidirectional(self._gen_cell_layer(num, test), merge_mode=self.merge_mode)
			else:
				return self._gen_cell_layer(num, test)
	def _gen_cell_layer(self, num=0, test=False):
		cell_func = self.cell_options[self.cell_type]
		input_shape = (None, None)
		if num == 0:
			input_shape = (self.n_steps, self.n_inputs)
		if self.stateful and not self.bidirectional:
			if test:
				logger.debug("Setting testing batch size to 1")
				bis = (1, self.n_steps, self.n_inputs)
			else:
				bis = (self.stateful, self.n_steps, self.n_inputs)
			if not self.gpu:
				return cell_func(self.n_neurons, return_sequences=True, \
					kernel_regularizer=self._gen_reg('kernel'), \
					bias_regularizer=self._gen_reg('bias'), \
					activity_regularizer=self._gen_reg('activity'), \
					stateful=bool(self.stateful), implementation=2, \
					batch_input_shape=bis, dropout=self.dropout)
			else:
				return cell_func(self.n_neurons, return_sequences=True, \
					kernel_regularizer=self._gen_reg('kernel'), \
					bias_regularizer=self._gen_reg('bias'), \
					activity_regularizer=self._gen_reg('activity'), \
					stateful=bool(self.stateful), \
					batch_input_shape=bis, dropout=self.dropout)
		if not self.gpu:
			return cell_func(self.n_neurons, return_sequences=True, input_shape=input_shape, \
				kernel_regularizer=self._gen_reg('kernel'), \
				bias_regularizer=self._gen_reg('bias'), \
				activity_regularizer=self._gen_reg('activity'), \
				implementation=2, \
				stateful=bool(self.stateful), dropout=self.dropout)
		else:
			return cell_func(self.n_neurons, return_sequences=True, input_shape=input_shape, \
				kernel_regularizer=self._gen_reg('kernel'), \
				bias_regularizer=self._gen_reg('bias'), \
				activity_regularizer=self._gen_reg('activity'), \
				stateful=bool(self.stateful), dropout=self.dropout)
	def _gen_reg(self, name):
		if name == 'kernel' and self.reg_kernel:
			return self._gen_l1_l2()
		elif name == 'bias' and self.reg_bias:
			return self._gen_l1_l2()
		elif name == 'activity' and self.reg_activity:
			return self._gen_l1_l2()
		else:
			return None
	def _gen_l1_l2(self):
		if self.l1:
			if self.l2:
				return l1_l2(self.l1)
			return l1(self.l1)
		elif self.l2:
			return l2(self.l2)
		else:
			return None
	def _detect_gpu(self):
		return "GPU" in [d.device_type for d in device_lib.list_local_devices()]
	def save(self, epoch=False):
		if not hvd or hvd.rank() == 0:
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
			if epoch:
				epoch_file = self.save_file.replace(".h5","_e%i.h5"%(epoch))
				self.model.save_weights(epoch_file)
			self.model.save_weights(self.save_file)
			logger.debug("Saved model")
	def _make_stateful_model(self):
		self.test_model = self._build_graph(test=True)
		self._compile_graph(self.test_model, 'mse', 'adam')
		logger.debug("Created test model")
		self.test = True
	def sync_stateful_online(self):
		if self.stateful:
			if not self.test:
				self._make_stateful_model()
			current_weights = self.model.get_weights()
			# update weights of new model
			self.test_model.set_weights(current_weights)
			logger.debug("Updated weights")
		else:
			logger.error("This should only be used with stateful models")
	def restore(self):
		self.model.load_weights(self.save_file)
		logger.debug("Restored model from %s"%(self.save_file))
	def __del__(self):
		clear_session()
	def train(self, x_batch, y_batch):
		# TODO add horovod code
		start_time = time()
		loss, accuracy = self.model.train_on_batch(x_batch, y_batch)
		total_time = time() - start_time
		#logger.debug("Finished training batch in %.1f seconds (%.1f sequences/second)"%(total_time, len(x_batch)/total_time))
		return (loss, accuracy, total_time)
	def predict(self, x_batch, return_time=False):
		start_time = time()
		y_pred = self.model.predict_on_batch(x_batch)
		total_time = time() - start_time
		#logger.debug("Finished predict batch in %.1f seconds (%.1f sequences/second)"%(total_time, len(x_batch)/total_time))
		if return_time:
			return np.abs(y_pred.round(0)).astype(np.uint32), total_time
		return np.abs(y_pred.round(0)).astype(np.uint32)

def mem_usage():
	process = psutil.Process(os.getpid())
	return process.memory_info().rss/1000000

if __name__ == "__main__":
	main()
