#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 06/24/2019
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

from operator import itemgetter
from pysam import FastaFile
import h5py, os, sys, logging
logger = logging.getLogger(__name__)
import numpy as np
from teamRNN.util import irange, iterdict, fivenum
from teamRNN.constants import gff3_f2i, gff3_i2f, contexts, strands, base2index, te_feature_names
from teamRNN.constants import te_order_f2i, te_order_i2f, te_sufam_f2i, te_sufam_i2f
from itertools import chain
import re
from quicksect import IntervalTree
from glob import glob
from collections import defaultdict as dd

class output_aggregator:
	'''
	>>> OA = output_aggregator(chrom_dict)
	>>> OA.vote(chrom, s, e, out_array)
	>>> OA.write_gff3()
	'''
	def __init__(self, fasta_file, h5_file='tmp_vote.h5'):
		self.fasta_file = fasta_file
		with FastaFile(fasta_file) as FA:
			self.chrom_dict = {c:FA.get_reference_length(c) for c in FA.references}
		self.cur_chrom = ''
		self.h5_file = h5_file
		self.H5 = h5py.File(h5_file, 'a')
		self._genome_init()
	def __del__(self):
		if self.H5:
			self.H5.close()
			os.remove(self.h5_file)
	def close(self):
		self.__del__()
	def _load_arrays(self, chrom):
		self.feature_vote_array = self.H5[chrom+'/votes/features']
		self.feature_total_array = self.H5[chrom+'/totals/features']
		self.te_order_array = self.H5[chrom+'/votes/tes/order']
		self.te_sufam_array = self.H5[chrom+'/votes/tes/sufam']
		#self.te_total_array = self.H5[chrom+'/totals/tes']
		self.cur_chrom = chrom
		self.te_feature_ids = set([gff3_f2i[s+f] for f in te_feature_names for s in '+-'])
	def _genome_init(self):
		n_features = len(gff3_i2f)
		n_order_ids = len(te_order_i2f)
		n_sufam_ids = len(te_sufam_i2f)
		for chrom, chrom_len in iterdict(self.chrom_dict):
			# total != sum
			self.feature_vote_array = self.H5.create_dataset(chrom+'/votes/features', \
				(chrom_len, n_features), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype=np.uint32)
			self.feature_total_array = self.H5.create_dataset(chrom+'/totals/features', \
				(chrom_len, 1), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype=np.uint32)
			# These do not need a total array since there is only a single value per location
			self.te_order_array = self.H5.create_dataset(chrom+'/votes/tes/order', \
				(chrom_len, n_order_ids), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype=np.uint32)
			self.te_sufam_array = self.H5.create_dataset(chrom+'/votes/tes/sufam', \
				(chrom_len, n_sufam_ids), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype=np.uint32)
		self.cur_chrom = chrom
		# Create counters
		self.features_tp = np.zeros(n_features)
		self.features_fn = np.zeros(n_features)
		self.features_fp = np.zeros(n_features)
		self.te_order_tp = np.zeros(n_order_ids)
		self.te_order_fn = np.zeros(n_order_ids)
		self.te_order_fp = np.zeros(n_order_ids)
		self.te_sufam_tp = np.zeros(n_sufam_ids)
		self.te_sufam_fn = np.zeros(n_sufam_ids)
		self.te_sufam_fp = np.zeros(n_sufam_ids)
	def vote(self, chrom, start, end, array, overwrite=False):
		#print "VOTE:", chrom, start, end, np.nonzero(array)
		# Split the array
		assert(array.shape[1] == len(gff3_i2f)+2)
		feature_array = array[:,:len(gff3_i2f)]
		te_order_array = array[:,-2]
		te_sufam_array = array[:,-1]
		# Load the current chromosome arrays
		if self.cur_chrom != chrom:
			self._load_arrays(chrom)
		# Track features
		#print "BEFORE", self.feature_total_array[start:end].flatten()
		if overwrite:
			self.feature_total_array[start:end] = 1
		else:
			self.feature_total_array[start:end] += 1
		#print "AFTER", self.feature_total_array[start:end].flatten()
		if np.sum(feature_array):
			if overwrite:
				self.feature_vote_array[start:end] = feature_array
			else:
				self.feature_vote_array[start:end] += feature_array
		# Track te class/family
		if sum(te_order_array):
			for i,v in enumerate(te_order_array):
				if overwrite:
					self.te_order_array[start+i,v] = 1
				else:
					self.te_order_array[start+i,v] += 1
		if sum(te_sufam_array):
			for i,v in enumerate(te_sufam_array):
				if overwrite:
					self.te_sufam_array[start+i,v] = 1
				else:
					self.te_sufam_array[start+i,v] += 1
	def compare(self, chrom, start, end, pred_array, true_array):
		assert(array.shape[1] == len(gff3_i2f)+2)
		assert(pred_array.shape == true_array.shape)
		# Split the array
		pred_feature_array = pred_array[:,:len(gff3_i2f)]
		pred_te_order_array = pred_array[:,-2]
		pred_te_sufam_array = pred_array[:,-1]
		true_feature_array = true_array[:,:len(gff3_i2f)]
		true_te_order_array = true_array[:,-2]
		true_te_sufam_array = true_array[:,-1]
		# Track features
		for i in irange(pread_array.shape[0]):
			for j in irange(len(gff3_i2f)):
				# Features
				if pred_feature_array[i,j] != true_feature_array[i,j]:
					if pred_feature_array[i,j] == 1:
						self.features_fp[j] += 1
					else:
						self.features_fn[j] += 1
				elif pred_feature_array[i,j] == 1:
					self.features_tp[j] += 1
					# TEs
					if j in self.te_feature_ids:
						# Order
						if true_te_order_array[i] == 0:
							if pred_te_order_array[i] != 0:
								self.te_order_fp[pred_te_order_array[i]] += 1
						else:
							if pred_te_order_array[i] == true_te_order_array[i]:
								self.te_order_tp[pred_te_order_array[i]] += 1
							else:
								self.te_order_fn[pred_te_order_array[i]] += 1
						# Super family
						if true_te_sufam_array[i] == 0:
							if pred_te_sufam_array[i] != 0:
								self.te_sufam_fp[pred_te_sufam_array[i]] += 1
						else:
							if pred_te_sufam_array[i] == true_te_sufam_array[i]:
								self.te_sufam_tp[pred_te_sufam_array[i]] += 1
							else:
								self.te_sufam_fn[pred_te_sufam_array[i]] += 1
	def comparison_stats(self, features=True, order=False, sufam=False):
		oStr = [', '.join(('Name','TP','FP','FN','Sensitivity','Precision'))]
		if features:
			for i in irange(len(gff3_i2f)):
				name = gff3_i2f[i]
				tp = self.te_features_tp[i]
				fp = self.te_features_fp[i]
				fn = self.te_features_fn[i]
				sen = str(tp/float(tp+fn))
				prec = str(tp/float(tp+fp))
				oStr.append('%s, %i, %i, %i, %s, %s'%(name, tp, fp, fn, sen, prec))
		return oStr
	def write_gff3(self, out_file='', threshold=0.5):
		total_feature_count = 0
		out_gff3 = ['##gff-version   3']
		for chrom in sorted(self.chrom_dict.keys()):
			chrom_len = self.chrom_dict[chrom]
			features = []
			se_array = [[0,0] for i in irange(len(gff3_i2f))]
			self._load_arrays(chrom)
			for index in irange(chrom_len):
				index_votes = self.feature_vote_array[index]
				index_totals = self.feature_total_array[index]
				for feat_index in gff3_i2f.keys():
					se = se_array[feat_index]
					#if index_votes[feat_index] > 0:
						#print chrom, index, feat_index, index_votes[feat_index], index_totals, index_votes[feat_index] >= threshold*index_totals
					if index_votes[feat_index] >= threshold*index_totals and index_votes[feat_index] > 0:
						if se[0] == 0:
							se[0] = index+1
						else:
							se[1] = index+1
						if index == chrom_len-1 and se[1]:
							features.append((se[0],se[1],feat_index))
							se[0],se[1] = 0,0
					else:
						if se[1] != 0:
							features.append((se[0],se[1],feat_index))
							se[0],se[1] = 0,0
			features.sort(key=itemgetter(0,1))
			for s,e,feat_index in features:
				full_name = gff3_i2f[feat_index]
				strand = full_name[0]
				feature_name = full_name[1:]
				feature_str = "%s\tteamRNN\t%s\t%i\t%i\t.\t%s\t.\tID=team_%i"%(chrom, feature_name, s, e, strand, total_feature_count)
				if feature_name in te_feature_names:
					argmax_order_sum = np.argmax(np.sum(self.te_order_array[s-1:e], axis=0))
					te_order = te_order_i2f[argmax_order_sum]
					argmax_sufam_sum = np.argmax(np.sum(self.te_sufam_array[s-1:e], axis=0))
					te_sufam = te_sufam_i2f[argmax_sufam_sum]
					feature_str += ';Order=%s;Superfamily=%s'%(te_order, te_sufam)
				out_gff3.append(feature_str)
				total_feature_count += 1
		if out_file:
			with open(out_file,'w') as OF:
				OF.write('\n'.join(out_gff3)+'\n')
		else:
			return out_gff3

class MSE_interval:
	def __init__(self, fasta_file, out_dir, hvd_rank):
		self.mse_dict = dd(IntervalTree)
		self.rank = hvd_rank
		self.out_dir = out_dir
		self.regex = re.compile(r"mse__([>\w]+)__(\d+).pkl")
		self.fasta_file = fasta_file
		with FastaFile(fasta_file) as FA:
			self.chrom_dict = {c:FA.get_reference_length(c) for c in FA.references}
		self.agg_method = {'median':np.median, 'mean':np.mean, 'sum':np.sum}
		self.c_method = {'midpoint':self._to_midpoint, 'range':self._to_range}
	def add_batch(self, cb, mse_value):
		for chrom, s, e in cb:
			self.mse_dict[chrom].add(s,e,mse_value)
	def dump(self):
		for chrom in self.mse_dict:
			if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
			out_name = os.path.join(self.out_dir, 'mse__%s__%i.pkl'%(chrom, self.rank))
			self.mse_dict[chrom].dump(out_name)
	def __del__(self):
		for p_file in glob("%s/mse__*__%i.pkl"%(self.out_dir, self.rank)):
			logger.debug("Deleting %s"%(p_file))
			os.remove(p_file)
	def load_all(self):
		if self.rank == 0:
			for p_file in glob("%s/mse__*__*.pkl"%(self.out_dir)):
				chrom, rank = self.regex.search(p_file).groups()
				rank = int(rank)
				assert(chrom in self.chrom_dict.keys())
				if rank != self.rank:
					logger.debug("loading %s"%(p_file))
					self.mse_dict[chrom].load(p_file)
				else:
					logger.debug("skipping %s"%(p_file))
	def _to_midpoint(self, s, e, v):
		half = (e-s)/2.0
		x, y = [s+half], [v]
		return x,y
	def _to_range(self, s, e, v):
		x = [s, e]
		y = [v, v]
		return x,y
	def _region_to_agg_value(self, chrom, start, end, method='mean'):
		values = []
		if chrom not in self.mse_dict:
			return -1
		for interval in self.mse_dict[chrom].search(start, end):
			v = interval.data
			assert(not hasattr(v, '__iter__'))
			N = min(end, interval.end)-max(start, interval.start)
			values += [v]*N
		agg_value = self.agg_method[method](values) if values else -1
		return agg_value
	def _region_to_xy(self, c, s, e, method='mean', coords='midpoint'):
		agg_value = self._region_to_agg_value(c, s, e, method)
		rx, ry = self.c_method[coords](s, e, agg_value)
		return rx, ry
	def to_array(self, chrom, width=1000, method='mean', coords='midpoint'):
		x,y = [], []
		if self.rank == 0:
			for start in irange(0, self.chrom_dict[chrom], width):
				end = min(start+width, self.chrom_dict[chrom])
				assert(start != end)
				rx, ry = self._region_to_xy(chrom, start, end, method, coords)
				x += rx
				y += ry
		return x,y
	def write(self, hvd=False, chroms=[], name='TRAIN', epoch=0, width=1000, method='mean', coords='midpoint'):
		assert(name in set(('TRAIN','TEST')))
		self.dump()
		if hvd: hvd.allgather([self.rank], name="Barrier")
		self.load_all()
		if hvd: hvd.allgather([self.rank], name="Barrier")
		if self.rank != 0: return
		for chrom in sorted(chroms):
			x,y = self.to_array(chrom, width=width, method=method, coords=coords)
			f_sum_str_list = map(str, fivenum(y))
			f_sum_str = ', '.join(f_sum_str_list)
			logger.info("%s - Epoch %3i - Chrom %s MSE summary [%s]"%(name, epoch+1, chrom, f_sum_str))
			out_file = os.path.join(self.out_dir, '%s_e%i_%s.tsv'%(name.lower(), epoch, chrom))
			np.savetxt(out_file, (x,y), delimiter='\t')
			logger.debug("Wrote %s"%(out_file))

#def main():

#if __name__ == "__main__":
#	main()
