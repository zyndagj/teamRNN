#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 01/27/2019
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
from teamRNN.util import irange, iterdict
from teamRNN.constants import gff3_f2i, gff3_i2f

#def main():

class output_aggregator:
	'''
	>>> OA = output_aggregator(chrom_dict, gff3_dict)
	>>> OA.vote(chrom, s, e, out_array)
	>>> OA.write_gff3()
	'''
	def __init__(self, chrom_dict, gff3_dict, h5_file='tmp_vote.h5'):
		self.gff3_dict = gff3_dict
		self.chrom_dict = chrom_dict
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
		self.vote_array = H5[chrom+'/votes']
		self.total_array = H5[chrom+'/totals']
	def _genome_init(self):
		n_features = len(gff3_dict)
		for chrom, chrom_len in iterdict(self.chrom_dict):
			self.vote_array = self.H5.create_dataset(chrom+'/votes', \
				(chrom_len, n_features), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype='u32')
			self.total_array = self.H5.create_dataset(chrom+'/totals', \
				(chrom_len, 1), compression='gzip', \
				compression_opts=6, chunks=True, fillvalue=0, dtype='u32')
		self.cur_chrom = chrom
	def vote(self, chrom, start, end, array):
		if self.cur_chrom != chrom:
			self._load_arrays(chrom)
			self.cur_chrom = chrom
		self.total_array[start:end] += 1
		if np.sum(array):
			self.cur_chrom[start:end] += array
	def write_gff3(self, out_file='', threshold=0.5)
		# tdict = {index:name, ...}
		tdict = {v,k for k,v in iterdict(self.gff3_dict)}
		total_feature_count = 0
		out_gff3 = []
		for chrom, chrom_len in iterdict(self.chrom_dict):
			features = []
			se_array = [[0,0] for i in irange(len(self.gff3_dict))]
			self._load_arrays(chrom)
			for index in irange(chrom_len):
				index_votes = self.vote_array[index]
				index_totals = self.total_array[index]
				for feat_index in tdict.keys():
					se = se_array[feat_index]
					if index_votes[feat_index] >= threshold*index_totals:
						if se[0] == 0:
							se[0] = index+1
						else:
							se[1] = index+1
						if index == chrom_len-1:
							features.append((se[0],se[1],feat_index))
							se[0],se[1] = 0,0
					else:
						if se[1] != 0:
							features.append((se[0],se[1],feat_index))
							se[0],se[1] = 0,0
			features.sort(key=itemgetter(0,1))
			for s,e,feat_index in features:
				full_name = tdict[feat_index]
				strand = full_name[0]
				feature_name = full_name[1:]
				# think about making the ID more descriptive or hash to a unique value
				out_gff3.append("%s\tteamRNN\t%s\t%i\t%i\t.\t%s\t.\tID=team_%i"%(chrom, feature_name, s, e, strand, total_feature_count))
				total_feature_count += 1
		if out_file:
			with open(out_file,'w') as OF:
				OF.write('\n'.join(out_gff3)
		return out_gff3

#if __name__ == "__main__":
#	main()
