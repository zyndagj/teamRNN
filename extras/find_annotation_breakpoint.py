#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 08/13/2019
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

from teamRNN.reader import gff3_interval
from pysam import FastaFile
from os import symlink, getcwd, mkdir, path
from shutil import rmtree, copyfile
import matplotlib.pyplot as plt
import numpy as np
from time import time
from random import choice, sample, randint
from glob import glob
import sys
from itertools import groupby

out_dir = "gff3_benchmark"
o_gff3 = "../arabidopsis_thaliana/arabidopsis_thaliana.gff3"
fa = "arabidopsis_thaliana/arabidopsis_thaliana.fa"
gff3 = path.join(out_dir, path.basename(o_gff3))
num_reps = 4

def main():
	if not path.exists(out_dir):
		mkdir(out_dir)
		symlink(o_gff3, gff3)
	GI = gff3_interval(gff3, fa=fa)
	with FastaFile(fa) as FA:
		#chroms = ['1','2','3','4','5']
		chroms = ['1']
		chrom_lens = {c:FA.get_reference_length(c) for c in chroms}
	X = []
	Y_h5 = []
	Y_tree = []
	for power in range(70):
		seq_len = int(100*2**(power/4.0))
		h5_times = []
		tree_times = []
		for c,s,e in random_gen(chroms, chrom_lens, seq_len, num_reps):
			h5_time, h5_region = fetch_h5(GI, c,s,e)
			tree_time, tree_region = fetch_tree(GI, c,s,e)
			#print seq_len,'-',c,s,e,'-',e-s,'-',h5_region.shape, tree_region.shape
			#mismatching_indices = np.unique(np.where(h5_region != tree_region)[0])
			#for mismatched_range in list(ranges(mismatching_indices)):
			#	s,e = mismatched_range
			#	print("Mismatch at %s:%i-%i"%(c,s,e))
			h5_times.append(h5_time)
			tree_times.append(tree_time)
		Y_h5.append(np.median(h5_times))
		Y_tree.append(np.median(tree_times))
		X.append(seq_len)
		print power, X[-1], Y_h5[-1], Y_tree[-1]
	plt.loglog(X,Y_h5,label="HDF5 file")#, basex=2, basey=2)
	plt.loglog(X,Y_tree,label="Interval tree")#, basex=2, basey=2)
	plt.xlabel("Region Size (bp)")
	plt.ylabel("Time (seconds)")
	plt.title("Time Required for Querying Annotation Regions")
	plt.legend(loc=2)
	plt.savefig("anno_performance.pdf")
	#plt.show()
			
def random_gen(chroms, chrom_lens, size, n):
	for i in range(n):
		chrom = choice(chroms)
		while chrom_lens[chrom] < size:
			chrom = choice(chroms)
		max_v = chrom_lens[chrom] - size + 1
		s = randint(0, max_v)
		e = s+size
		assert(e-s == size)
		yield (chrom, s, e)
		
def fetch_h5(GI, c,s,e):
	ts = time()
	region = GI._fetch_h5(c,s,e)
	te = time()
	assert(region.shape[0] == e-s)
	return te-ts, region
def fetch_tree(GI, c,s,e):
	ts = time()
	region = GI._fetch_tree(c,s,e)
	te = time()
	assert(region.shape[0] == e-s)
	return te-ts, region
def ranges(i):
	for a, b in groupby(enumerate(i), lambda (x, y): y - x):
		b = list(b)
		yield b[0][1], b[-1][1]

if __name__ == "__main__":
	if getcwd().split('/')[-1] == 'extras':
		main()
	else:
		print("Please run from the extras directory")

