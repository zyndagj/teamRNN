#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 06/14/2019
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

import numpy as np

try:
	from itertools import izip
	tmp = xrange(2)
	del tmp
	irange = xrange
	def iterdict(D): return D.iteritems()
except:
	izip = zip
	irange = range
	def iterdict(D): return D.items()

def fivenum(data):
	return np.percentile(data, [0, 25, 50, 75, 100], interpolation='midpoint')

def calcRegionBounds(bool_array, inclusive=False, null=0):
	'''
	Returns the new lower and upper bounds over overlapped regions. Returns [s, e)
	
	Parameters
	=============================
	bool_array	Binary interest array
	>>> calcRegionBounds(np.array([1,1,0,0,1,1,1,0,0,1,1]))
	array([[ 0,  1],
	       [ 4,  6],
	       [ 9, 10]])
	'''
	assert(bool_array.dtype == 'bool')
	if inclusive:
		idx = np.diff(bool_array).nonzero()[0]
		if bool_array[0]:
			idx = np.r_[-1, idx]
		if bool_array[-1]:
			idx = np.r_[idx, bool_array.size-1]
		idx.shape = (-1,2)
		idx[:,0] += 1
	else:
		idx = np.diff(np.r_[null, bool_array, null]).nonzero()[0]
		assert(len(idx)%2 == 0)
		idx.shape = (-1, 2)
	return idx

def bridge_array(bool_array, min_size=1, max_gap_size=1):
	'''
	Fills small gaps and removes small classifications

	Operates directly on `bool_array` and returns nothing

	# Parameters
	bool_array (np.ndarray): boolean classification array
	min_size (int): minimum feature size to be kept
	max_gap_size (int): maximum gap size to be filled
	'''
	# Clean up things like
	#1       teamRNN gene    425     425     .       +       .       ID=team_7
	#1       teamRNN gene    428     428     .       +       .       ID=team_8
	#1       teamRNN gene    437     437     .       +       .       ID=team_9
	#1       teamRNN gene    441     441     .       +       .       ID=team_11
	#1       teamRNN gene    445     471     .       +       .       ID=team_12
	#1       teamRNN gene    493     862     .       +       .       ID=team_16
	# This requires bridge then filter
	zero_intervals = calcRegionBounds(bool_array, null=1)
	filled_gap_count = 0
	# Bridge gaps <= max_gap_size
	for s,e in zero_intervals:
		gap_size = e-s
		if gap_size <= max_gap_size:
			filled_gap_count += 1
			assert(bool_array[s:e].sum() == 0)
			bool_array[s:e] = 1
	# Remove small regions
	removed_region_count = 0
	intervals = calcRegionBounds(bool_array)
	for s,e in intervals:
		interval_size = e-s
		if interval_size < min_size:
			removed_region_count += 1
			assert(bool_array[s:e].sum() == interval_size)
			bool_array[s:e] = 0
