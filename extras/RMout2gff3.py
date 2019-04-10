#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 04/08/2019
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

import sys, argparse, os, logging
logger = logging.getLogger(__name__)

def main():
	parser = argparse.ArgumentParser(description="Converts RepeatMasker output to GFF3 with class, family, and score information")
	parser.add_argument('-i', '--input', metavar='STR', help="RepeatMasker 'out' for conversion (can be piped with '-')", default='-', type=str, required=True)
	parser.add_argument('-o', '--output', metavar='STR', help="Output file (Default %(deafult)s)", default='STDOUT', type=str)
	args = parser.parse_args()

	# Open the input
	if args.input == '-':
		IF = sys.stdin
	elif os.path.splitext(args.input)[1] == '.out':
		IF = open(args.input, 'r')
	else:
		logger.error("Input file extension not '.out' - %s"%(args.input))
		sys.exit(1)
	# Open the output
	if args.output == 'STDOUT' or not args.output:
		OF = sys.stdout
	else:
		OF = open(args.output, 'w')
	
	# Check the first two lines of input
	line1 = IF.readline().lstrip(' ').split()
	line2 = IF.readline().lstrip(' ').split()
	line3 = IF.readline()
	if line1[0] != 'SW' or line2[0] != 'score' or not line3:
		logger.error("Input is not 'out' from RepeatMasker")
		sys.exit(2)
	# Parse the input
	for line in IF:
		split_line = line.lstrip(' ').rstrip('\n').split()
		score = int(split_line[0])
		p_divergence = float(split_line[1])
		chrom = split_line[4]
		start = int(split_line[5])
		end = int(split_line[6])
		strand = '+' if split_line[8] == '+' else '-'
		match = split_line[9]
		class_family = split_line[10]
		if '/' in class_family:
			c, f = class_family.split('/')
		else:
			c, f = class_family, ''
		t_id = int(split_line[14])
		# create output row
		attributes = "Score=%i;Match=%s;Class=%s"%(score, match,c)
		if f: attributes+=";Family=%s"%(f)
		out_list = [chrom, 'RepeatMasker', 'transposable_element', start, end, round(p_divergence, 1), strand, '.', attributes]
		out_str = '\t'.join(map(str, out_list))
		OF.write(out_str+'\n')
	OF.close()
	IF.close()

if __name__ == "__main__":
	main()
