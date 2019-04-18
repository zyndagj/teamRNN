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

import sys, argparse, os, logging, re
from collections import defaultdict as dd
logging.basicConfig()
logger = logging.getLogger('ZM_te_families')

def main():
	parser = argparse.ArgumentParser(description="Adds TE family information to Zea Mays B73 annotation")
	parser.add_argument('-i', '--input', metavar='GFF', help="B73 gff file (can be piped with '-')", default='-', type=str, required=True)
	parser.add_argument('-o', '--output', metavar='GFF', help="Output file (Default %(default)s)", default='STDOUT', type=str)
	args = parser.parse_args()
	# Convert names to class/family
	# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3554455/
	superfamily_dict = dd(lambda: 'Unassigned')
	copia_families = ('opie','ji','iteki','ruda','bygum','gori',\
		'leviathan','vuijon','tufe','nida','giepum','dijap',\
		'tiwewi','anar','eninu','sawujo','ekoj','gudyeg',\
		'guvi','raider','stonor','victim','wamenu','udav',\
		'wiwa','lusi','debeh','donuil','ibulaf','iwim','labe',\
		'machiavelli','totu')
	for family in copia_families:
		superfamily_dict[family] = 'Copia'
	gypsy_families = ('dagaf','grande','gyma','flip','doke','ansuya','huck',\
		'nihep','riiryl','prem1','cinful-zeon','neha','xilondiguus',\
		'ywyt','ajajog','fourf','lata','guhis','gyte','bosohe')
	for family in gypsy_families:
		superfamily_dict[family] = 'Gypsy'
	l1_families = ('etiti',)
	for family in l1_families:
		superfamily_dict[family] = 'L1'
	# superfamilies: copia, gypsy, l1, Unassigned
	order_dict = {'helitron':'RC', 'LINE_element':'LINE', 'LTR_retrotransposon':'LTR',\
		'SINE_element':'SINE', 'solo_LTR':'solo_LTR', 'terminal_inverted_repeat_element':'TIR'}
	# orders: RC, DNA, LINE, LTR, SINE, solo_LTR, TIR

	# Open the input
	if args.input == '-':
		IF = sys.stdin
	elif os.path.splitext(args.input)[1] in set(('.gff','.gtf','.gff3')):
		IF = open(args.input, 'r')
	else:
		logger.error("Input file extension not at GFF - %s"%(args.input))
		sys.exit(1)
	# Open the output
	if args.output == 'STDOUT' or not args.output:
		OF = sys.stdout
	else:
		OF = open(args.output, 'w')
	su_regex = re.compile("Name=[^_]+_([^_]+)")
	# Parse the input
	for line in IF:
		# 1       RepeatMasker    solo_LTR        14558312        14561392        .       +       .       ID=RLX11152B73v4S0001;Name=RLX11152B73v4S0001_NA_SoloLTR
		# 1       LTRharvest      LTR_retrotransposon     14631092        14640095        .       +       .       ID=RLC00002B73v400781;Name=RLC00002B73v400781_ji_LTRsimilarity95.91
		# 1       SineFinder      SINE_element    14660704        14661215        .       +       .       ID=RST00003B73v400001;Name=RST00003B73v400001_NA_TSDlen13_TSDmismat2
		# 1       HelitronScanner helitron        14752755        14770924        .       +       .       ID=DHH00004B73v400152;Name=DHH00004B73v400152_NA_LCV5p10_LCV3p10
		# 1       TARGeT  terminal_inverted_repeat_element        29447299        29447397        .       *       .       ID=DTA_1_29447299_29447397;Name=DTA,TSDlen8,TIRlen11
		# 1       TARGeT  LINE_element    29518381        29521882        .       *       .       ID=RIL00001B73v400008;Name=RIL00001B73v400008_okor
		if line[0] == '#':
			OF.write(line)
			continue
		split_line = line.rstrip('\n').split('\t')
		# Store the order and attributes
		input_order = split_line[2]
		attributes = split_line[8]
		# Set the feature to transposable_element
		split_line[2] = 'transposable_element'
		# Determine order
		order = order_dict[input_order]
		# Determine the superfamily
		if input_order == 'helitron':
			superfamily = 'Helitron'
		else:
			match = su_regex.search(attributes)
			superfamily = 'Unassigned'
			if match:
				family = match.group(1)
				superfamily = superfamily_dict[family.lower()]
		# Modify the attributes
		attributes += ';Order=%s;Super=%s'%(order, superfamily)
		split_line[8] = attributes
		out_str = '\t'.join(split_line)
		OF.write(out_str+'\n')
	OF.close()
	IF.close()

if __name__ == "__main__":
	main()
