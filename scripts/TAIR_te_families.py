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
logger = logging.getLogger('TAIR_te_families')

def main():
	parser = argparse.ArgumentParser(description="Adds TE family information to TAIR annotation")
	parser.add_argument('-i', '--input', metavar='GFF', help="TAIR gff file (can be piped with '-')", default='-', type=str, required=True)
	parser.add_argument('-t', '--txt', metavar='TXT', help="TAIR te identity file", type=str, required=True)
	parser.add_argument('-o', '--output', metavar='GFF', help="Output file (Default %(default)s)", default='STDOUT', type=str)
	args = parser.parse_args()

	# Parse the identity file
	ID = open(args.txt, 'r')
	name_dict = {} #{name:super_family, ...}
	alias_dict = dd(set) #{alias:set([super_family, ...]), ...}
	for line in ID:
		#Transposon_Name orientation_is_5prime   Transposon_min_Start    Transposon_max_End      Transposon_Family    Transposon_Super_Family
		#AT1TE52125      false   15827287        15838845        ATHILA2 LTR/Gypsy
		split_line = line.rstrip('\n').split('\t')
		if line[0] == '#' or split_line[0] == 'Transposon_Name':
			continue
		te_name = split_line[0]
		alias = split_line[4]
		family = split_line[5]
		name_dict[te_name] = family
		alias_dict[alias].add(family)
	ID.close()

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

	# Create regexes
	name_re = re.compile('Name=(?P<name>[^;:]+)')
	alias_re = re.compile('Alias=(?P<alias>[^;]+)')
	# Create chromosome name mapping
	new_chroms = {'Chr1':'1','Chr2':'2','Chr3':'3','Chr4':'4','Chr5':'5','ChrM':'Mt','ChrC':'Pt'}
	# Parse the input
	for line in IF:
		# ##gff-version   3
		# Chr1    Araport11       gene    3631    5899    .       +       .       ID=AT1G01010;Name=AT1G01010;Note=NAC domain containing protein 1;symbol=NAC001;Alias=ANAC001,NAC domain containing protein 1;full_name=NAC domain co
		# Chr1    Araport11       mRNA    3631    5899    .       +       .       ID=AT1G01010.1;Parent=AT1G01010;Name=AT1G01010.1;Note=NAC domain containing protein 1;conf_class=2;symbol=NAC001;Alias=ANAC001,NAC domain containing
		# Chr1    Araport11       transposable_element    11897   11976   .       +       .       ID=AT1TE00010;Name=AT1TE00010;Alias=ATCOPIA24
		# Chr1    Araport11       transposon_fragment     11897   11976   .       +       .       ID=AT1TE00010:transposon_fragment:1;Parent=AT1TE00010;Name=AT1TE00010:transposon_fragment:1
		if line[0] == '#':
			OF.write(line)
			continue
		split_line = line.rstrip('\n').split('\t')
		split_line[0] = new_chroms[split_line[0]]
		# Change chromosome name
		feature_type = split_line[2]
		if feature_type not in set(("transposable_element", 'transposon_fragment')):
			OF.write('\t'.join(split_line)+'\n')
			continue
		attributes = split_line[8]
		te_name = name_re.search(attributes).group("name")
		te_alias = ''
		if feature_type == 'transposable_element':
			te_alias = alias_re.search(attributes).group("alias")
		if te_name in name_dict:
			attributes += modify_attributes(name_dict[te_name])
		elif te_alias and te_alias in alias_dict:
			family_set = alias_dict[te_alias]
			if len(family_set) == 1:
				attributes += modify_attributes(family_set)
		split_line[8] = attributes
		out_str = '\t'.join(split_line)
		OF.write(out_str+'\n')
	OF.close()
	IF.close()

def modify_attributes(o_su):
	if '/' in o_su:
		o, su = o_su.split('/')
	else:
		o, su = (o_su, 'Unassigned')
	o = o.rstrip('?')
	su = su.rstrip('?')
	if 'RathE' in o:
		su = o.rstrip('_cons')
		o = 'SINE'
	return ';Order=%s;Superfamily=%s'%(o, su)
	

if __name__ == "__main__":
	main()
