#!/usr/bin/env python
#
###############################################################################
# Author: Greg Zynda
# Last Modified: 04/10/2019
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

# Pulled from all GFF3 files
features = ['CDS', 'RNase_MRP_RNA', 'SRP_RNA', 'antisense_RNA', 'antisense_lncRNA', 'biological_region', 'chromosome', 'contig', 'exon', 'five_prime_UTR', 'gene', 'lnc_RNA', 'mRNA', 'miRNA', 'ncRNA', 'ncRNA_gene', 'pre_miRNA', 'pseudogene', 'pseudogenic_exon', 'pseudogenic_tRNA', 'pseudogenic_transcript', 'rRNA', 'region', 'snRNA', 'snoRNA', 'supercontig', 'tRNA', 'three_prime_UTR', 'tmRNA', 'transposable_element', 'transposable_element_gene', 'transposon_fragment', 'uORF',]
contexts = ('CG','CHG','CHH')
strands = ('+', '-')

gff3_f2i = {v:i for i,v in enumerate([s+e for s in strands for e in features])}
gff3_i2f = {i:v for i,v in enumerate([s+e for s in strands for e in features])}

te_feature_names = set(('transposable_element', 'transposable_element_gene', 'transposon_fragment'))

# Pulled from arabidopsis and repeatmasker
orders = ['Unassigned', 'DNA', 'LINE', 'LTR', 'Low_complexity', 'RC', 'Retroposon', 'rRNA', 'Satellite', 'Simple_repeat', 'SINE', 'snRNA', 'TIR', 'tRNA']
superfamilies = ['Unassigned', 'Cassandra', 'Caulimovirus', 'centr', 'CMC-EnSpm', 'Copia', 'En-Spm', 'Gypsy', 'HAT', 'hAT-Ac', 'hAT-Charlie', 'hAT-Tag1', 'hAT-Tip100', 'Harbinger', 'Helitron', 'L1', 'L1-dep', 'Mariner', 'MuDR', 'MULE-MuDR', 'PIF-Harbinger', 'Pogo', 'RathE1_cons', 'RathE2_cons', 'RathE3_cons', 'Tc1', 'TcMar-Mariner', 'TcMar-Pogo', 'TcMar-Stowaway', 'tRNA', 'solo']

te_order_f2i = {v.lower():i for i,v in enumerate(orders)}
te_order_i2f = list(orders)

te_sufam_f2i = {v.lower():i for i,v in enumerate(superfamilies)}
te_sufam_i2f = list(superfamilies)

#classes = ['Unassigned', 'DNA', 'DNA?', 'DNA/CMC-EnSpm', 'DNA/En-Spm', 'DNA/HAT', 'DNA/hAT-Ac', 'DNA/hAT-Charlie', 'DNA/hAT-Tag1', 'DNA/hAT-Tip100', 'DNA/Harbinger', 'DNA/Mariner', 'DNA/MuDR', 'DNA/MULE-MuDR', 'DNA/PIF-Harbinger', 'DNA/Pogo', 'DNA/Tc1', 'DNA/TcMar-Mariner', 'DNA/TcMar-Pogo', 'DNA/TcMar-Stowaway', 'LINE/L1', 'LINE?', 'Low_complexity', 'LTR/Cassandra', 'LTR/Caulimovirus', 'LTR/Copia', 'LTR/Gypsy', 'RC/Helitron', 'RC/Helitron?', 'RathE1_cons', 'RathE2_cons', 'RathE3_cons', 'Retroposon', 'Retroposon/L1-dep', 'rRNA', 'Satellite', 'Satellite/centr', 'Simple_repeat', 'SINE', 'SINE/tRNA', 'SINE/tRNA?', 'snRNA', 'tRNA']
#te_cf_f2i = {v.lower():i for i,v in enumerate(classes)}
#te_cf_i2f = {i:v for i,v in enumerate(classes)}

# https://github.com/zyndagj/teamRNN#numerical-mapping-key---016
bases = 'ACGTURYKMSWBDHVN-'
base2index = {b:i for i,b in enumerate(bases)}
index2base = bases

# Process configuration
tacc_nodes = {'knl':(136,2), 'skx':(48,2), 'hikari':(24,2)}

#def main():
#
#if __name__ == "__main__":
#	main()
