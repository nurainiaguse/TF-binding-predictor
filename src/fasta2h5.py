# Converts fasta files to h5py representation
# Each 3-mer is represented as 1 or 0

from Bio import SeqIO
import numpy as np
import sys
import h5py
import itertools

if len(sys.argv) < 2:
	print("Usage: python fasta2h5.py <filename.fasta>")
	exit()

k = 3
seq_len = 500
comb = 4**k # possible combinations of k-mers
fasta_sequences = SeqIO.parse(open(sys.argv[1]),'fasta')
np.random.seed(1)

def create_feature_vector(k, comb):
	kmer_comb = itertools.combinations_with_replacement('ACTGN',3)
	kmer_dict = {}
	i = 0
	for kmer in kmer_comb:
		string_kmer = ''.join(kmer)
		sub_kmer_comb = itertools.permutations(string_kmer,3)
		for j, sub_kmer in enumerate(set(sub_kmer_comb)):
			f_vector = [0]*comb # each k-mer is a feature
			if not 'N' in sub_kmer:
				f_vector[i] = 1
				i = i+1
			kmer_dict[sub_kmer] = np.asarray(f_vector)
	if len(kmer_dict) < comb:
		raise Exception("Please check your math.")
	return kmer_dict

		
def fasta2h5(seqs, filename):
	seqmat = np.zeros((len(seqs), comb, seq_len-k))
	kmer_dict = create_feature_vector(k, comb)
	n = 0
	for seq in seqs:
		seq = seq.upper()
		for i in range(seq_len-k):
			seqmat[n,:,i] = kmer_dict[tuple(seq[i:i+k])]
		n = n + 1
	# kmer_dict = 
	# why do we need dataflip?
	# we're doing an np.concatenate? this means each sequence has two versions of data -- left to right and right to left
	seqmat = seqmat.reshape(seqmat.shape[0],comb,seq_len-k,1)
	print(seqmat.shape)
	seqmat = seqmat.astype(np.uint8)
	f = h5py.File(filename, 'w')
	f.create_dataset('traindata', data=seqmat, compression="gzip")
	f.close()
# create_feature_vector(k,comb)

seqs = [str(fasta.seq) for fasta in fasta_sequences]
# print(seqs[0])

fasta2h5(seqs, sys.argv[1]+'.ref.h5')