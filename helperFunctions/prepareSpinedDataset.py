import os
import sys

for line in open("./dataset_shalem_out.csv","r"):
	items = line.split(",")
	identifier = items[0].strip()
	sequence = items[24].strip()
	print identifier + " => " + sequence
	outfile = open("./spined/jobs/" + identifier + ".fasta","w")
	outfile.write(">" + identifier + "\n" + sequence + "\n")
	outfile.close()
	
