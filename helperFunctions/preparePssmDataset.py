import os
import sys

for line in open("./dataset_shalem_out.csv","r"):
	items = line.split(",")
	identifier = items[0].strip()
	sequence = items[16].strip() + items[23].strip() + items[17].strip()
	print identifier + " => " + sequence
	outfile = open("./pssm/jobs/" + identifier + ".fas","w")
	outfile.write(">" + identifier + "\n" + sequence + "\n")
	outfile.close()
	
