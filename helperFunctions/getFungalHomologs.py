import os
import sys
import urllib2
import copy
import operator
import traceback
from cogent.db.ensembl import Species, HostAccount, Genome


# GET GENE IDs
queries = {}
queriesOrganisms = {}
for line in open("./dataset_shalem_out.csv"):
	if(line.startswith("#") or (len(line.strip()) == 0)):
		continue
	queries[line.split(",")[0]] = True

print "> Number of loaded queries: " + str(len(queries))

# GET HOMOLOG IDs
hits = {}
hitsOrganisms = {}
i = 0
outfile = open("./dataset_fungal_homologs_OUT.csv","w")
for line in open("./dataset_fungal_homologs_out.csv"):
	items = line.split("\t")
	identifiers = []
	identifiers.append(items[0].strip())
	identifiers.append(items[1].strip())
	identifiers.append(items[5].strip())
	identifiers.append(items[6].strip())
	
	for identifier in identifiers:
		if(identifier in queries):
			for identifier2 in identifiers:
				hits[identifier2] = True
			outfile.write(line)
			break
	if(i == 0):
		outfile.write(line)
	i += 1
	#if(i == 5):
	#	break
	
	hitsOrganisms[items[2]] = True
	hitsOrganisms[items[7]] = True

print "> Number of loaded hits: " + str(len(hits))
outfile.close()

print "> SPECIES <"
for key in hitsOrganisms:
	print key

exit(1)


##### SET CONNECTION TO THE ENSEMBL DATABASE ##########################
Release = 67
account = None
yeast = Genome(Species='Neosartorya fischeris', Release=Release, account=account)
outfile = open("dataset_fungal_homologs_sequences.csv", "w")
print Species

##### GET SEQUENCES FROM ENSEMBL ######################################
i = 0
for hit in hits:
	i += 1
	print "i: " + str(i)
	
	identifier = hit
	print "\n***** ZPRACOVANI ZAZNAMU " + identifier + " *****"
	
	# Selection of species
	genes = yeast.getGenesMatching(StableId="CADNFIAP00000001")
	
	if(genes == None):
		print "> skip"
		continue
		
	for gene in genes:
		try:
			print "\n\n=================== GENE ===================="
			print "> gene.symbol: " + str(gene.Symbol)
			print "> gene.description: " + str(gene.Description)
			print "> gene.location: " + str(gene.Location)
			print "> gene.length: " + str(len(gene))
			print "> gene.full_info: " + str(gene)
			print "> gene.bio_type: " + str(gene.BioType)
			dnaSequence = str(gene.Seq).strip()
			transcriptSequence = str(gene.CanonicalTranscript.Cds).strip()
			
			print "> gene.seq: " + dnaSequence
			print "> gene.canonical_transcript_cds: " + transcriptSequence
						
			outfile.write(identifier + "," + dnaSequence + "," + transcriptSequence + "\n")
			
			outfilePath = "./fungalSequences/" + identifier + "_DNA.fasta"
			outfileSeq = open(outfilePath, "w")
			outfileSeq.write(">" + hit + "\n" + dnaSequence + "")
			outfileSeq.close()
			
			outfilePath = "./fungalSequences/" + identifier + "_TRANSCRIPT.fasta"
			outfileSeq = open(outfilePath, "w")
			outfileSeq.write(">" + hit + "\n" + transcriptSequence + "")
			outfileSeq.close()
			
			break
		except:
			traceback.print_exc()
	

outfile.close()

