###################################################################################################################
###################### HALF-LIFE PREDICTOR (STEP 1) - DATASET FROM ARTICLE WRITTEN BY SHALEM  #####################
##### Function: Get nucleotide, amino acid and protein sequence via Ensembl database ##############################
###################################################################################################################

import os
import sys
import urllib2
import copy
import operator
import traceback
from cogent.db.ensembl import Species, HostAccount, Genome

##### CONFIGURATION ###################################################
BELLI_DATASET = "./belli_dataset_initial.csv"                  # Dataset with measured half-life from dataset of Belli's article
SHALEM_DATASET = "./dataset_shalem_initial.csv"                # Dataset with measured half-life from dataset of Shalem's article without sequence
SHALEM_DATASET_OUT = "./shalem_dataset_sequence_assigned.csv"  # Dataset with measured half-life from dataset of Shalem's article with loaded sequences


##### GET MEASURED HALF-LIFE FROM BELLI'S DATASET #####
#halfLife = {}
#for line in open(DATASET_BELLI, "r"):
#	items = line.split(",")
#	identifier = items[0]
#	halfLife = float(items[3])
#	belliRecords[identifier] = halfLife

##### GET LAST PROCESSED ID FROM PREVIOUS RUN #########################
skipFlag = False
lastId = "XXX"
if(os.path.isfile(SHALEM_DATASET_OUT)):
	datasetFile = open(SHALEM_DATASET_OUT, "r")
	allLines = datasetFile.readlines()
	if(len(allLines) > 1):
		lastId = allLines[len(allLines)-1].split(",")[0]
		skipFlag = True

##### SET CONNECTION TO THE ENSEMBL DATABASE ##########################
Release = 67
account = None
yeast = Genome(Species='Homo Sapiens', Release=Release, account=account)
outfile = open(SHALEM_DATASET_OUT, "a")

##### GET SEQUENCES FROM ENSEMBL ######################################
for line in open(SHALEM_DATASET):
	if(line.startswith("#")):   # skip headline 
		continue
	items = line.split(",")
	identifier = items[0].strip()
	
	if(identifier == lastId):   # inverse skipFlag variable when last processed identifier from previous run was found
		skipFlag = False
		continue
	if(skipFlag == True):       # identifiers processed in previous flag will be skipped
		continue
	
	print "\n***** ZPRACOVANI ZAZNAMU " + identifier + " *****"
	
	# Selection of species
	genes = yeast.getGenesMatching(StableId="ERRB2")
		
	for gene in genes:
		try:
			print "\n\n=================== GENE ===================="
			print "> gene.symbol: " + str(gene.Symbol)
			print "> gene.description: " + str(gene.Description)
			print "> gene.location: " + str(gene.Location)
			print "> gene.length: " + str(len(gene))
			print "> gene.full_info: " + str(gene)
			print "> gene.bio_type: " + str(gene.BioType)
			dnaSequence = str(gene.Seq)
			proteinSequence = str(gene.CanonicalTranscript.ProteinSeq)
			transcriptSequence = str(gene.CanonicalTranscript.Cds)
			
			print "> gene.seq: " + dnaSequence[:40]
			print "> gene.canonical_transcript_protein_seq: " + proteinSequence[:40]
			print "> gene.canonical_transcript_cds: " + transcriptSequence[:40]
			
			belliRecord = "None"
			if(identifier in belliRecords):
				belliRecord = belliRecords[identifier]
			
			outfile.write(line.strip() + "," + str(belliRecord) + ",UNKNOWN," + dnaSequence + "," + proteinSequence + "," + transcriptSequence + "\n")
		
			break
		except:
			traceback.print_exc()
	

outfile.close()
