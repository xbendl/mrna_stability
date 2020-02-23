import os
import sys
import random
import copy

class Enum(set):
	def __getattr__(self, name):
		if name in self:
			return name
		raise AttributeError


RECORD_STATUS = Enum(["DINUCLEOTIDE_SHUFFLING", "PSSM", "PSSM_ASA", "RNAHEAT", "SPINED", "PSIPRED", "DISORDER"])

CALCULATED_FEATURE = sys.argv[1]
print "> CALCULATED_FEATURE = " + CALCULATED_FEATURE

# wormsik
SPINED_JOBS = "/storage/brno2/home/wormsik/halfLifeFeatureTester/spined/jobs/"
SPINED_PREDOUT = "/storage/brno2/home/wormsik/halfLifeFeatureTester/spined/predout/"
SPINED_BINARY = "/storage/brno2/home/wormsik/halfLifeFeatureTester/spined/bin/run_spine-d"
#BLAST_DB = "/auto/brno2/home/wormsik/shared/db/blast/nt/nt"
#BLAST_DB = "/auto/brno2/home/wormsik/shared/db/blast/fungi/alfa"
#BLAST_DB = "/auto/brno2/home/wormsik/shared/db/blast/saccharomyces/saccharomyces.fsa"
BLAST_DB = "/auto/brno2/home/wormsik/shared/db/blast/fungiSac/fungiSac"
BLASTN_BINARY = "/auto/brno2/home/wormsik/shared/apps/blast/ncbi-blast-2.2.30+/bin/blastn"

# bendl
DATASET_PATH = "/auto/brno2/home/bendl/halfLifeFeatureTester/halfLifeDataset.csv"
ROOT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/"
WORKING_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/tmp/"
OUTPUT_FILE = "/auto/brno2/home/bendl/halfLifeFeatureTester/results/results_" + CALCULATED_FEATURE + ".txt"
RNAHEAT_BINARY = "/auto/brno2/home/bendl/halfLifeFeatureTester/ViennaRNA-2.1.9/Progs/RNAheat"
RNAFOLD_BINARY = "/auto/brno2/home/bendl/halfLifeFeatureTester/ViennaRNA-2.1.9/Progs/RNAfold"
RNALFOLD_BINARY = "/auto/brno2/home/bendl/halfLifeFeatureTester/ViennaRNA-2.1.9/Progs/RNALfold"
RNAZ_BINARY = ROOT_FOLDER + "RNAz-2.1/rnaz/RNAz"
RNAZ_INPUT_FOLDER = ROOT_FOLDER + "RNAz-2.1/input/"
RNAZ_OUTPUT_FOLDER = ROOT_FOLDER + "RNAz-2.1/output/"
#PSSM_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/blast_output/"
#PSSM_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/blast_fungi_output/"
#PSSM_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/blast_sac_output/"
PSSM_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/blast_fungiSac_output/"
PSSM_ASA_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/output_asa/"
PSSM_RNA = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/stats_bla.py"
PSSM_ASA = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/pred_pssm.py"
PSSM_ASA_MODFILE = "/auto/brno2/home/bendl/halfLifeFeatureTester/pssm/o40_20.mod1"
PSIPRED_BINARY = "/auto/brno2/home/bendl/halfLifeFeatureTester/psipred/BLAST+/runpsipredplus"
PSIPRED_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/psipred/results"
DISORDER_BINARY = "/auto/brno2/home/bendl/halfLifeFeatureTester/disorder/Feature.sh"
DISORDER_OUTPUT_FOLDER = "/auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data"
MAX_SHUFFLING_ITERATION = 1

records = []
if(CALCULATED_FEATURE == "RNAHEAT"):
	for i in range(4):
		path = "/auto/brno2/home/wormsik/halfLifeFeatureTester/results/results_RNAHEAT.txt_HEAT_" + str(i)
		records.append({})
		for line in open(path):
			items = line.split(",")
			if(len(items) != 204):
				continue
			identifier = items[0].strip().replace("-","")
			records[i][identifier] = True
		print "len(records[" + str(i) +"]) = " + str(len(records[i]))

recordsPssm = {}
if(CALCULATED_FEATURE == "PSSM"):
	for name in os.listdir(PSSM_OUTPUT_FOLDER):
		if(name.find(".") < 0):
			continue
		recordsPssm[name.split(".")[0]] = True

recordsDisorder = {}
if(CALCULATED_FEATURE == "DISORDER"):   ### TODO ###
	for name in os.listdir(DISORDER_OUTPUT_FOLDER + "/general"):
		if(name.find(".") < 0):
			continue
		recordsDisorder[name.split(".")[0]] = True
		
recordsPssmAsa = {}
if(CALCULATED_FEATURE == "PSSM_ASA"):
	for name in os.listdir(PSSM_ASA_OUTPUT_FOLDER):
		if(name.find(".") < 0):
			continue
		recordsPssmAsa[name.split(".")[0]] = True
		
recordsRnaz = {}
if(CALCULATED_FEATURE == "RNAZ"):
	for name in os.listdir(RNAZ_OUTPUT_FOLDER):
		if(name.find(".") < 0):
			continue
		recordsRnaz[name.split(".")[0]] = True

class Experiment:
	jobID = int(0)   # identifier for identification of this experiment
	pathProcessedFile = ""
	pathComputedFile = ""
	pathTempFile = ""
	pathTempFile2 = ""
	pathOutTempFile = ""
	pathOut2TempFile = ""
	pathStatusFile = ""
	
	identifier = ""
	sequence = ""
	
	def __init__(self):
		self.jobID = random.randint(1, 1000000)
		
	def prepareFiles(self):
		"Prepare files for neccessary manipulation during evaluation of the individual mutation."
		
		if((os.path.exists(WORKING_FOLDER) == False) or (os.path.exists(DATASET_PATH) == False)):
			print "Working directory does not exist! Please edit the configuration file."
			exit(1)
		
		self.pathProcessedFile = str(WORKING_FOLDER + "/" + "processed_" + CALCULATED_FEATURE + "_SINGLE.txt")
		if(os.path.exists(self.pathProcessedFile) == False):
			newFile = open(self.pathProcessedFile, "w")
			newFile.close()
		self.pathTempFile = WORKING_FOLDER + "/" + str(self.jobID) + ".tmp"
		if(os.path.exists(self.pathTempFile) == False):
			newFile = open(self.pathTempFile, 'w')
			newFile.close()
		self.pathTempFile2 = WORKING_FOLDER + "/" + str(self.jobID) + ".tmp2"
		if(os.path.exists(self.pathTempFile2) == False):
			newFile = open(self.pathTempFile2, 'w')
			newFile.close()
		self.pathComputedFile = WORKING_FOLDER + "/computed_" + CALCULATED_FEATURE + ".csv"
		if(os.path.exists(self.pathComputedFile) == False):
			newFile = open(self.pathComputedFile, "w")
			newFile.close()
		self.pathOutTempFile = WORKING_FOLDER + "/" + str(self.jobID) + ".out"
		if(os.path.exists(self.pathOutTempFile) == False):
			newFile = open(self.pathOutTempFile, "w")
			newFile.close()
		self.pathOut2TempFile = WORKING_FOLDER + "/" + str(self.jobID) + ".out2"
		if(os.path.exists(self.pathOut2TempFile) == False):
			newFile = open(self.pathOut2TempFile, "w")
			newFile.close()
		self.pathStatusFile = WORKING_FOLDER + "/" + str(self.jobID) + "_status.txt"
		if(os.path.exists(self.pathStatusFile) == False):
			newFile = open(self.pathStatusFile, "w")
			newFile.close()
	
	def computeAll(self):
		"Maintains all computation."
		
		self.prepareFiles()
		
		# Get dataset into the list
		dataset = open(DATASET_PATH, 'r')		
		dataset_list = dataset.readlines()
		dataset.close()
		
		# Get processed items into the list
		processed = open(self.pathProcessedFile, 'r')
		processed_list = processed.readlines()
		processed.close()
		
		# MAIN LOOP --> go through the records of the dataset
		i = 0
		for line in dataset_list:
			print "\n\n***** NEW RECORD *****"
			if((len(line.strip()) == 0) or line.startswith("#")):
				continue
			
			items = line.split(",")
			self.identifier = items[0].strip()
			self.identifierShortened = self.identifier.replace("-","")
			print "> " + self.identifier + " / " + self.identifierShortened
			
			found = False
			for lineProc in processed_list:
				if((lineProc.strip() == self.identifier) or (lineProc.strip() == self.identifierShortened)):
					found = True
					break
			if found == True:
				continue
			
			print "> ... will be launched ... "
			
			# Append the current record to the file with processed records
			processedFile = open(self.pathProcessedFile, 'a')
			processedFile.write(self.identifier + "\n")
			processedFile.close()
			
			print "CALCULATED_FEATURE: " + CALCULATED_FEATURE
			
			if(CALCULATED_FEATURE == "DINUCLEOTIDE_SHUFFLING"):
				self.sequence = None
				self.sequence = copy.deepcopy([])
				
				for j in range(7):
					self.sequence.append(items[j+1])
				
				for j in range(8):
					print "> self.appendDinucleotideShuffling(" + str(j) + ")"
					self.appendDinucleotideShuffling(j)
			
			elif(CALCULATED_FEATURE == "PSSM"):
				print "***** RUN PSSM *****"
				self.sequence = None
				self.sequence = copy.deepcopy([])
				self.sequence.append(items[5])  # DNA
				self.sequence.append(items[6])  # TRANSCRIPT
				self.sequence.append(items[1])  # UPSTREAM
				self.sequence.append(items[2])  # DOWNSTREAM
				
				for j in range(4):
					print "> self.appendPssm(" + str(j) + ")"
					self.appendPssm(j)
			
			elif(CALCULATED_FEATURE == "PSSM_ASA"):
				print "***** RUN PSSM_ASA *****"
				self.sequence = None
				self.sequence = copy.deepcopy([])
				self.sequence.append(items[5])  # DNA
				self.sequence.append(items[6])  # TRANSCRIPT
				self.sequence.append(items[1])  # UPSTREAM
				self.sequence.append(items[2])  # DOWNSTREAM
				
				for j in range(4):
					print "> self.appendPssm2(" + str(j) + ")"
					self.appendPssm2(j)
					
			elif(CALCULATED_FEATURE == "RNAZ"):
				print "***** RUN RNAZ *****"
				self.sequence = None
				self.sequence = copy.deepcopy([])
				self.sequence.append(items[5])  # DNA
				self.sequence.append(items[6])  # TRANSCRIPT
				self.sequence.append(items[1])  # UPSTREAM
				self.sequence.append(items[2])  # DOWNSTREAM
				
				for j in range(4):
					print "> self.appendRNAz(" + str(j) + ")"
					self.appendRnaz(j)
			
			elif(CALCULATED_FEATURE == "RNAHEAT"):
				print "***** RUN RNA_HEAT *****"
				self.sequence = None
				self.sequence = copy.deepcopy([])
				self.sequence.append(items[5])  # DNA
				self.sequence.append(items[6])  # TRANSCRIPT
				self.sequence.append(items[1])  # UPSTREAM
				self.sequence.append(items[2])  # DOWNSTREAM
				
				for j in range(4):
					if((self.identifierShortened in records[j]) == False):
						print "> self.appendMeltingCurve(" + str(j) + ")"
						self.appendMeltingCurve(j)
			
			elif(CALCULATED_FEATURE == "SPINED"):
				print "***** RUN SPINED *****"
				
				resultFile = SPINED_PREDOUT + self.identifier
				if(os.path.isdir(resultFile)):
					print "> yet calculated: " + self.identifier
					continue
				
				command = SPINED_BINARY + " " + SPINED_JOBS + " " + str(self.identifier)
				print "Command: " + command
				
				os.system(command)
				
			elif(CALCULATED_FEATURE == "PSIPRED"):
				print "***** RUN PSIPRED *****"
				
				flagFile = PSIPRED_OUTPUT_FOLDER + "/" + self.identifier + ".ss2"
				if(os.path.isfile(flagFile)):
					print "> yet calculated: " + self.identifier
					continue
				
				tempFileIn = open(self.pathTempFile, "w")
				tempFileIn.write(">" + self.identifier + "\n" + items[8].strip() + "\n")
				tempFileIn.close()
		
				command = PSIPRED_BINARY + " " + self.pathTempFile + " " + self.identifier
				print "Command 1: " + command
				os.system(command)
				
				
			elif(CALCULATED_FEATURE == "DISORDER"):
				print "***** RUN DISORDER *****"
				
				self.sequence = None
				self.sequence = copy.deepcopy([])
				self.sequence.append(items[5])  # DNA
				self.sequence.append(items[6])  # TRANSCRIPT
				self.sequence.append(items[1])  # UPSTREAM
				self.sequence.append(items[2])  # DOWNSTREAM
				
				for j in range(4):
					if((self.identifierShortened in recordsDisorder) == False):
						print "> self.appendRnaDisorder(" + str(j) + ")"
						self.appendRnaDisorder(j)
				
			# Get processed items into the list
			processed = open(self.pathProcessedFile, 'r')
			processed_list = processed.readlines()
			processed.close()
	
	
	def appendDinucleotideShuffling(self, columnIndex):
		shuffledSequences = []
		shuffledSequencesEnergies1 = []     # RNALfold
		shuffledSequencesEnergies2 = []     # RNAFOLD_ENERGY
		shuffledSequencesEnergies3 = []     # RNAFOLD_ENSEMBLE_ENERGY
		shuffledSequencesEnergies4 = []     # RNAFOLD_CENTROID_ENERGY
		shuffledSequencesEnergies5 = []     # RNAFOLD_MEA_ENERGY
		shuffledSequencesEnergies6 = []     # RNAFOLD_MEA
		shuffledSequencesEnergies7 = []     # RNAFOLD_ENSEMBLE_DIVERSITY
		shuffledSequencesEnergies8 = []     # PREDICTED_STRUCTURE
		pValues = []
		
		initialSeq = ""
		if((columnIndex == 7)):
			if((len(self.sequence[0]) > 0) or (len(self.sequence[1]) > 0)):
				initialSeq = self.sequence[0] + self.sequence[4] + self.sequence[1]
		else:
			initialSeq = self.sequence[columnIndex].strip()
					
		if(len(initialSeq) == 0):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n\n\n"
			return
		
		# RNALfold applied on unchanged sequence
		tempFileIn = open(self.pathTempFile, "w")
		tempFileIn.write(initialSeq + "\n")
		tempFileIn.close()
		
		command = RNALFOLD_BINARY + "  < " + self.pathTempFile + " > " + self.pathTempFile2
		print "********** SEQUENCE " + str(columnIndex) + " **********"
		print "> Sequence: " + str(columnIndex) + " / " + initialSeq
		print "> Command " + str(self.identifier) + ": " + command
		os.system(command)
		resultFile = open(self.pathTempFile2,"r")
		allLines = resultFile.readlines()
		print "> Output RNALfold (all lines): " + str(allLines)
		stopValue1 = float(allLines[len(allLines)-1].strip()[1:-1])
		resultFile.close()
		
		command = RNAFOLD_BINARY + " --MEA -p < " + self.pathTempFile + " > " + self.pathTempFile2
		print "> Command " + str(self.identifier) + ": " + command
		os.system(command)
		resultFile = open(self.pathTempFile2, "r")
		allLines = resultFile.readlines()
		print "> Output RNAfold (all lines): " + str(allLines)
		predictedStructure = allLines[1].split(" ")[0].strip()
		
		print allLines[1][allLines[1].find(" ")+1:].strip()[1:-1].strip()
		stopValue2 = float(allLines[1][allLines[1].find(" ")+1:].strip()[1:-1].strip())
		stopValue3 = float(allLines[2][allLines[2].find(" ")+1:].strip()[1:-1].strip())
		stopValue4 = float(allLines[3][allLines[3].find(" ")+2:].strip().split(" ")[0])
		stopValue5 = float(allLines[4][allLines[4].find(" ")+2:].strip().split(" ")[0])
		stopValue6 = float(allLines[4][allLines[4].find(" ")+2:].strip().split(" ")[1].split("=")[1].strip()[:-1])
		stopValue7 = float(allLines[5].strip().split(" ")[len(allLines[5].strip().split(" "))-1])
		predictedStructureCut = copy.deepcopy(predictedStructure)
		if(columnIndex == 7):
			predictedStructureCut = predictedStructureCut[:len(self.sequence[0])] + predictedStructureCut[-len(self.sequence[1]):]
		stopValue8 = self.calculateUnboundedRatio(predictedStructureCut)		
		resultFile.close()
		
		# RNALfold & RNAfold applied on shuffled sequence
		#for i in range(MAX_SHUFFLING_ITERATION):
		#	if(columnIndex == 4):
		#		generatedSequence = "ATG" + self.tempShuffleDinucleotide(initialSeq[3:-3]) + initialSeq[-3:]
		#	if(columnIndex == 7):
		#		generatedSequence5end = self.tempShuffleDinucleotide(self.sequence[0])
		#		generatedSequence3end = self.tempShuffleDinucleotide(self.sequence[1])
		#		generatedSequence = generatedSequence5end + self.sequence[4] + generatedSequence3end
		#		
		#		print "> generatedSequence5end(" + generatedSequence5end + ") = " + generatedSequence5end
		#		print "> generatedSequence3end(" + generatedSequence3end + ") = " + generatedSequence3end
		#		print "> generatedSequence(" + initialSeq + ") = " + generatedSequence
		#		print "SHUFFLING 7: " + generatedSequence
		#	else:
		#		generatedSequence = self.tempShuffleDinucleotide(initialSeq)
		#		print "> INITIAL SEQUENCE:   " + initialSeq
		#		print "> GENERATED SEQUENCE: " + generatedSequence
		#		shuffledSequences.append(generatedSequence)
		#	
		#	tempFileIn = open(self.pathTempFile, "w")
		#	tempFileIn.write(generatedSequence + "\n")
		#	tempFileIn.close()
		#	
		#	# RNALfold
		#	command = RNALFOLD_BINARY + "  < " + self.pathTempFile + " > " + self.pathTempFile2
		#	print "> Command: " + str(self.identifier) + ": " + command
		#	os.system(command)
		#	resultFile = open(self.pathTempFile2,"r")
		#	allLines = resultFile.readlines()
		#	shuffledSequencesEnergies1.append(float(allLines[len(allLines)-1].strip()[1:-1]))
		#	resultFile.close()
		#	
		#	# RNAfold
		#	command = RNAFOLD_BINARY + " --MEA -p < " + self.pathTempFile + " > " + self.pathTempFile2
		#	print "> Command " + str(self.identifier) + ": " + command
		#	os.system(command)
		#	resultFile = open(self.pathTempFile2, "r")
		#	allLines = resultFile.readlines()
		#	#print "$ ALL_LINES: " + str(allLines)
		#	shuffledSequencesEnergies2.append(float(allLines[1][allLines[1].find(" ")+1:].strip()[1:-1].strip()))
		#	shuffledSequencesEnergies3.append(float(allLines[2][allLines[2].find(" ")+1:].strip()[1:-1].strip()))
		#	shuffledSequencesEnergies4.append(float(allLines[3][allLines[3].find(" ")+2:].strip().split(" ")[0]))
		#	shuffledSequencesEnergies5.append(float(allLines[4][allLines[4].find(" ")+2:].strip().split(" ")[0]))
		#	shuffledSequencesEnergies6.append(float(allLines[4][allLines[4].find(" ")+2:].strip().split(" ")[1].split("=")[1].strip()[:-1]))
		#	shuffledSequencesEnergies7.append(float(allLines[5].strip().split(" ")[len(allLines[5].strip().split(" "))-1]))
		#	predictedStructureCut = copy.deepcopy(allLines[1].split(" ")[0].strip())
		#	print "> predictedStructureCut: " + predictedStructureCut
		#	if(columnIndex == 7):
		#		predictedStructureCut = predictedStructureCut[:len(self.sequence[0])] + predictedStructureCut[-len(self.sequence[1]):]
		#	shuffledSequencesEnergies8.append(self.calculateUnboundedRatio(predictedStructureCut))
		#	print "> calculateUnboundedRatio: " + str(self.calculateUnboundedRatio(predictedStructureCut))
		#	resultFile.close()
 		#
		#pValue1 = self.getOrder(shuffledSequencesEnergies1, stopValue1)
		#pValue2 = self.getOrder(shuffledSequencesEnergies2, stopValue2)
		#pValue3 = self.getOrder(shuffledSequencesEnergies3, stopValue3)
		#pValue4 = self.getOrder(shuffledSequencesEnergies4, stopValue4)
		#pValue5 = self.getOrder(shuffledSequencesEnergies5, stopValue5)
		#pValue6 = self.getOrder(shuffledSequencesEnergies6, stopValue6)
		#pValue7 = self.getOrder(shuffledSequencesEnergies7, stopValue7)
		#pValue8 = self.getOrder(shuffledSequencesEnergies8, stopValue8)
		#
		#if(os.path.isfile(OUTPUT_FILE + "_" + str(columnIndex)) == False):
		#	resultFile = open(OUTPUT_FILE + "_" + str(columnIndex), "w")
		#	resultFile.write("IDENTIFIER,RNALFOLD_ENERGY,RNAFOLD_ENERGY,RNAFOLD_ENSEMBLE_ENERGY,RNAFOLD_CENTROID_ENERGY,RNAFOLD_MEA_ENERGY,RNAFOLD_MEA,RNAFOLD_ENSEMBLE_DIVERSITY\n")
		#	resultFile.close()
		#	
		#if(os.path.isfile(OUTPUT_FILE_2 + "_" + str(columnIndex)) == False):
		#	resultFile2 = open(OUTPUT_FILE_2 + "_" + str(columnIndex), "w")
		#	resultFile2.write("IDENTIFIER,RNALFOLD_ENERGY,RNAFOLD_ENERGY,RNAFOLD_ENSEMBLE_ENERGY,RNAFOLD_CENTROID_ENERGY,RNAFOLD_MEA_ENERGY,RNAFOLD_MEA,RNAFOLD_ENSEMBLE_DIVERSITY,PREDICTED_STRUCTURE\n")
		#	resultFile2.close()
		#
		#resultFile = open(OUTPUT_FILE + "_" + str(columnIndex), "a")
		#resultFile.write(self.identifier + "," + str(pValue1) + "," + str(pValue2) + "," + str(pValue3) + "," + str(pValue4) + "," + str(pValue5) + "," + str(pValue6) + "," + str(pValue7) + "\n")
		#resultFile.close()
		resultFile2 = open(OUTPUT_FILE + "_" + str(columnIndex), "a")
		resultFile2.write(self.identifier + "," + str(stopValue1) + "," + str(stopValue2) + "," + str(stopValue3) + "," + str(stopValue4) + "," + str(stopValue5) + "," + str(stopValue6) + "," + str(stopValue7) + "," + str(predictedStructure) + "\n")
		resultFile2.close()
	
	
	def appendPssm(self, columnIndex):
		fileIndex = []
		fileIndex.append(5)
		fileIndex.append(6)
		fileIndex.append(1)
		fileIndex.append(2)
		
		initialSeq = self.sequence[columnIndex].strip()
		print ">>> ::PSSM_1:: " + self.identifier + " <<<"
		isCalculatedYet = self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print "> isCalculatedYet ? " + isCalculatedYet + " => " + str((isCalculatedYet in recordsPssm)) 
		#sys.stdin.read(1)
		if((len(initialSeq) == 0) or (isCalculatedYet in recordsPssm)):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n\n\n"
			return
		
		tempFileIn = open(self.pathTempFile, "w")
		tempFileIn.write(">" + self.identifierShortened.strip() + "\n" + initialSeq + "\n")
		tempFileIn.close()
		
		blastFile = PSSM_OUTPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + ".blast"
		#command = BLASTN_BINARY + " -db " + BLAST_DB + " -query " + self.pathTempFile + " -out " + blastFile + " -num_descriptions 1 -num_threads 8 -num_alignments 50000"
		command = BLASTN_BINARY + " -evalue 10  -db " + BLAST_DB + " -query " + self.pathTempFile + " -out " + blastFile + " -gapextend 1 -gapopen 3 -word_size 10 -num_descriptions 1 -num_threads 4 -num_alignments 2500"
		
		print "> Command1 " + str(self.identifier) + ": " + command
		os.system(command)
		
		#outputPssmPath = PSSM_OUTPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + ".pssm"
		#command2 = PSSM_RNA + " " + self.pathTempFile + " " + blastFile + " > " + outputPssmPath
		#print "> Command2 " + str(self.identifier) + ": " + command2
		#os.system(command2)
	
	def appendPssm2(self, columnIndex):
		fileIndex = []
		fileIndex.append(5)
		fileIndex.append(6)
		fileIndex.append(1)
		fileIndex.append(2)
		
		initialSeq = self.sequence[columnIndex].strip()
		isCalculatedYet = self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print ">>> ::PSSM_2:: " + self.identifier + " <<<"
		print "> isCalculatedYet ? " + isCalculatedYet + " => " + str((isCalculatedYet in recordsPssmAsa)) 
		if((len(initialSeq) == 0) or (isCalculatedYet in recordsPssmAsa)):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n>>> LENGTH: " + str(len(initialSeq)) + "\n\n\n"
			return
		
		pssmFilePath = PSSM_OUTPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) +  ".pssm"
		pssmAsaFile = PSSM_ASA_OUTPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) +  ".txt"
		#~/work/RNA/train/misc/pred_pssm.py -w 40 --w2 20 --mod ~/work/RNA/train/pred_cpx/o40_20.mod1 $m
		command = PSSM_ASA + " -w 40 --w2 20 --mod " + PSSM_ASA_MODFILE + " " + pssmFilePath + " > " + pssmAsaFile
		print "> Command " + str(self.identifierShortened) + ": " + command
		os.system(command)
	
	def appendRnaz(self, columnIndex):
		fileIndex = []
		fileIndex.append(5)
		fileIndex.append(6)
		fileIndex.append(1)
		fileIndex.append(2)
		
		initialSeq = self.sequence[columnIndex].strip()
		isCalculatedYet = self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print ">>> ::RNAz:: " + self.identifier + " <<<"
		print "> isCalculatedYet ? " + isCalculatedYet + " => " + str((isCalculatedYet in recordsRnaz)) 
		if((len(initialSeq) == 0) or (isCalculatedYet in recordsPssmAsa)):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n>>> LENGTH: " + str(len(initialSeq)) + "\n\n\n"
			return
		
		rnazInputPath = RNAZ_INPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) +  ".aln"
		rnazOutputPath = RNAZ_OUTPUT_FOLDER + self.identifierShortened + "_" + str(fileIndex[columnIndex]) +  ".out"

		command = RNAZ_BINARY + " " + rnazInputPath + " > " + rnazOutputPath
		print "> Command " + str(self.identifierShortened) + ": " + command
		os.system(command)
	
	
	def appendRnaDisorder(self, columnIndex):
		fileIndex = []
		fileIndex.append(5)
		fileIndex.append(6)
		fileIndex.append(1)
		fileIndex.append(2)
		
		initialSeq = self.sequence[columnIndex].strip()
		isCalculatedYet = self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print ">>> ::RNA_DISORDER:: " + self.identifier + " <<<"
		print "> isCalculatedYet ? " + isCalculatedYet + " => " + str((isCalculatedYet in recordsDisorder)) 
		if((len(initialSeq) == 0) or (isCalculatedYet in recordsDisorder)):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n>>> LENGTH: " + str(len(initialSeq)) + "\n\n\n"
			return
		
		command1 = "mkdir " + DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print "> Command1: " + command1
		os.system(command1)
		
		tempFileInPath = DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex])
		tempFileIn = open(tempFileInPath, "w")
		tempFileIn.write(">" + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "\n" + initialSeq + "\n")
		tempFileIn.close()
		print "> Vytvoreny soubor: " + tempFileInPath
		
		tempFileInPath = DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "/list"
		tempFileIn = open(tempFileInPath, "w")
		tempFileIn.write(self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "\n")
		tempFileIn.close()
		
		command2 = DISORDER_BINARY + " " + DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex])
		print "> Command2: " + command2
		os.system(command2)
		
		command3 = "cp " + DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "/*.s " + DISORDER_OUTPUT_FOLDER + "/general/"
		print "> Command3: " + command3
		os.system(command3)
		
		command3 = "mv " + DISORDER_OUTPUT_FOLDER + "/" + self.identifierShortened + "_" + str(fileIndex[columnIndex]) + "/*.s " + ROOT_FOLDER + "/general2/"
		print "> Command3: " + command3
		os.system(command3)
		
		#sys.stdin.read(1)
	
	
	def appendMeltingCurve(self, columnIndex):
		initialSeq = self.sequence[columnIndex].strip()
		if(len(initialSeq) == 0):
			print "\n\n\n**********************************\n********** NULOVA DELKA **********\n**********************************\n\n\n"
			return
				
		# RNALfold applied on unchanged sequence
		tempFileIn = open(self.pathTempFile, "w")
		tempFileIn.write(initialSeq + "\n")
		tempFileIn.close()
		
		command = RNAHEAT_BINARY + " < " + self.pathTempFile + " > " + self.pathTempFile2
		print "> Command " + str(self.identifier) + ": " + command
		os.system(command)
		resultFile = open(self.pathTempFile2, "r")
		allLines = resultFile.readlines()
		outline = self.identifier + ","
		for line in allLines:
			print "\n> line: " + line.strip()
			try:
				temp = int(line[0:3])
				heat = float(line[4:])
				print str(temp) + "\t" + str(heat)
				outline += str(temp) + "," + str(heat) + ","
			except:
				print "ERROR"
				#sys.stdin.read(1)
		
		
		resultFile2 = open(OUTPUT_FILE + "_HEAT_" + str(columnIndex), "a")
		resultFile2.write(outline + "\n")
		resultFile2.close()
		#print "> Output RNAfold (all lines): " + str(allLines)
		#sys.stdin.read(1)
		
		
	def calculateUnboundedRatio(self, predictedStructure):
		### Calculate ratio of unbounded positions in RNA secondary structure of analyzed region ###
		unboundedPos = 0
		unboundedPosRatio = 0
		for position in predictedStructure:
			if(position == "."):
				unboundedPos += 1
		if(unboundedPos > 0):
			unboundedPosRatio = float(unboundedPos) / len(predictedStructure)
		print "=> " + str(unboundedPos) + " / " + str(len(predictedStructure)) + " = " + str(unboundedPosRatio)
		
		return unboundedPos
	
	
	def tempShuffleDinucleotide(self, sequence):
		nucleotides = ("A", "T", "G", "C")
		swapCount = 0
		expCount = 0
		maxCount = len(sequence) * 10
		#print "maxCount: " + str(maxCount)
		#print "before: " + sequence
		swaps = 0
		i = 0
		while((swapCount < maxCount) and (expCount < (3 * maxCount))):
			firstNucleotide = nucleotides[random.randint(0, 3)]
			thirdNucleotide = nucleotides[random.randint(0, 3)]
			
			positions = []
			for i in range(len(sequence) - 2):
				if((sequence[i] == firstNucleotide) and (sequence[i+2] == thirdNucleotide)):
					positions.append(i+1)
					
			for i in range(len(positions)):
				swap = random.randint(0, len(positions)-1)
				if((abs(positions[swap] - positions[i]) >= 3) and (sequence[positions[swap]] != sequence[positions[i]])):
					tmpChar = sequence[positions[swap]]
					sequenceOriginal = copy.deepcopy(sequence)
					sequence = sequenceOriginal[0:positions[swap]] + sequence[positions[i]] + sequenceOriginal[positions[swap]+1:]
					sequenceOriginal = copy.deepcopy(sequence)
					sequence = sequenceOriginal[0:positions[i]] + tmpChar + sequenceOriginal[positions[i]+1:]
					swapCount +=1
			expCount += 1
		
		return sequence
	
	
	def getOrder(self, searchedList, stopValue):
		lowerCount = 1
		suma = 1
		
		for energy in searchedList:
			if(float(energy) > float(stopValue)):
				lowerCount += 1
				
			if(float(energy) != float(stopValue)):
				suma += 1
		
		pValue = float(lowerCount) / suma
		return pValue
	
	
def main():
	exp = Experiment()
	exp.computeAll()


if __name__ == "__main__":
    main()
