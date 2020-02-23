###################################################################################################################
################################### HALF-LIFE PREDICTOR (STEP 2) ##################################################
##### Prerequisities: #############################################################################################
##### 1) Input dataset is complete (include nucleotide, amino acid and transcript sequences #######################
##### 2) RNAfold is downloaded and installed properly #############################################################
###################################################################################################################

import os
import sys
import urllib2
import copy
import operator
import subprocess
import traceback
import math
import random
import numpy
import pandas
import re
import scipy
import scipy.stats
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation, PredictionOutput, KernelClassifier, Kernel
from weka.core.classes import Random
from weka.core.converters import Loader, Saver
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel
import rpy2.robjects.numpy2ri
import rpy2.robjects as robj
import rpy2.robjects.pandas2ri    #for dataframe conversion
from rpy2 import robjects
from rpy2.robjects import Formula
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2 import robjects
from rpy2.robjects.packages import importr, data
import array
import random
from pandas import DataFrame

# krome nasledujici sekce odramovat
#from rpy2.robjects import Formula, Environment
#from rpy2.robjects.vectors import IntVector, FloatVector
#from rpy2.robjects.lib import grid
#from rpy2.rinterface import RRuntimeError
#import rpy2.robjects as robjects

rprint = robjects.globalenv.get("print")
graphics = importr('graphics')
grdevices = importr('grDevices')
base = importr('base')

LOG_FLAG = True
DEBUG_FLAG = True
DEBUG_IMG_FLAG = True
STATISTICS = []

class Enum(set):
	def __getattr__(self, name):
		if name in self:
			return name
		raise AttributeError


RECORD_STATUS = Enum(["TRAINING", "TESTING", "UNKNOWN"])
RECORD_STATUS_CATEGORY = Enum(["STABLE", "UNSTABLE", "TWILIGHT"])
SEQUENCE_TYPE = Enum(["DNA_SEQUENCE","UPSTREAM","DOWNSTREAM","TRANSCRIPT","PROTEIN_SEQUENCE"])
SEQUENCE_NUC_TYPE = Enum(["DNA_SEQUENCE","UPSTREAM","DOWNSTREAM","TRANSCRIPT"])
SEQUENCE_AMINO_TYPE = Enum(["PROTEIN_SEQUENCE"])


class ClassifierDefinition:
	classifierName = ""
	classifierDesc = ""
	classifierClass = ""
	classifierOptions = ""
	classifierModelFile = ""
	classifierInfoFile = ""
	comment = ""


class Statistics:
	key = ""                # Unique identifier (region + feature + detail)
	region = ""             # Region (UPSTREAM, DOWNSTREAM, ...)
	feature = ""            # Name of analyzed feature
	detail = ""             # Version of feature (for properties with many options)
	spearmanNumericHL = ""  # Spearman's correlation between half-life and analyzed feature
	spearmanBins = ""       # Spearman's correlation for binned distribution of half-lifes and analyzed feature (number of bins: CONFIG.BINS_COUNT
	pearsonNumericHL = ""   # Pearson's correlation between half-life and analyzed feature
	tTest = ""              # The calculated t-statistic
	tTestProb = ""          # The two-tailed p-value
	
	def __init__(self, region_arg, feature_arg, detail_arg):
		self.region = region_arg
		self.feature = feature_arg
		self.detail = detail_arg
		self.key = region_arg + feature_arg + detail_arg

	def calculateStatistics(self, recordsVector, propertyKey):
		### Calculate the relationship between half-life and analyze property ###
		
		# Pearson & Spearman correlation for numerical values
		vector1 = [float(item.var[CONFIG.TARGET_HL]) for item in recordsVector]
		vector2 = [float(item.var[propertyKey]) for item in recordsVector]
		self.spearmanNumericHL = (round(scipy.stats.spearmanr(vector1, vector2)[0], 2))
		self.pearsonNumericHL = (round(scipy.stats.pearsonr(vector1, vector2)[0], 2))
		
		if(DEBUG_FLAG): print "> Key: " + self.region + "\t" + self.feature + "\t" + self.detail
		if(DEBUG_FLAG): print "> Regression HL || spearman: " + str(self.spearmanNumericHL) + "\tpearson:" + str(self.pearsonNumericHL)
		
		# Spearman for "binned" values
		vector1 = []
		vector2 = []
		
		for i in range(CONFIG.BINS_COUNT):
			print "> BIN " + str(i) + " ... " + str(len([float(item.var[CONFIG.TARGET_HL]) for item in recordsVector if (int(item.var["STABILITY_BIN"]) == i)]))
			hlMean = numpy.mean([float(item.var[CONFIG.TARGET_HL]) for item in recordsVector if (int(item.var["STABILITY_BIN"]) == i)])
			vector1.append(hlMean)
			propertyMean = numpy.mean([float(item.var[propertyKey]) for item in recordsVector if (int(item.var["STABILITY_BIN"]) == i)])
			vector2.append(propertyMean)
		
		self.spearmanBins = (round(scipy.stats.spearmanr(vector1, vector2)[0], 2))
		if(DEBUG_FLAG): print "> Categorica HL || spearman: " + str(self.spearmanBins)
		
		# TODO: for tertiary-class hierarchy
		vectorStable = [float(item.var[propertyKey]) for item in recordsVector if ((item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.STABLE) and HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
		vectorTwilight = [float(item.var[propertyKey]) for item in recordsVector if ((item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.TWILIGHT) and HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
		vectorUnstable = [float(item.var[propertyKey]) for item in recordsVector if ((item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.UNSTABLE) and HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
		tTestTmp = scipy.stats.ttest_ind(vectorStable,vectorUnstable)
		self.tTest = float(tTestTmp[0])
		self.tTestProb = float(tTestTmp[1])
		if(DEBUG_FLAG): print "> mean(vectorStable): " + str(numpy.mean(vectorStable))
		if(DEBUG_FLAG): print "> len(vectorUnstable): " + str(len(vectorUnstable))
		if(DEBUG_FLAG): print "> mean(vectorUnstable): " + str(numpy.mean(vectorUnstable))
		if(DEBUG_FLAG): print "> two-sample t-test: " + str(self.tTest) + " / " + str(self.tTestProb)
		#print "vectorStable: " + str(vectorUnstable)
		#sys.stdin.read(1)
		
		if(DEBUG_FLAG):
			rpy2.robjects.numpy2ri.activate()
			robj.pandas2ri.activate()
			
			# ***** Scatter plot for bins
			testData = pandas.DataFrame( {'hlNumeric' : vector1, 'property' : vector2} )
			sourceData = DataFrame(testData)
			sourceDataR = robj.conversion.py2ri(sourceData)
			
			filename = "./ggplots/region_" + self.region.lower() + "~" + self.feature.lower() + "~" + self.detail.lower() + "~1.png"
			if(DEBUG_IMG_FLAG): grdevices = importr('grDevices')
			if(DEBUG_IMG_FLAG): grdevices.png(file=filename)
			
			title = "scatter plot for bins, r = " + str(round(self.spearmanBins, 2)) + " (" + self.region.lower() + ")"
			gp = ggplot2.ggplot(sourceDataR)
			pp = gp + ggplot2.aes_string(x='hlNumeric', y='property') + ggplot2.stat_smooth(colour='blue', span=0.2) + ggplot2.geom_point() + ggplot2.stat_smooth(colour='blue', span=0.2) + ggplot2.labs(title = title, x = "log2(half-life)", y = propertyKey.lower())
			pp.plot()
			
			if(DEBUG_IMG_FLAG==True):  grdevices.dev_off()
			if(DEBUG_IMG_FLAG==False): sys.stdin.read(1) 
			
			# ***** Scatter plot (all data)			
			vectorStability = [item.var["STABILITY_TERTIARY"] for item in recordsVector  if(HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
			vectorProperty = [float(item.var[propertyKey]) for item in recordsVector  if(HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
			vectoHlNumeric = [float(item.var[CONFIG.TARGET_HL]) for item in recordsVector  if(HalfLifePredictor.isParameterDefined(item, CONFIG.TARGET_HL) and HalfLifePredictor.isParameterDefined(item, propertyKey))]
			testData = pandas.DataFrame( {'stability' : vectorStability, 'property' : vectorProperty, 'hlNumeric' : vectoHlNumeric} )
			
			filename = "./ggplots/region_" + self.region.lower() + "~" + self.feature.lower() + "~" + self.detail.lower() + "~2.png"
			if(DEBUG_IMG_FLAG): grdevices = importr('grDevices')
			if(DEBUG_IMG_FLAG): grdevices.png(file=filename)
			
			sourceData = DataFrame(testData)
			sourceDataR = robj.conversion.py2ri(sourceData)
			
			title = "Scatter plot, r = " + str(round(self.spearmanNumericHL, 2)) + " (" + self.region.lower() + ")"
			gp = ggplot2.ggplot(sourceDataR)
			pp = gp + ggplot2.aes_string(x='hlNumeric', y='property', color='stability') + ggplot2.stat_smooth(colour='blue', span=0.2) + ggplot2.geom_point() + ggplot2.stat_smooth(colour='blue', span=0.2) + ggplot2.labs(title = title, x = "log2(half-life)", y = propertyKey.lower(),legend="stability class") + ggplot2.scale_colour_manual(name = "stability class", values = numpy.array(['green','grey','red']))
			pp.plot()
			
			if(DEBUG_IMG_FLAG==True):  grdevices.dev_off()
			if(DEBUG_IMG_FLAG==False): sys.stdin.read(1) 
			
			# ***** Densitity plot (distribution for stable & unstable)
			filename = "./ggplots/region_" + self.region.lower() + "~" + self.feature.lower() + "~" + self.detail.lower() + "~3.png"
			if(DEBUG_IMG_FLAG): grdevices = importr('grDevices')
			if(DEBUG_IMG_FLAG): grdevices.png(file=filename)
			
			title = "Density plot, t-test = " + str(round(self.tTest, 2)) + " / " + str(round(self.tTestProb, 2)) + " (" + self.region.lower() + ")"
			gp = ggplot2.ggplot(sourceDataR)
			pp = gp + ggplot2.aes_string(x='property',fill='factor(stability)') + ggplot2.geom_density(alpha = 0.5) + ggplot2.labs(title = title, x = propertyKey.lower(), y = "density") + ggplot2.scale_fill_manual(name = "stability class", values = numpy.array(['green','grey','red']))
			pp.plot()
			
			if(DEBUG_IMG_FLAG==True):  grdevices.dev_off()
			if(DEBUG_IMG_FLAG==False): sys.stdin.read(1) 
			
			# ***** Boxplot (stable / twilight / unstable)
			filename = "./ggplots/region_" + self.region.lower() + "~" + self.feature.lower() + "~" + self.detail.lower() + "~4.png"
			if(DEBUG_IMG_FLAG): grdevices = importr('grDevices')
			if(DEBUG_IMG_FLAG): grdevices.png(file=filename)
			
			title = "Boxplot plot, t-test = " + str(round(self.tTest,2)) + " / " + str(round(self.tTestProb,2)) + " (" + self.region.lower() + ")"
			gp = ggplot2.ggplot(sourceDataR)
			pp = gp + ggplot2.aes_string(x='factor(stability)', y='property',fill='factor(stability)') + ggplot2.geom_boxplot() + ggplot2.labs(title = title) + ggplot2.labs(title = title, x = "stability categories", y = propertyKey.lower()) + ggplot2.scale_fill_manual(name = "stability class", values = numpy.array(['green','grey','red']))
			pp.plot()
			
			if(DEBUG_IMG_FLAG==True):  grdevices.dev_off()
			if(DEBUG_IMG_FLAG==False): sys.stdin.read(1)
	
	
	def toString(self):
		outline = self.region + "," + self.feature + "," + self.detail + "," + str(self.spearmanNumericHL) + "," + str(self.pearsonNumericHL) + "," + str(self.spearmanBins) + "," + str(self.tTest) + "," + str(self.tTestProb) + "\n"
		return outline
	
	@staticmethod
	def getHeadline():
		outline = "REGION,FEATURE,DETAIL,SPEARMAN-NUMERIC,PEARSON-NUMERIC,SPEARMAN-BIN,TTEST,TTEST-PROB\n"
		return outline


class Config:
	### Configuration of application ###
	
	MAX_SHUFFLING_ITERATION = 1000                # Maximal iteration of dinucleotide shuffling
	DATASET_INPUT = "./dataset_shalem_out.csv"    # Path to dataset with all annotations
	DATASET_OUTPUT = "./dataset_shalem_OUT.csv"   # Path to dataset with all annotations
	CORRELATION_OUTPUT  = "./correlations_out.csv" # Path to file with output calculated correlations
	CONFIG_FILE = "./config.text"                 # Global configuration file
	CLASSIFIERS = {}                              # Loaded definition of classifier
	ROOT_FOLDER = "/home/jarda/Desktop/halfLifePredictor/"
	RNA_PSSM_BLAST_FOLDER = ROOT_FOLDER + "results/blast_fungiSac/"
	#RNA_PSSM_RESULT_FOLDER = ROOT_FOLDER + "results/pssm_fungiOnly/"
	RNA_PSSM_RESULT_FOLDER = ROOT_FOLDER + "results/pssm_phastcons/"
	RNA_DISORDER_RESULT_FOLDER = ROOT_FOLDER + "results/rna-disorder/"
	RNA_ASA_RESULT_FOLDER = ROOT_FOLDER + "results/asa/"
	PSSM_RNA = ROOT_FOLDER + "pssm/stats_bla.py"
	SPINED_RESULT_FOLDER = ROOT_FOLDER + "results/spined/"
	PSIPRED_RESULT_FOLDER = ROOT_FOLDER + "results/psipred/"
	RNA_RNAZ_RESULT_FOLDER = ROOT_FOLDER + "results/rnaz/"
	RNA_RNAZ_INPUT_FOLDER = ROOT_FOLDER + "rnaz2.1/input/"
	PHASTCONS_ALI_FOLDER = "/home/jarda/Desktop/halfLifePredictor/results/phastcons/multiz7way/maf/"
	TMP_FOLDER = ROOT_FOLDER + "tmp/"
	CDHIT_FOLDER = ROOT_FOLDER + "cdhit/"
	CDHIT_BINARY = ROOT_FOLDER + "cdhit/cd-hit"
	MEME_BINARY = ROOT_FOLDER + "meme/meme_4.10.1/bin/meme"
	FIMO_BINARY = ROOT_FOLDER + "meme/meme_4.10.1/bin/fimo"
	MEME_PSPGEN = ROOT_FOLDER + "meme/meme_4.10.1/bin/psp-gen"
	DREME_BINARY = ROOT_FOLDER + "meme/meme_4.10.1/bin/dreme"
	MEME_FOLDER = ROOT_FOLDER + "meme/"
	RNAZ_BINARY = ROOT_FOLDER + "rnaz2.1/rnaz/RNAz"
	RNAFOLD_BINARY = ROOT_FOLDER + "ViennaRNA-2.1.9/Progs/RNAfold"    # Path to binaries of RNAfold tool for calculation of minimum energy of RNA structure
	RNALFOLD_BINARY = ROOT_FOLDER + "ViennaRNA-2.1.9/Progs/RNALfold"  # Path to binaries of RNALfold tool for calculation of locally stable ensemble of RNA structures with minimum energy
	CODONW_BINARY = ROOT_FOLDER + "codonW/codonw"    # Path to binaries of CodonW tool for analysis of codon properties / frequencies
	CODONW_OUTPUT = ROOT_FOLDER + "codonW/dataset_presnyak_total.out"    # Path to output of CodonW analysis
	PHASTCONS_RESULTS = ROOT_FOLDER + "results/phastcons/phastCons7way/conservation.bed"
	DAMBE_ISOELECTRIC_POINT = ROOT_FOLDER + "results/dambe_isoelectric_point/dambe_isoelectric_point.txt"  # Path to the output of DAMBE tool (analysis: isoelectric point)
	DAMBE_CODONS = ROOT_FOLDER + "results/dambe_effective_codons/dambe_effective_codons.txt"               # Path to the output of DAMBE tool (analysis: effective number of codons)
	TARGET_HL = "PRESNYAK_HL_TOTAL"
	TARGET_SEQUENCE = SEQUENCE_TYPE.DNA_SEQUENCE
	RNA_FILE_COLUMNS = {"RNALFOLD_ENERGY" : "1", "RNAFOLD_ENERGY" : "2", "RNAFOLD_ENSEMBLE_ENERGY" : "3", "RNAFOLD_CENTROID_ENERGY" : "4", "RNAFOLD_MEA_ENERGY" : "5",  "RNAFOLD_MEA" : "6", "RNAFOLD_ENSEMBLE_DIVERSITY" : "7", "PREDICTED_STRUCTURE" : "8"}
	PHASTCONS_CHROMOSOMES = {}
	PHASTCONS_ALI_CHROMOSOMES = {}
	
	BINS_COUNT = 10
	GLOBAL_RSQUARE = 0.7                        # Threshold of minimal R-square for considering measurement as valid
	DEBUG = True                                # Flag for turn on / turn off debugging messages
	
	ATTR = {
		"GENE_ID" : 0,                       # Gene identifier (=ORF name)
		"SHALEM_HL" : 1,                     # Measured half-life from Shalem's article (log value)
		"SHALEM_RSQUARE" : 2,                # Calculated R-square for values from Shalem's article
		"BELLE_HL" : 3,                      # Measured half-life from Belli's article (in minutes)
		"WANG_HL_TOTAL" : 4,                 # Measured half-life from Wang's article / total (in minutes)
		"WANG_HL_POLYA" : 5,                 # Measured half-life from Wang's article / decap (in minutes) - preferred
		"PRESNYAK_HL_TOTAL" : 6,             # Measured half-life from Presnyak's article / total (in minutes)
		"PRESNYAK_HL_POLYA" : 7,             # Measured half-life from Presnyak's article / decap (in minutes) - preferred
		"GEISBERG_HL" : 8,                   # Measured half-life from Geisberg's article (oligonucleotide)
		"NAGALAKSHMI_TRANSCRIPTIONAL" : 9,   # Translational rate taken from Nagalakshmi's article
		"ARAVA_COPY_NO" : 10,                # mRNA copy number taken from dateset of Arava's article
		"ARAVA_PEAK" : 11,                   # mRNA peak taken from dataset of Arava's article
		"ARAVA_RIBOSOME" : 12,               # Ribosome count taken from dataset of Arava's article
		"ARAVA_DENSITY" : 13,                # Ribosome density taken from dataset of Arava's article
		"ARAVA_OCCUPANCY" : 14,              # Ribosome occupancy taken from dataset of Arava's article
		"ARAVA_TRANSL_RATE" : 15,            # Translation rate taken from dataset of Arava's article
		"UPSTREAM" : 16,                     # Upstream region recognized by method described in Nagalakshmi's article (not for each gene)
		"DOWNSTREAM" : 17,                   # Downstream region recognized by method described in Nagalakshmi's article (not for each gene)
		"UPSTREAM_1KB" : 18,                 # Upstream region taken from the border of gene (length: 1kb)
		"DOWNSTREAM_1KB" : 19,               # Downstream region taken from the border of gene (length: 1kb)
		"TRAINING_FLAG" : 20,                # Flag determining if the record will be in training or testing dataset
		"STABILITY_BIN" : 21,                # Bin number (according to the position in dataset sorted by half-life)
		"STABILITY_TERTIARY" : 22,           # Category stable / unstable / unknown 
		"DNA_SEQUENCE" : 23,                 # DNA sequence obtained from SGD 2007 (DEC)
		"PROTEIN_SEQUENCE" : 24,             # Translated protein sequence obtained from SGD 207 DEC
		"DNA_SEQUENCE_LENGTH" : 25,          # Length of DNA sequence (version SGD 2007 DEC)
		"PROTEIN_SEQUENCE_LENGTH" : 26,      # Length of translated protein sequence (version SGD 2007 DEC)
		"DNA_SEQUENCE_2015" : 27,            # DNA sequence obtained from SGD (latest version)
		"PROTEIN_SEQUENCE_2015" : 28,        # Translated protein sequence obtained from SGD (latest version)
		"DNA_SEQUENCE_LENGTH_2015" : 29,     # Length of DNA sequence (latest version)
		"PROTEIN_SEQUENCE_LENGTH_2015" : 30, # Length of translated protein sequence (latest version)
		"GENE_POSITION" : 31,                # Gene position: format: "chrXX:XX-YY;ZZ-AA"
		"RNAFOLD_STRUCTURE" : 32,            # Predicted RNA structure "(((.((((.......)).)))))........." for sequence "CGACGUAGAUGCUAGCUGACUCGAUGCCCCCC"
		"RNAFOLD_ENERGY" : 33,               # Free energy of predicted RNA structure
		"RNAFOLD_ENSEMBLE_ENERGY" : 34,      # Free energy of RNA ensemble structures
		"RNAFOLD_CENTROID_ENERGY" : 35,      # Free energy of centroid RNA structure
		"RNAFOLD_MEA_ENERGY" : 36,           # Free energy of MEA structure
		"RNAFOLD_MEA" : 37,                  # Maximum expected accuracy
		"RNAFOLD_ENSEMBLE_DIVERSITY" : 38,   # Ensemble diversity
		"RNALFOLD_ENERGY" : 39,              # Sum of energy of predicted locally stable RNA structures
		"RNALFOLD_STRUCTURE_COUNT" : 40,     # Number of predicted locally stable RNA structures
		"LOCAL_CONTIGUOUS_TRIPLET" : 41,     # Overall value representing the half-life length tendency as a result of SVM classifier (based on triplet of RNA secondary structure and central nucleotide)
		"DINUCLEOTIDE_SHUFFLING" : 42,       # P-value saying the order of transcript free energy in distribution of free energies of randomized sequences
		"DINUCLEOTIDE_FREQUENCY" : 43,       # Overall value representing the half-life length tendency as a result of SVM classifier (based on frequency of dinucleotide content)
		"AMINO_ACID_FREQUENCY" : 44,         # Overall value representing the half-life length tendency as a result of SVM classifier (based on frequency of amino acid content)
		"NUCL_ENTROPY_1" : 45,               # Entropy score calculated for individual nucleotides
		"NUCL_ENTROPY_2" : 46,               # Entropy score calculated for dinucleotides
		"NUCL_ENTROPY_3" : 47,               # Entropy score calculated for dinucleotides
		"SS_ENTROPY_1" : 48,                 # Secondary structure entrophy calculated for single elements
		"SS_ENTROPY_2" : 49,                 # Secondary structure entrophy calculated for couples of elements
		"SS_ENTROPY_3" : 50,                 # Secondary structure entrophy calculated for triplets of elements
		"SS_ENTROPY_4" : 51,                 # Secondary structure entrophy calculated for quadruplets of elements
		"TRINUCLEOTIDE_FREQUENCY" : 52,      # Frequency of nucleotide triplets
		"CODONW_ANALYSIS" : 53,              # CodonW analysis (codon bias, ...)
		"LENGTH_ANALYSIS" : 54,              # Lenght of sequence
		"UNBOUNDED_RATIO" : 55,              # Ratio of unbounded nucleotides in RNA structure ('.' characters)
		"RNAHEAT_ANALYSIS" : 56,             # Analysis of melting curve
		"RNA_PSSM" : 57,                     # Analysis of RNA PSSM matrixes
		"RNA_DISORDER" : 58,                 # Analysis of RNA disorder
		"RNA_ASA" : 59,                      # Analysis of RNA ASA
		"SPINED_DISORDER" : 60,              # Analysis of protein disorder (by SPINE-D)
		"ISOELECTRIC_POINT" : 61,            # Isoelectic point calculated by DAMBE tool
		"SYX13" : 62,                        # Number of effective codons DAMBE tool
		"PHASTCONS" : 63,                    # Conservation score from PhastCons
		"RNAZ" : 64,                         # RNA structure alignment conservation score
		"MEME" : 65,                         # Motif finder
		"DREME" : 66                         # Motif finder
	}
	
	ATTR_PRESENCE = {
		"GENE_ID" : True,
		"SHALEM_HL" : True,
		"SHALEM_RSQUARE" : True,
		"BELLE_HL" : True,
		"WANG_HL_TOTAL" : True,
		"WANG_HL_POLYA" : True,
		"PRESNYAK_HL_TOTAL" : True,
		"PRESNYAK_HL_POLYA" : True,
		"GEISBERG_HL" : True,
		"NAGALAKSHMI_TRANSCRIPTIONAL" : True,
		"ARAVA_COPY_NO" : True,
		"ARAVA_PEAK" : True,
		"ARAVA_RIBOSOME" : True,
		"ARAVA_DENSITY" : True,
		"ARAVA_OCCUPANCY" : True,
		"ARAVA_TRANSL_RATE" : True,
		"UPSTREAM" : True,
		"DOWNSTREAM" : True,
		"UPSTREAM_1KB" : True,
		"DOWNSTREAM_1KB" : True,
		"TRAINING_FLAG" : True,
		"STABILITY_BIN" : True,
		"STABILITY_TERTIARY" : True,
		"DNA_SEQUENCE" : True,
		"PROTEIN_SEQUENCE" : True,
		"DNA_SEQUENCE_LENGTH" : True,
		"PROTEIN_SEQUENCE_LENGTH" : True,
		"DNA_SEQUENCE_2015" : True,
		"PROTEIN_SEQUENCE_2015" : True,
		"DNA_SEQUENCE_LENGTH_2015" : True,
		"PROTEIN_SEQUENCE_LENGTH_2015" : True,
		"GENE_POSITION" : True,
		"RNAFOLD_STRUCTURE" : True,
		"RNAFOLD_ENERGY" : True,
		"RNAFOLD_ENSEMBLE_ENERGY" : True,
		"RNAFOLD_CENTROID_ENERGY" : True,
		"RNAFOLD_MEA_ENERGY" : True,
		"RNAFOLD_MEA" : True,
		"RNAFOLD_ENSEMBLE_DIVERSITY" : True,
		"RNALFOLD_ENERGY" : True,
		"RNALFOLD_STRUCTURE_COUNT" : True,
		"LOCAL_CONTIGUOUS_TRIPLET" : True,
		"DINUCLEOTIDE_SHUFFLING" : True,
		"DINUCLEOTIDE_FREQUENCY" : True,
		"AMINO_ACID_FREQUENCY" : True,
		"NUCL_ENTROPY_1" : True,
		"NUCL_ENTROPY_2" : True,
		"NUCL_ENTROPY_3" : True,
		"SS_ENTROPY_1" : True,
		"SS_ENTROPY_2" : True,
		"SS_ENTROPY_3" : True,
		"SS_ENTROPY_4" : True,
		"TRINUCLEOTIDE_FREQUENCY" : True,
		"CODONW_ANALYSIS" : True,
		"LENGTH_ANALYSIS" : True,
		"UNBOUNDED_RATIO" : True,
		"RNAHEAT_ANALYSIS" : True,
		"RNA_PSSM" : True,
		"RNA_DISORDER" : True,
		"RNA_ASA" : True,
		"SPINED_DISORDER" : True,
		"ISOELECTRIC_POINT" : True,
		"SYX13" : True,
		"PHASTCONS" : True,
		"RNAZ" : True,
		"MEME" : True,
		"DREME" : True
	}
	
	
	ATTR_STATISTICS = {
		"DNA_SEQUENCE_LENGTH" : True, 
		"PROTEIN_SEQUENCE_LENGTH" : True, 
		"RNAFOLD_ENERGY" : True, 
		"RNAFOLD_ENSEMBLE_ENERGY" : True,
		"RNAFOLD_CENTROID_ENERGY" : True,
		"RNAFOLD_MEA_ENERGY" : True,
		"RNAFOLD_MEA" : True,
		"RNAFOLD_ENSEMBLE_DIVERSITY" : True,
		"RNALFOLD_ENERGY" : True,
		"RNALFOLD_STRUCTURE_COUNT" : True,
		"LOCAL_CONTIGUOUS_TRIPLET" : True, 
		"DINUCLEOTIDE_SHUFFLING" : True, 
		"DINUCLEOTIDE_FREQUENCY" : True, 
		"AMINO_ACID_FREQUENCY" : True, 
		"NUCL_ENTROPY_1" : True, 
		"NUCL_ENTROPY_2" : True, 
		"NUCL_ENTROPY_3" : True, 
		"SS_ENTROPY_1" : True, 
		"SS_ENTROPY_2" : True, 
		"SS_ENTROPY_3" : True, 
		"SS_ENTROPY_4" : True,
		"TRINUCLEOTIDE_FREQUENCY" : True,
		"CODONW_ANALYSIS" : True,
		"LENGTH_ANALYSIS" : True,
		"UNBOUNDED_RATIO" : True,
		"RNAHEAT_ANALYSIS" : True,
		"RNA_PSSM" : True,
		"RNA_DISORDER" : True,
		"RNA_ASA" : True,
		"SPINED_DISORDER" : True,
		"ISOELECTRIC_POINT" : True,
		"SYX13" : True,
		"PHASTCONS" : True,
		"RNAZ" : True,
		"MEME" : True,
		"DREME" : True
		#"FINAL" : True
	}
	
	
	ATTR_ARFF = {
		"AMINO_ACID_FREQUENCY_R_PROTEIN_SEQUENCE" : True,
		"AMINO_ACID_FREQUENCY_A_PROTEIN_SEQUENCE" : True,
		"CODONW_C3s_DNA_SEQUENCE" : True,     # +0.16
		"CODONW_CAI_DNA_SEQUENCE" : True,     # +0.27
		"CODONW_CBI_DNA_SEQUENCE" : True,     # +0.27
		"CODONW_G3s_DNA_SEQUENCE" : True,     # +0.15 
		"CODONW_Gravy_DNA_SEQUENCE" : True,   # +0.10
		"CODONW_Nc_UPSTREAM" : True,          # -0.24
		"DINUCLEOTIDE_FREQUENCY_AA_DNA_SEQUENCE" : True,   # -0.13
		"DINUCLEOTIDE_FREQUENCY_AT_TRANSCRIPT" : True,     # +0.15
		"DINUCLEOTIDE_FREQUENCY_CT_DNA_SEQUENCE" : True,   # -0.17
		"LENGTH_DNA_SEQUENCE" : True,                      # +0.07
		"LOCAL_CONTIGUOUS_TRIPLET_A..._TRANSCRIPT" : True, # -0.18
		"NUCL_ENTROPY_1_DNA_SEQUENCE" : True,          # +0.19
		"SS_ENTROPY_2_DNA_SEQUENCE" : True,            # +0.09
		"RNA_PSSM_ENTROPHY_TRANSCRIPT" : True,         # +0.20
		"RNA_ASA_AVERAGE_BOUNDED_DNA_SEQUENCE" : True, # +0.13
		"RNA_DISORDER_AVERAGE_DNA_SEQUENCE" : True,    # +0.13
		"SPINED_DISORDER_AVERAGE" : True,              # +0.16
		"PHASTCONS" : True,                            # XX.YY 
		"RNAZ" : True,                                 # XX.YY
		"MEME" : True,                                 # XX.YY
		"DREME" : True
		#"ISOELECTRIC_POINT" : True,
		#"SYX13" : True
	}
	
	
	def __init__(self):
		self.loadInitConfiguration()	
		self.loadClassifiers()
	
	
	def loadInitConfiguration(self):
		### Load global configuration ###
		CONFIG_MODE = "UNKNOWN"
				
		for line in open("config.txt"):
			if(line.find("ATTRIBUTE_RECALCULATION") >= 0):
				CONFIG_MODE = "ATTRIBUTE_RECALCULATION"
				print "ATTRIBUTE_RECALCULATION :: start"
			
			if((len(line.strip()) == 0) or line.startswith("#")):
				continue
			
			if(CONFIG_MODE == "ATTRIBUTE_RECALCULATION"):
				varName = line.split("=")[0].strip()
				varValue = False if (line.split("=")[1].strip() == "False") else True
				self.ATTR_PRESENCE[varName] = varValue
				#print "self.ATTR_PRESENCE[" + varName + "] = " + str(varValue)
				#sys.stdin.read(1)
	
	
	def loadClassifiers(self):
		### Load configuration about employed sclassifiers ###
		newClassifier = ClassifierDefinition()
		for line in open("config_classifiers.txt","r"):
			if((newClassifier != None) and line.startswith("###############################################################################")):
				self.CLASSIFIERS[newClassifier.classifierName] = copy.deepcopy(newClassifier)
				self.newClassifier = None
			else:
				if(line.startswith("CLASSIFIER_NAME")):
					newClassifier.classifierName = line.split("=")[1].strip()
				elif(line.startswith("CLASSIFIER_DESC")):
					newClassifier.classifierDesc = line.split("=")[1].strip()
				elif(line.startswith("CLASSIFIER_CLASS")):
					newClassifier.classifierClass = line.split("=")[1].strip()
				elif(line.startswith("CLASSIFIER_OPTIONS")):
					newClassifier.classifierOptions = line.split("=")[1].strip()
				elif(line.startswith("CLASSIFIER_MODEL_FILE")):
					newClassifier.classifierModelFile = line.split("=")[1].strip()
				elif(line.startswith("CLASSIFIER_INFO_FILE")):
					newClassifier.classifierInfoFile = line.split("=")[1].strip()
				elif(line.startswith("COMMENT")):
					newClassifier.classifierComment = line.split("=")[1].strip()


###################################################################################################################
################################### GLOBAL VARIABLES ##############################################################
###################################################################################################################
CONFIG = Config()

###################################################################################################################
################################### IMPLEMENTATION ################################################################
###################################################################################################################

class DatasetRecord:
	### Container for records of the dataset ###
	var = {}
	identifierShortened = ""
	
	def __init__(self, line):
		self.var = None
		self.var = copy.deepcopy({})
		items = line.split(",")

		for key in CONFIG.ATTR:
			if(CONFIG.ATTR[key] > len(items)):
				CONFIG.ATTR_PRESENCE[key] = False
				self.var[key] = ""
			else:
				keyValue = items[CONFIG.ATTR[key]].strip()
				self.var[key] = keyValue
				#print "> self.var[" + key + "] = " + str(keyValue)
		
		#sys.stdin.read(1)
		self.var["TRANSCRIPT"] = self.var["UPSTREAM"] + self.var["DNA_SEQUENCE"] + self.var["DOWNSTREAM"]
		self.identifierShortened = copy.deepcopy(self.var["GENE_ID"]).replace("-","")

class HalfLifePredictor:
	### Manager of computation ###
	
	datasetAll = {}        # all records of the dataset
	datasetTraining = []   # training dataset
	datasetTesting = []    # testing dataset

	def __init__(self):
		### Load dataset ###
		
		print "***** HalfLifePredictor *****"
		for line in open(CONFIG.DATASET_INPUT, "r"):
			if(line.startswith("#") or line.startswith("GENE_ID")):
				continue
			if(DEBUG_FLAG): print "***** PROCESSING RECORD " + line.split(",")[0].strip() + " *****"
			newRecord = copy.deepcopy(DatasetRecord(line))
			self.datasetAll[newRecord.identifierShortened] = newRecord
		print "*** Dataset loaded (" + str(len(self.datasetAll)) + " records)"
	
	
	# TODO: DODELAT
	def isParameterDefined(self, record, parameterName):
		if((parameterName in record.var) and (len(record.var[parameterName]) > 0)):
			return True
		else:
			return False
		
	@staticmethod
	def isParameterDefined(record, parameterName):
		if((parameterName in record.var) and (len(str(record.var[parameterName])) > 0)):
			return True
		else:
			return False
	
	def calculateAttributes(self):
		### Calculate value of attributes ###
		
		sortedAttr = sorted(CONFIG.ATTR.items(), key=operator.itemgetter(1))
		
		for targetSequence in SEQUENCE_TYPE:
			print "\n***** " + targetSequence + " *****"
			if((targetSequence in SEQUENCE_NUC_TYPE)):
				self.loadMfe(targetSequence)# TODO: ODRAMOVAT
				self.loadDinucleotideShuffling(targetSequence)  # TODO: ODRAMOVAT
						
			for attr in sortedAttr:
				if(CONFIG.ATTR_PRESENCE[attr[0]] == False):
					if(attr[0] == "TRAINING_FLAG"):               # Dataset cut to create the training & testing datasets
						self.generateTrainingTestingDataset(targetSequence)
					elif((attr[0] == "RNAFOLD_STRUCTURE") and (targetSequence in SEQUENCE_NUC_TYPE)):         # Minimum of free energy feature
						self.appendMfe(targetSequence)
					elif((attr[0] == "LOCAL_CONTIGUOUS_TRIPLET") and (targetSequence in SEQUENCE_NUC_TYPE)):  # Local contiguous triple feature
						self.appendLocalContiguousTriplet(targetSequence)
					elif((attr[0] == "RNAHEAT_ANALYSIS") and (targetSequence in SEQUENCE_NUC_TYPE)):          # Minimum of free energy feature
						self.appendRnaHeat(targetSequence)
					elif((attr[0] == "RNA_PSSM") and (targetSequence in SEQUENCE_NUC_TYPE)):                  # RNA PSSM
						#self.appendRnaPssmAlt(targetSequence)
						#self.appendRnaPssmBlast(targetSequence)
						self.appendRnaPssm(targetSequence)
					elif((attr[0] == "RNA_DISORDER") and (targetSequence in SEQUENCE_NUC_TYPE)):              # RNA DISORDER
						self.appendRnaDisorder(targetSequence)
					elif((attr[0] == "RNA_ASA") and (targetSequence in SEQUENCE_NUC_TYPE)):                   # RNA ASA
						self.appendRnaAsa(targetSequence)
					elif((attr[0] == "SPINED_DISORDER") and (targetSequence in SEQUENCE_AMINO_TYPE)):         # SPINE-D DISORDER
						self.appendPsipredSecondaryStructure()
						self.appendSpinedDisorder()
					elif((attr[0] == "DINUCLEOTIDE_SHUFFLING") and (targetSequence in SEQUENCE_NUC_TYPE)):    # P-value of RNA sec.str. free energy distribution among randomly shuffled sequences
						self.appendDinucleotideShuffling(targetSequence)
					elif((attr[0] == "DINUCLEOTIDE_FREQUENCY") and (targetSequence in SEQUENCE_NUC_TYPE)):    # Dinucleotide frequency
						self.appendDinucleotideFrequency(targetSequence)
					elif((attr[0] == "TRINUCLEOTIDE_FREQUENCY") and (targetSequence in SEQUENCE_NUC_TYPE)):   # Trinucleotide frequency
						self.appendTrinucleotideFrequency(targetSequence)
					elif((attr[0] == "AMINO_ACID_FREQUENCY") and (targetSequence in SEQUENCE_AMINO_TYPE)):    # Amino acid frequency
						self.appendAminoAcidFrequency(targetSequence)
					elif((attr[0] == "NUCL_ENTROPY_1") and (targetSequence in SEQUENCE_NUC_TYPE)):            # Entropy on nucleotide level (single _1, pairs _2, triples _3)
						self.appendNucleotideEntropy(targetSequence)
					elif((attr[0] == "SS_ENTROPY_1") and (targetSequence in SEQUENCE_NUC_TYPE)):              # Entropy on sec.str. element level (single _1, pairs _2, triples _3, quadruplets _4)
						self.appendStructureEntropy(targetSequence)
					elif((attr[0] == "CODONW_ANALYSIS") and (targetSequence in SEQUENCE_NUC_TYPE)):           # CodonW analysis (codon bias etc)
						self.appendCodonAnalysis(targetSequence)
					elif(attr[0] == "LENGTH_ANALYSIS"):                                                       # Analysis of correlation between sequence length and HL
						self.appendLengthAnalysis(targetSequence)
					elif((attr[0] == "UNBOUNDED_RATIO") and (targetSequence in SEQUENCE_NUC_TYPE)):           # Ratio of unbounded nucleotides in structure ('.' in predicted secondary structure)
						self.appendUnboundedRatio(targetSequence)
					elif((attr[0] == "ISOELECTRIC_POINT") and (targetSequence in SEQUENCE_AMINO_TYPE)):       # Isoelectric point calculated by DAMBE
						self.appendIsoelectricPoint()
					elif((attr[0] == "SYX13") and (targetSequence in SEQUENCE_NUC_TYPE.DNA_SEQUENCE)):        # Effective number of codons calculated by DAMBE
						self.appendEffectiveNumberOfCodons()
					elif((attr[0] == "PHASTCONS") and (targetSequence in SEQUENCE_NUC_TYPE)):                 # PhastCons conservation score
						self.appendPhastConsScore(targetSequence)
					elif((attr[0] == "RNAZ") and (targetSequence in SEQUENCE_NUC_TYPE)):                      # Structure conservation RNA_ALIGN
						self.appendRnaStrucConsScoreCalculated(targetSequence)
						#self.appendRnaStrucConsScoreCalculated(targetSequence)
					elif((attr[0] == "MEME") and (targetSequence in SEQUENCE_NUC_TYPE)): # Find motif by MEME
						self.appendMemeMotifs(targetSequence)
					elif((attr[0] == "DREME") and (targetSequence in SEQUENCE_NUC_TYPE.DNA_SEQUENCE) or (targetSequence in SEQUENCE_NUC_TYPE.TRANSCRIPT)): # Find motif by MEME
						self.appendDremeMotifs(targetSequence)
	
	
	def appendNucleotideEntropy(self, targetSequence):
		### Append nucleotide entropy feature ###
		print "*** Feature calculation: NUCL_ENTROPY_1/2/3 (" + targetSequence + ")"
		
		variants1 = {}
		variants2 = {}
		variants3 = {}
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in dataset:
			variants1sum = 0
			variants2sum = 0
			variants3sum = 0
		
			nucleotides = ("A", "T", "G", "C")
			
			# Set frequencies of singles, couples, triplets to zero
			for nucl1 in nucleotides:
				variants1[nucl1] = 0
				for nucl2 in nucleotides:
					variants2[nucl1+nucl2] = 0
					for nucl3 in nucleotides:
						variants3[nucl1+nucl2+nucl3] = 0
			
			# Calculate frequencies
			for i in range(len(record.var[targetSequence])):
				if(record.var[targetSequence][i] in variants1):
					variants1[record.var[targetSequence][i]] += 1
					variants1sum += 1
				if((len(record.var[targetSequence]) - i) > 1):
					key = record.var[targetSequence][i:i+2]
					if(key in variants2):
						variants2[key] += 1
						variants2sum += 1
				if((len(record.var[targetSequence]) - i) > 2):
					key = record.var[targetSequence][i:i+3]
					if(key in variants3):
						variants3[key] += 1
						variants3sum += 1
			
			# Derive entropy score according to the calculated frequencies
			entropy1 = 0
			for var in variants1:
				if(variants1[var] != 0):
					entropy1 -= (float(variants1[var]) / variants1sum) * math.log(float(variants1[var]) / variants1sum, 2)
			entropy2 = 0
			for var in variants2:
				if(variants2[var] != 0):
					entropy2 -= (float(variants2[var]) / variants2sum) * math.log(float(variants2[var]) / variants2sum, 2)
			entropy3 = 0
			for var in variants3:
				if(variants3[var] != 0):
					entropy3 -= (float(variants3[var]) / variants3sum) * math.log(float(variants3[var]) / variants3sum, 2)
			
			record.var["NUCL_ENTROPY_1_" + targetSequence] = entropy1
			record.var["NUCL_ENTROPY_2_" + targetSequence] = entropy2
			record.var["NUCL_ENTROPY_3_" + targetSequence] = entropy3
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]

		newRecordNucl1 = Statistics(targetSequence, "NUCL_ENTROPY_1", "")
		newRecordNucl1.calculateStatistics(selectedRecordsVector, "NUCL_ENTROPY_1_" + targetSequence)
		STATISTICS.append(newRecordNucl1)
		
		newRecordNucl2 = Statistics(targetSequence, "NUCL_ENTROPY_2", "")
		newRecordNucl2.calculateStatistics(selectedRecordsVector, "NUCL_ENTROPY_2_" + targetSequence)
		STATISTICS.append(newRecordNucl2)
		
		newRecordNucl3 = Statistics(targetSequence, "NUCL_ENTROPY_3", "")
		newRecordNucl3.calculateStatistics(selectedRecordsVector, "NUCL_ENTROPY_3_" + targetSequence)
		STATISTICS.append(newRecordNucl3)
	
	
	def appendLengthAnalysis(self, targetSequence):
		### Append analysis of sequence length ###
		print "*** Feature calculation: LENGTH (" + targetSequence + ")"
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in dataset:
			record.var["LENGTH_" + targetSequence] = len(record.var[targetSequence])
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		newRecordNucl3 = Statistics(targetSequence, "LENGTH", "")
		newRecordNucl3.calculateStatistics(selectedRecordsVector, "LENGTH_" + targetSequence)
		STATISTICS.append(newRecordNucl3)
	
	
	def appendUnboundedRatio(self, targetSequence):
		### Append analysis of sequence length ###
		print "*** Feature calculation: UNBOUNDED RATIO (" + targetSequence + ")"
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence))]
		for record in dataset:
			unboundedCount = len([i for i in record.var["MFE_PREDICTED_STRUCTURE_" + targetSequence] if(i == '.')])
			unboundedRatio = float(unboundedCount) / len(record.var["MFE_PREDICTED_STRUCTURE_" + targetSequence])
			record.var["UNBOUNDED_RATIO_" + targetSequence] = unboundedRatio
			if(DEBUG_FLAG): print "> record.var[UNBOUNDED_RATIO_" + targetSequence + "] = " + str(unboundedRatio)
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence))]
		newRecordNucl3 = Statistics(targetSequence, "UNBOUNDED_RATIO", "")
		newRecordNucl3.calculateStatistics(selectedRecordsVector, "UNBOUNDED_RATIO_" + targetSequence)
		STATISTICS.append(newRecordNucl3)
		
	def appendIsoelectricPoint(self):
		print "*** Feature calculation: DAMBE_ISOELECTRIC_POINT (PROTEIN_SEQUENCE)"
		
		for line in open(CONFIG.DAMBE_ISOELECTRIC_POINT):
			items = items = re.split(' +', line.strip())
			if((len(line.strip()) == 0) or line.startswith("#")):
				continue
			
			identifier = items[0].strip().replace("-", "")
			isoelPoint = float(items[9].replace(",", "."))
			print "> isoelPoint: " + str(isoelPoint)
			if(identifier in self.datasetAll):
				self.datasetAll[identifier].var["ISOELECTRIC_POINT"] = isoelPoint
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, SEQUENCE_TYPE.DNA_SEQUENCE) and self.isParameterDefined(item, "ISOELECTRIC_POINT"))]
		newRecordSyx13NcObs = Statistics(SEQUENCE_TYPE.PROTEIN_SEQUENCE, "ISOELECTRIC_POINT", "")
		newRecordSyx13NcObs.calculateStatistics(selectedRecordsVector, "ISOELECTRIC_POINT")
		STATISTICS.append(newRecordSyx13NcObs)
		
		#sys.stdin.read(1)
	
	
	def appendEffectiveNumberOfCodons(self):
		print "*** Feature calculation: DAMBE_EFFECTIVE_NUMBER_OF_CODONS (DNA_SEQUENCE)"
		
		for line in open(CONFIG.DAMBE_CODONS):
			items = items = re.split(' +', line.strip())
			if((len(line.strip()) == 0) or line.startswith("#")):
				continue
			
			identifier = items[0].strip().replace("-", "")
			ncObs = float(items[5].replace(",", "."))
			ncExp = float(items[6].replace(",", "."))
			ncObsExpRatio = ncObs / ncExp
			#print "> ncObs: " + str(ncObs)
			#print "> ncExp: " + str(ncExp)
			#print "> ratio: " + str(ncObsExpRatio) + "\n"
			if(len(items[5].strip()) == 0):
				sys.stdin.read(1)
			
			if((identifier in self.datasetAll) == False):
				continue
			
			self.datasetAll[identifier].var["SYX13_NC_OBS"] = int(ncObs)
			self.datasetAll[identifier].var["SYX13_NC_EXP"] = ncExp
			self.datasetAll[identifier].var["SYX13_NC_OBS_EXP"] = ncObsExpRatio
			
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, SEQUENCE_TYPE.DNA_SEQUENCE) and self.isParameterDefined(item, "SYX13_NC_OBS"))]
		newRecordSyx13NcObs = Statistics(SEQUENCE_TYPE.DNA_SEQUENCE, "SYX13", "NC_OBS")
		newRecordSyx13NcObs.calculateStatistics(selectedRecordsVector, "SYX13_NC_OBS")
		STATISTICS.append(newRecordSyx13NcObs)
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, SEQUENCE_TYPE.DNA_SEQUENCE) and self.isParameterDefined(item, "SYX13_NC_EXP"))]
		newRecordSyx13NcExp = Statistics(SEQUENCE_TYPE.DNA_SEQUENCE, "SYX13", "NC_EXP")
		newRecordSyx13NcExp.calculateStatistics(selectedRecordsVector, "SYX13_NC_EXP")
		STATISTICS.append(newRecordSyx13NcExp)
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, SEQUENCE_TYPE.DNA_SEQUENCE) and self.isParameterDefined(item, "SYX13_NC_OBS_EXP"))]
		newRecordSyx13NcObsExp = Statistics(SEQUENCE_TYPE.DNA_SEQUENCE, "SYX13", "NC_OBS_EXP")
		newRecordSyx13NcObsExp.calculateStatistics(selectedRecordsVector, "SYX13_NC_OBS_EXP")
		STATISTICS.append(newRecordSyx13NcObsExp)
	
	
	def appendPhastConsScore(self, targetSequence):
		print "*** Feature calculation: PHASTCONS (" + targetSequence + ")"
				
		if(len(CONFIG.PHASTCONS_CHROMOSOMES) == 0):
			for line in open(CONFIG.PHASTCONS_RESULTS):
				items = line.split("\t")
				chrom = items[0].strip()
				position = int(items[1].strip())
				
				if((chrom in CONFIG.PHASTCONS_CHROMOSOMES) == False):
					CONFIG.PHASTCONS_CHROMOSOMES[chrom] = []
				
				#print str(position) + " > " + str((len(CONFIG.PHASTCONS_CHROMOSOMES[chrom])))
				if(position > (len(CONFIG.PHASTCONS_CHROMOSOMES[chrom]))):
					while(position > (len(CONFIG.PHASTCONS_CHROMOSOMES[chrom]))):
						CONFIG.PHASTCONS_CHROMOSOMES[chrom].append(None)
				
				conservation = float(items[3].strip())
				CONFIG.PHASTCONS_CHROMOSOMES[chrom].append(conservation)
				#print "CONFIG.PHASTCONS_CHROMOSOMES[" + chrom + "][" + str(len(CONFIG.PHASTCONS_CHROMOSOMES[chrom]) - 1)  + "] = " + str(CONFIG.PHASTCONS_CHROMOSOMES[chrom][len(CONFIG.PHASTCONS_CHROMOSOMES[chrom])-1])
				#sys.stdin.read(1)
			
		print "> PhastCons conservation loaded for all chromosomes."
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence)  and self.isParameterDefined(item, "GENE_POSITION"))]
		
		index = 0 
		for record in dataset:
			identifier = record.var["GENE_ID"].replace("-","")
			avgConservation = []
			
			chrom = record.var["GENE_POSITION"].split(":")[0].strip()              # get chromosome
			ranges = record.var["GENE_POSITION"].split(":")[1].strip().split(";")  # get ranges of nucleotides
			
			for i in range(len(ranges)):
				rang = ranges[i]
				print "> rang: " + rang
				oStart = int(rang.split("-")[0])  # start position in current range
				oEnd = int(rang.split("-")[1])  # start position in current range
				start = oStart
				end = oEnd
				
				if(targetSequence == SEQUENCE_NUC_TYPE.UPSTREAM):
					if(i == 0):
						if(oStart < oEnd):
							start = oStart - len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
							end = oStart
						else:
							end = oStart + len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
							
					else:
						continue
				
				if(targetSequence == SEQUENCE_NUC_TYPE.DOWNSTREAM):
					if(i == (len(ranges) - 1)):
						if(oStart < oEnd):
							start = oEnd
							end = oEnd + len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
						else:
							start = oEnd - len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
					else:
						continue
				
				if(targetSequence == SEQUENCE_NUC_TYPE.TRANSCRIPT):
					if(i == 0):
						if(oStart < oEnd):
							start = oStart - len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
						else:
							end = oStart + len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
					if(i == (len(ranges) - 1)):
						if(oStart < oEnd):
							end = oEnd + len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
						else:
							start = oEnd - len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
									
				if(start > end):
					tmp = start
					#start = len(chromosomes[chrom]) - end
					#end = len(chromosomes[chrom]) - tmp
					start = end
					end = tmp
					#print "> rozsah: " + str(start) + " - " + str(end)
								
				# Data consistency check
				sequenceLen = len(self.datasetAll[identifier].var[targetSequence])
				structureLen = len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])
				
				if((sequenceLen != structureLen)):# or (sequenceLen != asaLen) or (structureLen != asaLen)):
					print identifier
					#sys.stdin.read(1)
					continue
				
				avgSecStr = {"UNBOUNDED" : [], "BOUNDED" : []}
				for j in range(end - start):
					if(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j] != None):
						avgConservation.append(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j])
						#print ">>>>> avgConservation.append(chromosomes[" + chrom + "][" + str(start+j) + "]) ... " + str(chromosomes[chrom][start+j])
						secStructure = "BOUNDED" if(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence][j] == ".") else "UNBOUNDED"
						avgSecStr[secStructure].append(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j])
						
				if(len(avgSecStr["UNBOUNDED"]) > 0): 
					self.datasetAll[identifier].var["PHASTCONS_UNBOUNDED_" + targetSequence] = numpy.mean(avgSecStr["UNBOUNDED"])
					#print "avg(unbounded) = " + str(self.datasetAll[identifier].var["PHASTCONS_UNBOUNDED_" + targetSequence])
				if(len(avgSecStr["BOUNDED"]) > 0): 
					self.datasetAll[identifier].var["PHASTCONS_BOUNDED_" + targetSequence] = numpy.mean(avgSecStr["BOUNDED"])
					#print "avg(bounded) = " + str(self.datasetAll[identifier].var["PHASTCONS_BOUNDED_" + targetSequence])
				
			if(len(avgConservation) > 0):
				avgConservation = numpy.mean(avgConservation)
				propertyKey = "PHASTCONS_AVERAGE_" + targetSequence
				self.datasetAll[identifier].var[propertyKey] = avgConservation
			
			index += 1
			print "> CONSERVATION / " + targetSequence + " / " + str(index)
		
		sumVariants = ["AVERAGE", "UNBOUNDED", "BOUNDED"]
		for key in sumVariants:
			propertyKey = "PHASTCONS_" + key + "_" + targetSequence
			print "> propertyKey: " + propertyKey
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey))]
			print "> len(selectedRecordsVector): " + str(len(selectedRecordsVector))
			newRecord = Statistics(targetSequence, "PHASTCONS", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
		
		
		phastConsVector = [item.var["PHASTCONS_AVERAGE_" + targetSequence] for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, "RNA_PSSM_CONS_" + targetSequence) and self.isParameterDefined(item, "PHASTCONS_AVERAGE_" + targetSequence))]
		rnaPssmVector = [item.var["RNA_PSSM_CONS_" + targetSequence] for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, "RNA_PSSM_CONS_" + targetSequence) and self.isParameterDefined(item, "PHASTCONS_AVERAGE_" + targetSequence))]
		print "VALUES: " + str(len(phastConsVector)) 
		print "> pearson : " + str(round(scipy.stats.pearsonr(phastConsVector, rnaPssmVector)[0], 2))
		print "> spearman: " + str(round(scipy.stats.spearmanr(phastConsVector, rnaPssmVector)[0], 2))
			
		#sys.stdin.read(1)
		
	def appendDremeMotifs(self, targetSequence):
		print "*** Feature calculation: DREME MOTIFS (" + targetSequence + ")"
		
		fileMapping = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		
		stableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_STABLE.fasta"
		unstableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_UNSTABLE.fasta"
		twilightFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_TWILIGHT.fasta"
		self.createStableUnstableFiles(targetSequence)
		
		dremeStableMotifPath = CONFIG.MEME_FOLDER + "dreme_stable_" + fileMapping[targetSequence] + ".txt"
		dremeUnstableMotifPath = CONFIG.MEME_FOLDER + "dreme_unstable_" + fileMapping[targetSequence] + ".txt"
		currentSeqFilePath = CONFIG.MEME_FOLDER + "temp.txt"
		
		#dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, "SHALEM_HL") and self.isParameterDefined(item, "BELLE_HL") and self.isParameterDefined(item, "WANG_HL_TOTAL") and self.isParameterDefined(item, "WANG_HL_POLYA") and self.isParameterDefined(item, "PRESNYAK_HL_TOTAL") and self.isParameterDefined(item, "PRESNYAK_HL_POLYA") and self.isParameterDefined(item, "GEISBERG_HL") and self.isParameterDefined(item, targetSequence))]
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		
		i = 0
		for record in dataset:
			if(record.var["TRAINING_FLAG"] == True):
				continue
			
			currentStableFolder = CONFIG.MEME_FOLDER + "results/" + record.identifierShortened + "_" + fileMapping[targetSequence]
			currentUnstableFolder = CONFIG.MEME_FOLDER + "results/" + record.identifierShortened + "_unstable_" + fileMapping[targetSequence]
			
			if(os.path.isdir(currentStableFolder) == False):
				currentSeqFile = open(currentSeqFilePath, "w")
				currentSeqFile.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "")
				currentSeqFile.close()
				command = CONFIG.FIMO_BINARY + " --oc . --verbosity 1 --thresh 1.0E-4 --o " + currentStableFolder + " " + dremeStableMotifPath + " " + currentSeqFilePath
				os.system(command)
			if(os.path.isdir(currentUnstableFolder) == False):
				currentSeqFile = open(currentSeqFilePath, "w")
				currentSeqFile.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "")
				currentSeqFile.close()
				command = CONFIG.FIMO_BINARY + " --oc . --verbosity 1 --thresh 1.0E-4 --o " + currentUnstableFolder + " " + dremeUnstableMotifPath + " " + currentSeqFilePath
				os.system(command)
			
			# TODO: rozbehnout
			dremeStableOutputFilePath = currentStableFolder + "/fimo.txt"
			dremeUnstableOutputFilePath = currentUnstableFolder + "/fimo.txt"
			#print "> dremeStableOutputFilePath: " + dremeStableOutputFilePath + "\t" + str(os.path.isfile(dremeStableOutputFilePath))
			#print "> dremeUnstableOutputFilePath: " + dremeUnstableOutputFilePath + "\t" + str(os.path.isfile(dremeUnstableOutputFilePath))
			#sys.stdin.read(1)
			if(os.path.isfile(dremeStableOutputFilePath) and os.path.isfile(dremeUnstableOutputFilePath)):
				stableHits = len(open(dremeStableOutputFilePath, "r").readlines()) - 1
				#print "> stableHits 1 : " + str(open(dremeStableOutputFilePath, "r").readlines())
				#print "> stableHits 2: " + str(stableHits)
				unstableHits = len(open(dremeUnstableOutputFilePath, "r").readlines()) - 1
				#print "> unstableHits 1: " + str(open(dremeUnstableOutputFilePath, "r").readlines())
				#print "> unstableHits 2: " + str(unstableHits)
			
			record.var["DREME_AVERAGE_" + targetSequence] = float(stableHits - unstableHits) / len(record.var[CONFIG.TARGET_SEQUENCE])
		
		sumVariants = ["AVERAGE"]
		for key in sumVariants:
			propertyKey = "DREME_" + key + "_" + targetSequence
			print "> propertyKey: " + propertyKey
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey))]
			print "> len(selectedRecordsVector): " + str(len(selectedRecordsVector))
			newRecord = Statistics(targetSequence, "DREME", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
			#print "> " + str(i)
			#sys.stdin.read(1)
		
		#sys.stdin.read(1)
	
	
	def createStableUnstableFiles(self, targetSequence):
		stableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_STABLE.fasta"
		inputFileStable = open(stableFilePath, "w")
		unstableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_UNSTABLE.fasta"
		inputFileUnstable = open(unstableFilePath, "w")
		twilightFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_TWILIGHT.fasta"
		inputFileTwilight = open(twilightFilePath, "w")
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and (item.var["TRAINING_FLAG"] == True))]
		for record in dataset:
			if(len(record.var[targetSequence]) < 10):
				continue
			if(record.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.STABLE):
				inputFileStable.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
			elif(record.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.UNSTABLE):
				inputFileUnstable.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
			elif(record.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.TWILIGHT):
				inputFileTwilight.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
		
		inputFileStable.close()
		inputFileUnstable.close()
		inputFileTwilight.close()
		
	
	def appendMemeMotifs(self, targetSequence):
		print "*** Feature calculation: MEME MOTIFS (" + targetSequence + ")"
		
		#dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, "SHALEM_HL") and self.isParameterDefined(item, "BELLE_HL") and self.isParameterDefined(item, "WANG_HL_TOTAL") and self.isParameterDefined(item, "WANG_HL_POLYA") and self.isParameterDefined(item, "PRESNYAK_HL_TOTAL") and self.isParameterDefined(item, "PRESNYAK_HL_POLYA") and self.isParameterDefined(item, "GEISBERG_HL") and self.isParameterDefined(item, targetSequence))]
		#print "DATASET SIZE: " + str(len(dataset))
		#sys.stdin.read(1)
		
		stableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_STABLE.fasta"
		unstableFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_UNSTABLE.fasta"
		twilightFilePath = CONFIG.MEME_FOLDER + "input_" + targetSequence + "_TWILIGHT.fasta"
		self.createStableUnstableFiles()
		
		priorsFilePath = CONFIG.MEME_FOLDER + "meme_priors.psp"
		
		# Documentation URL: http://meme.ebi.edu.au/meme/doc/meme.html?man_type=web
		if(tool == "MEME"):
			command1 = CONFIG.MEME_PSPGEN + " -pos " + stableFilePath + " -neg " + unstableFilePath + " -alpha dna -minw 4 -maxw 10 > " + priorsFilePath
			print "> Command 1: " + command1
			os.system(command1)
		
			command2 = CONFIG.MEME_BINARY + " " + stableFilePath + " -dna -oc . -nostatus -time 18000 -mod zoops -nmotifs 5 -minw 4 -maxw 10 -maxsize 10000000 -revcomp -psp " + priorsFilePath
			print "> Command 2: " + command2
			os.system(command2)
		
		if(tool == "DREME"):
			dremeFilePath = CONFIG.MEME_FOLDER + "dreme_" + targetSequence + ".out"			
			command1 = CONFIG.DREME_BINARY + " -png -v 1 -oc . -t 18000 -p " + stableFilePath + " -n " + unstableFilePath + " -e 0.05 > " + dremeFilePath
			print "> Command 1: " + command1
			os.system(command1)
		
		sys.stdin.read(1)
		
	def appendRnaStrucConsScoreCalculated(self, targetSequence):
		print "*** Feature calculation: RNAZ (CALCULATED) (" + targetSequence + ")"
		
		fileMapping = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		errors = []
		
		# Browse RNAz output files
		for fileName in os.listdir(CONFIG.RNA_RNAZ_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().split("_")[0]
			version = fileName.split(".")[0].strip().split("_")[1]
			
			if((version != fileMapping[targetSequence]) or (fileName.endswith(".out") == False)):  # process only files for currently analyzed type of sequence (transcript /  upstream / ...)
				continue
			
			for line in open(CONFIG.RNA_RNAZ_RESULT_FOLDER + fileName):
				#print line
				if(line.find("Mean z-score:") >= 0):
					propertyKey = "RNAZ_ZSCORE_" + targetSequence
					self.datasetAll[identifier].var[propertyKey] = float(line.strip().split(":")[1].strip())
				if(line.find("Structure conservation index:") >= 0):
					propertyKey = "RNAZ_STRUCT_CONSERVATION_" + targetSequence
					self.datasetAll[identifier].var[propertyKey] = float(line.strip().split(":")[1].strip())
		
		sumVariants = ["ZSCORE", "STRUCT_CONSERVATION"]
		for key in sumVariants:
			propertyKey = "RNAZ_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence)) and self.isParameterDefined(item, propertyKey)]
			newRecord = Statistics(targetSequence, "RNAZ", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
		
		#propertyKey1 = "RNAZ_STRUCT_CONSERVATION_" + targetSequence
		#propertyKey2 = "RNA_PSSM_CONS_" + targetSequence
		#propertyKey3 = "PHASTCONS_AVERAGE_" + targetSequence
		#vectorTesting1 = [item.var[propertyKey1] for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey1) and self.isParameterDefined(item, propertyKey2) and self.isParameterDefined(item, propertyKey3))]
		#vectorTesting2 = [item.var[propertyKey2] for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey1) and self.isParameterDefined(item, propertyKey2) and self.isParameterDefined(item, propertyKey3))]
		#vectorTesting3 = [item.var[propertyKey3] for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey1) and self.isParameterDefined(item, propertyKey2) and self.isParameterDefined(item, propertyKey3))]

		#print scipy.stats.pearsonr(vectorTesting1, vectorTesting2)[0]
		#print "> " + propertyKey1 + " / " + propertyKey2 + ": " + str(round(scipy.stats.spearmanr(vectorTesting1, vectorTesting2)[0], 2)) + "\t" + str(round(scipy.stats.pearsonr(vectorTesting1, vectorTesting2)[0], 2))
		#print "> " + propertyKey2 + " / " + propertyKey3 + ": " + str(round(scipy.stats.spearmanr(vectorTesting2, vectorTesting3)[0], 2)) + "\t" + str(round(scipy.stats.pearsonr(vectorTesting2, vectorTesting3)[0], 2))
		#print "> " + propertyKey1 + " / " + propertyKey3 + ": " + str(round(scipy.stats.spearmanr(vectorTesting1, vectorTesting3)[0], 2)) + "\t" + str(round(scipy.stats.pearsonr(vectorTesting1, vectorTesting3)[0], 2))
		print "> RNAz calculated ... press any key to continue "
		sys.stdin.read(1)
	
	def appendRnaStrucConsScore(self, targetSequence):
		print "*** Feature calculation: RNAZ (" + targetSequence + ")"

		ORGANISM_MAPPING = {'sacCer3' : 0, 'sacPar' : 1, 'sacMik' : 2, 'sacKud' : 3, 'sacBay' : 4, 'sacCas' : 5, 'sacKlu' : 6}
		emptyVector = ['-', '-', '-', '-', '-', '-', '-']
		
		if(len(CONFIG.PHASTCONS_ALI_CHROMOSOMES) == 0):
			for fileName in os.listdir(CONFIG.PHASTCONS_ALI_FOLDER):
				if(fileName.endswith("maf") == False):
					continue
				
				chrom = fileName.split(".")[0].strip()
				CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom] = []
				print "> LOADED CHROMOSOME: " + fileName + " : " + chrom + " ***"
				
				alns = {}
				sourceStart = ""
				sourceSize = ""
				
				#if(chrom != "chrI"):
				#	continue
				
				for line in open(CONFIG.PHASTCONS_ALI_FOLDER + fileName):
					if(line.startswith("# ") or len(line.strip()) == 0):
						continue
					
					if(line.startswith("a ") or line.startswith("##eof maf")):
						if(len(alns) == 0):
							continue
						
						#print "*** NOVY BLOK, STARTUJE OD POZICE: " + str(sourceStart) + " ***"
						while(sourceStart > len(CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom])):
							CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom].append(copy.deepcopy(emptyVector))
							#print "< Position " + str(len(CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom]) - 1) + ": " + str(emptyVector)
						
						for i in range(len(alns["sacCer3"])):
							currentVector = copy.deepcopy(emptyVector)
							for organism in ORGANISM_MAPPING:
								if(organism in alns):
									currentVector[ORGANISM_MAPPING[organism]] = alns[organism][i]
							if(alns["sacCer3"][i] == "-"):
								continue
							
							CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom].append(copy.deepcopy(currentVector))
							#print "> Position " + str(len(CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom]) - 1) + ": " + str(currentVector)
							
						alns = {}
						#sys.stdin.read(1)
					
					elif(line.startswith("s ")):
						items = re.split(' +', line.strip())
						sourceName = items[1].strip().split(".")[0]
						#print "> line: " + line.strip()
						#print "> sourceName: " + sourceName + " ... " + str(items[6])
						if(sourceName == "sacCer3"):
							sourceStart = int(items[2].strip())
							sourceSize = int(items[3].strip())
						alns[sourceName] = items[6].strip()
		
		print "> Sequence alignments loaded for all chromosomes."
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence)  and self.isParameterDefined(item, "GENE_POSITION"))]
		print "> records containing gene position: " + str(len(dataset))
		
		alns = {'sacCer3' : "", 'sacPar' : "", 'sacMik' : "", 'sacKud' : "", 'sacBay' : "", 'sacCas' : "", 'sacKlu' : ""}
		for record in dataset:
			identifier = record.var["GENE_ID"].replace("-","")
			avgConservation = []
			
			print "***** IDENTIFIER: " + identifier + " / " + record.var["GENE_POSITION"]
			
			chrom = record.var["GENE_POSITION"].split(":")[0].strip()              # get chromosome
			ranges = record.var["GENE_POSITION"].split(":")[1].strip().split(";")  # get ranges of nucleotides
			
			#if(chrom != "chrI"):
			#	continue
			
			alns = {'sacCer3' : "", 'sacPar' : "", 'sacMik' : "", 'sacKud' : "", 'sacBay' : "", 'sacCas' : "", 'sacKlu' : ""}
		
			for i in range(len(ranges)):
				rang = ranges[i]
				oStart = int(rang.split("-")[0])  # start position in current range
				oEnd = int(rang.split("-")[1])  # start position in current range
				start = oStart
				end = oEnd
				forwardStrand = True if (oStart < oEnd) else False
				
				if(targetSequence == SEQUENCE_NUC_TYPE.UPSTREAM):
					if(i == 0):
						if(oStart < oEnd):
							start = oStart - len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM]) - 1
							end = oStart - 1
						else:
							end = oStart + len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
							
					else:
						continue
				
				if(targetSequence == SEQUENCE_NUC_TYPE.DOWNSTREAM):
					if(i == (len(ranges) - 1)):
						if(oStart < oEnd):
							start = oEnd - 1
							end = oEnd + len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM]) - 1
						else:
							start = oEnd - len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
					else:
						continue
				
				if(targetSequence == SEQUENCE_NUC_TYPE.TRANSCRIPT):
					if(i == 0):
						#print "T"
						#sys.stdin.read(1)
						if(oStart < oEnd):
							start = oStart - len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM]) - 1
							end -= 1
						else:
							end = oStart + len(record.var[SEQUENCE_NUC_TYPE.UPSTREAM])
					if(i == (len(ranges) - 1)):
						
						#print "TRANSCRIPTe"
						#sys.stdin.read(1)
						if(oStart < oEnd):
							end = oEnd + len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
						else:
							start = oEnd - len(record.var[SEQUENCE_NUC_TYPE.DOWNSTREAM])
				
				if(targetSequence == SEQUENCE_NUC_TYPE.DNA_SEQUENCE):
					if(start > end):
						tmp = start
						start = end
						end = tmp
					else:
						start -= 1
						end -= 1
				
				#start = 136914
				#print str(start) + ":" + CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom][start][ORGANISM_MAPPING["sacCer3"]]
				#sys.stdin.read(1)
				
				for j in range(end - start):
					#print "len(CONFIG.PHASTCONS_ALI_CHROMOSOMES[" + chrom + "][" + str(start+j) + "][sacCer3] = " + CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom][(start+j)][ORGANISM_MAPPING["sacCer3"]]
					if(CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom][start+j][ORGANISM_MAPPING["sacCer3"]] == "-"):
						continue
					for org in alns:
						alns[org] += CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom][start+j][ORGANISM_MAPPING[org]]
						#print "CONFIG.PHASTCONS_ALI_CHROMOSOMES[" + chrom + "][" + str(start+j) + "][" + str(ORGANISM_MAPPING[org]) + "] = " + str(CONFIG.PHASTCONS_ALI_CHROMOSOMES[chrom][start+j][ORGANISM_MAPPING[org]])
						#sys.stdin.read(1)
					#if(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j] != None):
					#	avgConservation.append(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j])
					#	#print ">>>>> avgConservation.append(chromosomes[" + chrom + "][" + str(start+j) + "]) ... " + str(chromosomes[chrom][start+j])
					#	secStructure = "BOUNDED" if(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence][j] == ".") else "UNBOUNDED"
					#	avgSecStr[secStructure].append(CONFIG.PHASTCONS_CHROMOSOMES[chrom][start+j])
			
			# RUN RNAz
			fileMappingIndex = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
			rnazInputFile = CONFIG.TMP_FOLDER + "rnaz/" + identifier + "_" + fileMappingIndex[targetSequence] + ".aln"
			inputFile = open(rnazInputFile, "w")
			inputFile.write("CLUSTAL W (1.83) multiple sequence alignment\n\n\n")
			
			for org in alns:
				if(len(alns[org].replace("-","")) > 0):
					if(forwardStrand == False):
						alns[org] = alns[org][::-1]
						alns[org] = alns[org].replace("A", "X")
						alns[org] = alns[org].replace("T", "Y")
						alns[org] = alns[org].replace("X", "T")
						alns[org] = alns[org].replace("Y", "A")
						alns[org] = alns[org].replace("C", "X")
						alns[org] = alns[org].replace("G", "Y")
						alns[org] = alns[org].replace("X", "G")
						alns[org] = alns[org].replace("Y", "C")
					
					inputFile.write(org.ljust(16) + alns[org] + "\n")
			inputFile.close()
			
			print "> Input file " + rnazInputFile + " has been created."
			
			continue
			
			rnazOutputFile = CONFIG.TMP_FOLDER + "rnaz_output.txt"
			command1 = "rm " + rnazOutputFile
			command2 = CONFIG.RNAZ_BINARY + " " + rnazInputFile + " > " + rnazOutputFile
			os.system(command1 + ";" + command2)
			#sys.stdin.read(1)
			
			print "> RNAz output: " + rnazOutputFile + " / " + str(os.path.isfile(rnazOutputFile))
			
			if(os.path.isfile(rnazOutputFile)):
				for line in open(rnazOutputFile):
					print line
					if(line.find("Mean z-score:") >= 0):
						record.var["RNA_ZSCORE: " + targetSequence] = float(line.strip().split(":")[1].strip())
					if(line.find("Structure conservation index:") >= 0):
						record.var["RNAZ_STRUCT_CONSERVATION_" + targetSequence] = float(line.split(":")[1].strip())
			
			print "record.var[RNAZ_ZSCORE_" + targetSequence + "] = " + str(record.var["RNAZ_ZSCORE_" + targetSequence])
			print "record.var[RNAZ_STRUCT_CONSERVATION_" + targetSequence + "] = " +  str(record.var["RNAZ_STRUCT_CONSERVATION_" + targetSequence])
			
			#sys.stdin.read(1)
			
		#sys.stdin.read(1)
		#sumVariants = ["ZSCORE", "STRUCT_CONSERVATION"]
		#for key in sumVariants:
		#	propertyKey = "PHASTCONS_" + key + "_" + targetSequence
		#	print "> propertyKey: " + propertyKey
		#	selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey))]
		#	print "> len(selectedRecordsVector): " + str(len(selectedRecordsVector))
		#	newRecord = Statistics(targetSequence, "PHASTCONS", key)
		#	newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
		#	STATISTICS.append(newRecord)

		print "TARGET_SEQUENCE: " + targetSequence
		sys.stdin.read(1)
		
	
	
	def appendCodonAnalysis(self, targetSequence):
		### Append analysis of codons by CodonW tool ###
		print "*** Feature calculation: CODONW (" + targetSequence + ")"
		
		# Run CodonW
		#inputDatasetPath = CONFIG.TMP_FOLDER + "inputCodonW.fasta"
		#inputDataset = open(inputDatasetPath,"w")
		#
		#dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		#for record in dataset:
		#	inputDataset.write(">" + record.var["GENE_ID"] + "\n")
		#	
		#	sequence = record.var["DNA_SEQUENCE"]
		#	for i in range((len(sequence) / 60 + 1)):
		#		start = i * 60
		#		stop = (i + 1) * 60
		#		if(len(sequence[start : stop]) == 0):
		#			continue
		#		inputDataset.write(sequence[start : stop].strip() + "\n")
		#inputDataset.close()
		#
		#command = "yes \"yes\"|" + CONFIG.CODONW_BINARY + " " + inputDatasetPath + " -nomenu"
		#print "> command: " + command
		#os.system(command)
		
		# THE EASIEST CHOICE IS TO PREPARE OUTPUT FILE OF CODONW MANUALLY
		# SETTINGS / Change defaults: 6-4, 7-2 (switch to E.coli)
		# SETTINGS / Codon usage analysis: 12 (select all)
		# Calculate => Run: >R<
		
		# Analyze results of CodonW
		headline = []
		for line in open(CONFIG.CODONW_OUTPUT, "r"):
			line = line.strip()   # remove tab at the end of file
			items = line.split("\t")
			if(line.startswith("title")):
				for item in items:
					headline.append(item.strip())
				continue
			identifierShortened = items[0].strip().replace("-","")
			if((identifierShortened in self.datasetAll) == False):
				continue
			record = self.datasetAll[identifierShortened]
			for i in range(len(items) - 1):
				propertyName = "CODONW_" + headline[i+1] + "_" + targetSequence
				#print "> propertyName: " + propertyName
				try:
					value = float(items[i+1].strip())
					record.var[propertyName] = value
				except:
					if(DEBUG_FLAG): print "> Error: Conversion of " + propertyName + " (value " + items[i+1].strip() + ") failed for record " + identifierShortened
					
				#print "> record.var[" + propertyName + "] = " + str(items[i+1].strip())
				
			#sys.stdin.read(1)
		
		for key in headline:
			if(key == "title"):
				continue
			propertyKey = "CODONW_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey))]
			newRecord = Statistics(targetSequence, "CODONW", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)

	
	def appendStructureEntropy(self, targetSequence):
		### Append secondary structure entropy feature ###
		print "*** Feature calculation: SEC_STRUCT_ENTROPY_1/2/3/4 (" + targetSequence + ")"
		
		variants1 = {}
		variants2 = {}
		variants3 = {}
		variants4 = {}
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence))]
		for record in dataset:
			variants1sum = 0
			variants2sum = 0
			variants3sum = 0
			variants4sum = 0
		
			structElements = (".", ")", "(")
			
			# Set frequencies of singles, couples, triplets to zero
			for struct1 in structElements:
				variants1[struct1] = 0
				for struct2 in structElements:
					variants2[struct1+struct2] = 0
					for struct3 in structElements:
						variants3[struct1+struct2+struct3] = 0
						for struct4 in structElements:
							variants4[struct1+struct2+struct3+struct4] = 0
			
			# Calculate frequencies
			predictedStructureKey = "MFE_PREDICTED_STRUCTURE_" + targetSequence
			for i in range(len(record.var[predictedStructureKey])):
				if(record.var[predictedStructureKey][i] in variants1):
					variants1[record.var[predictedStructureKey][i]] += 1
					variants1sum += 1
				if((len(record.var[predictedStructureKey]) - i) > 1):
					key = record.var[predictedStructureKey][i:i+2]
					if(key in variants2):
						variants2[key] += 1
						variants2sum += 1
				if((len(record.var[predictedStructureKey]) - i) > 2):
					key = record.var[predictedStructureKey][i:i+3]
					if(key in variants3):
						variants3[key] += 1
						variants3sum += 1
				if((len(record.var[predictedStructureKey]) - i) > 3):
					key = record.var[predictedStructureKey][i:i+4]
					if(key in variants4):
						variants4[key] += 1
						variants4sum += 1
			
			# Derive entropy score according to the calculated frequencies
			entropy1 = 0
			for var in variants1:
				if(variants1[var] != 0):
					entropy1 -= (float(variants1[var]) / variants1sum) * math.log(float(variants1[var]) / variants1sum, 2)
			entropy2 = 0
			for var in variants2:
				if(variants2[var] != 0):
					entropy2 -= (float(variants2[var]) / variants2sum) * math.log(float(variants2[var]) / variants2sum, 2)
			entropy3 = 0
			for var in variants3:
				if(variants3[var] != 0):
					entropy3 -= (float(variants3[var]) / variants3sum) * math.log(float(variants3[var]) / variants3sum, 2)
			entropy4 = 0
			for var in variants4:
				if(variants4[var] != 0):
					entropy4 -= (float(variants4[var]) / variants4sum) * math.log(float(variants4[var]) / variants4sum, 2)
					
			record.var["SS_ENTROPY_1_" + targetSequence] = entropy1
			record.var["SS_ENTROPY_2_" + targetSequence] = entropy2
			record.var["SS_ENTROPY_3_" + targetSequence] = entropy3
			record.var["SS_ENTROPY_4_" + targetSequence] = entropy4
			
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence))]

		newRecordNucl1 = Statistics(targetSequence, "SS_ENTROPY_1", "")
		newRecordNucl1.calculateStatistics(selectedRecordsVector, "SS_ENTROPY_1_" + targetSequence)
		STATISTICS.append(newRecordNucl1)
		
		newRecordNucl2 = Statistics(targetSequence, "SS_ENTROPY_2", "")
		newRecordNucl2.calculateStatistics(selectedRecordsVector, "SS_ENTROPY_2_" + targetSequence)
		STATISTICS.append(newRecordNucl2)
		
		newRecordNucl3 = Statistics(targetSequence, "SS_ENTROPY_3", "")
		newRecordNucl3.calculateStatistics(selectedRecordsVector, "SS_ENTROPY_3_" + targetSequence)
		STATISTICS.append(newRecordNucl3)
		
		newRecordNucl4 = Statistics(targetSequence, "SS_ENTROPY_4", "")
		newRecordNucl4.calculateStatistics(selectedRecordsVector, "SS_ENTROPY_4_" + targetSequence)
		STATISTICS.append(newRecordNucl4)

	def appendAminoAcidFrequency(self, targetSequence):
		### Append dinucleotide frequency feature ###
		print "*** Feature calculation: AMINO_ACID_FREQUENCY (" + targetSequence + ")"
		
		aminoAcids = ("A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
		
		# Initialization of summary array with all possibilities of input (combinations of couple of nucleotides)
		sumVariants = {}
		for aminoAcid in aminoAcids:
			sumVariants[aminoAcid] = []
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in dataset:
			# Initialization of array with all possibilities of input (combinations of couple of nucleotides)
			variants = {}
			for aminoAcid in aminoAcids:
				variants[aminoAcid] = 0
				
			# Calculation of frequency for individual variants (combinations of one central nucleotide and corresponding triplet of RNA secondary structures
			adjustedSequence = record.var[targetSequence]
			aminoacidCount = 0
			for i in range(len(adjustedSequence)):	
				if(adjustedSequence[i] in aminoAcids):
					variants[adjustedSequence[i]] += 1
					aminoacidCount += 1
							
			# Normalization of variant occurrencies to reach overall sum of 1.0
			for rec in variants:
				if(variants[rec] > 0):
					variants[rec] = float(variants[rec]) / aminoacidCount
				sumVariants[rec].append(variants[rec])
				
				propertyKey = "AMINO_ACID_FREQUENCY_" + rec + "_" + targetSequence
				record.var[propertyKey] = variants[rec]
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for key in sumVariants:
			propertyKey = "AMINO_ACID_FREQUENCY_" + key + "_" + targetSequence
			newRecord = Statistics(targetSequence, "AMINO_ACID_FREQUENCY", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
	
	
	def wekaTraining(self, featureName, trainFile, allRecordsFile, convertorAll):
		print ">>> wekaTraining(" + featureName + ")"
		loader = Loader(classname="weka.core.converters.ArffLoader")
		dataTrain = loader.load_file(trainFile)
		dataTrain.class_is_last()
		dataAll = loader.load_file(allRecordsFile)
		dataAll.class_is_last()
		
		cls = KernelClassifier(classname="weka.classifiers.functions.SMOreg", options=["-C", "1.0", "-N", "0", "-I", "weka.classifiers.functions.supportVector.RegSMOImproved -L 0.001 -W 1 -P 1.0E-12 -T 0.001 -V"])
		kernel = Kernel(classname="weka.classifiers.functions.supportVector.RBFKernel", options=["-C", "250007", "-G", "0.01"])
		cls.kernel = kernel
		cls.build_classifier(dataTrain)
		
		# Assigning prediction to all instances
		testSetHalfLife = []
		testSetFeature = []
		for index, inst in enumerate(dataAll):
			pred = cls.classify_instance(inst)
			convertorAll[index].var[featureName] = float(pred)
			if(convertorAll[index].var["TRAINING_FLAG"] == "TESTING"):
				testSetHalfLife.append(float(convertorAll[index].var[CONFIG.TARGET_HL]))
				testSetFeature.append(float(convertorAll[index].var[featureName]))
	
		
	def appendLocalContiguousTriplet(self, targetSequence):
		### Append local contiguous triplet feature ###
		print "*** Feature calculation: LOCAL_CONTIGUOUS_TRIPLET (" + targetSequence + ")"
		
		nucleotides = ("A", "T", "G", "C")
		triplets = ("(((", "((.", "(.(", "(..", ".((", ".(.", "..(", "...")
		
		# Initialization of summary array with all possibilities of input (combinations of tuples "central nucleotide" and "sec.str.elem.triplets")
		sumVariants = {}
		for nucleotide in nucleotides:
			for triplet in triplets:
				sumVariants[nucleotide + triplet] = []
		
		dataset = [item for item in self.datasetAll.itervalues() if self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence)]
		
		for record in dataset:
			# Initialization of array with all possibilities of input (combinations of tuples "central nucleotide" and "sec.str.elem.triplets")
			variants = {}
			for nucleotide in nucleotides:
				for triplet in triplets:
					variants[nucleotide + triplet] = 0
			 
			# Calculation of frequency for individual variants (combinations of one central nucleotide and corresponding triplet of RNA secondary structures
			adjustedSequence = record.var["MFE_PREDICTED_STRUCTURE_" + targetSequence].replace(")","(")   # types of secondary RNA structure element "(" and ")" is considered as the same
			tripletsCount = 0
			for i in range(len(adjustedSequence)-2):	
				try:
					triplet = adjustedSequence[i:i+3]
					nucleotide = record.var[targetSequence][i+1]

					# Triplets with non-standard nucleotides ("N" => any nucleotide etc) will not be evaluated
					flag = True if ((nucleotide == 'A') or (nucleotide == 'T') or (nucleotide == 'C') or (nucleotide == 'G')) else False
					if(flag == True):
						variants[nucleotide + triplet] += 1
						tripletsCount += 1
				
				except:
					print "!!! ERROR !!!"
					traceback.print_exc()
					sys.stdin.read(1)
				
			# Normalization of variant occurrencies to reach overall sum of 1.0
			for rec in variants:
				if(variants[rec] > 0):
					variants[rec] = float(variants[rec]) / tripletsCount
				sumVariants[rec].append(variants[rec])
				
				propertyKey = "LOCAL_CONTIGUOUS_TRIPLET_" + rec + "_" + targetSequence
				record.var[propertyKey] = variants[rec]
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "MFE_PREDICTED_STRUCTURE_" + targetSequence))]
		
		for key in sumVariants:
			propertyKey = "LOCAL_CONTIGUOUS_TRIPLET_" + key + "_" + targetSequence
			newRecord = Statistics(targetSequence, "LOCAL_CONTIGUOUS_TRIPLET_", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
	
	
	def appendDinucleotideFrequency(self, targetSequence):
		### Append dinucleotide frequency feature ###
		print "*** Feature calculation: DINUCLEOTIDE_FREQUENCY (" + targetSequence + ")"
		
		# Initialization of summary array with all possibilities of input (combinations of couple of nucleotides)
		nucleotides = ("A", "T", "G", "C")
		sumVariants = {}
		for nucl1 in nucleotides:
			for nucl2 in nucleotides:
				sumVariants[nucl1+nucl2] = []
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in dataset:
			# Initialization of array with all possibilities of input (combinations of couple of nucleotides)
			variants = {}
			for nucl1 in nucleotides:
				for nucl2 in nucleotides:
					variants[nucl1+nucl2] = 0
				
			# Calculation of frequency for individual variants (combinations of one central nucleotide and corresponding triplet of RNA secondary structures
			adjustedSequence = record.var[targetSequence]
			dinucleotidesCount = 0
			for i in range(len(adjustedSequence)-1):	
				try:
					dinucleotide = adjustedSequence[i:i+2]

					# Triplets with non-standard nucleotides ("N" => any nucleotide etc) will not be evaluated
					flag1 = True if ((dinucleotide[0] == 'A') or (dinucleotide[0] == 'T') or (dinucleotide[0] == 'C') or (dinucleotide[0] == 'G')) else False
					flag2 = True if ((dinucleotide[1] == 'A') or (dinucleotide[1] == 'T') or (dinucleotide[1] == 'C') or (dinucleotide[1] == 'G')) else False
					if((flag1 == True) and (flag2 == True)):
						variants[dinucleotide] += 1
						dinucleotidesCount += 1
				
				except:
					if(DEBUG_FLAG): print "!!! ERROR !!!" + traceback.print_exc()
					#traceback.print_exc()
					#sys.stdin.read(1)
				
			# Normalization of variant occurrencies to reach overall sum of 1.0
			for rec in variants:
				if(variants[rec] > 0):
					variants[rec] = float(variants[rec]) / dinucleotidesCount
				sumVariants[rec].append(variants[rec])
				
				propertyKey = "DINUCLEOTIDE_FREQUENCY_" + rec + "_" + targetSequence
				record.var[propertyKey] = variants[rec]
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for key in sumVariants:
			propertyKey = "DINUCLEOTIDE_FREQUENCY_" + key + "_" + targetSequence
			newRecord = Statistics(targetSequence, "DINUCLEOTIDE_FREQUENCY", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
	
	
	def appendTrinucleotideFrequency(self, targetSequence):
		### Append dinucleotide frequency feature ###
		print "*** Feature calculation: TRINUCLEOTIDE_FREQUENCY (" + targetSequence + ")"
		
		# Initialization of summary array with all possibilities of input (combinations of couple of nucleotides)
		nucleotides = ("A", "T", "G", "C")
		sumVariants = {}
		for nucl1 in nucleotides:
			for nucl2 in nucleotides:
				for nucl3 in nucleotides:
					sumVariants[nucl1+nucl2+nucl3] = []
		
		dataset = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in dataset:
			# Initialization of array with all possibilities of input (combinations of couple of nucleotides)
			variants = {}
			for nucl1 in nucleotides:
				for nucl2 in nucleotides:
					for nucl3 in nucleotides:
						variants[nucl1+nucl2+nucl3] = 0
				
			# Calculation of frequency for individual variants (combinations of one central nucleotide and corresponding triplet of RNA secondary structures
			adjustedSequence = record.var[targetSequence]
			trinucleotidesCount = 0
			for i in range(len(adjustedSequence)-2):
				try:
					dinucleotide = adjustedSequence[i:i+3]

					# Triplets with non-standard nucleotides ("N" => any nucleotide etc) will not be evaluated
					flag1 = True if ((dinucleotide[0] == 'A') or (dinucleotide[0] == 'T') or (dinucleotide[0] == 'C') or (dinucleotide[0] == 'G')) else False
					flag2 = True if ((dinucleotide[1] == 'A') or (dinucleotide[1] == 'T') or (dinucleotide[1] == 'C') or (dinucleotide[1] == 'G')) else False
					flag3 = True if ((dinucleotide[2] == 'A') or (dinucleotide[2] == 'T') or (dinucleotide[2] == 'C') or (dinucleotide[2] == 'G')) else False
					if((flag1 == True) and (flag2 == True)):
						variants[dinucleotide] += 1
						trinucleotidesCount += 1
				
				except:
					if(DEBUG_FLAG): print "!!! ERROR !!!" + traceback.print_exc()
					traceback.print_exc()
					#sys.stdin.read(1)
				
			# Normalization of variant occurrencies to reach overall sum of 1.0
			for rec in variants:
				if(variants[rec] > 0):
					variants[rec] = float(variants[rec]) / trinucleotidesCount
				sumVariants[rec].append(variants[rec])
				
				propertyKey = "TRINUCLEOTIDE_FREQUENCY_" + rec + "_" + targetSequence
				record.var[propertyKey] = variants[rec]
		
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for key in sumVariants:
			propertyKey = "TRINUCLEOTIDE_FREQUENCY_" + key + "_" + targetSequence
			newRecord = Statistics(targetSequence, "TRINUCLEOTIDE_FREQUENCY", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
	
	
	def generateTrainingTestingDatasetNew(self, targetSequence):
		### Divide all records into training and testing part ###
		print "*** NEW Dataset cut (CD-HIT) ... (" + targetSequence + ")"
				
		# Generate FASTA file with all valid sequences for CD-HIT
		#outputFileTrain = open(CONFIG.CDHIT_FOLDER + "tmp/sequencesTrain_" + TARGET_SEQUENCE + ".fasta", "w")
		#outputFileTest = open(CONFIG.CDHIT_FOLDER + "tmp/sequencesTest_" + TARGET_SEQUENCE + ".fasta", "w")
		inputFileAllPath = CONFIG.CDHIT_FOLDER + "tmp/sequences_" + targetSequence + ".fasta"
		inputFileAll = open(inputFileAllPath, "w")
		
		for record in self.datasetAll.itervalues():
			if(len(record.var[targetSequence]) > 0):  # if(self.isParameterDefined(record, CONFIG.TARGET_HL) and self.isParameterDefined(record, targetSequence)):
				inputFileAll.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
				#rand = random.randint(0,1)
				#if(rand == 0):
				#	outputFile1.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
				#else:
				#	outputFile2.write(">" + record.var["GENE_ID"] + "\n" + record.var[targetSequence] + "\n")
		
		inputFileAll.close()
		
		# Run CD-HIT (or check if output file exist and skip calculation)
		outputFileAll = CONFIG.CDHIT_FOLDER + "tmp/output_" + targetSequence + ".clstr"
		print "outputFile: " + outputFileAll + " | " + str(os.path.isfile(outputFileAll))
		if(os.path.isfile(outputFileAll) == False):
			command = CONFIG.CDHIT_BINARY + " -i " + inputFileAllPath + " -c 0.8 -n 4 -M 6000 -o " + outputFileAll + "; echo \">clusterZarazka\" >> " + outputFileAll + ";"
			os.system(command)
			if(CONFIG.DEBUG == True): print "> CD-HIT sucessfully ended."
		else:
			if(CONFIG.DEBUG == True): print "> CD-HIT calculation skipped."
		
		
		# Divide dataset to training and testing subset
		clusterIds = []
		trainingFlag = True
		for line in open(outputFileAll):
			if(line.startswith(">")):
				trainingFlag = True if (trainingFlag == False) else False
				for record in clusterIds:
					self.datasetAll[record].var["TRAINING_FLAG"] = trainingFlag
				clusterIds = []
			elif(len(line.strip()) > 0):
				clusterIds.append(line.split(">")[1].split(".")[0].replace("-",""))
		
		trainingCases = 0
		testingCases = 0
		for record in self.datasetAll:
			if(bool(self.datasetAll[record].var["TRAINING_FLAG"]) == True):
				trainingCases += 1
			else:
				testingCases += 1
		
		print "> TRAINING CASES: " + str(trainingCases)
		print "> TESTING CASES: " + str(testingCases)
		
		#print "> Generate training & testing dataset ... press any key to continue "
		#sys.stdin.read(1)

	def generateTrainingTestingDataset(self, targetSequence):
		### Divide all records into training and testing part ###
		print "*** Dataset cut (binning / tertiary classes) ... (" + targetSequence + ")"
			
		print "generateTrainingTestingDatasetNew(" + targetSequence + ")"
		self.generateTrainingTestingDatasetNew(targetSequence)
			
		# Generate labels training / testing / unknown (for those record without measured HL on target dataset)
		datasetTraining = [float(item.var[CONFIG.TARGET_HL]) for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and (item.var["TRAINING_FLAG"] == True))]
		mean = numpy.mean(datasetTraining)		
		thresholdTop = mean * 2	
		thresholdBottom = mean / 2
		
		if(LOG_FLAG):
			mean = numpy.mean([math.pow(item,2) for item in datasetTraining])
			thresholdTop = math.log((mean * 2), 2)
			thresholdBottom = math.log((mean / 2), 2)
		
		# Generate labels stable / unstable / twilight
		for record in self.datasetAll.itervalues():
			if(self.isParameterDefined(record, CONFIG.TARGET_HL) and self.isParameterDefined(record, targetSequence)):
				if(float(record.var[CONFIG.TARGET_HL]) < thresholdBottom):
					record.var["STABILITY_TERTIARY"] = RECORD_STATUS_CATEGORY.UNSTABLE
				elif(float(record.var[CONFIG.TARGET_HL]) > thresholdTop):
					record.var["STABILITY_TERTIARY"] = RECORD_STATUS_CATEGORY.STABLE
				else:
					record.var["STABILITY_TERTIARY"] = RECORD_STATUS_CATEGORY.TWILIGHT
			else:
				record.var["STABILITY_TERTIARY"] = ""
		
		if(DEBUG_FLAG): print "> thresholdTop: " + str(thresholdTop) + " ... log ? " + str(LOG_FLAG)
		if(DEBUG_FLAG): print "> thresholdBottom: " + str(thresholdBottom) + " ... log ? " + str(LOG_FLAG)
		if(DEBUG_FLAG): print "> len(stable): " + str(len([item for item in self.datasetAll.itervalues() if (item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.STABLE)]))
		if(DEBUG_FLAG): print "> len(unstable): " + str(len([item for item in self.datasetAll.itervalues() if (item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.UNSTABLE)]))
		if(DEBUG_FLAG): print "> len(twilight): " + str(len([item for item in self.datasetAll.itervalues() if (item.var["STABILITY_TERTIARY"] == RECORD_STATUS_CATEGORY.TWILIGHT)]))
		
		# Generate bins for categories of HL
		thresholds = []
		if(DEBUG_FLAG): print "> Binning distribution"
		for i in range(CONFIG.BINS_COUNT):
			thresholds.append(scipy.stats.scoreatpercentile(datasetTraining, (i * 10)))
			if(DEBUG_FLAG):print ">> i " + str(i) + " \t" + str(thresholds[i])
				
		#dataset = [float(item.var[CONFIG.TARGET_HL]) for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence))]
		for record in self.datasetAll.itervalues():
			if(self.isParameterDefined(record, CONFIG.TARGET_HL) and self.isParameterDefined(record, targetSequence)):
				record.var["STABILITY_BIN"] = 0
				for i in range(CONFIG.BINS_COUNT):
					threshold = thresholds[CONFIG.BINS_COUNT-1-i]
					if(float(record.var[CONFIG.TARGET_HL]) >= threshold):
						record.var["STABILITY_BIN"] = CONFIG.BINS_COUNT-1-i
						break
			else:
				record.var["STABILITY_BIN"] = 5
		#sys.stdin.read(1)
		
	
	
	def loadDinucleotideShuffling(self, targetSequence):
		# Load informations about 
		
		print "*** Feature extraction from files: RNA FREE ENERGY - SHUFFLING(" + targetSequence + ")"
		fileMappingIndex = {"UPSTREAM" : "0", "DOWNSTREAM" : "1", "UPSTREAM_1KB" : "2", "DOWNSTREAM_1KB" : "3", "DNA_SEQUENCE" : "4", "CONCAT" : "5", "CONCAT_2015" : "6", "TRANSCRIPT" : "7"}
		inputFile = "./results/results_DINUCLEOTIDE_SHUFFLING.txt_" + str(fileMappingIndex[targetSequence])
		
		for line in open(inputFile):
			if(line.startswith("IDENTIFIER") or (len(line.strip()) == 0) or (len(line.split(",")) != 8)):
				continue
			
			items = line.split(",")
			identifierShortened = items[0].strip().replace("-","")
			for columnName in CONFIG.RNA_FILE_COLUMNS.iterkeys():
				if(columnName == "PREDICTED_STRUCTURE"):
					continue
				propertyKey = "SHUFFLING_" + columnName + "_" + targetSequence
				if(identifierShortened in self.datasetAll):
					self.datasetAll[identifierShortened].var[propertyKey] = items[int(CONFIG.RNA_FILE_COLUMNS[columnName])].strip()
	
	
	def appendDinucleotideShuffling(self, targetSequence):
		### Append values of minimum of free energy of the secondary structure predicted by RNAfold ###
		
		for key in CONFIG.RNA_FILE_COLUMNS.iterkeys():
			if(key == "PREDICTED_STRUCTURE"):
				continue
			propertyKey = "SHUFFLING_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, propertyKey))]
			newRecord = Statistics(targetSequence, "SHUFFLING", key)
			
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
			
			# Normalized (divided by sequence length)
			propertyKeyNormalized = "SHUFFLING_" + key + "_NORM_" + targetSequence
			for record in selectedRecordsVector:
				record.var[propertyKeyNormalized] = float(record.var[propertyKey]) / len(record.var[targetSequence])
			newRecord = Statistics(targetSequence, "SHUFFLING", key + "_NORM")
			newRecord.calculateStatistics(selectedRecordsVector, propertyKeyNormalized)
			STATISTICS.append(newRecord)
	
	
	def loadMfe(self, targetSequence):
		# Load informations about 
		
		print "*** Feature extraction from files: RNA FREE ENERGY (" + targetSequence + ")"
		#fileMappingIndex = {"0" : "UPSTREAM", "1" : "DOWNSTREAM", "2" : "UPSTREAM_1KB", "3" : "DOWNSTREAM_1KB", "4" : "DNA_SEQUENCE", "5" : "CONCAT", "6" : "CONCAT_2015", "7" : "TRANSCRIPT"}
		fileMappingIndex = {"UPSTREAM" : "0", "DOWNSTREAM" : "1", "UPSTREAM_1KB" : "2", "DOWNSTREAM_1KB" : "3", "DNA_SEQUENCE" : "4", "CONCAT" : "5", "CONCAT_2015" : "6", "TRANSCRIPT" : "7"}
		inputFile = "./results/results_DINUCLEOTIDE_SHUFFLING_2.txt_" + str(fileMappingIndex[targetSequence])
		
		for line in open(inputFile):
			if(line.startswith("IDENTIFIER") or (len(line.strip()) == 0) or (len(line.split(",")) != 9)):
				continue
			
			items = line.split(",")
			identifierShortened = items[0].strip().replace("-","")
			for columnName in CONFIG.RNA_FILE_COLUMNS.iterkeys():
				propertyKey = "MFE_" + columnName + "_" + targetSequence
				if(identifierShortened in self.datasetAll):
					self.datasetAll[identifierShortened].var[propertyKey] = items[int(CONFIG.RNA_FILE_COLUMNS[columnName])].strip()
	
	
	def appendRnaHeat(self, targetSequence):
		# Load informations about 
		
		print "*** Feature extraction from files: RNA HEAT (" + targetSequence + ")"
		fileMappingIndex = {"DNA_SEQUENCE" : "0", "TRANSCRIPT" : "1","UPSTREAM" : "2", "DOWNSTREAM" : "3"}
		inputFile = "./results/results_RNAHEAT.txt_HEAT_" + str(fileMappingIndex[targetSequence])
		
		for line in open(inputFile):
			if(len(line.split(",")) != 204):
				continue
			
			items = line.split(",")
			identifierShortened = items[0].strip().replace("-","")
			
			#meltingCurve = copy.deepcopy([])
			maxDegree = 0
			maxHeat = 0 
			for i in range(101):
				temperature = float(items[i*2+2])
				#meltingCurve.append(temperature)
				if((i > 15) and (temperature > maxHeat)):
					maxDegree = i
					maxHeat = temperature
			
			self.datasetAll[identifierShortened].var["RNAHEAT_HEAT_" + targetSequence] = maxHeat
			self.datasetAll[identifierShortened].var["RNAHEAT_DEGREE_" + targetSequence] = maxDegree
			
			for columnName in CONFIG.RNA_FILE_COLUMNS.iterkeys():
				propertyKey = "RNAHEAT_" + columnName + "_" + targetSequence
				self.datasetAll[identifierShortened].var[propertyKey] = items[int(CONFIG.RNA_FILE_COLUMNS[columnName])].strip()
		
		# Normalized (divided by sequence length)
		propertyName = "RNAHEAT_HEAT_" + targetSequence
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "RNAHEAT_HEAT_" + targetSequence))]
		newRecord = Statistics(targetSequence, "RNAHEAT_HEAT", "")
		newRecord.calculateStatistics(selectedRecordsVector, "RNAHEAT_HEAT_" + targetSequence)
		STATISTICS.append(newRecord)
		selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, "RNAHEAT_DEGREE_" + targetSequence))]
		newRecord = Statistics(targetSequence, "RNAHEAT_DEGREE", "")
		newRecord.calculateStatistics(selectedRecordsVector, "RNAHEAT_DEGREE_" + targetSequence)
		STATISTICS.append(newRecord)
	
	
	def appendRnaPssmAlt(self, targetSequence):
		print "*** GENERATE PSSM MATRICES"
		fileMappingIndex = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		
		for fileName in os.listdir(CONFIG.RNA_RNAZ_INPUT_FOLDER):
			if(fileName.endswith(".aln") == False):
				continue
			
			inputFileName = CONFIG.RNA_RNAZ_INPUT_FOLDER + fileName
			print "> PROCESSING FILE: " + inputFileName
			
			version = fileName.split("_")[1][0]
			print "> version: " + version + " / " + fileMappingIndex[targetSequence]
			
			if(version != fileMappingIndex[targetSequence]):
				continue
			
			alnMatrix = {}
			for line in open(inputFileName):
				if(line.startswith("sac")):
					items = re.split(' +', line.strip())
					identifier = items[0].strip()
					sequence = items[1].strip()
					alnMatrix[identifier] = sequence
			
			if(len(alnMatrix) == 0):
				continue
			
			outputText = ""
			pssmOutputFilePath = CONFIG.RNA_PSSM_RESULT_FOLDER + fileName.split(".")[0] + ".pssm"
			pssmOutputPath = open(pssmOutputFilePath, "w")
			
			for i in range(len(alnMatrix["sacCer3"])):
				nucl = {"A" : 0, "T" : 0, "C" : 0, "G" : 0, "-" : 0, "N" : 0}
				for org in alnMatrix:
					nucl[alnMatrix[org][i]] += 1
				outline = alnMatrix["sacCer3"][i] + "  " + str(nucl["A"]) + " " + str(nucl["T"]) + " " + str(nucl["G"]) + " " + str(nucl["C"]) + " " + str(nucl["-"]) + "\n"
				pssmOutputPath.write(outline)
					
			pssmOutputPath.close()
		
		sys.stdin.read(1)
		
	
	
	def appendRnaPssmBlast(self, targetSequence):
		print "*** Feature extraction from files: RNA PSSM BLAST (" + targetSequence + ")"
		fileMapping = {"1" : "UPSTREAM", "2" : "DOWNSTREAM", "5" : "DNA_SEQUENCE", "6" : "TRANSCRIPT"}
		fileMapping2 = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		
		#i = 0
		#for fileName in os.listdir(CONFIG.RNA_PSSM_BLAST_FOLDER):	
		#	fullPath = CONFIG.RNA_PSSM_BLAST_FOLDER + fileName
		#	if(fullPath.endswith("blast") == False):
		#		continue
		#	command = "echo \">ref|ARTIFICIAL| Saccharomyces cerevisiae\n\" >> " + fullPath + ";"
		#	i += 1
		#	print "> " + str(i)
		#	os.system(command)
		#print "Zarazky pridany :-)"
		#sys.stdin.read(1)
		
		validBlocksList = []
		invalidBlocksList = []
		zeroHl = []
		zeroHl2 = []
		
		i = 0
		for fileName in os.listdir(CONFIG.RNA_PSSM_BLAST_FOLDER):			
			if(fileName.find("_" + fileMapping2[targetSequence]) < 0):
				continue
			
			blastFilePath = CONFIG.RNA_PSSM_BLAST_FOLDER + fileName
			if(blastFilePath.endswith("blast") == False):
				continue
			
			identifier = fileName.split(".")[0]
			identifierShort = identifier.split("_")[0].strip()
			
			print "> Probiha filtrovani souboru: " + blastFilePath
			outputBlastFilePath = CONFIG.RNA_PSSM_BLAST_FOLDER + identifier + ".blastNewSacOnly"
			
			outputBlastFile = open(outputBlastFilePath, "w")
			
			block = ""
			blockValidity = False
			currentIdentity = 0
			validBlocks = 0
			invalidBlocks = 0
			beforeStart = True
			
			for line in open(blastFilePath):
				if(beforeStart):
					outputBlastFile.write(line)
					
				if(line.startswith(">")):
					# TODO: NEW BLOCK
					beforeStart = False
					if(blockValidity):
						outputBlastFile.write(block)
						validBlocks += 1
					else:
						invalidBlocks += 1
						
					block  = ""
					blockValidity = True
					
					if(line.find("Saccharomyces cerevisiae") >= 0):
						blockValidity = False
					if(line.find("Saccharomyces") < 0):
						blockValidity = False
				
				if(line.find("Expect") >= 0):
					expect = float(line.split("Expect = ")[1].strip())
					if(expect > 0.001):
						blockValidity = False
					#sys.stdin.read(1)
				else:
					if(line.startswith(" Identities =") == True):
						currentIdentity = int(line.split("(")[1].split("%")[0])
						#print ">> Current identity: " + str(currentIdentity)
						if(currentIdentity >= 100):
							blockValidity = False
				block += line
				
			outputBlastFile.close()
			print "SUMMARY"
			print ">> validBlocks: " + str(validBlocks)
			print ">> invalidBlocks: " + str(invalidBlocks)
			#sys.stdin.read(1)
			validBlocksList.append(validBlocks)
			
			if(((identifierShort in self.datasetAll) == True) and (validBlocks == 0) and (len(self.datasetAll[identifierShort].var["PRESNYAK_HL_TOTAL"]) > 0)):
				print "> self.datasetAll[" + identifierShort + "].var[" + CONFIG.TARGET_HL + "] = " + str(self.datasetAll[identifierShort].var[CONFIG.TARGET_HL])
				zeroHl.append(float(self.datasetAll[identifierShort].var["PRESNYAK_HL_TOTAL"]))
				print identifierShort
				#sys.stdin.read(1)
			#elif(((identifierShort in self.datasetAll) == True) and (len(self.datasetAll[identifierShort].var["PRESNYAK_HL_TOTAL"]) > 0)):
			#	print "a"
			#	zeroHl.append(float(self.datasetAll[identifierShort].var["PRESNYAK_HL_TOTAL"]))
			
			invalidBlocksList.append(invalidBlocks)
			#sys.stdin.read(1)
			
			sequenceFilePath = CONFIG.TMP_FOLDER + "sequence.fasta"
			sequenceFile = open(sequenceFilePath, "w")
			if((identifierShort in self.datasetAll) == False):
				continue
			sequenceFile.write(">" + identifier + "\n" + self.datasetAll[identifierShort].var[fileMapping[identifier.split("_")[1].strip()]])
			sequenceFile.close()
			
			i += 1
			print "> Iterace " + str(i)
			
			outputPssmFilePath = CONFIG.RNA_PSSM_RESULT_FOLDER + identifier + ".pssm"
			command = CONFIG.PSSM_RNA + " " + sequenceFilePath + " " + outputBlastFilePath + " > " + outputPssmFilePath
			print "> Command: " + command
			os.system(command)
			#sys.stdin.read(1)
		
		print "> LENGTH: " + str(len(validBlocksList))
		print "> zeros   (" + targetSequence + "): " + str(len([x for x in validBlocksList if(x == 0)]))
		print "> validBlocksMean   (" + targetSequence + "): " + str(numpy.mean(validBlocksList))
		print "> invalidBlocksMean (" + targetSequence + "): " + str(numpy.mean(invalidBlocksList))
		print "> zeroAvg (" + targetSequence + ") / " + str(len(zeroHl)) + " ... " + str(zeroHl)
		print ">>: " + str(numpy.mean(zeroHl))
		
		sys.stdin.read(1)
		
		
	
	def appendRnaPssm(self, targetSequence):
		# Load informations about RNA PSSM
		
		print "*** Feature extraction from files: RNA PSSM (" + targetSequence + ")"
		
		fileMapping = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		percentiles = {"p20" : [0], "p40" : [0], "p60" : [0], "p80" : [0]}
		columnMapping = {"A" : 1, "T" : 2, "G" : 3, "C" : 4}
		index = 0
		
		xSuma = {}
		xInsufficientSuma = {}
		ok = 1
		notOk = 1
		
		# Calculate average RNA PSSM conservation & entrophy
		for fileName in os.listdir(CONFIG.RNA_PSSM_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().split("_")[0]
			version = fileName.split(".")[0].strip().split("_")[1]
			
			if(version != fileMapping[targetSequence]):  # process only files for currently analyzed type of sequence (transcript /  upstream / ...)
				continue
			
			print "> CONSERVATION / " + targetSequence + " / " + identifier + " / " + str(index)
			index += 1
			conservationVector = []
			entrophyVector = []
			error = False     # usually in case of PSSM file containing rows with sum equal to 0 (no homologs - short sequences)
			
			
			print "> IDENTIFIER: " + identifier
			if(((identifier in self.datasetAll) == False)):
				continue
			
			# Loop for every valid file 
			index2 = 0
			for line in open(CONFIG.RNA_PSSM_RESULT_FOLDER + fileName):
				items = re.split(' +', line.strip())
				currentNucleotide = items[0].strip()
					
				suma = 0
				for i in range(4):
					suma += int(items[i+1])
				
				if(suma == 0):
					error = True
					break
				
				entrophyValue = 0
				for i in range(4):
					if(int(items[i+1]) > 0):
						entrophyValue -= (float(items[i+1]) / suma) * math.log(float(items[i+1]) / suma, 2)
				
				#if(suma >= 5): # pozor na zmenu cislovani pak
				#	ok += 1
				#else:
				#	notOk += 1
				#	index2 += 1
				#	continue
				
				if((identifier in xSuma) == False):
					xSuma[identifier] = 0
				else:
					if(suma < 10):
						xSuma[identifier] += 1
					
				# Data consistency check				
				sequenceLen = len(self.datasetAll[identifier].var[targetSequence])
				structureLen = len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])
				
				if((sequenceLen != structureLen)):
					print identifier
					continue
			
				avgSecStr = {"UNBOUNDED" : [], "BOUNDED" : []}
				secStructure = "BOUNDED" if(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence][index2] == ".") else "UNBOUNDED"
				avgSecStr[secStructure].append(entrophyValue)
				#print "avgSecStr[" + secStructure + "].append(" + str(entrophyValue) + ")"
				#if(floatAsa[i] > maxi[self.datasetAll[identifier].var[targetSequence][i]]):
				#	maxi[self.datasetAll[identifier].var[targetSequence][i]] = floatAsa[i]
						
				if(len(avgSecStr["UNBOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_PSSM_ENTROPHY_UNBOUNDED_" + targetSequence] = numpy.mean(avgSecStr["UNBOUNDED"])
				if(len(avgSecStr["BOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_PSSM_ENTROPHY_BOUNDED_" + targetSequence] = numpy.mean(avgSecStr["BOUNDED"])
				
				conservationValue = float(items[columnMapping[currentNucleotide]]) / suma
				
				conservationVector.append(conservationValue)  # vector of conservation values
				entrophyVector.append(entrophyValue)          # vector of entrophies
				index2 += 1
			
			if(error):
				continue
			
			# Data consistency check
			sequenceLen = len(self.datasetAll[identifier].var[targetSequence])
			pssmLen = len(entrophyVector)
			
			#if(sequenceLen != pssmLen):
			#	print "> ERROR: IDENTIFIER: " + identifier + " > " + str(sequenceLen) + " =VS= " + str(pssmLen)
			#	#sys.stdin.read(1)
			#	continue
			
			# Set threshold percentiles of distribution
			percentileValue = scipy.stats.scoreatpercentile(entrophyVector, 20)
			if(percentileValue > 0): percentiles["p20"].append(percentileValue)
			percentileValue = scipy.stats.scoreatpercentile(entrophyVector, 40)
			if(percentileValue > 0): percentiles["p40"].append(percentileValue)
			percentileValue = scipy.stats.scoreatpercentile(entrophyVector, 60)
			if(percentileValue > 0): percentiles["p60"].append(percentileValue)
			percentileValue = scipy.stats.scoreatpercentile(entrophyVector, 80)
			if(percentileValue > 0): percentiles["p80"].append(percentileValue)
			
			self.datasetAll[identifier].var["RNA_PSSM_CONS_" + targetSequence] = numpy.mean(conservationVector)
			self.datasetAll[identifier].var["RNA_PSSM_ENTROPHY_" + targetSequence] = numpy.mean(entrophyVector)
				
		#print ">> OK: " + str(ok) + "\t" + str(float(ok)/(ok+notOk))
		#print ">> NOT_OK: " + str(notOk) + "\t" + str(float(notOk)/(ok+notOk))
		sys.stdin.read(1)
		
		newVector = []
		for rec in xSuma:
			newVector.append(float(xSuma[rec]) / len(self.datasetAll[identifier].var[targetSequence]))
				
		# Set percentile thresholds for entrophy
		percentiles["p20"] = numpy.mean(percentiles["p20"])
		percentiles["p40"] = numpy.mean(percentiles["p40"])
		percentiles["p60"] = numpy.mean(percentiles["p60"])
		percentiles["p80"] = numpy.mean(percentiles["p80"])
		
		# Calculate prediction power on thesholds set on percentiles
		index = 0
		for fileName in os.listdir(CONFIG.RNA_PSSM_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().split("_")[0]
			version = fileName.split(".")[0].strip().split("_")[1]
			
			if(version != fileMapping[targetSequence]):
				continue
			
			if(((identifier in self.datasetAll) == False)):
				continue
			
			for percentile in percentiles:
				propertyKey = "RNA_PSSM_" + percentile + "_" + targetSequence
				self.datasetAll[identifier].var[propertyKey] = 0
					
			# Calculate ratio of RNA's nucleotides with entrophy over threshold
			error = False
			for line in open(CONFIG.RNA_PSSM_RESULT_FOLDER + fileName):
				items = re.split(' +', line.strip())
				currentNucleotide = items[0].strip()
					
				suma = 0
				for i in range(4):
					suma += int(items[i+1])
				if(suma == 0):
					error = True
					break
				
				entrophyValue = 0
				for i in range(4):
					if(int(items[i+1]) > 0):
						entrophyValue -= (float(items[i+1]) / suma) * math.log(float(items[i+1]) / suma, 2)
				
				for threshold in percentiles:
					if(entrophyValue > percentiles[threshold]):
						propertyKey = "RNA_PSSM_" + threshold + "_" + targetSequence
						self.datasetAll[identifier].var[propertyKey] += 1
						#print str(entrophyValue) + " > " + str(percentiles[threshold])
						#sys.stdin.read(1)
			
			for threshold in percentiles:
				propertyKey = "RNA_PSSM_" + threshold + "_" + targetSequence
				self.datasetAll[identifier].var[propertyKey] = float(self.datasetAll[identifier].var[propertyKey]) / len(self.datasetAll[identifier].var[targetSequence])
			
			if(error):   # usually in case of PSSM file containing rows with sum equal to 0 (no homologs - short sequences)
				continue
			
			index += 1
			print "> CONSERVATION / " + targetSequence + " / " + str(index)
		
		sumVariants = ["CONS", "ENTROPHY", "p20", "p40", "p60", "p80", "ENTROPHY_BOUNDED", "ENTROPHY_UNBOUNDED"]		
		for key in sumVariants:
			propertyKey = "RNA_PSSM_" + key + "_" + targetSequence
			print "> propertyKey: " + propertyKey
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence) and self.isParameterDefined(item, propertyKey))]
			print "> len(selectedRecordsVector): " + str(len(selectedRecordsVector))
			newRecord = Statistics(targetSequence, "RNA_PSSM", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
	
	
	def appendRnaAsa(self, targetSequence):
		# Load informations about RNA ASA
		
		print "*** Feature extraction from files: RNA ASA (" + targetSequence + ")"
		
		fileMapping = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		percentiles = {"p20" : [0], "p40" : [0], "p60" : [0], "p80" : [0]}
		errors = []
		
		# Calculate average RNA ASA conservation
		for fileName in os.listdir(CONFIG.RNA_ASA_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().split("_")[0]
			version = fileName.split(".")[0].strip().split("_")[1]
			
			if(version != fileMapping[targetSequence]):  # process only files for currently analyzed type of sequence (transcript /  upstream / ...)
				continue
			
			# Load disorder predictions
			floatAsa = []
			for line in open(CONFIG.RNA_ASA_RESULT_FOLDER + fileName):
				#print "> LINE: " + line + "\n\n"
				if((len(line.strip()) == 0) or (line.startswith(">"))):
					continue
				else:
					items = re.split(' +', line.strip())
					for item in items:
						floatAsa.append(float(item))
		
			# Data consistency check
			sequenceLen = len(self.datasetAll[identifier].var[targetSequence])
			structureLen = len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])
			asaLen = len(floatAsa)
			
			if((sequenceLen != structureLen) or (sequenceLen != asaLen) or (structureLen != asaLen)):
				errors.append(identifier)
				continue
			
			# Calculate average ASA score
			propertyKey = "RNA_ASA_AVERAGE_" + targetSequence
			self.datasetAll[identifier].var[propertyKey] = numpy.mean(floatAsa)
			
			# Calculate average disorder for different elements of secondary structure
			avgSecStr = {"UNBOUNDED" : [], "BOUNDED" : []}
			for i in range(len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])):
				secStructure = "BOUNDED" if(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence][i] == ".") else "UNBOUNDED"
				avgSecStr[secStructure].append(floatAsa[i])
					
			if(len(avgSecStr["UNBOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_ASA_AVERAGE_UNBOUNDED_" + targetSequence] = numpy.mean(avgSecStr["UNBOUNDED"])
			if(len(avgSecStr["BOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_ASA_AVERAGE_BOUNDED_" + targetSequence] = numpy.mean(avgSecStr["BOUNDED"])
		
		sumVariants = ["AVERAGE", "AVERAGE_UNBOUNDED", "AVERAGE_BOUNDED"]
		for key in sumVariants:
			propertyKey = "RNA_ASA_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence)) and self.isParameterDefined(item, propertyKey)]
			newRecord = Statistics(targetSequence, "RNA_ASA", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
		
		if(len(errors) > 0):
			print "> " + str(errors)
			sys.stdin.read(1)
	
	
	def appendRnaDisorder(self, targetSequence):
		# Load informations about RNA DISORDER
		
		print "*** Feature extraction from files: RNA DISORDER (" + targetSequence + ")"
		
		fileMapping = {"UPSTREAM" : "1", "DOWNSTREAM" : "2", "DNA_SEQUENCE" : "5", "TRANSCRIPT" : "6"}
		percentiles = {"p20" : [0], "p40" : [0], "p60" : [0], "p80" : [0]}
		index = 0
		errors = []
		
		# Calculate average RNA PSSM conservation
		for fileName in os.listdir(CONFIG.RNA_DISORDER_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().split("_")[0]
			version = fileName.split(".")[0].strip().split("_")[1]
			
			if(version != fileMapping[targetSequence]):  # process only files for currently analyzed type of sequence (transcript /  upstream / ...)
				continue
			
			# Load disorder predictions
			itemsRaw = None
			try:
				itemsRaw = open(CONFIG.RNA_DISORDER_RESULT_FOLDER + fileName,"r").readlines()[0].strip().split("\t")
			except:
				continue
			
			# Data consistency check
			itemsFloat = [float(x) for x in itemsRaw]
			sequenceLen = len(self.datasetAll[identifier].var[targetSequence])
			structureLen = len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])
			disorderLen = len(itemsFloat)
			
			if((sequenceLen != structureLen) or (sequenceLen != disorderLen) or (structureLen != disorderLen)):
				errors.append(identifier + "_" + str(version))
				print str(identifier) + "-" + str(sequenceLen) + "-" + str(structureLen) + "-" + str(disorderLen)
				continue
			
			# Calculate average disorder score
			propertyKey = "RNA_DISORDER_AVERAGE_" + targetSequence
			self.datasetAll[identifier].var[propertyKey] = numpy.mean(itemsFloat)
			
			# Calculate average disorder for different elements of secondary structure
			avgSecStr = {"UNBOUNDED" : [], "BOUNDED" : []}
			for i in range(len(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence])):
				secStructure = "BOUNDED" if(self.datasetAll[identifier].var["MFE_PREDICTED_STRUCTURE_" + targetSequence][i] == ".") else "UNBOUNDED"
				positionDisorderProbability = itemsFloat[i]
				avgSecStr[secStructure].append(positionDisorderProbability)
			
			#if(CONFIG.DEBUG == True): print ">> SEQUENCE  = " + str(self.datasetAll[identifier].var["PROTEIN_SEQUENCE"])
			#if(CONFIG.DEBUG == True): print ">> STRUCTURE = " + str(self.datasetAll[identifier].var["PROTEIN_SS"]) 
			#if(CONFIG.DEBUG == True): print ">> lenSecStr[UNBOUNDED] = " + str(len(avgSecStr["UNBOUNDED"]))
			#if(CONFIG.DEBUG == True): print ">> lenSecStr[BOUNDED] = " + str(len(avgSecStr["BOUNDED"]))
			#if(CONFIG.DEBUG == True and len(avgSecStr["UNBOUNDED"]) > 0): print ">> avgSecStr[UNBOUNDED] = " + str(numpy.mean(avgSecStr["UNBOUNDED"]))
			#if(CONFIG.DEBUG == True and len(avgSecStr["BOUNDED"]) > 0): print ">> avgSecStr[BOUNDED] = " + str(numpy.mean(avgSecStr["BOUNDED"]))
			
			if(len(avgSecStr["UNBOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_DISORDER_AVERAGE_UNBOUNDED_" + targetSequence] = numpy.mean(avgSecStr["UNBOUNDED"])
			if(len(avgSecStr["BOUNDED"]) > 0): self.datasetAll[identifier].var["RNA_DISORDER_AVERAGE_BOUNDED_" + targetSequence] = numpy.mean(avgSecStr["BOUNDED"])
			
		sumVariants = ["AVERAGE", "AVERAGE_UNBOUNDED", "AVERAGE_BOUNDED"]
		for key in sumVariants:
			propertyKey = "RNA_DISORDER_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, targetSequence)) and self.isParameterDefined(item, propertyKey)]
			print "> LEN(" + propertyKey + ") = " + str(len(selectedRecordsVector)) 
			newRecord = Statistics(targetSequence, "RNA_DISORDER", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
		
		if(len(errors) > 0):
			print "> " + str(errors)
			for case in errors:
				print "cd /auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data/" + case + "; rm Y*.*  Y*_100* Y*_400* Y*_nu15;"
				print "./Feature.sh /auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data/" + case
				print "cp /auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data/" + case + "/" + case + ".s /auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data/general/" 
				print "cp /auto/brno2/home/bendl/halfLifeFeatureTester/disorder/data/" + case + "/" + case + ".s /auto/brno2/home/bendl/halfLifeFeatureTester/general2/\n" 
			sys.stdin.read(1)
		
	def appendPsipredSecondaryStructure(self):
		# Load predictions of secondary structure
		
		print "\n*** Feature extraction from files: PSIPRED_PREDICTIONS"
		
		for fileName in os.listdir(CONFIG.PSIPRED_RESULT_FOLDER):
			identifier = fileName.strip().split(".")[0].replace("-","")
			
			secondaryStructure = ""
			for line in open(CONFIG.PSIPRED_RESULT_FOLDER + fileName):
				if((len(line.strip()) == 0) or (line.startswith("#"))):
					continue
				items = re.split(' +', line.strip())
				secondaryStructure += items[2]
			self.datasetAll[identifier].var["PROTEIN_SS"] = secondaryStructure
				
		print ">> PSIPRED calculations loaded records)"
		
	
	def appendSpinedDisorder(self):
		# Load informations about PROTEIN DISORDER (calculated by SPINE-D)
		
		print "\n*** Feature extraction from files: SPINED_DISORDER"
		errors = []
		
		# Calculate average protein disorder
		for fileName in os.listdir(CONFIG.SPINED_RESULT_FOLDER):
			identifier = fileName.split(".")[0].strip().replace("-","")
			
			# Load protein disorder predictions calculated by SPINE-D
			itemsFloat = []
			for line in open(CONFIG.SPINED_RESULT_FOLDER + fileName):
				items = re.split(' +', line.strip())
				itemsFloat.append(float(items[2]))
			
			# Calculate average disorder score
			propertyKey = "SPINED_DISORDER_AVERAGE"
			self.datasetAll[identifier].var[propertyKey] = numpy.mean(itemsFloat)
			#print "> mean (" + identifier + "): " + str(self.datasetAll[identifier].var[propertyKey])
			
			# Calculate average disorder for different elements of secondary structure
			avgSecStr = {"H": [], "E" : [], "C" : []}
			for i in range(len(self.datasetAll[identifier].var["PROTEIN_SS"])):
				secStructure = self.datasetAll[identifier].var["PROTEIN_SS"][i]
				positionDisorderProbability = itemsFloat[i]
				avgSecStr[secStructure].append(positionDisorderProbability)
			
			# Data consistency check
			sequenceLen = len(self.datasetAll[identifier].var[SEQUENCE_AMINO_TYPE.PROTEIN_SEQUENCE])
			psipredLen = len(self.datasetAll[identifier].var["PROTEIN_SS"])
			spinedLen = len(itemsFloat)
			
			if((sequenceLen != psipredLen) or (sequenceLen != spinedLen) or (psipredLen != spinedLen)):
				errors.append(identifier)
				continue
			
			#if(CONFIG.DEBUG == True): print ">> SEQUENCE  = " + str(self.datasetAll[identifier].var["PROTEIN_SEQUENCE"])
			#if(CONFIG.DEBUG == True): print ">> STRUCTURE = " + str(self.datasetAll[identifier].var["PROTEIN_SS"]) 
			#if(CONFIG.DEBUG == True): print ">> lenSecStr[H] = " + str(len(avgSecStr["H"]))
			#if(CONFIG.DEBUG == True): print ">> lenSecStr[E] = " + str(len(avgSecStr["E"]))
			#if(CONFIG.DEBUG == True): print ">> lenSecStr[C] = " + str(len(avgSecStr["C"])) + "\n"
			#if(CONFIG.DEBUG == True and len(avgSecStr["H"]) > 0): print ">> avgSecStr[H] = " + str(numpy.mean(avgSecStr["H"]))
			#if(CONFIG.DEBUG == True and len(avgSecStr["E"]) > 0): print ">> avgSecStr[E] = " + str(numpy.mean(avgSecStr["E"]))
			#if(CONFIG.DEBUG == True and len(avgSecStr["C"]) > 0): print ">> avgSecStr[C] = " + str(numpy.mean(avgSecStr["C"]))
			
			if(len(avgSecStr["H"]) > 0): self.datasetAll[identifier].var["SPINED_DISORDER_AVERAGE_H"] = numpy.mean(avgSecStr["H"])
			if(len(avgSecStr["E"]) > 0): self.datasetAll[identifier].var["SPINED_DISORDER_AVERAGE_E"] = numpy.mean(avgSecStr["E"])
			if(len(avgSecStr["C"]) > 0): self.datasetAll[identifier].var["SPINED_DISORDER_AVERAGE_C"] = numpy.mean(avgSecStr["C"])
			
		sumVariants = ["AVERAGE", "AVERAGE_H", "AVERAGE_E", "AVERAGE_C"]		
		for key in sumVariants:
			propertyKey = "SPINED_DISORDER_" + key
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, SEQUENCE_AMINO_TYPE.PROTEIN_SEQUENCE)) and self.isParameterDefined(item, propertyKey)]
			newRecord = Statistics(SEQUENCE_AMINO_TYPE.PROTEIN_SEQUENCE, "SPINED_DISORDER", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
		
		if(len(errors) > 0):
			print "***** SPINE-D ERRORS *****\n" + str(errors)
		
	
	def appendMfe(self, targetSequence):
		### Append values of minimum of free energy of the secondary structure predicted by RNAfold ###
		
		for key in CONFIG.RNA_FILE_COLUMNS.iterkeys():		
			if(key == "PREDICTED_STRUCTURE"):
				continue
			propertyKey = "MFE_" + key + "_" + targetSequence
			selectedRecordsVector = [item for item in self.datasetAll.itervalues() if (self.isParameterDefined(item, CONFIG.TARGET_HL) and self.isParameterDefined(item, propertyKey))]
			newRecord = Statistics(targetSequence, "MFE", key)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKey)
			STATISTICS.append(newRecord)
			
			# Normalized (divided by sequence length)
			propertyKeyNormalized = "MFE_" + key + "_NORM_" + targetSequence
			for record in selectedRecordsVector:
				record.var[propertyKeyNormalized] = float(record.var[propertyKey]) / len(record.var[targetSequence])
			newRecord = Statistics(targetSequence, "MFE", key + "_NORM_" + targetSequence)
			newRecord.calculateStatistics(selectedRecordsVector, propertyKeyNormalized)
			STATISTICS.append(newRecord)
	
	
	def calculateCorrelations(self):
		print "> dataset (all):      " + str(len(self.datasetAll))
		print "> dataset (training): " + str(len(self.datasetTraining))
		print "> dataset (testing):  " + str(len(self.datasetTesting))
		
		#data = recfromcsv(dataset1, names = True)
		#xvars = ['exp','exp_sqr','wks','occ','ind','south','smsa','ms','union','ed','fem','blk']
		#y = data['lwage']
		#X = data[xvars]
		#c = ones_like(data['lwage'])
		#X = add_field(X, 'constant', c)
		
		for attr in CONFIG.ATTR_STATISTICS:
			print "***** CORRELATION WITH ATTRIBUTE " + attr + " / " + str(CONFIG.ATTR[attr]) +  " *****"
			vectorTraining1 = []
			vectorTraining2 = []
			
			dataset = [float(item.var[CONFIG.TARGET_HL]) for item in self.datasetAll.itervalues() if self.isParameterDefined(item, CONFIG.TARGET_HL)]
			
			for record in self.datasetTraining:
				if(len(str(record.var[attr]).strip()) > 0):   # NEBUDE CASEM POTREBA (AZ BUDOU SHUFFLING CELE)
					vectorTraining1.append(float(record.var[CONFIG.TARGET_HL]))
					vectorTraining2.append(float(record.var[attr]))
			
			vectorTesting1 = []
			vectorTesting2 = []
			for record in self.datasetTesting:
				if(len(str(record.var[attr]).strip()) > 0):   # NEBUDE CASEM POTREBA (AZ BUDOU SHUFFLING CELE)
					vectorTesting1.append(float(record.var[CONFIG.TARGET_HL]))
					vectorTesting2.append(float(record.var[attr]))
					print "> " + str(record.var[attr])
					if(attr == "DINUCLEOTIDE_SHUFFLING"):
						print "record[" + record.var["GENE_ID"] + "] = " + str(record.var[attr])
			
			#print "vectorTesting1: " + str(len(vectorTesting1))
			#print "vectorTraining1: " + str(len(vectorTraining1))
			
			
			# Print Pearson & Spearman correlation
			#print scipy.stats.pearsonr(vectorTesting1, vectorTesting2)[0]
			#print "> pearson : " + str(round(scipy.stats.pearsonr(vectorTesting1, vectorTesting2)[0], 2)) + "\t" + str(round(scipy.stats.pearsonr(vectorTraining1, vectorTraining2)[0], 2))
			#print "> spearman: " + str(round(scipy.stats.spearmanr(vectorTesting1, vectorTesting2)[0], 2)) + "\t" + str(round(scipy.stats.spearmanr(vectorTraining1, vectorTraining2)[0], 2))
			#print attr + "\t" + str(abs(round(scipy.stats.pearsonr(vectorTesting1, vectorTesting2)[0], 2))) + "\t" + str(abs(round(scipy.stats.spearmanr(vectorTesting1, vectorTesting2)[0], 2)))
	
	
	def saveCorrelations(self):
		### Save calculated correlations into output file. ###
		
		outfile = open(CONFIG.CORRELATION_OUTPUT, "w")
		outfile.write(Statistics.getHeadline())
		
		for statistics in STATISTICS:
			outfile.write(statistics.toString())
		outfile.close()
		
		print "> Saved to " + CONFIG.CORRELATION_OUTPUT + " (" + str(len(STATISTICS)) + ")"
	
	
	def createArffFinalClassifier(self):
		print "***** createArffFinalClassifier *****"
		
		# Create headline for training & testing ARFF files
		headlines = "@relation finalClassifier\n"
		for attr in CONFIG.ATTR_ARFF:
			headlines += "@attribute " + attr + " numeric\n"
		headlines += "@attribute stability {STABLE,UNSTABLE}\n"
		headlines += "@attribute halfLife numeric\n"
		headlines += "@data\n"
		arffFileTraining = open("tempfile_train.arff", "w")
		arffFileAll = open("tempfile_test.arff", "w")
		arffFileTraining.write(headlines)
		arffFileAll.write(headlines)
		convertorAll = []
		
		# Create content of ARFF training & testing files
		for record in self.datasetAll.itervalues():
			#for key in record.var:
			#	print key + "\t" + str(record.var[key])
			#sys.stdin.read(1)
			outline = ""
			problem = False
			record.var["FINAL"] = ""
			for attr in CONFIG.ATTR_ARFF:
				if(((attr in record.var) == False) or (len(str(record.var[attr])) == 0)):
					problem = True
				else:
					outline += str(record.var[attr]) + ","
				
			if(problem == True):
				continue
			outline += record.var["STABILITY_TERTIARY"] + "," + record.var[CONFIG.TARGET_HL] + "\n"
			
			
			if(record.var["TRAINING_FLAG"] == "TRAINING"):
				arffFileTraining.write(outline)
			arffFileAll.write(outline)
			convertorAll.append(record)
		arffFileTraining.close()
		arffFileAll.close()
		
		self.wekaTraining("FINAL", "tempfile_train.arff", "tempfile_test.arff", convertorAll)
	
	
	def saveDataset(self):
		""" Save dataset with newly calculated attributes """
		sortedAttr = sorted(CONFIG.ATTR.items(), key=operator.itemgetter(1))
		outfile = open(CONFIG.DATASET_OUTPUT, "w")
		
		headline = "#"
		for attrIndex in sortedAttr:
			headline += attrIndex[0] + ","
		outfile.write(headline[:-1] + "\n")
		
		for rec in self.datasetAll.itervalues():
			outline = ""
			for attrIndex in sortedAttr:
				outline += str(rec.var[attrIndex[0]]) + ","
			outfile.write(outline[:-1] + "\n")
				
		outfile.close()
		print "> Saved to " + CONFIG.DATASET_OUTPUT


def main():
	# BASH: export CLASSPATH=/home/jarda/Desktop/halfLifePredictor/weka-3-7-5/
	#jvm.start(max_heap_size="512m", system_cp=True, packages=True)            # Start WEKA
	jvm.start(max_heap_size="512m")            # Start WEKA
	hlPredictor = HalfLifePredictor()          # Initialize Half-Life prediction
	hlPredictor.calculateAttributes()          # Calculate missing prediction attributes
	hlPredictor.saveDataset()                  # Save current version of the dataset
	#hlPredictor.createArffFinalClassifier()    # Create ARFF file for final classifier
	hlPredictor.calculateCorrelations()        # TODO: fill after implementation of this method
	
	
	
	hlPredictor.saveCorrelations()
	jvm.stop()
	
if __name__ == "__main__":
	main()
		
