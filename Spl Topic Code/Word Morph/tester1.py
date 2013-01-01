import numpy
import math
import AtoB
import matplotlib.pyplot as plt
import random
import networkx as nx
import pdb
import pickle
import generateGraph

print "Word Morph Game Network:"
print "The Path Concatenation Algorithm (PCA) has been applied to the above graph"
while(True):
	word1 = raw_input("Enter word 1: ")
	word2 = raw_input("Enter word 2: ")
	AtoB.main(0, "RW", 'wordmorph', [word1,word2])
	condition = raw_input("\nContinue? (Y\N): ")
	if condition != 'Y':
		break
