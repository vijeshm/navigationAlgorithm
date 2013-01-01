import numpy
import math
import AtoB
import matplotlib.pyplot as plt
import random
import networkx as nx
import pdb
import pickle
import generateGraph

print "Gnutella P2P Network:"
print "Source: http://snap.stanford.edu/data/p2p-Gnutella08.html"
print "The Path Concatenation Algorithm (PCA) has been applied to an induced subgraph of the above graph consisting of 500 nodes"
while(True):
	node1 = int(input("Enter node 1: "))
	node2 = int(input("Enter node 2: "))
	AtoB.main(0, "RW", 'gnutella', [node1,node2])
	condition = raw_input("\nContinue? (Y\N): ")
	if condition != 'Y':
		break
