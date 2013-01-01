import numpy
import math
import AtoB
import matplotlib.pyplot as plt
import random
import networkx as nx
import pdb
import pickle
import generateGraph

word1 = raw_input("Enter word 1: ")
word2 = raw_input("Enter word 2: ")

generateGraph.runthis()
AtoB.main(0, "RW", 'wordmorph', [word1,word2])
#AtoB.main(0, "RW", 'wordmorph', ['van','rat'])
