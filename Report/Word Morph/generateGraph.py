import networkx as nx
import numpy
import matplotlib.pyplot as plt
import pickle
import AtoB

def runthis():
	fin = open("common.txt")
	String = fin.readlines()
	fin.close()
	
	for i in range(len(String)):
		String[i] = String[i][:-1]
	
	Mapping = {'a': 1, 'b': 2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26}
	
	'''
	numericals = []
	for i in range(len(String)):
			numericals.append((String[i], Mapping[String[i][0]], Mapping[String[i][1]], Mapping[String[i][2]]))
	print numericals
	'''
	
	G = nx.Graph()
	for i in range(len(String)):
		for j in range(i+1, len(String)):
			if String[i][0] == String[j][0] and String[i][1] == String[j][1] and String[i][2] != String[j][2] or String[i][0] == String[j][0] and String[i][1] != String[j][1] and String[i][2] == String[j][2] or String[i][0] != String[j][0] and String[i][1] == String[j][1] and String[i][2] == String[j][2]:
				G.add_edge(String[i], String[j])


	fin = open("wordmorph.Mygml", 'w')
	string = pickle.dumps(G)
	fin.write(string)
	fin.close()
	
	AtoB.createRealWorld("wordmorph")
	
	print "Number of words:", len(String)
	print "Drawing Graph..."
	nx.draw(G)
	plt.show()
