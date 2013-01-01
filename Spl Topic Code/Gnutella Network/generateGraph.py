import networkx as nx
import numpy
import matplotlib.pyplot as plt
import pickle
import AtoB

def runthis():
	fin = open('p2p-Gnutella08.txt','r')
	String = fin.read()
	fin.close()
	ConnectionList = String.rsplit()[29:]
	EdgeList = []
	for i in range(0,len(ConnectionList),2):
		EdgeList.append((int(ConnectionList[i]), int(ConnectionList[i+1])))
	G = nx.Graph()
	G.add_edges_from(EdgeList)
	G = nx.subgraph(G,range(500))
	
	fin = open("gnutella.Mygml", 'w')
	string = pickle.dumps(G)
	fin.write(string)
	fin.close()
	
	AtoB.createRealWorld("gnutella")
	
	#print "Number of words:", len(String)
	#print "Drawing Graph..."
	#nx.draw(G)
	#plt.show()
