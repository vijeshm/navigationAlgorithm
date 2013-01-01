import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import numpy
import pdb
import pickle

def findHit(G, A, B, pathA, pathB):
	'''
	G: Graph which is undergoing machine learning
	A: Vertex #1
	B: Vertex #2
	pathA: contains the drunkard walk starting from A
	pathB: contains the drunkard walk starting from B
	Takes 2 vertices A and B from a graph G. Takes a random walk starting from A and takes another random walk starting from B and simultaneously builds the paths. If an intersection is found, the path is established and the corresponding intersection is returned. pathA and pathB are also dynamically updated.
	'''
	walkerA = A #walkers(robots) that take a random walk from given vertices
	walkerB = B

	pathA.append(A) #actual paths that the walkers take
	pathB.append(B)

	while( True ):
		walkerAAdj = G.neighbors(walkerA) #Adjacent vertices for current vertex
		walkerBAdj = G.neighbors(walkerB)

		randA = random.sample(walkerAAdj,1) #improvement Technique!!! - Why shouldnt we use 2 choices instead of just 1???????
		randB = random.sample(walkerBAdj,1)
		
		pathA.append(randA[0]) #add the randomly selected edge to the set
		if set(pathA).isdisjoint(set(pathB)):
			pathB.append(randB[0])

		if( set(pathA).isdisjoint(set(pathB)) ): #If the sets are disjoint, then there is no common point that both the walkers now. So, proceed one step further for the next loop
			walkerA = randA[0]
			walkerB = randB[0]
		else:
			break

	hit = list(set(pathA).intersection(set(pathB)))[0]
	
	return hit

def createPath(pathA, pathB, hit):
	'''
	pathA: drunkard WALK starting from A
	pathB: drunkard WALK starting from B
	hit: the point at which hit has occured.
	Given two paths pathA and pathB and the intersection point hit, then this function integrates them into a path and returns the path. This path may contain cycles and must be remmoved.
	'''
	Path = []

	index = pathA.index(hit)
	for i in range(index):
		Path.append(pathA[i])

	index = pathB.index(hit)
	for i in range(index, -1, -1):
		Path.append(pathB[i])		
	
	return Path

def removeCycles(Path):
	'''
	Given a path, this function removes all the cycles and returns the acyclic path
	Order: N
	'''
	i = 0 #i is the walker
	while i < len(Path): #the length of the path keeps on decreasing as the control flow of the program progresses
		#while( Path.count(Path[i]) > 1):
		while(Path[i] in Path[i+1:]):
			lowerIndex = i + 1 #lower index of the block to be removed
			upperIndex = Path[i+1:].index( Path[i] ) #upper index of the block to be removed relative to the subarray starting from i+1 to end 
			#remove all entries lowerIndex to UpperIndex
			for j in range(lowerIndex, (i + 1) + (upperIndex) + 1 ): #1 is added at the end to make the last element 'inclusive' rather than exclusive
				Path.pop(lowerIndex) #on every pop, the next element to be removed comes to the same position. Hence, always remove from the same position
			#print Path		
		i += 1			
	return Path

def findPath(A,B,G,Flagger):
	'''
	A: Vertex #1
	B: Vertex #2
	This function takes in 2 vertices A and B in a graph G. It finds a path from A to B through the method of random walks. It returns the path and the intersection node of the random walk. 
	'''
	
	pathA = []
	pathB = []

	hit = findHit(G, A, B, pathA, pathB) #Take a random walk and stop when an intersection occurs. Return the intersection point.
	Path = createPath(pathA, pathB, hit) #Create a path from A to B. This path may contain cycles too.
	Path = removeCycles(Path) #Remove all the cycles from the current path.
	Flagger[hit] += 1 #Flag the hit point at every stage, rather than only for the minimum path case
	return [Path, hit]

def createWeightMatrix(G):
	'''
	This function creates an empty 2-d matrix (List of Lists) and initializes all elements to zero. This matrix is later used to keep track of the edge weights in the re-enforcement algorithm.
	'''
	wtMatrix = [] #create an empty list
 	for i in range( G.number_of_nodes() ): #make it a 2-d List
		wtMatrix.append( [] ) 

	for i in range( G.number_of_nodes() ): #append all the elements with 0's
		for j in range( G.number_of_nodes() ):
			wtMatrix[i].append(0) #wtMatrix[i] returns the list in the second dimension. Hence we just use the append function

	return wtMatrix

def createFlagger(G):
	'''
	This function returns a list of length equal to the number of nodes in the graph G. The list is filled with zeroes. 
	'''
	Flagger = [] 
	for i in range( G.number_of_nodes() ): #appends the empty Flagger List with all 0'. Its length is same as the number of nodes.
		Flagger.append(0)
	return Flagger
	
def updateWeightMatrix(P, hit, wtMatrix):
	'''
	P: path from A to B, with cycles removed
	hit: the actual hit point in the path
	wtMatrix: the matrix that represents the directed graph with edge weights
	This function updates the weight matrix for every collision of random walks starting from A and B. This function doesnt return anything.
	'''
	Alength = P.index(hit) #path from A to hitMin must be rewarded with 1/(length). Hence, Alength will contain the length of this path.
	Blength = (len(P) - 1) - Alength #(len(Pmin) - 1) gives the total number of edges in the complete path. Subtracting Alength from it will give the path length from B to hitMin
	for i in range(Alength): #Assigning rewards to all the edges along the path from A to hitMin
		wtMatrix[P[i]][P[i+1]] += 1.0/Alength #PMin[i] is the starting vertex of the ith egde. The edge terminates at PMin[i+1]
			
	for i in range(Blength): #Assigning rewards to all the edges along the path from B to hitMin
		wtMatrix[ P[Alength+i+1] ][ P[Alength+i] ] += 1.0/Blength #we just just need to add Alength to the index in order to get the index for the path from B to hitMin.
	#IMPORTANT: The direction is reversed here. That is, we are incrementing (j,i) pair instead of the (i,j) pair.

def findWeightMatrix(G, Flagger, wtMatrix):
	'''
	G: Graph for which reinforcement learning must be done
	Flagger: The empty flagger list that will be filled after undergoing reinforcement learning
	wtMatrix: The 2-d array filled with zeroes as initial weights
	'''
	for i in range(5*500*499):
		A = random.choice(G.nodes())
		B = random.choice(G.nodes())
		if A!=B:
	#for A in range( G.number_of_nodes() ): 
		#for B in range( A+1, G.number_of_nodes() ): #We need not analyze (j,i) pair if we've analyzed (i,j) pair since the ordered pair just increases the redundancy.
			P = findPath(A,B,G,Flagger) #P contains an acyclic path from A to B and the corresponding hit point. While computing the path, the Flagger is also updated.
			#IMPORTANT: Flagger is updated on EVERY random walk, not just the one with the minimum length. 
			hit = P[1] #Hit contains the intersection point
			P = P[0] #P contains the path from A to B
			updateWeightMatrix(P, hit, wtMatrix) #updates the weight matrix by adding 1/(length) to the edge-weights of the corresponding paths

def findDeviation(G,wtMatrix,hotSpots):
	'''
	Given a graph G and its corresponding weight matrix, this function finds the difference in the oppositely directed edges between every pair of vertices.
	'''
	
	dev = []
	for i in range( G.number_of_nodes() ):
		for j in range(i+1, G.number_of_nodes() ): #Analyzing half the matrix is enough since both (i,j) and (j,i) can be captured in the same loop.
			if (not(wtMatrix[i][j]==0 and wtMatrix[j][i]==0)):
				dev.append(wtMatrix[i][j] - wtMatrix[j][i])
	dev = list(numpy.absolute(dev)) #finds the absolute value of the difference
	return dev
	
	'''//This module considers the edge difference between those edges that are adjacent to the hotSpots ONLY'''

	'''
	EdgeAdj = set([])
	for i in hotSpots:
		EdgeAdj = EdgeAdj.union(set(G.neighbors(i)))
	EdgeAdj = list(EdgeAdj)

	dev = []
	for i in hotSpots:
		for j in G.neighbors(i):
			print wtMatrix[i][j], wtMatrix[j][i]
			dev.append(wtMatrix[i][j] - wtMatrix[j][i])
	dev = list(numpy.absolute(dev)) #finds the absolute value of the difference
	#print dev
	return devthob
	'''

def attachNode(G, Flagger):
	'''
	G: Graph that is undergoing reinforcement learning
	Flagger: list containing the number of flags planted at each node.
	Given a graph G and a Flagger list, this function creates a list of 2-tuples. The first element of the 2-tuple is the flagger value and the second value is the corresponding vertex (at which the intersections have occured). This action is done so that the ordering information is not lost when/if the list is sorted.
	'''
	for i in range( G.number_of_nodes() ):
		Flagger[i] = ( (Flagger[i], i) ) #This tuple contains the actual frequency of hits as the first entry and its corresponding vertex as the second element. Meaning: The Flagger[i][1] vertex has been the intersection point for Flagger[i][0] times 

def getHotSpots(G, startIndex, Flagger):
	'''
	Given a sorted Flagger list which contains a list of 2-tuples (First element is the actual element and second element is the corresponding index at intersections had occured) and the startIndex from which the HotSpots are to be selected. 
	'''
	hotSpots = []
	for i in range( startIndex, G.number_of_nodes() ): #Just choose the last 'f(n)' elements from the sorted Flagger list
		hotSpots.append(Flagger[i][1]) #copy the second element(vertex) from the tuple into the hotSpots list
	return hotSpots

def getFlags(G, startIndex, Flagger):
	'''
	Given a sorted Flagger list which contains a list of 2-tuples (First element is the actual element and second element is the corresponding index at intersections had occured) and the startIndex from which the HotSpots are to be selected. 
	'''
	Flags = []
	for i in range( startIndex, G.number_of_nodes() ): #Just choose the last 'f(n)' elements from the sorted Flagger list
		Flags.append(Flagger[i][0]) #copy the second element(vertex) from the tuple into the hotSpots list
	return Flags

def createLookup(hotSpots, G):
	'''
	hotSpots: List of hotSpots in descending order of importance
	G: considered Graph
	Given the Graph G and the list of HotSpots, this function returns the lookup table for the shortest path between each of the hotSpots. The return value is a 2 dimensional matrix.
	'''
	hotSpotLookup = []
	for i in range(len(hotSpots)):
		hotSpotLookup.append( [] ) #Create an empty 2-d Array List
	
	for i in range(len(hotSpots)):
		for j in range(len(hotSpots)):
			hotSpotLookup[i].append( nx.shortest_path(G,hotSpots[i],hotSpots[j],False) ) #Load the actual Shortest Path as an element of the 2-d array created 
	return hotSpotLookup

def histogram(list, binsize, title, xlabel, ylabel):
	'''
	list: the list whose histogram is to be plotted
	binsize: sizeof the each bin at which the histogram is to be drawn
	Draws the histogram for the list of the given binSize.
	'''

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	numberOfBins = int(len(list) / binsize) #the argument to the 'hist' function is the number of bins. Hence, it is calculated, converted to int and sent to plt.hist to plot the histogram
	n, bins, patches = plt.hist(list, bins = numberOfBins) #Create the histogram, draw and show
	#plt.draw()	
	#plt.show()

def findHotSpotPath(G, A, hotSpots, wtMatrix):
	'''
	G: Graph
	A: vertex from which the path must be deduced (based on maximum edge selection) to any one of the hotSpots
	hotSpots: List of Hot Spots
	wtMatrix: Reinforced matrix of edge-weights.

	This method finds a path from the given vertex A to the nearest hotspot and returns the path.
	Method used: At any vertex choose that edge which has the highest weight of the lot and traverse along that edge. Sometimes, it may lead to a vertex that has already been visited. Hence it may form an cycle. In such a case, choose the second best edge and traverse along it. If the second too forms a cycle, try for the third one and so on until you hit one of the Hot Spots
	'''

	walker = A #walker is the current vertex
	path = [A] #actual path from vertex A which will eventually lead to a hotSpot
	while(not(walker in hotSpots)): #loop until you hit a hotSpot
		walkerAdj =  G.neighbors(walker) #walkerAdj contains the list of neighbors		
		weights = [] #weights is a list of edge weights from A to all its neighbors in the order corresponding to walkerAdj
		for i in range(len(walkerAdj)):
			weights.append(wtMatrix[A][ walkerAdj[i] ])
		index = weights.index(numpy.max(weights)) #index contains the index of the largest weight in the weights list
		maxNeighbor = walkerAdj[index] #the neighbor at that particular index is the one where we have to go next
		
		while(maxNeighbor in path): #as long as the maxNeighbor is in the visited list, find the next best maxNeighbor. That is, cycles are to be forbidden since it gives rise to infinite loop.
			if(len(weights)!=1):
				walkerAdj.remove(maxNeighbor) #remove the maxNeighbor from the neighbors list
				weights.remove(numpy.max(weights)) #remove the corresponding weight entry from the weights list
				index = weights.index(numpy.max(weights)) #calculate index again
				maxNeighbor = walkerAdj[index]	#calculate maxNeighbor again to verify whether it is present in the existing path
			else:
				print "Random Choice:", random.choice(G.neighbors(walkerAdj[0]))
				maxNeighbor = random.choice(G.neighbors(walkerAdj[0]))
				
		path.append(maxNeighbor) #append the maxNeighbor
		walker = maxNeighbor #move the Walker to the next neighbor
	path = removeCycles(path) #Will there be any cycles?? I dont think so. This is just for the safer side.
	return path

def findFullPath(pathA, pathB, hotSpots, hotSpotLookup):
	'''
	pathA: path from A to any one of the hotSpots, based on the max Weighted Edge Traversal
	pathB: path from B to any one of the hotSpots, based on the max Weighted Edge Traversal
	hotSpots: list of Hot Spots
	hotSpotLookup: Lookup Table for shortest path between every pair of HotSpots
	This method integrates the path from A to hotSpot1, hotSpot1 to hotSpot2, hotSpot2 to B and returns the full list
	'''

	fullPath = pathA[:] #copy the complete path from A to hotSpot1 (including hotSpot1) to the fullPath
	fullPath.extend(hotSpotLookup[ hotSpots.index(pathA[-1]) ][ hotSpots.index(pathB[-1]) ][1:] ) #Lookup for the shortest Path between hotSpot1 and hotSpot2 and extend it to the list fullPath excluding the first element
	pathBReversed = pathB[:] #copy pathB - pathB is a path from B to hotSpot2. Hence, it needs to be reversed before appending
	pathBReversed.reverse()	#Reverse pathB
	pathBReversed.pop(0) #Since hotSpot2 is already included in the list, pop it out from the list.
	fullPath.extend( pathBReversed ) #extend the fullPath with the reversed and popped pathB
	return fullPath

def test(G,A,B,hotSpots,hotSpotLookup,wtMatrix):
	'''
	G: Graph
	A, B: vertices between which the approximate shortest path is to be estimated
	hotSpots: List of hotSpots
	hotSpotLookup: Lookup Table for shortest path between any two pairs of hotSpots
	wtMatrix: Matrix containing weights of all the edges
	This function is used to test the hypothesis. It returns the approximate Shortest Path.
	'''

	pathA = findHotSpotPath(G, A, hotSpots, wtMatrix) #Find a path from A to any of the Hot Spot
	pathB = findHotSpotPath(G, B, hotSpots, wtMatrix) #Find a path from B to any of the Hot Spot
	fullPath = findFullPath(pathA, pathB, hotSpots, hotSpotLookup) #Integrate the two paths A and B
	fullPath = removeCycles(fullPath)
	return fullPath

def edgeWeightSequence(path, wtMatrix):
	'''
	path: Path whose corresponding edge weights are to be returned
	wtMatrix: matrix of weights formed by the reinforcement algorithm
	This function returns the list of edge weights as one travels along the path from start vertex to terminal vertex
	'''
	x = []
	for i in range(len(path)-1):
		x.append( wtMatrix[ path[i] ][ path[i+1] ] )	
	return x

def createRealWorld(name):
	'''
	name: The name of the real world graph
	This function creates a .graph file from a .gml file, and runs the machine learning alogorithm on it.
	'''	
	#G = nx.read_gml(name + ".gml")
	
	fin = open(name + ".Mygml")
	string = fin.read()
	G = pickle.loads(string)
	fin.close()

	print name + ".Mygml" + " file read"
	#faprint G.nodes()

	'''
	H = nx.convert_node_labels_to_integers(G, ordering='sorted')
	print "Nodes labels are converted to integers"
	'''
	nodes = G.nodes()
	nodes.sort()
	for i in range(len(nodes)):
		nodes[i] = (nodes[i], i)

	fout = open(name + ".graph", 'w')
	string = pickle.dumps(G)
	fout.write(string)
	fout.close()
	print name + ".graph File Generated"
	
	fout = open(name + ".map", 'w')
	string = pickle.dumps(nodes)
	fout.write(string)
	fout.close()
	print name + ".map File Generated"
	
	print "Undergoing Machine Learning..."
	reinforce(G, "RW", name)
	#main(G.number_of_nodes(), "RW", name, math.e*numpy.log(G.number_of_nodes()), 1000)


def createScaleFreeNetwork(numOfNodes, degree):
	'''
	numOfNodes: The number of nodes that the scale free network should have
	degree: The degree of the Scale Free Network
	This function creates a Scale Free Network containing 'numOfNodes' nodes, each of degree 'degree'
	It generates the required graph and saves it in a file. It runs the Reinforcement Algorithm to create a weightMatrix and an ordering of the vertices based on their importance by Flagging.
	'''
	G = nx.barabasi_albert_graph(numOfNodes, degree) #Create a Scale Free Network of the given number of nodes and degree
	fout = open("SCN_" + str(numOfNodes) + "_" + str(degree) + ".graph", 'w') #create a file handler and open the corresponding file in write mode
	string = pickle.dumps(G) #Convert the generated graph into the string and store it into a file. It is later retrieved in the main module
	fout.write(string) #Write the String into the file and close the file handler
	fout.close()
	print "Scale Free Network successfully generated and written into the file."
	print "File Name: ", "SCN_" + str(numOfNodes) + "_" + str(degree) + ".graph"
	print "Undergoing Machine Learning..."
	reinforce(G,"SCN", degree) #Enforce Machine Learning to generate two files.

def createErdos(numOfNodes, edgeProb):
	'''
	numOfNodes: The number of nodes that the Eldish graph will have
	edgeProb: The probability of existance of an edge between any two vertices
	This function creates an Erdos Graph containing 'numOfNodes' nodes, with the probability of an edge existing between any two vertices being 'edgeProb'
	It generates the required graph and saves it in a file. It runs the Reinforcement Algorithm to create a weightMatrix and an ordering of the vertices based on their importance.
	'''
	G = nx.erdos_renyi_graph(numOfNodes, edgeProb) #Creates an Eldish Graph of the given number of nodes and edge Probability
	if nx.is_connected(G):
		fout = open("EG_" + str(numOfNodes) + "_" + str(edgeProb) + ".graph", 'w') #create a file handler and open the corresponding file in write mode
		string = pickle.dumps(G) #Convert the generated graph into the string and store it into a file. It is later retrieved in the main module
		fout.write(string) #write the string into the file and close the file handler
		fout.close()
		print "Erdos Renyi Graph successfully generated and written into the file."
		print "File Name: ", "EG_" + str(numOfNodes) + "_" + str(edgeProb) + ".graph"
		print "Undergoing Machine Learning..."
		reinforce(G,"EG", edgeProb) #Enforce Machine Learning to generate two files

def createBridge(numOfNodes, edgeProb, bridgeNodes):
	'''
	numOfNodes: Number of nodes in the clustered part of the Bridge Graph
	edgeProb: Probability of existance of an edge between any two vertices.
	bridgeNodes: Number of nodes in the bridge
	This function creates a Bridge Graph with 2 main clusters connected by a bridge.
	'''
	G1 = nx.erdos_renyi_graph(2*numOfNodes + bridgeNodes, edgeProb) #Create an ER graph with number of vertices equal to twice the number of vertices in the clusters plus the number of bridge nodes.
	G = nx.Graph() #Create an empty graph so that it can be filled with the required components from G1
	G.add_edges_from(G1.subgraph(range(numOfNodes)).edges()) #Generate an induced subgraph of the nodes, ranging from 0 to numOfNodes, from G1 and add it to G
	G.add_edges_from(G1.subgraph(range(numOfNodes + bridgeNodes,2*numOfNodes + bridgeNodes)).edges()) #Generate an induced subgraph of the nodes, ranging from (numOfNodes + bridgeNodes) to (2*numOfNodes + bridgeNodes)

	A = random.randrange(numOfNodes) #Choose a random vertex from the first component
	B = random.randrange(numOfNodes + bridgeNodes,2*numOfNodes + bridgeNodes) #Choose a random vertex from the second component

	prev = A #creating a connection from A to B via the bridge nodes
	for i in range(numOfNodes, numOfNodes + bridgeNodes):
		G.add_edge(prev, i)
		prev = i
	G.add_edge(i, B)

	fout = open("BG_" + str(2*numOfNodes + bridgeNodes) + "_" + str(edgeProb) + ".graph", 'w') #create a file handler and open the corresponding file in write mode
	string = pickle.dumps(G) #Convert the generated graph into the string and store it into a file. It is later retrieved in the main module
	fout.write(string) #write the string into the file and close the file handler
	fout.close()
	print "Bridge Graph successfully generated and written into the file."
	print "File Name: ", "BG_" + str(2*numOfNodes + bridgeNodes) + "_" + str(edgeProb) + ".graph"
	print "Undergoing Machine Learning..."
	reinforce(G,"BG", edgeProb) #Enforce Machine Learning to generate two files

def reinforce(G, Type, characteristic):
	'''
	G: Graph G which has to undergo machine learning
	Type: Type of the Graph. This parameter is just given due to generic programming reasons.
	characteristic: 
			Scale Free Networks are associated with number of connections
			Erdos Renyi Graphs are associated with probability
			Real World Networks are associated with their names
			Bridge Graphs are associated with their edge probability
	'''
	wtMatrix = createWeightMatrix(G) #create a 2-d matrix with all zeroes
	Flagger = createFlagger(G) #create a Flagger list whose length is same as the number of nodes and its filled with zeroes.
	findWeightMatrix(G, Flagger, wtMatrix) #Considers all pairs of vertices, takes 1 random walk between each of them, updates the Flagger at every instance of the random walk, updates wtMatrix only for the minimum path risen from the random walk between each vertex pair
	
	attachNode(G, Flagger) #affixes the vertex information along with the actual flag value so that the vertex information isnt lost when the list is sorted
	Flagger.sort() #Sort the Flagger so that the top k elements can be used as the hotSpot list. This k may depend upon the number of nodes in the graph

	if(Type == "SCN"): #If the given graph type is a Scale Free Network, then store its weightMatrix and the Flagger List in their corresponding files
		fout = open("SCN_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix", 'w')
		string = pickle.dumps(wtMatrix)
		fout.write(string)
		fout.close()
		print "Weight Matrix successfully written into the file", "SCN_", str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix"

		fout = open("SCN_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger", 'w')
		string = pickle.dumps(Flagger)
		fout.write(string)
		fout.close()
		print "Flagger List successfully written into the file", "SCN_", str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger"

	if(Type == "EG"): #If the given graph type is an Erdos Renyi Graph, then store its weightMatrix and the Flagger List in their corresponding files
		fout = open("EG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix", 'w')
		string = pickle.dumps(wtMatrix)
		fout.write(string)
		fout.close()
		print "Weight Matrix successfully written into the file", "EG_", str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix"

		fout = open("EG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger", 'w')
		string = pickle.dumps(Flagger)
		fout.write(string)
		fout.close()
		print "Flagger List successfully written into the file", "EG_", "EG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger"
	
	if(Type == "BG"):
		fout = open("BG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix", 'w')
		string = pickle.dumps(wtMatrix)
		fout.write(string)
		fout.close()
		print "Weight Matrix successfully written into the file", "BG_", str(G.number_of_nodes()) + "_" + str(characteristic) + ".wtMatrix"

		fout = open("BG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger", 'w')
		string = pickle.dumps(Flagger)
		fout.write(string)
		fout.close()
		print "Flagger List successfully written into the file", "BG_" + str(G.number_of_nodes()) + "_" + str(characteristic) + ".Flagger"

	if(Type == "RW"):
		fout = open(characteristic + ".wtMatrix", 'w')
		string = pickle.dumps(wtMatrix)
		fout.write(string)
		fout.close()
		print "Weight Matrix Successfully written into the file", characteristic + ".wtMatrix"
	
		fout = open(characteristic + ".Flagger", 'w')
		string = pickle.dumps(Flagger)
		fout.write(string)
		fout.close()
		print "Flagger List successfully written into the file", characteristic + ".Flagger"
		
def SortDictionary(dict):
	items = [(k,v) for v,k in dict.items()]
	items.sort()
	items.reverse()
	vertices = [v for k,v in items]
	return vertices

def DegreeOrdering(G, fn):
	'''
	G: Graph
	fn: funtion of n used to delimit the number of hotspots
	'''
	local_a=[]
	for i in G.nodes():
		local_a.append([])
	for i in G.nodes():
		local_a[G.degree(i)].append(i)
	while([] in local_a):
		local_a.remove([])
	ordered_array=[]
	local_a.reverse()
	for i in range(0,len(local_a)):
		for j in range(0,len(local_a[i])):
			ordered_array.append(local_a[i][j])
	#print ordered_array
	return ordered_array[:int(fn)]

def Betweenness_Centrality(G, fn):
	'''
	G: Graph
	fn: funtion of n used to delimit the number of hotspots
	'''
	betDict = nx.betweenness_centrality(G)
	betweenness = SortDictionary(betDict)[:int(fn)]
	return betweenness

def Eigen_Centrality(G, fn):
	'''
	G: Graph
	fn: funtion of n used to delimit the number of hotspots
	'''
	eigDict = nx.eigenvector_centrality(G)
	eigenVector = SortDictionary(eigDict)[:int(fn)]
	return eigenVector

def Closeness_Centrality(G, fn):
	'''
	G: Graph
	fn: funtion of n used to delimit the number of hotspots
	'''
	closeDict = nx.closeness_centrality(G)
	closeness = SortDictionary(closeDict)[:int(fn)]
	return closeness

def Betweenness_Edge_centrality(G, fn):
	'''
	G: Graph
	fn: funtion of n used to delimit the number of hotspots
	'''
	edgeBetDict = nx.edge_betweenness_centrality(G)
	edgeBetCentr = SortDictionary(edgeBetDict)[:int(fn)]
	return edgeBetCentr

def FlaggingBasedOnActualSP(G, hotSpots, fn):
	FlaggerNew = [0 for i in range(G.number_of_nodes())]
	for A in range(G.number_of_nodes()):
		for B in range(A+1, G.number_of_nodes()):
			shortest = nx.shortest_path(G, A, B, False)
			for vertex in shortest:
				FlaggerNew[vertex] += 1

	for i in range(G.number_of_nodes()):
		FlaggerNew[i] = (FlaggerNew[i], i)

	FlaggerNew.sort()
	FlaggerNew.reverse()
	FlaggerNew = [v for f,v in FlaggerNew]
	print "Flagging based on Actual Shortest Path:", FlaggerNew[:int(fn)]
	hotSpotRev = hotSpots[:]
	hotSpotRev.reverse()	
	print "HotSpots:                              ", hotSpotRev

def AverageStepsFromAllVtoH(G, hotSpots, wtMatrix):
	'''
	This snippet calculates the average length it takes to get from any node to ANY ONE of the hotSpots. This code loops over the entire vertex set and finds the path length from each vertex to ANY ONE of the hotSpots. This ensures that all the vertices are chosen exactly once (unlike choosing the vertex randomly). Once the weight Matrix is known, the path from any vertex to the hotSpot set is deterministic.
	'''
	length = []
	for i in range( G.number_of_nodes() ): #Loop through every vertex and find the path traversed using the reinforced algorithm. Then find the average path length travelled to reach ANY ONE of the hotspots. 
		if i not in hotSpots: #Neglect all the 'i's that are hotSpots since their length is already known to be 0.
			path = findHotSpotPath(G,i,hotSpots,wtMatrix) #Finds the path from the given vertex to ANY ONE of the hotSpots
			length.append(len(path)-1) #We dont really need the path, just the path length is enough
	print "Average number of steps needed to reach a hot spot from all vertices: ", numpy.average(length) #Computes the average of the path lengths from any vertex to any hotSpot.


def AverageStepsFromRandomVtoH(G, hotSpots, wtMatrix, VtoHTrials):
	'''
	This snippet calculates the average length it takes to get from any node to ANY ONE of the hotSpots. This code chooses a vertex at random and calculates the path from itself to any one of the hotspots. It stores the path length in an array. This procedure repeats for 'VtoHtrials' number of trials.
	'''
	length = []
	for i in range(int(VtoHTrials)):
		A = random.randrange(G.number_of_nodes())
		path = findHotSpotPath(G, A, hotSpots, wtMatrix)
		#print "From", A, "to", path[-1], ":", path 
		length.append(len(path)-1)
	#print "Length of the Path:", length
	print "Average number of steps needed to reach a hot spot from any random vertex: ", numpy.average(length)


def ReinVSbfs(G, hotSpots, hotSpotLookup, wtMatrix, ReinVSshortestTrials):
	'''This snippet calculates the ratio of the path length given by the algorithm, to the actual shortest path. The two vertex pairs are chosen in a random fashion. The average ratio tells us how many times longer we'll have to travel to get to the destination compared to the actual shortest path'''
	ratios = []
	for i in range(ReinVSshortestTrials): #ReinVSshortestTrials is the number of trials carried out to calculate the average ratio of len(path by our algo) to len(actual Shortest path)
		A = random.randrange( G.number_of_nodes() )
		B = random.randrange( G.number_of_nodes() )
		if( A != B):
			fullPath = test(G,A,B,hotSpots,hotSpotLookup,wtMatrix) #Find out the full path between the chosen random vertices
			shortestPath = nx.shortest_path(G,A,B,False) #Find out the actual Shortest Path
			#print "Full Path by our Algorithm: ", fullPath
			#print "Shortest Path by inbuilt Algorithm: ", shortestPath
			#print ""
			#print "length of Full Path/length of Actual Shortest Path:", float(len(fullPath))/len(shortestPath)
			ratios.append(float(len(fullPath))/len(shortestPath)) #Since we have to find the average of the ratios, append the ratios itself.
	print "Average of ratios of lengthByReinforcement / lengthbyBFS:", numpy.average(ratios)


def RandWalkVSReinWalk(G, hotSpots, hotSpotLookup, wtMatrix, RandVSReinTrials):
	'''
	This snippet calculates the ratio of the path length of the random walk, to the path length given by the reinforced Algorithm. The two vertex pairs are chosen at Random. An absolutely random walk is constructed starting from the first vertex and it continues till the second vertex is reached. Then the path length is found using the reinforced algorithm. Their ratio is computed.
	'''
	RandVSRein = [] #RandVSRein contains the ratio of len(random walk from A to B)  to len(Reinforced Walk)
	for k in range(RandVSReinTrials): #This trail generates two random vertices i and j and finds the ratio len(random walk from i to j) to len(Reinforced Walk)
		i = random.randrange(G.number_of_nodes())
		j = random.randrange(G.number_of_nodes())
	
		if( i!=j):
			walkerI = i #WalkerI contains the current position of the walker starting from I
			pathI = [i] #pathI contains the path along which the walker has travalled
			while( j not in pathI ): #Loop until you hit j
				neighborI = G.neighbors(walkerI) #Get the neighbors of the current walker position
				walkerI = random.sample(neighborI, 1)[0] #Choose a random neighbor and move walker to that position
				pathI.append(walkerI) #append the walker to that path
		
			fullPath = test(G,i,j,hotSpots,hotSpotLookup,wtMatrix) #Find the path based on Reinforced Algorithm
			RandVSRein.append( len(pathI) / len(fullPath) )	#Append the ratio of len(random walk) to len(Reinforeced Path)
	print "Average of lengths of RandomWalk to ReinforcedWalk between nP2-2n pairs:",numpy.average(RandVSRein) 				

def VtoHshortestPathAverage(G, hotSpots):
	'''
	Consider all vertices. Consider the shortest path from each vertex to all the hotSpots. Choose the nearest hotSpot and append its path length to a list. Take the average of all such shortest paths.
	'''
	AShortest = [] #shortest path from all the vertices A to the 
	for i in range(G.number_of_nodes()):		
		AtoHshortest = [] #AtoHShortest contains all the shortest paths from A to all of the hotspots
		for j in hotSpots:
			AtoHshortest.append(len(nx.shortest_path(G,i,j,False)))
		AShortest.append(numpy.min(AtoHshortest))
	print "Average Shortest Path from any vertex to the nearest hotspot:", numpy.average(AShortest)

def RandomWalkHotSpotHit(G, hotSpots):
	'''Consider all vertices. Take Random walks ( => without using reinforcement algortihm) from that vertex until we hit a hotSpot. This snippet calculates the average number of steps required to hit a hotSpot.'''
	HitPathLen = [] #length of the paths from all vertices to any one of the hotspots
	for i in range(G.number_of_nodes()):
		walkerI = i
		HitPath = [i]
		while walkerI not in hotSpots:
			walkerI = random.sample(G.neighbors(walkerI),1)[0]
			HitPath.append(walkerI)
		HitPathLen.append(len(HitPath) - 1) #path length = number of vertices - 1

def extentOfDisjoint(hotSpots, DegreeOrder, betweenness, eigenVector, closeness):
	'''
	This method returns a list of extent of disjoints of the hotspots with all the centrality measures.
	'''
	#Convert all the lists to sets
	hotSpotsDisjoint = set(hotSpots[:])
	DegreeOrderDisjoint = set(DegreeOrder[:])
	BetweennessDisjoint = set(betweenness[:])
	EigenVectorDisjoint = set(eigenVector[:])
	ClosenessDisjoint = set(closeness[:])

	#Find the elements that are in (A union B) - (A intersection B)
	HvsDeg = hotSpotsDisjoint.union(DegreeOrderDisjoint).difference(hotSpotsDisjoint.intersection(DegreeOrderDisjoint))
	HvsBet = hotSpotsDisjoint.union(BetweennessDisjoint).difference(hotSpotsDisjoint.intersection(DegreeOrderDisjoint))
	HvsEig = hotSpotsDisjoint.union(eigenVector).difference(hotSpotsDisjoint.intersection(eigenVector))
	HvsClo = hotSpotsDisjoint.union(ClosenessDisjoint).difference(hotSpotsDisjoint.intersection(ClosenessDisjoint))

	#Disjoint Factor: number of elements in the above list / total number of elements
	DegDisjointFactor = float(len(HvsDeg)) /len(hotSpotsDisjoint.union(DegreeOrderDisjoint))
	BetDisjointFactor = float(len(HvsBet)) / len(hotSpotsDisjoint.union(BetweennessDisjoint))
	EigDisjointFactor = float(len(HvsEig)) / len(hotSpotsDisjoint.union(EigenVectorDisjoint))
	CloDisjointFactor = float(len(HvsClo)) / len(hotSpotsDisjoint.union(ClosenessDisjoint))

	return [DegDisjointFactor, BetDisjointFactor, EigDisjointFactor, CloDisjointFactor]

def correctnessMeasure(predictedHotSpots, hotSpotRev):
	#summationMeasure: farther the element is from the beginning, even if it is unordered, it must be given lesser weightage. 
	summationMeasure = 0
	for i in predictedHotSpots:
		if(i in predictedHotSpots and i in hotSpotRev): #i may not be in both the predicted lists and the hotspots list.
			summationMeasure += numpy.abs(predictedHotSpots.index(i) - hotSpotRev.index(i)) / float(predictedHotSpots.index(i) + 1)
	print "Sum of deviations of Indices:", summationMeasure

def PredictHotSpots(G, hotSpots, fn, Weights):
	'''
	This function predicts the hotspots based on the weighted average of indices technique for Degree, Betweenness, Eigen and Closeness Centrality
	'''
	DegreeOrder = DegreeOrdering(G, fn)
	print "Degree Ordering:", DegreeOrder

	betweenness = Betweenness_Centrality(G, fn)
	print "Betweenness:", betweenness

	eigenVector = Eigen_Centrality(G, fn)
	print "Eigen Vector:", eigenVector

	closeness = Closeness_Centrality(G, fn)
	print "Closeness:", closeness
	
	DBECdisjoints = extentOfDisjoint(hotSpots, DegreeOrder, betweenness, eigenVector, closeness)
	print "\nExtent of Disjointness of HotSpots with:"
	print "Top few vertices with a high degree:", DBECdisjoints[0]
	print "Betweenness Centrality:", DBECdisjoints[1]
	print "Eigen Centrality:", DBECdisjoints[2]
	print "Closeness Centrality:", DBECdisjoints[3]

	DegWeight = Weights[0]
	BetWeight = Weights[1]
	EigWeight = Weights[2]
	CloWeight = Weights[3]

	Sum = float(DegWeight + BetWeight + EigWeight + CloWeight)
	DegWeight = DegWeight / Sum
	BetWeight = BetWeight / Sum
	EigWeight = EigWeight / Sum
	CloWeight = CloWeight / Sum

	union = list(set(DegreeOrder).union(set(betweenness)).union(set(eigenVector)).union(set(closeness)))

	predictedHotSpots = []
	for i in union:
		element = i
	
		if(element in DegreeOrder):
			DegPos = DegreeOrder.index(element)
		else: 
			DegPos = int(fn)

		if(element in betweenness):
			BetPos = betweenness.index(element)
		else:
			BetPos = int(fn)

		if(element in eigenVector):
			EigPos = eigenVector.index(element)
		else:
			EigPos = int(fn)
	
		if(element in closeness):
			CloPos = closeness.index(element)
		else:		
			CloPos = int(fn)

		avgIndex = DegWeight*DegPos + BetWeight*BetPos + EigWeight*EigPos + CloWeight*CloPos
		predictedHotSpots.append((avgIndex, element))

	predictedHotSpots.sort()
	predictedIndices = [k for k,v in predictedHotSpots]
	predictedIndices = predictedIndices[:int(fn)]
	predictedHotSpots = [v for k,v in predictedHotSpots]
	predictedHotSpots = predictedHotSpots[:int(fn)]

	correctnessMeasure(predictedHotSpots, hotSpots)
	
	return [predictedHotSpots, predictedIndices]

def givenKfindAlpha(G, hotSpots, K):
	Flagger = [0 for i in range(G.number_of_nodes())]	

	newHotSpotList = []
	Set = set([])
	for vertex in hotSpots:
		KSpan = bfs(G, vertex, K)
		for element in KSpan:
			Flagger[element] += 1
		
		if not set(KSpan).issubset(Set):
			newHotSpotList.append(vertex)
			Set.union(set(KSpan))
		
		if 0 not in Flagger:
			break
		
	return newHotSpoList

def compareVertexCentralities(G, Flagger):
	'''
	G: Graph
	Flagger: This is list of tuples of the form (no. of Flags, node)
	This function calculates the flag centrality, degree centrality, betweenness centrality, eigen centrality and closeness centrality. It then plots a graph with x-axis as the node and y-axis as the importance level 
	'''
	FlagCentrality = [ v for k,v in Flagger]
	DegreeCentrality = DegreeOrdering(G, G.number_of_nodes())
	BetweennessCentrality = Betweenness_Centrality(G, G.number_of_nodes())
	EigenCentrality = Eigen_Centrality(G, G.number_of_nodes())
	ClosenessCentrality = Closeness_Centrality(G, G.number_of_nodes())

	DegreeCentrality = [ DegreeCentrality.index(i) for i in range(G.number_of_nodes())]
	BetweennessCentrality = [BetweennessCentrality.index(i) for i in range(G.number_of_nodes())]
	EigenCentrality = [EigenCentrality.index(i) for i in range(G.number_of_nodes())]
	ClosenessCentrality = [ClosenessCentrality.index(i) for i in range(G.number_of_nodes())]

	plt.xlabel("Node")
	plt.ylabel("Level of Importance")
	plt.plot(FlagCentrality)
	plt.plot(DegreeCentrality)
	plt.plot(BetweennessCentrality)
	plt.plot(EigenCentrality)
	plt.plot(ClosenessCentrality)
	x = ["Blue = Flag Centrality", "Green = Degree Centrality", "Red = Betweenness Centrality", " Cyan = Eigen Centrality", "Pink = Closeness Centrality"]
	plt.text(0,0,x)
	plt.show()

def compareEdgeCentralities(G, wtMatrix, numberOfEdges):
	'''
	G: Graph
	wtMatrix: A matrix of weights that represents a directed graph
	This function calculates the 
	'''
	wtMatrixWithEdges = [] #Creating a deepcopy of the the weight matrix
	for i in range(G.number_of_nodes()):
		wtMatrixWithEdges.append([])
		for j in range(G.number_of_nodes()):
			wtMatrixWithEdges[i].append(wtMatrix[i][j])
	
	for i in range(G.number_of_nodes()): # Making the graph undirected by forcing new-edge-weight = sum of edge weights in both the directions
		for j in range(i+1, G.number_of_nodes()):
			wtMatrixWithEdges[i][j] = (wtMatrixWithEdges[i][j] + wtMatrixWithEdges[j][i], i, j) #Append the row, column info too.
	
	wtList = []
	for i in range(G.number_of_nodes()): #creating a linear list of items of edge weights, nodes.
		for j in range(i+1, G.number_of_nodes()):
			if G.has_edge(i,j):
				wtList.append(wtMatrixWithEdges[i][j]) 

	wtList.sort()
	wtList.reverse()

	hotEdges = [] #deducing the hot edges
	for i in wtList:
		hotEdges.append((i[1],i[2]))
	
	edgeBetweenness =  Betweenness_Edge_centrality(G, G.number_of_edges()) #Calculates the hot edges based on edge centrality
	
	print "Machine Learnt Hot Edges:", hotEdges[:numberOfEdges]
	print "Edge Betweenness Centrality", edgeBetweenness[:numberOfEdges]
	
	indices = [] #Consider the egdes in hotEdges. 'indices' contains the position of this edge in the in the edgeBetweenness
	for i in hotEdges: 
		indices.append( edgeBetweenness.index(i) )
	print indices[:numberOfEdges]

def SudarshansOutput(G, wtMatrix, hotSpots, hotSpotLookup, Trials):

	SudarshansList = []
	for i in range(Trials):
		SudarshansList.append([])
		u = random.randrange(G.number_of_nodes())
		v = random.randrange(G.number_of_nodes())
	
		shortestPathLen = float(len(nx.shortest_path(G,u,v,False)))

		pathU = []
		pathV = []
		hit = findHit(G,u,v,pathU,pathV)
		Path = createPath(pathU, pathV, hit) #Create a path from A to B. This path may contain cycles too.
		Path = removeCycles(Path) #Remove all the cycles from the current path.
		SudarshansList[i].append(len(Path)/shortestPathLen)

		walkerU = u
		HitPathU = [u]	
		while walkerU not in hotSpots:
			walkerU = random.sample(G.neighbors(walkerU),1)[0]
			HitPathU.append(walkerU)
		walkerV = v
		HitPathV = [v]
		while walkerV not in hotSpots:
			walkerV = random.sample(G.neighbors(walkerV),1)[0]
			HitPathV.append(walkerV)
		FullPath = findFullPath(HitPathU, HitPathV, hotSpots, hotSpotLookup)
		SudarshansList[i].append(len(FullPath)/shortestPathLen)

		fullPath = test(G,u,v,hotSpots,hotSpotLookup,wtMatrix) #Find out the full path between the chosen random vertices
		SudarshansList[i].append(len(fullPath)/shortestPathLen)

		#print u,v,SudarshansList[i]

	AverageList = [0,0,0]
	for i in SudarshansList:
		AverageList[0] += i[0]
		AverageList[1] += i[1]
		AverageList[2] += i[2]
	AverageList[0] /= len(SudarshansList)
	AverageList[1] /= len(SudarshansList)
	AverageList[2] /= len(SudarshansList)
	print "Average List:", AverageList

def query(G, hotSpots, hotSpotLookup, wtMatrix):
	i = 1
	while i == 1:
		A = int(input("Enter vertex number 1:"))
		B = int(input("Enter vertex number 2:"))
		FullPath = test(G, A, B, hotSpots, hotSpotLookup, wtMatrix)
		ShortestPath = nx.shortest_path(G, A, B)
		print "Approx Path:", FullPath, "Length of Approx Path:", len(FullPath)
		print "Actual Path:", ShortestPath, "Length of Approx Path:", len(ShortestPath)
		i = int(input("Continue?? (1/0) "))

def main(numOfNodes, Type, characteristic, twoWords):
	'''
	numOfNodes: No. of nodes in the Graph
	Type: type of the input graph
	characteristic:
		Scale Free Networks: Degree
		Erdos Renyi Graph: Probability
		Real World Network: name
		Bridge Graph: Probability
	fn: The function of n that is used to find out the number of hotspots
	VtoHTrials: The number of trials to take for computing the average steps from randomly chosen vertices to enter the hotspot zone
	ReinVSshortestTrials: The number of trials to compute the average ratio of (length of Reinforced Path)/(length of Actual Shortest Path) for a randomly chosen pair of vertices
	RandVSReinTrials: The number of trials to compute the average ratio of (length of single source drunkard walk)/(length of Reinforced Paths) for a randomly chosen pair of vertices
	SudarshanTrials: The number of trials to compute the average values of several quantities and ratios
	'''
	
	if( Type == "RW"):
		fin = open(characteristic + ".graph")
		string = fin.read()
		G = pickle.loads(string)
		fin.close()
	
		fin = open(characteristic + ".wtMatrix")
		string = fin.read()
		wtMatrix = pickle.loads(string)
		fin.close()
	
		fin = open(characteristic + ".Flagger")
		string = fin.read()
		Flagger = pickle.loads(string)
		fin.close()
	
		fin = open(characteristic + ".map")
		string = fin.read()
		Maps = pickle.loads(string)
		#print Maps
		
		MappingItoW = {}
		for pair in Maps:
			MappingItoW[pair[1]] = pair[0]

		MappingWtoI = {}
		for pair in Maps:
			MappingWtoI[pair[0]] = pair[1]

		Flagger.reverse() #Reverse the Flagger List. Hence, its now in descending order.

		Flags = [f for f,v in Flagger]
		'''
		Maximum Sum
		'''
		maxLen = 0
		maxPoint = -1
		last = int((len(Flags) - 1)/5)
		for i in range(1,last):
			Len = math.sqrt( math.pow(i-0,2) + math.pow(Flags[i] - Flags[0],2) ) + math.sqrt( math.pow(last - i,2) + math.pow(Flags[i] - Flags[last],2) )
			if Len > maxLen:
				maxLen = Len
				maxPoint = i
		#print "Cut Point based on Sum:", maxPoint
		fn = maxPoint
		
		hotSpots = Flagger[:int(fn)] #Choose the top fn number of tuples
		hotSpots = [v for f,v in hotSpots] #Convert the tuples to vertices
		hotSpotLookup = createLookup(hotSpots,G) #Create a Lookup Table for the hotSpot List
		
		#print Flagger
		#print hotSpots

		twoWords = random.sample([v for v,i in Maps],2)
		print twoWords

		hotWords = []	
		for h in hotSpots:
			 hotWords.append(MappingItoW[h])
		print "Hot Spots:", hotWords
		
		transitionDA = []
		transitionML = []
		pathDA = nx.shortest_path(G,MappingWtoI[twoWords[0]], MappingWtoI[twoWords[1]], False)
		pathML = test(G,MappingWtoI[twoWords[0]],MappingWtoI[twoWords[1]],hotSpots,hotSpotLookup,wtMatrix)
		
		for vertex in pathDA:
			transitionDA.append(MappingItoW[vertex])
		
		for vertex in pathML:
			transitionML.append(MappingItoW[vertex])
		
		print "Djikstras: ", transitionDA
		print "The Deduced Path: ", transitionML
		
		#fullPath = test(G,MappingWtoI[twoWords[0]],MappingWtoI[twoWords[1]],hotSpots,hotSpotLookup,wtMatrix)
		'''
		CloseCent = nx.closeness_centrality(G)
		closeness = []
		for v in pathML:
			closeness.append(CloseCent[v])

		plt.plot(closeness)
		plt.xlabel('Vertex Number')
		plt.ylabel('Closeness Centrality Index')
		plt.text(0, numpy.average(closeness), [MappingItoW[v] for v in pathML])
		plt.show()
		'''
		#print MappingItoW[v], v, CloseCent[v]

		'''
		pathA = findHotSpotPath(G, MappingWtoI[twoWords[0]], MappingWtoI[twoWords[1]], hotSpots, wtMatrix) #Find a path from A to any of the Hot Spot
		pathB = findHotSpotPath(G, MappingWtoI[twoWords[1]], MappingWtoI[twoWords[0]], hotSpots, wtMatrix) #Find a path from B to any of the Hot Spot
		pathH = hotSpotLookup[ hotSpots.index(pathA[-1]) ][ hotSpots.index(pathB[-1]) ]
		fullPath = findFullPath(pathA, pathB, hotSpots, hotSpotLookup) #Integrate the two paths A and B

		transitionA = []
		transitionB = []
		transitionH = []
		for vertex in pathA:
			transitionA.append(MappingItoW[vertex])
		for vertex in pathB:
			transitionB.append(MappingItoW[vertex])
		for vertex in pathH:
			transitionH.append(MappingItoW[vertex])	

		print "Path A:", transitionA
		print "Path B:", transitionB
		print "Path H:", transitionH

		#plt.plot(Flags)
		#plt.show()
		'''
