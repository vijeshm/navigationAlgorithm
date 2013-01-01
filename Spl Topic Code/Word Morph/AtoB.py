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
	for A in range( G.number_of_nodes() ): 
		for B in range( A+1, G.number_of_nodes() ): #We need not analyze (j,i) pair if we've analyzed (i,j) pair since the ordered pair just increases the redundancy.
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
			walkerAdj.remove(maxNeighbor) #remove the maxNeighbor from the neighbors list
			weights.remove(numpy.max(weights)) #remove the corresponding weight entry from the weights list
			index = weights.index(numpy.max(weights)) #calculate index again
			maxNeighbor = walkerAdj[index]	#calculate maxNeighbor again to verify whether it is present in the existing path
		
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

	H = nx.convert_node_labels_to_integers(G, ordering='sorted')	
	print "Nodes labels are converted to integers"

	nodes = G.nodes()
	nodes.sort()
	for i in range(len(nodes)):
		nodes[i] = (nodes[i], i)

	fout = open(name + ".graph", 'w')
	string = pickle.dumps(H)
	fout.write(string)
	fout.close()
	print name + ".graph File Generated"
	
	fout = open(name + ".map", 'w')
	string = pickle.dumps(nodes)
	fout.write(string)
	fout.close()
	print name + ".map File Generated"
	
	for i in nodes:
		if G.degree(i[0]) != H.degree(i[1]):
			print "Hoge"

	print "Undergoing Machine Learning..."
	reinforce(H, "RW", name)
	main(G.number_of_nodes(), "RW", name, math.e*numpy.log(G.number_of_nodes()), 1000)

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
		fn = maxPoint
		
		hotSpots = Flagger[:int(fn)] #Choose the top fn number of tuples
		hotSpots = [v for f,v in hotSpots] #Convert the tuples to vertices
		hotSpotLookup = createLookup(hotSpots,G) #Create a Lookup Table for the hotSpot List
		
		hotWords = []	
		for h in hotSpots:
			 hotWords.append(MappingItoW[h])

		transitionDA = []
		transitionML = []
		pathDA = nx.shortest_path(G,MappingWtoI[twoWords[0]], MappingWtoI[twoWords[1]], False)
		pathML = test(G,MappingWtoI[twoWords[0]],MappingWtoI[twoWords[1]],hotSpots,hotSpotLookup,wtMatrix)
		
		for vertex in pathDA:
			transitionDA.append(MappingItoW[vertex])
		
		for vertex in pathML:
			transitionML.append(MappingItoW[vertex])
		
		print "Path using Djikstras: ", transitionDA
		print "Path using Path Concatenation Algorithm: ", transitionML
