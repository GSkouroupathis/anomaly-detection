from attackNode import AttackNode

class AttackTree(object):

	def __init__(self, nodeID, startSignal, threshold, targetTemp, condProbTable):
		self.nodeID = nodeID
		self.startSignal = startSignal
		self.threshold = threshold
		self.targetTemp = targetTemp
		self.condProbTable = condProbTable
		
	def set_start_node(self, atckNode):
		self.startNode = atckNode
	
	def tree_attack(self, nodeID, sensorsSegmentsReadingsDic):
		# attack signals
		atckPaths = []
		nodesLeft = [AttackNode(self, nodeID, segmentInfo) for segmentInfo in sensorsSegmentsReadingsDic[nodeID]]
		nodesToIter = [self.startNode]
		
		while len(nodesLeft) != 0:
			if len(nodesToIter) == 0: break
			newLevelNodesIn = [] # dummy to create new nodesToIter & nodesLeft
			for node in nodesToIter:
				print len(nodesLeft)
				# find if target temp is reached
				dataset = map(lambda x: x[-1], node.segmentInfo[6])
				endDatapoints = [i for (i, x) in \
				enumerate(dataset) if abs(self.targetTemp-x)<self.threshold]
				if len(endDatapoints) != 0:
					atckPaths.append(self.startSignal + node.signalUntil + dataset[:endDatapoints[0]])
					
				#find segs in next cluster & stitchable
				nodeCluster = node.segmentInfo[5]
				nextCluster = self.condProbTable[nodeCluster].index(max(self.condProbTable[nodeCluster]))
				childrenSegsIn = [i for (i,n) in enumerate(nodesLeft) if abs(dataset[-1]-n.segmentInfo[6][0][-1])<self.threshold]
				'''childrenSegsIn = [i for (i,n) in enumerate(nodesLeft) if n.segmentInfo[5] == nextCluster and abs(dataset[-1]-n.segmentInfo[6][0][-1])<self.threshold]'''
				
				# set node's children
				node.set_children([nodesLeft[i].set_signal_until(node.signalUntil + dataset) for i in childrenSegsIn])
				
				# manage nodesLeft & nodesToIter
				newLevelNodesIn += childrenSegsIn
				
			if len(atckPaths) != 0: 
				return atckPaths
			
			nodesToIter = [nodesLeft[i] for i in newLevelNodesIn]
			nodesLeft = list(set(nodesLeft) - set(nodesToIter))
		return atckPaths
	#Class representation
	def __repr__(self):
		return "I am a tree for node " + str(self.nodeID)