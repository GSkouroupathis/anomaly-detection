from attackNode import AttackNode

class AttackTree(object):

	def __init__(self, nodeID, threshold, targetTemp, condProbTable):
		self.nodeID = nodeID
		self.threshold = threshold
		self.targetTemp = targetTemp
		self.condProbTable = condProbTable
		
	def set_start_node(self, atckNode):
		self.startNode = atckNode
	
	def tree_attack(self, nodeID, sensorsSegmentsReadingsDic):
		nodesLeft = [AttackNode(self, nodeID, segmentInfo) for segmentInfo in sensorsSegmentsReadingsDic[nodeID]]
		nodesToIter = [self.startNode]
		
		for node in nodesToIter:
			pass