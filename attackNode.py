class AttackNode(object):

	def __init__(self, tree, nodeID, segmentInfo):
		self.tree = tree
		self.nodeID = nodeID
		self.segmentInfo = segmentInfo
		self.children = []

	def isEndSegment(self):
		'''
		try:
			targetTime = next(i for (i,d) in enumerate( map(lambda x: x-self.tree.targetTemp, self.dataset) if abs(d) < self.tree.threshold))-1
		except:
			targetTime = -1
		if targetTime != -1:
			return (True, targetTime)
		else:
			return (False, targetTime)
		'''
		
	def set_children(segmentsNodes):
		self.children = segmentsNodes
		
		
		