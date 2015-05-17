class AttackNode(object):

	def __init__(self, tree, nodeID, segmentInfo):
		self.tree = tree
		self.nodeID = nodeID
		self.segmentInfo = segmentInfo
		self.children = []
		self.signalUntil = []

	def set_signal_until(self, signal):
		self.signalUntil = signal
		return self
		
	def set_children(self, segmentsNodes):
		self.children = segmentsNodes
		
	#Class representation
	def __repr__(self):
		return "tree:" + str(self.tree) + "\nnodeID: " + str(self.nodeID) + "\nsegmentInfo: " + str(self.segmentInfo) + "\nchildren: " + str(self.children)
		