import numpy as np

class Segment(object):
	
	#Class constructor
	def __init__(self, nodeID, cluster, startDate, startTime, endDate, endTime, dataset):
		self.nodeID = nodeID
		self.cluster = cluster
		self.startDate = startDate
		self.startTime = startTime
		self.endDate = endDate
		self.endTime = endTime
		self.dataset = np.array(dataset)
		self.der1 = np.gradient(self.dataset)
		self.der2 = np.gradient(self.der1)