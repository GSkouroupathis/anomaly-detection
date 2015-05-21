import numpy as np

class Segment(object):
	
	# Class constructor
	def __init__(self, nodeID, cluster, startDate, startTime, endDate, endTime, readings=None):
		self.nodeID = nodeID
		self.cluster = cluster
		self.startDate = startDate
		self.startTime = startTime
		self.endDate = endDate
		self.endTime = endTime
		if readings:
			self.readings = readings
			dataset = map(lambda r:r.temperature, readings)
			self.dataset = np.array(dataset)
			self.der1 = np.gradient(self.dataset)
			self.der2 = np.gradient(self.der1)
			
	# Set segment dataset
	def set_readings(self, readings):
		self.readings = readings
		dataset = map(lambda r:r.temperature, readings)
		self.dataset = np.array(dataset)
		self.der1 = np.gradient(self.dataset)
		self.der2 = np.gradient(self.der1)
		
	def __str__(self):
		return str(self.nodeID)+'|'+str(self.cluster)+'|'+str(self.startDate)+'|'+str(self.startTime)+'|'+str(self.endDate)+'|'+str(self.endTime)+'|'+str(self.readings)