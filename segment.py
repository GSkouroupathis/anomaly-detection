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
			self.meanDer1 = np.mean(self.der1)
			self.der2 = np.gradient(self.der1)
			self.meanDer2 = np.mean(self.der2)
			self.dTemp = dataset[-1] - dataset[0]
			
	# Set segment dataset
	def set_readings(self, readings):
		self.readings = readings
		dataset = map(lambda r:r.temperature, readings)
		self.dataset = np.array(dataset)
		self.der1 = np.gradient(self.dataset)
		self.meanDer1 = np.mean(self.der1)
		self.der2 = np.gradient(self.der1)
		self.meanDer2 = np.mean(self.der2)
		self.dTemp = dataset[-1] - dataset[0]
	
	# Interpolate for 1 point
	def forecast(self):
		y1 = self.dataset[-1]
		y0 = self.dataset[-2]
		return (y1-y0)*2+y0
	
	# Calculate relative variance of first derivative
	def calc_der1_rel_var(self, mean):
		return np.mean(abs((self.der1-mean)**2))
	
	def __repr__(self):
		return str(self.nodeID)+'|'+str(self.cluster)+'|'+str(self.startDate)+'|'+str(self.startTime)+'|'+str(self.endDate)+'|'+str(self.endTime)+'|'+str(self.readings)
			
	def __str__(self):
		return str(self.nodeID)+'|'+str(self.cluster)+'|'+str(self.startDate)+'|'+str(self.startTime)+'|'+str(self.endDate)+'|'+str(self.endTime)+'|'+str(self.readings)