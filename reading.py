class Reading(object):
	
	#Class constructor
	def __init__(self, nodeID, date, time, temperature):
		self.nodeID = nodeID
		self.date = date
		self.time = time
		self.temperature = temperature