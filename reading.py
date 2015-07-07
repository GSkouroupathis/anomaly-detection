# Reading Class
class Reading(object):
	
	#Class constructor
	def __init__(self, nodeID, date, time, temperature):
		self.nodeID = nodeID
		self.date = date
		self.time = time
		self.temperature = temperature
		
	def __repr__(self):
		return str(self.nodeID)+' '+str(self.date)+' '+str(self.time)+' '+str(self.temperature)
		
	def __str__(self):
		return str(self.nodeID)+' '+str(self.date)+' '+str(self.time)+' '+str(self.temperature)