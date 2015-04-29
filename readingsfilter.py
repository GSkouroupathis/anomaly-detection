class ReadingsFilter(object):
	
	def __init__(self, readings=None, threshold=None):
		self.readings = readings
		self.threshold = threshold
	
	def set_readings(self, readings):
		self.readings = readings
		self.wat = self.readings[0][0]
		
	def set_threshold(self, threshold):
		self.threshold = threshold
		
	def filter_readings(self):
		filteredReadings = []
		dataset = map(lambda x: x[3], self.readings)
		
		if len(dataset) < 3:
			return []
		
		# Find if first value is anomalous
		if abs(dataset[0] - dataset[1]) < self.threshold \
		or abs(dataset[0] - dataset[2]) < self.threshold:
			filteredReadings.append(self.readings[0])
		else:
			print ' ** Reading 0 dropped due to high variance'

		# Find if middle values are anomalous
		prevVal = dataset[0]
		thisVal = dataset[1]
		for (i, nextVal) in enumerate(dataset[2:]):
			i+= 2
			if (i-1)%500==0:	print ' -- filtering record', i-1
			if (abs(thisVal - prevVal) < self.threshold \
			or abs(thisVal - nextVal) < self.threshold) \
			or (thisVal >= prevVal and thisVal <= nextVal \
			or thisVal <= prevVal and thisVal >= nextVal):
				filteredReadings.append(self.readings[i-1])
			else:
				pass
				'''
				if i > 40000 and self.wat==3:
					print ' ** Reading', i-1, 'dropped due to high variance'
					print prevVal, thisVal, nextVal
					print self.readings[i-1]
					'''
			prevVal = thisVal
			thisVal = nextVal

		# Find if last value is anomalous
		li = len(dataset)-1
		if abs(dataset[li] - dataset[li-1]) < self.threshold \
		or abs(dataset[li] - dataset[li-2]) < self.threshold:
			filteredReadings.append(self.readings[li])
		else:
			print ' ** Reading', li, 'dropped due to high variance'
	
		return filteredReadings