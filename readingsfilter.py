# Readings Filter Class
class ReadingsFilter(object):
	
	# Constructor
	def __init__(self, readings=None, threshold=None):
		self.readings = readings
		self.threshold = threshold
	
	def set_readings(self, readings):
		self.readings = readings
	
	# Sets filtering threshold
	def set_threshold(self, threshold):
		self.threshold = threshold
		
	# Filters Readings
	def filter_readings(self):
		filteredReadings = []
		dataset = map(lambda x: x[3], self.readings)
		
		if len(dataset) < 3:
			return []
		
		# Find if first value is anomalous
		if abs(dataset[0] - dataset[1]) < self.threshold \
		or abs(dataset[0] - dataset[2]) < self.threshold:
			filteredReadings.append(self.readings[0])
			if self.readings[0][3] < 10:
				print self.readings[i-1]
		else:
			print ' ** Reading 0 dropped due to high variance'

		# Find if middle values are anomalous
		prevVal = dataset[0]
		thisVal = dataset[1]
		for (i, nextVal) in enumerate(dataset[2:]):
			i+= 2
			if (i-1)%500==0:	print ' -- filtering record', i-1
			if ((abs(thisVal - prevVal) < self.threshold \
			or abs(thisVal - nextVal) < self.threshold) \
			or (thisVal > prevVal and thisVal < nextVal \
			or thisVal < prevVal and thisVal > nextVal)) \
			and abs(thisVal - filteredReadings[-1][3]) < self.threshold:
				filteredReadings.append(self.readings[i-1])
				if self.readings[i-1][3] < 10:
					print self.readings[i-1]
					print filteredReadings[-1]
					print thisVal
			else:
				pass
			prevVal = thisVal
			thisVal = nextVal

		# Find if last value is anomalous
		li = len(dataset)-1
		if (abs(dataset[li] - dataset[li-1]) < self.threshold \
		or abs(dataset[li] - dataset[li-2]) < self.threshold) \
		and abs(dataset[li] - filteredReadings[-1][3]) < self.threshold:
			filteredReadings.append(self.readings[li])
			if self.readings[li][3] < 10:
				print self.readings[i-1]
		else:
			print ' ** Reading', li, 'dropped due to high variance'
	
		return filteredReadings