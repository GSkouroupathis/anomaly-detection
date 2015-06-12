import datetime

# Testing set: first day
# Training set: rest of them
def getTrainingTesting(readings):
	# get dataset
	dataset = map(lambda x: x[-1], readings)
	
	# get first datetime
	firstDaTi = datetime.datetime.strptime(readings[0][1]+' '+readings[0][2], '%Y-%m-%d %H:%M:%S')

	# add two day
	dateAfter = firstDaTi + datetime.timedelta(2, 0)

	# find end of testing data
	endTestI = 0
	minDiff = abs(dateAfter - datetime.datetime.strptime(readings[0][1]+' '+readings[0][2], '%Y-%m-%d %H:%M:%S'))
	for (i, r) in enumerate(readings):
		rDateTime = datetime.datetime.strptime(r[1]+' '+r[2], '%Y-%m-%d %H:%M:%S')
		diff = abs(dateAfter - rDateTime)
		if diff <= minDiff:
			minDiff = diff
			endTestI = i
		else:
			break
	
	return (dataset[endTestI+1:], dataset[:endTestI+1], readings[endTestI+1:], readings[:endTestI+1])
	
	
	