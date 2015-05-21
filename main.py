from fileparser import *
from readingsfilter import *
import dbOp, datetime, testing, attack, data
from reading import *
from segment import *
from ekfdetector import *
from cusumdetector import *
from terattack import *
from mcattack import *
from plotter import *
import numpy
	
# Initialises tables
# parses the readings,
# filters the dataset
# and stores them into DB
def init():
	# Parse readings
	parsedReadings = parse_file("data/data.txt")

	# Filter readings
	allFilteredReadings = []
	readingsFilter = ReadingsFilter(threshold=4)

	for node in parsedReadings.keys():

		parsedReadings[node].sort(key=lambda x: datetime.datetime.strptime(x[1]+' '+x[2], '%Y-%m-%d %H:%M:%S'))
		
		readingsFilter.set_readings( parsedReadings[node] )
		filteredReadings = readingsFilter.filter_readings()
		allFilteredReadings += filteredReadings

	# Initialise tables
	dbOp.connectToDatabase("data/db")
	print '>> Initializing tables'
	dbOp.initTables()
	print '>> DONE'

	# Insert readings (first phase)
	dbOp.insertAllReadings(allFilteredReadings)

	# Calculate and insert Covariances
	'''
	nodes = map(lambda x: x[0], dbOp.selectAllNodes())
	for (i, node1) in enumerate(nodes):
		readings1 = dbOp.selectReadingsFromNode(node1)
		for node2 in nodes[i+1:]:
			readings2 = dbOp.selectReadingsFromNode(node2)
			sL = min(len(readings1), len(readings2))
			cov = numpy.cov(readings1[:sL], readings2[:sL])[0][1]
			dbOp.insertCov(node1, node2, cov)
	'''
	dbOp.closeConnectionToDatabase()
	
#init()

## Stuff ##
dbOp.connectToDatabase("data/db")
r3 = dbOp.selectReadingsFromNode(3)
dbOp.closeConnectionToDatabase()
(d3Training, d3Testing, r3Tr, r3Te) = data.getTrainingTesting(r3)


## CUSUM detection ##
'''
EKFd = EKFDetector(readings)
CUSUMd = CUSUMDetector(readings, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()
'''



# Calculates correlations
'''
cor = []
for (i,r) in readings:
	wat = min(len(r),len(r3))
	cor.append((i, numpy.cov(r3[:wat],r[:wat])[0][1]))
cor = sorted(cor, key=lambda x: x[1], reverse=True)
for i in cor:
	print i
'''



# Terence mimicry
'''
terMimicry = TerMimicry()
terSignal = terMimicry.attack(d3Training, 28, 0, [d3Training])[0]
'''

# MC mimicry
# Prepare Attack
def prep():
	dbOp.connectToDatabase("data/db")
	nodes = map(lambda x: x[0], dbOp.selectAllNodes())
	for node in nodes:
		print ">>", node
		readings = dbOp.selectReadingsFromNode(node)
		dataset = map(lambda x: x[0], dbOp.selectDatasetFromNode(node))
		mcMimicry = MCMimicry(dataset)
		(w, segments, centroids, labels, condProbTable, K, score) = mcMimicry.prepare()
		# insert cluster group
		dbOp.insertClusterGroup(node, K, w)
		# insert clusters
		for (i, centroid) in enumerate(centroids):
			dbOp.insertCluster(node, i, str(centroid))
			print ">>>cluster", i
		# insert conditional probabilities
		for (i, bClusterList) in enumerate(condProbTable):
			for (j, aClusterProb) in enumerate(bClusterList):
				dbOp.insertConditionalProbability(node, i, j, aClusterProb)
				print ">>>", i,j
		# insert reading segments
		uouo=len(segments)
		for (i, segment) in enumerate(segments):
			print ">>>segment", i,"/",uouo
			start_date = readings[i*w][1]
			start_time = readings[i*w][2]
			end_date = readings[i*w + w -1][1]
			end_time = readings[i*w + w -1][2]
			print start_time, '-', end_time
			cluster_id = int(labels[i])
			dbOp.insertReadingSegment(node, start_date, start_time, end_date, end_time, node, cluster_id)
	dbOp.closeConnectionToDatabase()

#prep()

# Launch Attack
def launch(atck):
	attackedSensor = 3
	usedSensors = (3,)
	goal = 28
	tdelay = 1
	
	dbOp.connectToDatabase("data/db") ##
	
	# readings & info & also segments

	noOfDimensions = dbOp.getNoOfDimensions(rootNodeID = attackedSensor)
	nodesSegmentsDic = {}
	
	for nodeID in usedSensors:
	
		readingsInfo = dbOp.selectReadingsFromNode(nodeID)
		if len(filter(lambda x:x[-1]>29, readingsInfo)) > 0:
			print 33
			asd=raw_input('d')
		readings = [Reading(r[0],r[1],r[2],r[3]) for r in readingsInfo]
		(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
		# get only segments after first training datetime
		firstTrainDateTime = datetime.datetime.strptime(rTr[0][1]+' '+rTr[0][2], '%Y-%m-%d %H:%M:%S')
		segsInfo = filter(lambda segInfo: datetime.datetime.strptime(segInfo[1]+' '+segInfo[2], '%Y-%m-%d %H:%M:%S')>=firstTrainDateTime, dbOp.selectSegmentsFromNode(nodeID))
		segments = [Segment(segInfo[0],segInfo[6],segInfo[1],segInfo[2],segInfo[3],segInfo[4]) for segInfo in segsInfo]
		
		for (i, segment) in enumerate(segments):
			segStartDateTime = datetime.datetime.strptime(segment.startDate+' '+segment.startTime, '%Y-%m-%d %H:%M:%S')
			segEndDateTime = datetime.datetime.strptime(segment.endDate+' '+segment.endTime, '%Y-%m-%d %H:%M:%S')
			#set readings
			segReadings = readings[i*noOfDimensions:(i+1)*noOfDimensions]
			segment.set_readings(segReadings)
			
		# set segments
		nodesSegmentsDic[nodeID] = segments
		
	# conditional probabilities table
	K = dbOp.selectClusterGroup(root_node_id=attackedSensor)[0][1]
	condProbsTable = [[0]*K for i in range(K)]
	condProbs = dbOp.selectCondProbs(root_node_id=attackedSensor)
	for probArray in condProbs:
		bCluster = probArray[1]
		aCluster = probArray[2]
		prob = probArray[3]
		condProbsTable[bCluster][aCluster] = prob

	dbOp.closeConnectionToDatabase() ##
	
	mcMimicry = MCMimicry()
	if atck == 0:
		(startSignal, iSignal) = mcMimicry.tree_attack(attackedSensor, goal, tdelay, sensorsSegmentsReadingsDic, cond_probs_table)
	elif atck == 1:
		(startSignal, iSignal) = mcMimicry.random_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	else:
		return None
	return (startSignal, iSignal)


(startSignal, iSignal) = launch(1)
#iSignal = iSignal[0]
#badSignal = iSignal[len(startSignal):]

#iSignal = [20.204, 19.4396, 19.4102, 19.4102, 19.4004, 19.371, 19.3612, 19.3612, 19.3612, 19.3612, 19.3514, 19.371, 20.204, 19.4396, 19.4102, 19.4102, 19.4004, 19.371, 19.3612, 19.3612, 19.3612, 19.3612, 19.3514, 19.371, 19.1652, 19.1456, 19.1848, 19.1848, 19.175, 19.1652, 19.1652, 19.5474, 19.8316, 20.302, 21.1644, 21.8504, 22.0856, 21.9876, 22.1052, 22.2032, 22.5168, 23.4184, 24.0456, 24.9472, 25.4666, 25.3784, 25.4568, 25.251, 25.398, 25.4078, 25.4862, 25.5254, 25.6822, 25.6626, 25.8194, 25.8194, 25.8194, 25.8488, 25.9076, 26.0056, 26.0056, 25.8488, 26.1624, 26.133, 26.2016, 26.5544, 26.7406, 26.9268, 27.113, 27.2796, 27.2502, 27.26, 27.2992, 27.5736, 27.603, 27.7304, 27.6618, 27.9166]

EKFd = EKFDetector(iSignal, d3Training)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]
#badRes = res[len(startSignal):]

'''
EKFd = EKFDetector(terSignal, d3Training)
CUSUMd = CUSUMDetector(terSignal, h=0.4, w=10, EKFd=EKFd)
terRes  = CUSUMd.detect()[0]
'''

EKFd = EKFDetector(d3Testing, d3Training)
CUSUMd = CUSUMDetector(d3Testing, h=0.4, w=10, EKFd=EKFd)
res3  = CUSUMd.detect()[0]

yo=len(iSignal)
top = 0
for i in res:
	if i > 1.3 or i <-1.3:
		top += 1
print "Detection rate:", top*1.0/len(res)

# Plot stuff
import matplotlib.pyplot as plt
#plt.axis('equal')
plt.plot(d3Testing, 'g', res3, 'y', iSignal, 'r', res, 'm')
plt.show()
