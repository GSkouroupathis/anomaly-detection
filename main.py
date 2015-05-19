from fileparser import *
from readingsfilter import *
import dbOp, datetime, testing, attack
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
	inp = 0
	for node in parsedReadings.keys():

		parsedReadings[node].sort(key=lambda x: datetime.datetime.strptime(x[1]+' '+x[2], '%Y-%m-%d %H:%M:%S'))
		inp += len(parsedReadings[node])
		
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
		i += 1
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
r3 = map(lambda x: x[0], dbOp.selectDatasetFromNode(3))
dbOp.closeConnectionToDatabase()




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

terMimicry = TerMimicry()
falseSignal = terMimicry.attack(r3, 28, 0, [r3])[0]




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
	
	# readings & info
	nodes_readings = {}
	allReadings = dbOp.selectReadingsFromNodes(usedSensors)
	for r in allReadings:
		nodeID = r[0]
		if nodeID in nodes_readings.keys():
			nodes_readings[nodeID].append(r[1:])
		else:
			nodes_readings[nodeID] = [r[1:]]
	
	# segments from used sensors
	noOfDimensions = dbOp.getNoOfDimensions(rootNodeID = attackedSensor)

	nodes_segments_reads = {}
	node_segments_reads = dbOp.selectSegments(root_node_id=attackedSensor, node_list=usedSensors)
	for (i, n_segment) in enumerate(node_segments_reads):
		nodeID = n_segment[0]
		if nodeID in nodes_segments_reads.keys():
			nodes_segments_reads[nodeID].append(n_segment[1:])
		else:
			nodes_segments_reads[nodeID] = [n_segment[1:]]
		nodes_segments_reads[nodeID][-1] += (nodes_readings[nodeID][i*noOfDimensions:i*noOfDimensions+noOfDimensions],)
	
	# conditional probabilities table
	K = dbOp.selectClusterGroup(root_node_id=attackedSensor)[0][1]
	cond_probs_table = [[0]*K for i in range(K)]
	cond_probs = dbOp.selectCondProbs(root_node_id=attackedSensor)
	for probArray in cond_probs:
		bCluster = probArray[1]
		aCluster = probArray[2]
		prob = probArray[3]
		cond_probs_table[bCluster][aCluster] = prob
	
	dbOp.closeConnectionToDatabase() ##
	
	mcMimicry = MCMimicry()
	if atck == 0:
		(startSignal, iSignal) = mcMimicry.tree_attack(attackedSensor, goal, tdelay, nodes_segments_reads, cond_probs_table)
	elif atck == 1:
		(startSignal, iSignal) = mcMimicry.random_attack(attackedSensor, goal, tdelay, nodes_segments_reads, cond_probs_table)
	else:
		return None
	return (startSignal, iSignal)


(startSignal, iSignal) = launch(1)
#iSignal = iSignal[0]
#badSignal = iSignal[len(startSignal):]

#iSignal = [20.204, 19.4396, 19.4102, 19.4102, 19.4004, 19.371, 19.3612, 19.3612, 19.3612, 19.3612, 19.3514, 19.371, 20.204, 19.4396, 19.4102, 19.4102, 19.4004, 19.371, 19.3612, 19.3612, 19.3612, 19.3612, 19.3514, 19.371, 19.1652, 19.1456, 19.1848, 19.1848, 19.175, 19.1652, 19.1652, 19.5474, 19.8316, 20.302, 21.1644, 21.8504, 22.0856, 21.9876, 22.1052, 22.2032, 22.5168, 23.4184, 24.0456, 24.9472, 25.4666, 25.3784, 25.4568, 25.251, 25.398, 25.4078, 25.4862, 25.5254, 25.6822, 25.6626, 25.8194, 25.8194, 25.8194, 25.8488, 25.9076, 26.0056, 26.0056, 25.8488, 26.1624, 26.133, 26.2016, 26.5544, 26.7406, 26.9268, 27.113, 27.2796, 27.2502, 27.26, 27.2992, 27.5736, 27.603, 27.7304, 27.6618, 27.9166]

EKFd = EKFDetector(iSignal)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]
#badRes = res[len(startSignal):]

'''
EKFd = EKFDetector(falseSignal)
CUSUMd = CUSUMDetector(falseSignal, h=0.4, w=10, EKFd=EKFd)
res2  = CUSUMd.detect()[0]
'''

EKFd = EKFDetector(r3)
CUSUMd = CUSUMDetector(r3, h=0.4, w=10, EKFd=EKFd)
res3  = CUSUMd.detect()[0]

yo=50
top = 0
for i in res:
	if i > 1.3 or i <-1.3:
		top += 1
print "Detection rate:", top*1.0/len(res)

# Plot stuff
import matplotlib.pyplot as plt
#plt.axis('equal')

plt.plot(r3, 'b', res3, 'm', iSignal, 'r', res, 'g')

plt.show()