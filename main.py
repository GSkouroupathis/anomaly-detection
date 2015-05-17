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
falseSignal = terMimicry.attack(r3, 29, 0, [r3])[0]




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
def launch():
	attackedSensor = 3
	usedSensors = (3,)
	target = 29
	tdelay = 1300
	
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
	iSignal = mcMimicry.tree_attack(attackedSensor, target, tdelay, nodes_segments_reads, cond_probs_table)
	return iSignal
	
iSignal = launch()[0]
print iSignal
sss=raw_input()
EKFd = EKFDetector(iSignal)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]

EKFd = EKFDetector(falseSignal)
CUSUMd = CUSUMDetector(falseSignal, h=0.4, w=10, EKFd=EKFd)
res2  = CUSUMd.detect()[0]

# Plot stuff
import matplotlib.pyplot as plt
#plt.axis('equal')
plt.plot(r3[:15], 'b', iSignal, 'r', res, 'g', falseSignal, 'k', res2, 'c')
plt.show()