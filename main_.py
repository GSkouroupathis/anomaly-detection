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
import scipy as sp
from scipy import stats

	
# Initialises tables
# parses the readings,
# filters the dataset
# and stores them into DB
def init():
	# Parse readings
	parsedReadings = parse_file("data/data.txt", maxTemp=32)

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
	# not ran in this project
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

# Prepare MC Attack
def prep():
	#dummy = raw_input('This is gonna take forever (press any key to continue)')
	dbOp.connectToDatabase("data/db")
	nodes = map(lambda x: x[0], dbOp.selectAllNodes())
	for node in nodes:
		try:
			print ">>", node
			readings = dbOp.selectReadingsFromNode(node)
			(dTr, dTe, rTr, rTe) = data.getTrainingTesting(readings)
			if len(dTr) == 0: raise Exception('No training data')
			mcMimicry = MCMimicry(dTr)
			(w, segments, centroids, labels, condProbTable, K, score) = mcMimicry.prepare()
			if w is None: continue
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
				print ">>>segment", i+1,"/",uouo
				start_date = rTr[i*w][1]
				start_time = rTr[i*w][2]
				end_date = rTr[i*w + w -1][1]
				end_time = rTr[i*w + w -1][2]
				print start_time, '-', end_time
				cluster_id = int(labels[i])
				dbOp.insertReadingSegment(node, start_date, start_time, end_date, end_time, node, cluster_id)
			dbOp.setNodeAvail(node)
		except:
			print "Node", node, "failed to initialize"
	dbOp.closeConnectionToDatabase()

#prep()

# Launch Attack
def launch(atck, attackedSensor, usedSensors, goal, tdelay):
	
	dbOp.connectToDatabase("data/db") ##
	
	# readings & info & also segments

	noOfDimensions = dbOp.getNoOfDimensions(rootNodeID = attackedSensor)
	nodesSegmentsDic = {}
	
	for nodeID in usedSensors:
	
		readingsInfo = dbOp.selectReadingsFromNode(nodeID)
		(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
		readings = [Reading(r[0],r[1],r[2],r[3]) for r in rTr]
		
		# get segments
		segsInfo = dbOp.selectSegmentsFromNode(nodeID)
		segments = [Segment(segInfo[0],segInfo[6],segInfo[1],segInfo[2],segInfo[3],segInfo[4]) for segInfo in segsInfo]
		
		for (i, segment) in enumerate(segments):
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
		(startSignal, iSignal) = mcMimicry.greedy_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 2:
		(startSignal, iSignal) = mcMimicry.random_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 3:
		(startSignal, iSignal) = mcMimicry.greedy_smooth_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 4:
		(startSignal, iSignal) = mcMimicry.first_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 5:
		(startSignal, iSignal) = mcMimicry.super_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 6:
		(startSignal, iSignal) = mcMimicry.rgv_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.65, propdifftemp=0.35, comeBack=1)
	elif atck == 7:
		(startSignal, iSignal) = mcMimicry.softmax_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.75, temp=0.03, comeBack=1)
	elif atck == 8:
		(startSignal, iSignal) = mcMimicry.not_working_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, comeBack=True)
	elif atck == 9:
		(startSignal, iSignal) = mcMimicry.cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable)
	elif atck == 10:
		(startSignal, iSignal) = mcMimicry.ditto_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, comeBack=True)
	elif atck == 11:
		(startSignal, iSignal) = mcMimicry.rwgm_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propMean=0.45, propWDer1=0.2, propDTemp=0.65, comeBack=True)
	else:
		return None
	return (startSignal, iSignal, dTraining, dTesting)


####

# Example attack
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
(startSignal, iSignal, dTraining, dTesting) = launch(11, attackedSensor, usedSensors, goal, tdelay)

# Detect anomalies in attack
EKFd = EKFDetector(iSignal, dTraining)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]

# Terence mimicry
terMimicry = TerMimicry()
terSignal = terMimicry.attack(dTraining, 30, 0, [dTraining])[0]
EKFd = EKFDetector(terSignal, dTraining)
CUSUMd = CUSUMDetector(terSignal, h=0.4, w=10, EKFd=EKFd)
terRes  = CUSUMd.detect()[0]

# Detect anomalies in Terence attack
EKFd = EKFDetector(dTesting, dTraining)
CUSUMd = CUSUMDetector(dTesting, h=0.4, w=10, EKFd=EKFd)
res3  = CUSUMd.detect()[0]

# Detect Original signal
EKFd = EKFDetector(dTraining, dTraining)
CUSUMd = CUSUMDetector(dTraining, h=0.4, w=10, EKFd=EKFd)
baseRes  = CUSUMd.detect()[0]

#threshold
thr = 1.362

# Find detection rates
# --

# Original signal
top = 0
for i in res3:
	if i > thr or i <-thr:
		top += 1
print "Real Detection rate:", top*1.0/len(res3)

# MC attack signal
top = 0
for i in res:
	if i > thr or i <-thr:
		top += 1
print "MC attack Detection rate:", top*1.0/len(res)

# Terence attack signal
top = 0
for i in terRes:
	if i > thr or i <-thr:
		top += 1
print "Terence Detection rate:", top*1.0/len(terRes)

# Plot stuff
thrL=[thr] * len(dTesting)
thrLNeg=[-thr] * len(dTesting)

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(iSignal, linewidth=2, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
plt.plot(dTesting, linewidth=3, linestyle="--",  c="green", solid_capstyle="butt", label="Temperature")
plt.plot(terSignal, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt")
plt.xlabel('Time')

plt.subplot(2,1,2)
plt.plot(thrL, linewidth=2, linestyle="--", c="black", solid_capstyle="butt", label="Threshold")
plt.plot(thrLNeg, linewidth=2, linestyle="--", c="black", solid_capstyle="butt")
plt.plot(res3, linewidth=2, linestyle="--", c="green", solid_capstyle="butt", label="Temperature")
plt.plot(res, linewidth=1, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
plt.plot(terRes, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt", label="Past Attack")
plt.legend(loc='upper right')
plt.ylabel('$S_N$ Values')
plt.xlabel('Time')

plt.show()