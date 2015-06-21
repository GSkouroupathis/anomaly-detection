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

# MC mimicry
# Prepare Attack
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
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
(startSignal, iSignal, dTraining, dTesting) = launch(11, attackedSensor, usedSensors, goal, tdelay)

EKFd = EKFDetector(iSignal, dTraining)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]

# Terence mimicry
terMimicry = TerMimicry()
terSignal = terMimicry.attack(dTraining, 30, 0, [dTraining])[0]
EKFd = EKFDetector(terSignal, dTraining)
CUSUMd = CUSUMDetector(terSignal, h=0.4, w=10, EKFd=EKFd)
terRes  = CUSUMd.detect()[0]

EKFd = EKFDetector(dTesting, dTraining)
CUSUMd = CUSUMDetector(dTesting, h=0.4, w=10, EKFd=EKFd)
res3  = CUSUMd.detect()[0]


#### Trick #
trickI = 18.8
trick=[]
while trickI<30:
	trick.append(trickI)
	trickI += 0.015
EKFd = EKFDetector(trick, dTraining)
CUSUMd = CUSUMDetector(trick, h=0.4, w=10, EKFd=EKFd)
trickRes  = CUSUMd.detect()[0]

EKFd = EKFDetector(dTraining, dTraining)
CUSUMd = CUSUMDetector(dTraining, h=0.4, w=10, EKFd=EKFd)
baseRes  = CUSUMd.detect()[0]

#threshold
thr = 1.362

top = 0
for i in baseRes:
	if i > thr or i < -thr:
		top += 1
print "base Detection rate:", top*1.0/len(baseRes)
top = 0
for i in res3:
	if i > thr or i <-thr:
		top += 1
print "real Detection rate:", top*1.0/len(res3)
top = 0
for i in res:
	if i > thr or i <-thr:
		top += 1
print "my Detection rate:", top*1.0/len(res)
top = 0
for i in terRes:
	if i > thr or i <-thr:
		top += 1
print "ter Detection rate:", top*1.0/len(terRes)
top = 0
for i in trickRes:
	if i > thr or i <-thr:
		top += 1
print "trick Detection rate:", top*1.0/len(trickRes)

thrL=[thr] * len(dTesting)
thrLNeg=[-thr] * len(dTesting)
# Plot stuff
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(dTesting, linewidth=2, linestyle="--",  c="green", solid_capstyle="butt", label="Temperature")
plt.plot(iSignal, linewidth=1, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
#plt.plot(trick, linewidth=2, linestyle="-", c="red", solid_capstyle="butt", label="Linear Attack")
#plt.plot(terSignal, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt")
#plt.legend(loc='upper right')
plt.ylabel('Temperature')

plt.subplot(2,1,2)
plt.plot(thrL, linewidth=2, linestyle="--", c="black", solid_capstyle="butt", label="Threshold")
plt.plot(thrLNeg, linewidth=2, linestyle="--", c="black", solid_capstyle="butt")
plt.plot(res3, linewidth=2, linestyle="--", c="green", solid_capstyle="butt", label="Temperature")
plt.plot(res, linewidth=1, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
#plt.plot(trickRes, linewidth=2, linestyle="-", c="red", solid_capstyle="butt", label="Linear Attack")
#plt.plot(terRes, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt", label="Past Attack")
plt.legend(loc='upper right')
plt.ylabel('$S_N$ Values')
plt.xlabel('Time')
plt.show()