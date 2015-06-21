import matplotlib.pyplot as plt
import dbOp, data
from segment import *
from reading import *
from ekfdetector import *
from cusumdetector import *
from mcattack import *
import numpy as np
import numpy.random

################################################################################
'''
dbOp.connectToDatabase("data/db") ##

readingsInfo = dbOp.selectReadingsFromNode(3)
(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
d3 = dTraining[:7000]

readingsInfo = dbOp.selectReadingsFromNode(10)
(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
d10 = dTraining[:7000]

readingsInfo = dbOp.selectReadingsFromNode(11)
(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
d11 = dTraining[:7000]

readingsInfo = dbOp.selectReadingsFromNode(26)
(dTraining, dTesting, rTr, rTe) = data.getTrainingTesting(readingsInfo)
d26 = dTraining[:7000]

dbOp.closeConnectionToDatabase() ##

plt.plot(d3, linewidth=1, linestyle="-", c="blue", solid_capstyle="butt", label="Sensor 3")
plt.plot(d10, linewidth=1, linestyle="--", c="red", solid_capstyle="butt", label="Sensor 10")
plt.plot(d11, linewidth=1, linestyle="-", c="green", solid_capstyle="butt", label="Sensor 11")
plt.plot(d26, linewidth=1, linestyle="--", c="magenta", solid_capstyle="butt", label="Sensor 26")

plt.ylabel('Temperature')
plt.xlabel('Time')
plt.legend(loc='upper right')
plt.show()
'''



################################################################################
'''
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
thr=1.362

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

allDete = []
for dummy in range(100):
	print '>', dummy+1
	(startSignal, iSignal) = mcMimicry.superer_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.65, propdifftemp=0.35, comeBack=1)
	
	EKFd = EKFDetector(iSignal, dTraining)
	CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
	res  = CUSUMd.detect()[0]

	top = 0
	for i in res:
		if i > thr or i <-thr:
			top += 1
	dete = top*1.0/len(res)
	allDete.append(dete)

print allDete
'''


################################################################################
'''
varDerivTests = [0.25, 0.5, 0.75]
tempTests = [0.01, 0.02, 0.03, 0.04]
for varDeriv in varDerivTests:
	allDete = []
	allTime = []
	for temp in tempTests:
		mnDete = 0
		for dummy in range(3):
			print 'temp:', temp, 'dummy:', dummy
			(startSignal, iSignal) = mcMimicry.softmax_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=varDeriv, temp=temp, comeBack=0)

			EKFd = EKFDetector(iSignal, dTraining)
			CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
			res  = CUSUMd.detect()[0]

			top = 0
			for i in res:
				if i > thr or i <-thr:
					top += 1
			dete = top*1.0/len(res)
			
			mnDete += dete
		mnDete /= 3.0
		allTime.append(len(iSignal))
		allDete.append(mnDete)
	print allDete
	
	plt.subplot(2,1,1)
	#plt.plot([0]*int(map(lambda x: x*100, dTempTests)[-1]+1), linewidth=1, linestyle="-", c="black", solid_capstyle="butt")
	plt.plot(tempTests, allDete, linewidth=1, linestyle="--", marker='o', markersize=10, c="red", solid_capstyle="butt", label=str(varDeriv))
	plt.title('$relGradVarPerc$ = '+ str(int(varDeriv*100)) + '%')
	plt.ylabel('Detection Rate')
	#plt.xlabel('$deltaTempPerc$')
	
	plt.subplot(2,1,2)
	#plt.plot([0]*int(map(lambda x: x*100, dTempTests)[-1]+1), linewidth=1, linestyle="-", c="black", solid_capstyle="butt")
	plt.plot(tempTests, allTime, linewidth=1, linestyle="--", marker='o', markersize=10, c="blue", solid_capstyle="butt", label=str(varDeriv))
	plt.ylabel('Attack Time')
	plt.xlabel('$T$')
	plt.show()

'''





################################################################################
'''
varDerivTests = [0.25, 0.5, 0.75]
dTempTests = [0.2, 0.4, 0.6, 0.8]
for varDeriv in varDerivTests:
	allDete = []
	allTime = []
	for dTemp in dTempTests:
		mnDete = 0
		for dummy in range(3):
			print 'dTemp:', dTemp, 'dummy:', dummy
			(startSignal, iSignal) = mcMimicry.superer_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=varDeriv, propdifftemp=dTemp, comeBack=0)

			EKFd = EKFDetector(iSignal, dTraining)
			CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
			res  = CUSUMd.detect()[0]

			top = 0
			for i in res:
				if i > thr or i <-thr:
					top += 1
			dete = top*1.0/len(res)
			
			mnDete += dete
		mnDete /= 3.0
		allTime.append(len(iSignal))
		allDete.append(mnDete)
	print allDete
	
	plt.subplot(2,1,1)
	plt.plot([0]*int(map(lambda x: x*100, dTempTests)[-1]+1), linewidth=1, linestyle="-", c="black", solid_capstyle="butt")
	plt.plot(map(lambda x: x*100, dTempTests), allDete, linewidth=1, linestyle="--", marker='o', markersize=10, c="red", solid_capstyle="butt", label=str(varDeriv))
	plt.title('$relGradVarPerc$ = '+ str(int(varDeriv*100)) + '%')
	plt.ylabel('Detection Rate')
	#plt.xlabel('$deltaTempPerc$')
	
	plt.subplot(2,1,2)
	plt.plot([0]*int(map(lambda x: x*100, dTempTests)[-1]+1), linewidth=1, linestyle="-", c="black", solid_capstyle="butt")
	plt.plot(map(lambda x: x*100, dTempTests), allTime, linewidth=1, linestyle="--", marker='o', markersize=10, c="blue", solid_capstyle="butt", label=str(varDeriv))
	plt.ylabel('Attack Time')
	plt.xlabel('$deltaTempPerc$')
	plt.show()
'''	
	
	
	
	
################################################################################	
'''
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
thr=1.362

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
allISignals = []
allResults = []

for dummy in range(5):
	print '>', dummy+1
	(startSignal, iSignal) = mcMimicry.superer_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.65, propdifftemp=0.35, comeBack=1)

	EKFd = EKFDetector(iSignal, dTraining)
	CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
	res  = CUSUMd.detect()[0]
	
	top = 0
	for i in res:
		if i > thr or i <-thr:
			top += 1
	dete = top*1.0/len(res)
	print 'Detection:', dete
	
	allISignals.append(iSignal)
	allResults.append(res)	
	
thrL=[thr] * len(dTesting)
thrLNeg=[-thr] * len(dTesting)
# Plot stuff
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(dTesting, linewidth=2, linestyle="--",  c="green", solid_capstyle="butt", label="Temperature")
plt.plot(allISignals[0], linewidth=1, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
plt.plot(allISignals[1], linewidth=1, linestyle="-", c="blue", solid_capstyle="butt", label="Attack")
plt.plot(allISignals[2], linewidth=1, linestyle="-", c="cyan", solid_capstyle="butt", label="Attack")
plt.plot(allISignals[3], linewidth=1, linestyle="-", c="magenta", solid_capstyle="butt", label="Attack")
plt.plot(allISignals[4], linewidth=1, linestyle="-", c="#993300", solid_capstyle="butt", label="Attack")


plt.ylabel('Temperature')

plt.subplot(2,1,2)
plt.plot(thrL, linewidth=2, linestyle="--", c="black", solid_capstyle="butt")
plt.plot(thrLNeg, linewidth=2, linestyle="--", c="black", solid_capstyle="butt")
plt.plot(allResults[0], linewidth=1, linestyle="-", c="red", solid_capstyle="butt", label="#1")
plt.plot(allResults[1], linewidth=1, linestyle="-", c="blue", solid_capstyle="butt", label="#2")
plt.plot(allResults[2], linewidth=1, linestyle="-", c="cyan", solid_capstyle="butt", label="#3")
plt.plot(allResults[3], linewidth=1, linestyle="-", c="magenta", solid_capstyle="butt", label="#4")
plt.plot(allResults[4], linewidth=1, linestyle="-", c="#993300", solid_capstyle="butt", label="#5")
plt.ylabel('$S_N$ Values')
plt.xlabel('Time')
plt.legend(loc='upper right')
plt.show()	
'''



################################################################################
'''
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
thr=1.362

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
allDete = []


for dummy in range(100):
	print '>', dummy+1
	(startSignal, iSignal) = mcMimicry.smart3_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propMean=0.35, propWDer1=0.2, propDTemp=0.7, comeBack=1)

	EKFd = EKFDetector(iSignal, dTraining)
	CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
	res  = CUSUMd.detect()[0]
	
	top = 0
	for i in res:
		if i > thr or i <-thr:
			top += 1
	dete = top*1.0/len(res)
	print 'Detection:', dete
	allDete.append(dete)

print 'Unsuccessful:', len(filter(lambda x:x>=0.001, allDete))
print 'Successful:', len(filter(lambda x:x<0.001, allDete))
'''


################################################################################
'''
attackedSensor = 37
usedSensors = (attackedSensor,)
goal = 30
tdelay = 1
thr=1.362

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

meanTests = [0.25, 0.5, 0.75]
wDer1Tests = [0.2, 0.4, 0.6, 0.8]
dTempTests = [0.2, 0.4, 0.6, 0.8]
for meann in meanTests:
	allDete = []
	allTime = []
	for wDer1 in wDer1Tests:
		for dTemp in dTempTests:
			mnDete = 0
			mnTime = 0
			for dummy in range(3):
				print 'wDer1', wDer1, 'dTemp:', dTemp, 'dummy:', dummy
				(startSignal, iSignal) = mcMimicry.smart3_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propMean=meann, propWDer1=wDer1, propDTemp=dTemp, comeBack=0)

				EKFd = EKFDetector(iSignal, dTraining)
				CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
				res  = CUSUMd.detect()[0]

				top = 0
				for i in res:
					if i > thr or i <-thr:
						top += 1
				dete = top*1.0/len(res)
			
				mnDete += dete
				mnTime += len(iSignal)
			mnDete /= 3.0
			mnTime /= 3.0
			allDete.append(mnDete)
			allTime.append(mnTime)
	print allDete
	print allTime
	print map(lambda x: 1000*x+10, allDete)
	#plt.scatter(map(lambda x:100*x,wDer1Tests), map(lambda x:100*x,dTempTests), s=map(lambda x: 1200*x+10, allDete), c="red", alpha=0.5)
	plt.scatter(map(lambda x:100*x,[0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.6,0.6,0.6,0.6,0.8,0.8,0.8,0.8]), map(lambda x:100*x,[0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8]), s=map(lambda x: 30000*x+5, allDete), c="red", alpha=0.7)
	plt.title('$meanPerc$ = '+ str(int(meann*100)) + '%')
	plt.xlabel('$relWeightGradMeanPerc$')
	plt.ylabel('$deltaTempPerc$')
	plt.show()
	plt.scatter(map(lambda x:100*x,[0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.6,0.6,0.6,0.6,0.8,0.8,0.8,0.8]), map(lambda x:100*x,[0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8,0.2, 0.4, 0.6, 0.8]), allTime, c="blue", alpha=0.7)
	plt.title('$meanPerc$ = '+ str(int(meann*100)) + '%')
	plt.xlabel('$relWeightGradMeanPerc$')
	plt.ylabel('$deltaTempPerc$')
	plt.show()
	'''
	
	
	
################################################################################
attackedSensor = 37
usedSensors = (attackedSensor,)
tdelay = 1
thr=1.362

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

goals = [25,26,27,28,29,30,31,32,33,34,35]
tdelays = [50*i for i in range(15)]
detRates1 = []
detRates2 = []

for goal in goals:
	print '>goal:', goal
	atck1Mean = 0
	atck2Mean = 0
	for tdelay in tdelays:
		print ' >>tdelay:', tdelay
			
		(startSignal, iSignal1) = mcMimicry.superer_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.65, propdifftemp=0.35, comeBack=1)
		
		EKFd = EKFDetector(iSignal1, dTraining)
		CUSUMd = CUSUMDetector(iSignal1, h=0.4, w=10, EKFd=EKFd)
		res1  = CUSUMd.detect()[0]
		top = 0
		for i in res1:
			if i > thr or i <-thr:
				top += 1
		dete1 = top*1.0/len(res1)
		atck1Mean += dete1
		
		(startSignal, iSignal2) = mcMimicry.smart3_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propMean=0.45, propWDer1=0.2, propDTemp=0.65, comeBack=True)

		EKFd = EKFDetector(iSignal2, dTraining)
		CUSUMd = CUSUMDetector(iSignal2, h=0.4, w=10, EKFd=EKFd)
		res2  = CUSUMd.detect()[0]
		top = 0
		for i in res2:
			if i > thr or i <-thr:
				top += 1
		dete2 = top*1.0/len(res2)
		atck2Mean += dete2

	atck1Mean /= len(tdelays)
	atck2Mean /= len(tdelays)
	print atck1Mean
	print atck2Mean
	print '-_-_-_-_-'
	detRates1.append(atck1Mean)
	detRates2.append(atck2Mean)

plt.plot(goals, map(lambda x:x*100, detRates1), linewidth=3, linestyle="-", c="red", solid_capstyle="butt", label="Attack 1")
plt.plot(goals, map(lambda x:x*100, detRates2), linewidth=3, linestyle="-", c="blue", solid_capstyle="butt", label="Attack 2")
plt.ylabel('Detection Rate %')
plt.xlabel('Target Temperature')
plt.legend(loc='upper left')
plt.show()


