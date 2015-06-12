import matplotlib.pyplot as plt
import dbOp, data
from segment import *
from reading import *
from ekfdetector import *
from cusumdetector import *
from mcattack import *

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
	
	
	
	
	
	
	
	
	
	
	
	
	