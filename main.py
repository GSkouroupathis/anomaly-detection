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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--s", nargs='?', default="37", help="Sensor ID")
parser.add_argument("--a", nargs='?', default="rgv", help="Attack Type")
parser.add_argument("--t", nargs='?', default="30", help="Target Temperature")
parser.add_argument("--c", nargs='?', default="y", help="Come Back")
parser.add_argument("--p", nargs='?', default="n", help="Compare with Previous Work")
parser.add_argument("--d", nargs='?', default="1", help="Delay in seconds")
args = parser.parse_args()

##########################
# s
##########################
attackedSensor = int(args.s)
usedSensors = (attackedSensor,)
dbOp.connectToDatabase("data/db") ##
	
# readings & info & segments
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

##########################
# t
##########################
goal = int(args.t)

##########################
# c
##########################
if args.c=='n' or args.p=='y':
	comeBack = 0
else:
	comeBack = 1

##########################
# d
##########################
tdelay = int(args.d)

##########################
# a
##########################
if args.a == 'rgv':
	(startSignal, iSignal) = mcMimicry.rgv_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.65, propdifftemp=0.35, comeBack=comeBack)
elif args.a == 'rwgm':
	(startSignal, iSignal) = mcMimicry.rwgm_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propMean=0.2, propWDer1=0.12, propDTemp=0.65, comeBack=comeBack)
elif args.a == 'softmax':
	(startSignal, iSignal) = mcMimicry.softmax_cluster_attack(attackedSensor, goal, tdelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv=0.75, temp=0.03, comeBack=comeBack)
elif args.a == 'uniform':
	(startSignal, iSignal) = mcMimicry.uniform_attack(goal, dTesting)
# Detect Stuff
EKFd = EKFDetector(iSignal, dTraining)
CUSUMd = CUSUMDetector(iSignal, h=0.4, w=10, EKFd=EKFd)
res  = CUSUMd.detect()[0]

##########################
# p
##########################
if args.p != 'n':
	terMimicry = TerMimicry()
	terSignal = terMimicry.attack(dTraining, goal, 0, [dTraining])[0]
	EKFd = EKFDetector(terSignal, dTraining)
	CUSUMd = CUSUMDetector(terSignal, h=0.4, w=10, EKFd=EKFd)
	terRes  = CUSUMd.detect()[0]

# Testing Detection Rate
EKFd = EKFDetector(dTesting, dTraining)
CUSUMd = CUSUMDetector(dTesting, h=0.4, w=10, EKFd=EKFd)
testRes  = CUSUMd.detect()[0]
	
# Threshold
thr = 1.362

top = 0
for i in testRes:
	if i > thr or i <-thr:
		top += 1
print "Testing Detection Rate:", top*1.0/len(testRes)
top = 0
for i in res:
	if i > thr or i <-thr:
		top += 1
print "Detection Rate:", top*1.0/len(res)
if args.p != 'n':
	top = 0
	for i in terRes:
		if i > thr or i <-thr:
			top += 1
	print "Past Detection Rate:", top*1.0/len(terRes)

# Plot stuff

thrL=[thr] * len(dTesting)
thrLNeg=[-thr] * len(dTesting)

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(dTesting, linewidth=3, linestyle="--",  c="green", solid_capstyle="butt", label="Temperature")
plt.plot(iSignal, linewidth=2, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
if args.p != 'n':
	plt.plot(terSignal, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt")
plt.ylabel('Temperature')

plt.subplot(2,1,2)
plt.plot(thrL, linewidth=2, linestyle="--", c="black", solid_capstyle="butt", label="Threshold")
plt.plot(thrLNeg, linewidth=2, linestyle="--", c="black", solid_capstyle="butt")
plt.plot(testRes, linewidth=3, linestyle="--", c="green", solid_capstyle="butt", label="Temperature")
plt.plot(res, linewidth=2, linestyle="-", c="red", solid_capstyle="butt", label="Attack")
if args.p != 'n':
	plt.plot(terRes, linewidth=2, linestyle="-.", c="blue", solid_capstyle="butt", label="Past Attack")
plt.legend(loc='upper right')
plt.ylabel('$S_N$ Values')
plt.xlabel('Time')
plt.show()