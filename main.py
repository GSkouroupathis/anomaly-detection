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

readings = []
dbOp.connectToDatabase("data/db")
r3 = map(lambda x: x[0], dbOp.selectReadingsFromNode(3))
r4 = map(lambda x: x[0], dbOp.selectReadingsFromNode(4))
r6 = map(lambda x: x[0], dbOp.selectReadingsFromNode(6))
r8 = map(lambda x: x[0], dbOp.selectReadingsFromNode(8))
r13 = map(lambda x: x[0], dbOp.selectReadingsFromNode(13))
r18 = map(lambda x: x[0], dbOp.selectReadingsFromNode(18))
r23 = map(lambda x: x[0], dbOp.selectReadingsFromNode(23))
r25 = map(lambda x: x[0], dbOp.selectReadingsFromNode(25))
r49 = map(lambda x: x[0], dbOp.selectReadingsFromNode(49))
'''
for i in map(lambda x: x[0], dbOp.selectAllNodes()):
	if i != 3: readings.append( (i, map(lambda x: x[0], dbOp.selectReadingsFromNode(i))) )
'''
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
'''
terMimicry = TerMimicry()
falseSignal = terMimicry.attack(r3, 28, 0, [r3])[0]
'''



# MC mimicry
'''
mcMimicry = MCMimicry(r3)
'''



# Plot stuff
import matplotlib.pyplot as plt
#plt.axis('equal')

plt.plot(r6, 'b')
plt.show()
plt.plot(r13, 'b')
plt.show()
plt.plot(r18, 'b')
plt.show()
