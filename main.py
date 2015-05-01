from fileparser import *
from readingsfilter import *
import dbOp, datetime, testing, attack
from ekfdetector import *
from cusumdetector import *
from plotter import *
from scipy.stats.stats import pearsonr
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
	readingsFilter = ReadingsFilter(threshold=25)
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
	dbOp.closeConnectionToDatabase()

#init()

readings = []
dbOp.connectToDatabase("data/db")
r3 = map(lambda x: x[0], dbOp.selectReadingsFromNode(3))
r10 = map(lambda x: x[0], dbOp.selectReadingsFromNode(6))
r26 = map(lambda x: x[0], dbOp.selectReadingsFromNode(26))
r11 = map(lambda x: x[0], dbOp.selectReadingsFromNode(11))
r42 = map(lambda x: x[0], dbOp.selectReadingsFromNode(42))
for i in map(lambda x: x[0], dbOp.selectAllNodes()):
	if i != 3: readings.append( (i, map(lambda x: x[0], dbOp.selectReadingsFromNode(i))) )
dbOp.closeConnectionToDatabase()

## CUSUM detection ##
'''
EKFd = EKFDetector(readings)
CUSUMd = CUSUMDetector(readings, h=0.4, w=10, EKFd=EKFd)
(a, b)  = (CUSUMd.detect()[0], CUSUMd.detect()[1])
'''

# Calculates correlations

cor = []
for (i,r) in readings:
	wat = min(len(r),len(r3))
	cor.append((i, numpy.cov(r3[:wat],r[:wat])[0][1]))
cor = sorted(cor, key=lambda x: x[1], reverse=True)
for i in cor:
	print i

# Terence mimicry
#mimicry = attack.terence_mimicry(7, 25, 0, [8])[0]

# Plot stuff

import matplotlib.pyplot as plt
plt.plot([0], 'w', [40], 'w', r3, 'b', r26, 'r', r42, 'c', r10, 'g')
plt.show()
################







