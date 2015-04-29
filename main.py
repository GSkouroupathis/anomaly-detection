from fileparser import *
from readingsfilter import *
import dbOp, datetime, testing
from ekfdetector import *
from cusumdetector import *
from plotter import *

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

dbOp.connectToDatabase("data/db")
readings = map(lambda x: x[0], dbOp.selectReadingsFromNode(3))
dbOp.closeConnectionToDatabase()

## CUSUM detection ##

EKFd = EKFDetector(readings)
CUSUMd = CUSUMDetector(readings, h=0.4, w=10, EKFd=EKFd)
(a, b)  = (CUSUMd.detect()[0], CUSUMd.detect()[1])

import matplotlib.pyplot as plt
plt.plot([0 for i in a][200:300], 'g', a[200:300], 'r', b[200:300], 'b')
plt.show()

################