import os
import sqlite3

#Connect to a database. Create it if it doesn't exist
def connectToDatabase(db_path):
	global dbConnection
	global dbCursor
	dbConnection = sqlite3.connect(db_path)
	dbCursor = dbConnection.cursor()
	if not dbConnection:
		print 'Connection to database', database, 'failed'
		return
	if not dbCursor:
		print 'Setting the cursor of database', database, 'failed'
		return

#Close the connection to a database.
def closeConnectionToDatabase():
	dbCursor.close()
	dbConnection.close()

#Initializes all the tables in the database
def initTables():
	global dbCursor

	#Allow foreign keys
	try:
		dbCursor.execute('PRAGMA foreign_keys = ON;')
	except:
		print "Foreign keys could not be enabled"
	
	#Drop tables in case they already exist
	try:
		dbCursor.execute('DROP TABLE IF EXISTS reading_segments_table')
	except Exception,e:
		print str(e)
		print "data segments error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS covariance_table')
	except Exception,e:
		print str(e)
		print "covariance error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS readings_table')
	except Exception,e:
		print str(e)
		print "readings error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS conditional_probabilities_table')
	except Exception,e:
		print str(e)
		print "conditional probabilities error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS clusters_table')
	except Exception,e:
		print str(e)
		print "clusters error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS cluster_groups_table')
	except Exception,e:
		print str(e)
		print "cluster_groups error"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS nodes_table')
	except Exception,e:
		print str(e)
		print "nodes error"

	######################### nodes_table #########################
	dbCursor.execute('''
	CREATE TABLE nodes_table (
	node_id int,
	available int,
	PRIMARY KEY (node_id)
	)
	''')

	######################### cluster_groups_table #########################
	dbCursor.execute('''
	CREATE TABLE cluster_groups_table (
	root_node_id int,
	no_of_clusters int,
	no_of_dimensions int,
	PRIMARY KEY (root_node_id),
	FOREIGN KEY(root_node_id) REFERENCES nodes_table(node_id)
	)
	''')
	
	######################### clusters_table #########################
	dbCursor.execute('''
	CREATE TABLE clusters_table (
	root_node_id int,
	cluster_id int,
	centroid varchar(255),
	PRIMARY KEY (root_node_id, cluster_id),
	FOREIGN KEY(root_node_id) REFERENCES cluster_groups_table(root_node_id)
	)
	''')
	
	###################### conditional_probabilities_table ######################
	dbCursor.execute('''
	CREATE TABLE conditional_probabilities_table (
	root_node_id int,
	cluster_before_id int,
	cluster_after_id int,
	probability real,
	PRIMARY KEY (root_node_id, cluster_before_id, cluster_after_id),
	FOREIGN KEY(root_node_id) REFERENCES cluster_table(root_node_id),
	FOREIGN KEY(cluster_before_id) REFERENCES cluster_table(cluster_id),
	FOREIGN KEY(cluster_after_id) REFERENCES cluster_table(cluster_id)
	)
	''')
	
	######################### readings_table #########################
	dbCursor.execute('''
	CREATE TABLE readings_table (
	node_id int,
	date varchar(10),
	time varchar(15),
	temperature real,
	PRIMARY KEY (node_id, date, time),
	FOREIGN KEY(node_id) REFERENCES nodes_table(node_id)
	)
	''')
	
	######################### covariance_table #########################
	dbCursor.execute('''
	CREATE TABLE covariance_table (
	node1_id int,
	node2_id int,
	covariance real,
	PRIMARY KEY (node1_id, node2_id),
	FOREIGN KEY(node1_id) REFERENCES nodes_table(node_id),
	FOREIGN KEY(node2_id) REFERENCES nodes_table(node_id)
	)
	''')
	
	######################### reading_segments_table #########################
	dbCursor.execute('''
	CREATE TABLE reading_segments_table (
	node_id int,
	start_date varchar(10),
	start_time varchar(15),
	end_date varchar(10),
	end_time varchar(15),
	root_node_id int,
	cluster_id int,
	PRIMARY KEY (node_id, start_date, start_time, end_date, end_time),
	FOREIGN KEY(node_id) REFERENCES readings_table(node_id),
	FOREIGN KEY(start_date) REFERENCES readings_table(date),
	FOREIGN KEY(end_date) REFERENCES readings_table(date),
	FOREIGN KEY(start_time) REFERENCES readings_table(time),
	FOREIGN KEY(end_time) REFERENCES readings_table(time),
	FOREIGN KEY(root_node_id) REFERENCES clusters_table(root_node_id),
	FOREIGN KEY(cluster_id) REFERENCES clusters_table(cluster_id)
	)
	''')

###############	###############	###############	
#nodes_table
###############	###############	###############	
def insertNode(node_id):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO nodes_table (node_id, available)
	VALUES (?, 0)
	''', (node_id,))
	dbConnection.commit()

def selectAllNodes():
	global dbCursor
	dbCursor.execute('''
	SELECT node_id FROM nodes_table
	''')
	return dbCursor.fetchall()

def setNodeAvail(node_id):
	global dbCursor
	dbCursor.execute('''
	UPDATE nodes_table
	SET available=1
	WHERE node_id = ?
	''', (node_id,))
	dbConnection.commit()
	
###############	###############	###############	
#covariance_table
###############	###############	###############	
def insertCov(node1_id, node2_id, cov):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO covariance_table (node1_id, node2_id, covariance)
	VALUES (?,?,?)
	''', (node1_id, node2_id, cov))
	dbConnection.commit()

def selectOrderedCov(node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM covariance_table
	WHERE node1_id = ?
		OR node2_id = ?
	ORDER BY covariance DESC
	''', (node_id, node_id))
	return dbCursor.fetchall()

###############	###############	###############	
#readings_table
###############	###############	###############	
def insertAllReadings(readings):
	#and nodes
	nodes = []
	global dbCursor
	for (i,r) in enumerate(readings):
		print 'Inserting', i
		if r[0] not in nodes:
			insertNode(r[0])
			nodes.append(r[0])
		try:
			dbCursor.execute('''
		INSERT INTO readings_table (node_id, date, time, temperature)
		VALUES (?, ?, ?, ?)
		''', (r[0], r[1], r[2], r[3]))
		except Exception,e:
			print str(e)
			print ' -------->' ,r
	dbConnection.commit()
	
def insertReading(node_id, date, time, temperature):
	global dbCursor
	
	dbCursor.execute('''
	INSERT INTO readings_table (node_id, date, time, temperature)
	VALUES (?, ?, ?, ?)
	''', (node_id, date, time, temperature))
	dbConnection.commit()
	
def selectReading(node_id, date, time):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM readings_table
	WHERE node_id = ? AND date = ?
	AND time = ?
	''', (node_id, date, time))
	return dbCursor.fetchall()
	
def selectReadingsFromNode(node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM readings_table
	WHERE node_id = ?
	''', (node_id,))
	return dbCursor.fetchall()

def selectDatasetFromNode(node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT temperature FROM readings_table
	WHERE node_id = ?
	''', (node_id,))
	return dbCursor.fetchall()
	
def selectAllReadings():
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM readings_table
	''')
	return dbCursor.fetchall()

def selectReadingsFromNodes(nodesList):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM readings_table
	WHERE node_id IN (%s)
	''' % ','.join('?'*len(nodesList)), nodesList
	)
	return dbCursor.fetchall()

###############	###############	###############	
#cluster_groups_table
###############	###############	###############		
def insertClusterGroup(nodeID, K, W):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO cluster_groups_table
	(root_node_id, no_of_clusters, no_of_dimensions)
	VALUES (?, ?, ?)
	''', (nodeID, K, W))
	dbConnection.commit()
	
def selectClusterGroup(root_node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM cluster_groups_table
	WHERE root_node_id = ?
	''', (root_node_id,)
	)
	return dbCursor.fetchall()
	
def getNoOfDimensions(rootNodeID):
	global dbCursor
	dbCursor.execute('''
	SELECT no_of_dimensions FROM cluster_groups_table
	WHERE root_node_id = ?
	''', (rootNodeID,)
	)
	return dbCursor.fetchall()[0][0]

###############	###############	###############	
#clusters_table
###############	###############	###############		
def insertCluster(nodeID, clusterID, centroid):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO clusters_table (root_node_id, cluster_id, centroid)
	VALUES (?, ?, ?)
	''', (nodeID, clusterID, centroid))
	dbConnection.commit()

###############	###############	###############	
#conditional_probabilities_table
###############	###############	###############		
def insertConditionalProbability(nodeID, cbID, caID, prob):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO conditional_probabilities_table
	(root_node_id, cluster_before_id, cluster_after_id, probability)
	VALUES (?, ?, ?, ?)
	''', (nodeID, cbID, caID, prob))
	dbConnection.commit()
	
def selectCondProbs(root_node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM conditional_probabilities_table
	WHERE root_node_id = ?
	''', (root_node_id,)
	)
	return dbCursor.fetchall()

###############	###############	###############	
#reading_segments_table
###############	###############	###############		
def insertReadingSegment(nodeID, startDate, startTime, endDate, endTime, rNodeID, clusterId):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO reading_segments_table
	(node_id, start_date, start_time, end_date, end_time, root_node_id, cluster_id)
	VALUES (?, ?, ?, ?, ?, ?, ?)
	''', (nodeID, startDate, startTime, endDate, endTime, rNodeID, clusterId))
	dbConnection.commit()
	
def selectSegments(root_node_id, node_list):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM reading_segments_table
	WHERE root_node_id = ?
	AND node_id IN (%s)
	''' % ','.join('?'*len(node_list)), (root_node_id,) + node_list
	)
	return dbCursor.fetchall()

def selectSegmentsFromNode(nodeID):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM reading_segments_table
	WHERE node_id = ?
	''', (nodeID,)
	)
	return dbCursor.fetchall()








	