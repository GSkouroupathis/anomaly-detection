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
		dbCursor.execute('DROP TABLE IF EXISTS data_segments_table')
	except Exception,e:
		print str(e)
		print "ds"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS clusters_table')
	except Exception,e:
		print str(e)
		print "c"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS readings_table')
	except Exception,e:
		print str(e)
		print "r"
	try:
		dbCursor.execute('DROP TABLE IF EXISTS nodes_table')
	except Exception,e:
		print str(e)
		print "n"

	######################### nodes_table #########################
	dbCursor.execute('''
	CREATE TABLE nodes_table (
	node_id int,
	PRIMARY KEY (node_id)
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
	
	######################### clusters_table #########################
	dbCursor.execute('''
	CREATE TABLE clusters_table (
	cluster_id int,
	PRIMARY KEY (cluster_id)
	)
	''')
	
	######################### data_segments_table #########################
	dbCursor.execute('''
	CREATE TABLE data_segments_table (
	node_id int,
	start_date varchar(10),
	end_date varchar(10),
	start_time varchar(15),
	end_time varchar(15),
	mean real,
	cluster_id int,
	PRIMARY KEY (node_id, start_date, end_date, start_time, end_time),
	FOREIGN KEY(node_id) REFERENCES nodes_table(node_id),
	FOREIGN KEY(start_date) REFERENCES readings_table(date),
	FOREIGN KEY(end_date) REFERENCES readings_table(date),
	FOREIGN KEY(start_time) REFERENCES readings_table(time),
	FOREIGN KEY(end_time) REFERENCES readings_table(time),
	FOREIGN KEY(cluster_id) REFERENCES clusters_table(cluster_id)
	)
	''')

###############	###############	###############	
#nodes_table
###############	###############	###############	
def insertNode(node_id):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO nodes_table (node_id)
	VALUES (?)
	''', (node_id,))
	dbConnection.commit()

def selectNode(node_id):
	global dbCursor
	dbCursor.execute('''
	SELECT * FROM nodes_table
	WHERE node_id = ?
	''', (node_id,))
	return dbCursor.fetchall()

def selectAllNodes():
	global dbCursor
	dbCursor.execute('''
	SELECT node_id FROM nodes_table
	''')
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
	
###############	###############	###############	
#clusters_table
###############	###############	###############		
def insertCluster(cluster_id):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO clusters_table (cluster_id)
	VALUES (?)
	''', (cluster_id,))
	dbConnection.commit()
	
###############	###############	###############	
#data_segments_table
###############	###############	###############		
def insertDataSegment(node_id, start_date, end_date, start_time, end_time, mean, cluster_id):
	global dbCursor
	dbCursor.execute('''
	INSERT INTO data_segments_table (cluster_id)
	VALUES (?, ?, ?, ?, ?, ?, ?)
	''', (node_id, start_date, end_date, start_time, end_time, mean, cluster_id))
	dbConnection.commit()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	