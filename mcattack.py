from attack import *
from segment import *
from attackTree import AttackTree
from attackNode import AttackNode
import dbOp, math, numpy, random, datetime
from scipy.cluster.vq import kmeans2

class MCMimicry(Attack):
	# Constructor
	def __init__(self, dataset=None):
		self.dataset = dataset
				
	# 1 ##########################################################################
	# Chooses the appropriate window size w
	def prepare(self, dataset=None):
		if not dataset: dataset = self.dataset
		
		bestScore = -1
		bestW=bestSegments=bestCentroids=bestLabels=bestCondProbTable=K = None
		for w in [7, 8, 9, 10, 12, 15, 20, 25, 30]:
			segment_list = numpy.array(self.segment_signal(dataset, w, 0))
			if len(segment_list) == 0: continue
			#norm_segments = map(lambda x: x-numpy.mean(x), segment_list)
			(centroids, labels) = self.cluster(segment_list)
			# the sum of segments in each cluster
			if labels is None: continue
			clusterCount = numpy.bincount(labels)
			condProbTable = self.create_cond_prob_table(centroids, labels, clusterCount)
			tableScore = self.eval_cond_prob_table(condProbTable, clusterCount)
			if tableScore > bestScore:
				bestW = w
				bestSegments = segment_list
				(bestCentroids, bestLabels) = (centroids, labels)
				bestCondProbTable = condProbTable
				bestScore = tableScore
			
			K = len(bestCentroids)
			
		return (bestW, bestSegments, bestCentroids, bestLabels, bestCondProbTable, K, bestScore)

	# 2 ##########################################################################
	# Divides signal into segments
	# a: signal
	# w: window size
	# h: hop size
	def segment_signal(self, z, w, h):
		segment_list = []
		for (i, r) in enumerate(z[::w]):
			segment = z[i*w:i*w+w]
			if len(segment) == w:
				segment_list.append(segment)
		return segment_list

	# Clusters the segments into K clusters
	def cluster(self, segments):
		prevAvDist = 99999999
		prevCentroids = numpy.array([])
		prevLabels = []
		# tries to find K
		for K in range(3,80):
			(centroids, labels) = self.k_means(segments, K)
			# check to see if no cluster is empty
			allClusters = True
			for cntr in range(K):
				if cntr not in labels:
					allClusters = False
					break
			if not allClusters: continue
			avDist = self.get_average_distance(centroids, labels, segments)
			isKnee = self.find_knee(prevAvDist, avDist)
			# if correct K found
			if isKnee:
				return (prevCentroids, prevLabels)
			else:
				prevAvDist = avDist
				prevCentroids = centroids
				prevLabels = labels
		return (None, None)
			
	# Creates conditional probabilities table
	def create_cond_prob_table(self, centroids, labels, clusterCount):
		# the sum of segments in cluster given previous cluster
		countTable = [[0]*len(centroids) for i in range(len(centroids))]
		for (i, thisCluster) in enumerate(labels[1:]):
			lastCluster = labels[i]
			countTable[lastCluster][thisCluster] += 1
		condProbTable = self.dirichlet(countTable, clusterCount, K=len(centroids))
		return condProbTable
			
	# Evaluates conditional probabilities table
	def eval_cond_prob_table(self, condProbTable, clusterCount):
		negentropies = []
		for row in condProbTable:
			negentropies.append(self.negentropy(row))
			
		score = self.w_m_negentropy(negentropies, clusterCount)
		return score
		
	# 3 ##########################################################################
	# Performs K-means
	# K: number of clusters
	def k_means(self, segments, K):
		(centroids, labels) = kmeans2(segments, K)
		return (centroids, labels)
		
	# Gets average distance of segments to
	# their corresponding cluster centers
	def get_average_distance(self, centroids, labels, segments):
		totalDist = 0
		for (i, segment) in enumerate(segments):
			centroid = centroids[labels[i]]
			dist = numpy.linalg.norm(segment-centroid)
			totalDist += dist
		return totalDist / len(segments)
		
	# Finds knee on graph by computing the derivative
	def find_knee(self, prevAvDist, avDist):
		return avDist - prevAvDist > -0.3
	
	# 3 ##########################################################################
	# Applies Dirichlet distribution to create Conditional Probabilities Table
	def dirichlet(self, countTable, clusterCount, K):
		for (prevCluster, row) in enumerate(countTable):
			for (thisCluster, thisClusterCount) in enumerate(row):
				row[thisCluster] = (thisClusterCount + 1)*1.0 / (clusterCount[prevCluster] + K)
		return countTable

	# 3 ##########################################################################	# Computes entropy of distribution
	def entropy(self, distribution):

		return -sum([p*math.log(p,10) for p in distribution])
		
	# Computes negentropy of distribution against uniform distribution
	def negentropy(self, distribution):
		uniform = [1.0/len(distribution) for i in distribution]
		return self.entropy(uniform) - self.entropy(distribution)
	
	# Weighted mean negentropy
	def w_m_negentropy(self, negentropies, clusterCount):
		totalClusterCount = sum(clusterCount)
		return sum(map(lambda x: x[0]*(x[1]*1.0/totalClusterCount), zip(negentropies, clusterCount)))



################################################################################
################################################################################
################################################################################


	def prepare_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic):
		
		# find actual time of attack start
		firstDateTime = nodesSegmentsDic[sensorID][0].startDate + ' ' + nodesSegmentsDic[sensorID][0].startTime
		
		atkStartDateTime = datetime.datetime.strptime(firstDateTime, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(0,atckDelay)
		
		# constructs starting signal
		startSignal = []
		startSegment = None
		for (i, segment) in enumerate(nodesSegmentsDic[sensorID]):
			firstSegmentDateTime = datetime.datetime.strptime(segment.startDate+' '+segment.startTime, '%Y-%m-%d %H:%M:%S')

			# find if the segment will be added into the attack
			if firstSegmentDateTime < atkStartDateTime:
				startSignal += segment.dataset.tolist()
			else:
				startSegment = nodesSegmentsDic[sensorID][i-1]
				break
			
		return (startSignal, startSegment)

	# 1 ##########################################################################	
	# Tree Attacks sensor_id until goal
	def tree_attack(self, sensorID, goal, atckDelay, sensorsSegmentsReadingsDic, condProbTable):
		if atckDelay<100000:atckDelay+=100000
		(startSignal, startSegment, sensorsSegmentsReadingsDic) = self.prepare_attack(sensorID, goal, atckDelay, sensorsSegmentsReadingsDic, condProbTable)
		
		iSignal = self.tree_attack_(sensorID, startSignal, startSegment, goal, sensorsSegmentsReadingsDic, condProbTable)
		return (startSignal, iSignal)
		
	# 2 ##########################################################################	
	def tree_attack_(self, sensorID, startSignal, startSegment, goal, sensorsSegmentsReadingsDic, condProbTable):
		# first build attack tree & root node
		atckTree = AttackTree(sensorID, startSignal, 0.2, goal, condProbTable)
		rootNode = AttackNode(atckTree, sensorID, startSegment)
		atckTree.set_start_node(rootNode)
		
		# launch the attack
		return atckTree.attack(sensorID, sensorsSegmentsReadingsDic)
		
		
		
		
		
		
	###
	def greedy_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):
		
		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)

		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:

			# filter by goal
			candidateSegments = [segment for segment in segments if abs(segment.dataset[-1] - goal) < diff]
		
			# filter by stitching point
			candidateSegmentsTemp = [segment for segment in candidateSegments if abs(segment.dataset[0] - lastSegment.dataset[-1]) < 0.3]
			if len(candidateSegmentsTemp) > 0:
				candidateSegments = candidateSegmentsTemp
			else:
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dataset[0]-lastSegment.dataset[-1])
			lastSegment = candidateSegments[0] # take first & best stitch
			
			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
			
			
			
			
	######
	#####
	####
	###
	##		
	def random_cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:
		
			lastCluster = lastSegment.cluster

			# we pick the next cluster probabilistically from condProbsTable
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break
				
			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
					
			# get best segment
			lastSegment = random.choice(candidateSegments)
			#print lastSegment.dataset,
			
			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
		
		
		
		
		
		
	#
	##
	###
	####		
	def greedy_smooth_cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:
		
			lastCluster = lastSegment.cluster

			# we pick the next cluster probabilistically from condProbsTable
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break
				
			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			if finalValue < goal:
				candidateSegments = filter(lambda s:s.dTemp>0, candidateSegments)
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)
				newSegment = candidateSegments[0]
			else:
				candidateSegments = filter(lambda s:s.dTemp<0, candidateSegments)
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)
				newSegment = candidateSegments[0]

			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment
			
			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]

			diff = abs(finalValue - goal)
			#print diff
			
		return (startSignal, iSignal)
			
	#
	##
	###
	####		
	def first_cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:
		
			lastCluster = lastSegment.cluster

			# we pick the next cluster probabilistically from condProbsTable
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break

			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			if finalValue < goal:
				candidateSegments = filter(lambda s:s.dTemp>0, candidateSegments)
				#candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)
				newSegment = candidateSegments[0]

			else:
				candidateSegments = filter(lambda s:s.dTemp<0, candidateSegments)
				#candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)
				newSegment = candidateSegments[0]

			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment
			
			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]

			diff = abs(finalValue - goal)
			#print diff
			
		return (startSignal, iSignal)
		
	###
	####
	#####
	####
	def super_cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:
		
			lastCluster = lastSegment.cluster

			# we pick the next cluster probabilistically from condProbsTable
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break

			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			if finalValue < goal:
				candidateSegments = filter(lambda s:s.dTemp>0, candidateSegments)
				# filter by mean determinant
				lastMeanDer1 = lastSegment.meanDer1
				candidateSegmentsTemp = filter(lambda s: abs(s.meanDer1-lastMeanDer1)<0.005, candidateSegments)
				if len(candidateSegmentsTemp) != 0:
					candidateSegments = sorted(candidateSegmentsTemp, key=lambda s: s.dTemp, reverse=True)[:5]
					newSegment = random.choice(candidateSegments)
				else:
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.meanDer1-lastMeanDer1))[:4]
					newSegment = random.choice(candidateSegments)
			else:
				candidateSegments = filter(lambda s:s.dTemp<0, candidateSegments)
				# filter by mean determinant
				lastMeanDer1 = lastSegment.meanDer1
				candidateSegmentsTemp = filter(lambda s: abs(s.meanDer1-lastMeanDer1)<0.005, candidateSegments)
				if len(candidateSegmentsTemp) != 0:
					candidateSegments = sorted(candidateSegmentsTemp, key=lambda s: s.dTemp)[:5]
					newSegment = random.choice(candidateSegments)
				else:
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.meanDer1-lastMeanDer1))[:5]
					newSegment = random.choice(candidateSegments)
			
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment
			print lastSegment.dataset
			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]

			diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
		
		
		
		
	###
	####
	#####
	####
	def rgv_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv, propdifftemp, comeBack=False):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
		
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)

		while diff > 0.1 and len(iSignal)<len(dTesting):

			# we pick the next cluster probabilistically from condProbsTable
			lastCluster = lastSegment.cluster
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break

			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			if finalValue < goal:
				# filter by relative variance of first derivative
				lastMeanDer1 = lastSegment.meanDer1
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propdifftemp*len(candidateSegments))]
				newSegment = random.choice(candidateSegments)
			else:
				# filter by relative variance of first derivative
				lastMeanDer1 = lastSegment.meanDer1
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)[:int(propdifftemp*len(candidateSegments))]					
				newSegment = random.choice(candidateSegments)
				
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment

			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				if comeBack:
					iSignal += lastSegment.dataset.tolist()
				else:
					iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
		
		'''////////////////////////////////////////////		
		/////////////////// comeBack /////////////////	
		 the signal must go back to the real dataset
		'''
		if comeBack and len(iSignal)<len(dTesting):
			atckI = len(iSignal) - 1
			finalValue = lastSegment.dataset[-1]
			goal = dTesting[atckI]
			diff = abs(finalValue - goal)
		
			while diff > 0.1:

				# we pick the next cluster probabilistically from condProbsTable
				lastCluster = lastSegment.cluster
				rand = random.uniform(0,1)
				for (i, prob) in enumerate(condProbsTable[lastCluster]):
					rand -= prob
					if rand <= 0:
						nextCluster = i
						break

				# filter by cluster
				candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
				# get segment w/ greedy strategy
				if finalValue < goal:
					# filter by relative variance of first derivative
					lastMeanDer1 = lastSegment.meanDer1
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propdifftemp*len(candidateSegments))]
					newSegment = random.choice(candidateSegments)
				else:
					# filter by relative variance of first derivative
					lastMeanDer1 = lastSegment.meanDer1
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)[:int(propdifftemp*len(candidateSegments))]					
					newSegment = random.choice(candidateSegments)
				
				# shift everything so that it can be stitched
				startDiff = newSegment.dataset[0] - lastSegment.forecast()
				newSegment.dataset -= startDiff
				lastSegment = newSegment
				
				# we have to check every datapoint on its own
				brkDummy = False
				for d in lastSegment.dataset:
					iSignal.append(d)
					atckI += 1
					if abs(d-dTesting[atckI]) < 0.1:
						brkDummy = True
						break
					if len(iSignal) >= len(dTesting):
						brkDummy = True
						break
				if brkDummy:
					iSignal += dTesting[atckI+1:]
					break		
				
				finalValue = lastSegment.dataset[-1]
				goal = dTesting[atckI]
				diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
		
		
		
		
		
		
		
		
		
		
	# softmax cluster attack
	def softmax_cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable, dTesting, propvarderiv, temp, comeBack=False):
		
		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)

		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
		
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)

		while diff > 0.1 and len(iSignal)<len(dTesting):

			# we pick the next cluster probabilistically from condProbsTable
			lastCluster = lastSegment.cluster
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break
			
			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			
			# filter by relative variance of first derivative
			lastMeanDer1 = lastSegment.meanDer1
			candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))
			candidateSegments = candidateSegments[:int(propvarderiv*len(candidateSegments))]

			if finalValue < goal:
				derivvars = [ numpy.exp( s.dTemp/temp ) for s in candidateSegments]
			else:
				maxTemp = max([s.dTemp for s in candidateSegments])
				derivvars = [ numpy.exp( (-s.dTemp)/temp ) for s in candidateSegments]
			derivvars = numpy.array(derivvars)
			probs = derivvars/np.sum(derivvars)
			choice = numpy.where(numpy.random.multinomial(len(candidateSegments),pvals=probs))[0][0]
			newSegment = candidateSegments[choice]
				
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment

			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				if comeBack:
					iSignal += lastSegment.dataset.tolist()
				else:
					iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
		
		'''////////////////////////////////////////////		
		/////////////////// comeBack /////////////////	
		 the signal must go back to the real dataset
		'''
		if comeBack and len(iSignal)<len(dTesting):

			atckI = len(iSignal) - 1
			finalValue = lastSegment.dataset[-1]
			goal = dTesting[atckI]
			diff = abs(finalValue - goal)
		
			while diff > 0.1 and len(iSignal)<len(dTesting):

				# we pick the next cluster probabilistically from condProbsTable
				lastCluster = lastSegment.cluster
				rand = random.uniform(0,1)
				for (i, prob) in enumerate(condProbsTable[lastCluster]):
					rand -= prob
					if rand <= 0:
						nextCluster = i
						break

				# filter by cluster
				candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]

				# get segment w/ greedy strategy
				lastMeanDer1 = lastSegment.meanDer1
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))
				candidateSegments = candidateSegments[:int(propvarderiv*len(candidateSegments))]

				if finalValue < goal:
					derivvars = [ numpy.exp( s.dTemp/temp ) for s in candidateSegments]
				else:
					derivvars = [ numpy.exp( (-s.dTemp)/temp ) for s in candidateSegments]
				derivvars = numpy.array(derivvars)
				probs = derivvars/np.sum(derivvars)
				choice = numpy.where(numpy.random.multinomial(len(candidateSegments),pvals=probs))[0][0]
				newSegment = candidateSegments[choice]
				
				# shift everything so that it can be stitched
				startDiff = newSegment.dataset[0] - lastSegment.forecast()
				newSegment.dataset -= startDiff
				lastSegment = newSegment
				
				# we have to check every datapoint on its own
				brkDummy = False
				for d in lastSegment.dataset:
					iSignal.append(d)
					atckI += 1
					if abs(d-dTesting[atckI]) < 0.1:
						brkDummy = True
						break
					if len(iSignal) >= len(dTesting):
						brkDummy = True
						break
				if brkDummy:
					iSignal += dTesting[atckI+1:]
					break		
				
				finalValue = lastSegment.dataset[-1]
				goal = dTesting[atckI]
				diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
		
		
		
	#  attack
	def not_working_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable, dTesting, comeBack=False):
		propvarderiv = 0.7
		propdifftemp = 0.07
		
		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
		
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		while diff > 0.1:
			# we pick the next cluster probabilistically from condProbsTable
			lastCluster = lastSegment.cluster
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break
			
			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get segment w/ greedy strategy
			if finalValue < goal:
				# check if signal is on its way
				if nextCluster != 3:
					print 'filtering', lastSegment.dTemp
					# filter by relative variance of first derivative
					lastMeanDer1 = lastSegment.meanDer1
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propdifftemp*len(candidateSegments))]
				else:
					print 'only cluster', nextCluster
				newSegment = random.choice(candidateSegments)
			else:
				# check if signal is on its way
				if lastSegment.dTemp > -0.2:
					print 'filtering'
					# filter by relative variance of first derivative
					lastMeanDer1 = lastSegment.meanDer1
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)[:int(propdifftemp*len(candidateSegments))]					
				else:
					print 'only cluster'
				newSegment = random.choice(candidateSegments)
				
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment

			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				if comeBack:
					iSignal += lastSegment.dataset.tolist()
				else:
					iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
		
		'''////////////////////////////////////////////		
		/////////////////// comeBack /////////////////	
		 the signal must go back to the real dataset
		'''
		if comeBack and len(iSignal)<len(dTesting):
			atckI = len(iSignal) - 1
			finalValue = lastSegment.dataset[-1]
			goal = dTesting[atckI]
			diff = abs(finalValue - goal)
		
			while diff > 0.1:

				# we pick the next cluster probabilistically from condProbsTable
				lastCluster = lastSegment.cluster
				rand = random.uniform(0,1)
				for (i, prob) in enumerate(condProbsTable[lastCluster]):
					rand -= prob
					if rand <= 0:
						nextCluster = i
						break

				# filter by cluster
				candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
				
				# get segment w/ greedy strategy
				if finalValue < goal:
					# check if signal is on its way
					if lastSegment.dTemp < 0.2:
						# filter by relative variance of first derivative
						lastMeanDer1 = lastSegment.meanDer1
						candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
						candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propdifftemp*len(candidateSegments))]
					newSegment = random.choice(candidateSegments)
				else:
					# check if signal is on its way
					if nextCluster != 2:
						# filter by relative variance of first derivative
						lastMeanDer1 = lastSegment.meanDer1
						candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.calc_der1_rel_var(lastMeanDer1)))[:int(propvarderiv*len(candidateSegments))]
						candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp)[:int(propdifftemp*len(candidateSegments))]					
					else:
						print '>>', nextCluster
					newSegment = random.choice(candidateSegments)
				
				# shift everything so that it can be stitched
				startDiff = newSegment.dataset[0] - lastSegment.forecast()
				newSegment.dataset -= startDiff
				lastSegment = newSegment
				
				# we have to check every datapoint on its own
				brkDummy = False
				for d in lastSegment.dataset:
					iSignal.append(d)
					atckI += 1
					if abs(d-dTesting[atckI]) < 0.1:
						brkDummy = True
						break
					if len(iSignal) >= len(dTesting):
						brkDummy = True
						break
				if brkDummy:
					iSignal += dTesting[atckI+1:]
					break		
				
				finalValue = lastSegment.dataset[-1]
				goal = dTesting[atckI]
				diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)	
		
		# cluster
		# random
		# shift
	def cluster_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
		
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
		
		while diff > 0.1:
		
			lastCluster = lastSegment.cluster

			# we pick the next cluster probabilistically from condProbsTable
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break

			# filter by cluster
			candidateSegments = [segment for segment in segments if segment.cluster == nextCluster]
			
			# get new cluster
			newSegment = random.choice(candidateSegments)
			
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.dataset -= startDiff
			lastSegment = newSegment

			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]

			diff = abs(finalValue - goal)
			#print diff
			
		return (startSignal, iSignal)
	
	
	# ditto attack
	def ditto_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable, dTesting, comeBack=False):
		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		return (startSignal, startSignal + dTesting[len(startSignal)-1:])
		
		
		
		
		
		
		
	###
	####
	#####
	####
	def rwgm_attack(self, sensorID, goal, atckDelay, nodesSegmentsDic, condProbsTable, dTesting, propMean, propWDer1, propDTemp, comeBack=False):

		(startSignal, startSegment) = self.prepare_attack(sensorID, goal, atckDelay, nodesSegmentsDic)
		
		# merge all segments
		segments = []
		for nodeID in nodesSegmentsDic.keys():
			segments += nodesSegmentsDic[nodeID]
			
		iSignal = startSignal[:]
		lastSegment = startSegment
		finalValue = lastSegment.dataset[-1]
		diff = abs(finalValue - goal)
	
		while diff > 0.1 and len(iSignal)<len(dTesting):
	
			# we pick the next cluster probabilistically from condProbsTable
			lastCluster = lastSegment.cluster
			rand = random.uniform(0,1)
			for (i, prob) in enumerate(condProbsTable[lastCluster]):
				rand -= prob
				if rand <= 0:
					nextCluster = i
					break

			# filter by cluster
			#candidateSegments = [segment for segment in segments if segment.cluster == nextCluster or segment.cluster == lastCluster]
			candidateSegments = segments
			# get segment w/ greedy strategy
			if finalValue < goal:
				# filter by
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.dMean-lastSegment.dMean), reverse=False)[:int(propMean*len(candidateSegments))+1]
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.startWMeanDer1-lastSegment.endWMeanDer1), reverse=False)[:int(propWDer1*len(candidateSegments))+1]
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propDTemp*len(candidateSegments))+1]
				newSegment = random.choice(candidateSegments)
			else:
				# filter by 
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.dMean-lastSegment.dMean), reverse=False)[:int(propMean*len(candidateSegments))+1]
				candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.startWMeanDer1-lastSegment.endWMeanDer1), reverse=False)[:int(propWDer1*len(candidateSegments))+1]
				candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=False)[:int(propDTemp*len(candidateSegments))+1]
				newSegment = random.choice(candidateSegments)

			
			# shift everything so that it can be stitched
			startDiff = newSegment.dataset[0] - lastSegment.forecast()
			newSegment.upd_dataset(newSegment.dataset-startDiff)
			lastSegment = newSegment

			# see if we reached goal temperature
			endPoints = [ i for (i,r) in enumerate(lastSegment.dataset) if abs(r-goal)< 0.1 ]
			
			if len(endPoints) != 0:
				if comeBack:
					iSignal += lastSegment.dataset.tolist()
				else:
					iSignal +=  lastSegment.dataset[:endPoints[0]+1].tolist()
				break
			else:
				iSignal += lastSegment.dataset.tolist()
				
			finalValue = lastSegment.dataset[-1]
			diff = abs(finalValue - goal)
		
		'''////////////////////////////////////////////		
		/////////////////// comeBack /////////////////	
		 the signal must go back to the real dataset
		'''
		if comeBack and len(iSignal)<len(dTesting):
			atckI = len(iSignal) - 1
			finalValue = lastSegment.dataset[-1]
			goal = dTesting[atckI]
			diff = abs(finalValue - goal)
		
			while diff > 0.1:
				
				# we pick the next cluster probabilistically from condProbsTable
				lastCluster = lastSegment.cluster
				rand = random.uniform(0,1)
				for (i, prob) in enumerate(condProbsTable[lastCluster]):
					rand -= prob
					if rand <= 0:
						nextCluster = i
						break

				# filter by cluster
				#candidateSegments = [segment for segment in segments if segment.cluster == nextCluster or segment.cluster == lastCluster]
				candidateSegments = segments			
				# get segment w/ greedy strategy
				if finalValue < goal:
					# filter by
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.dMean-lastSegment.dMean), reverse=False)[:int(propMean*len(candidateSegments))+1]
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.startWMeanDer1-lastSegment.endWMeanDer1), reverse=False)[:int(propWDer1*len(candidateSegments))+1]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=True)[:int(propDTemp*len(candidateSegments))+1]
					newSegment = random.choice(candidateSegments)
				else:
					# filter by 
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.dMean-lastSegment.dMean), reverse=False)[:int(propMean*len(candidateSegments))+1]
					candidateSegments = sorted(candidateSegments, key=lambda s: abs(s.startWMeanDer1-lastSegment.endWMeanDer1), reverse=False)[:int(propWDer1*len(candidateSegments))+1]
					candidateSegments = sorted(candidateSegments, key=lambda s: s.dTemp, reverse=False)[:int(propDTemp*len(candidateSegments))+1]
					newSegment = random.choice(candidateSegments)

			
				# shift everything so that it can be stitched
				startDiff = newSegment.dataset[0] - lastSegment.forecast()
				newSegment.upd_dataset(newSegment.dataset-startDiff)
				lastSegment = newSegment
				
				# we have to check every datapoint on its own
				brkDummy = False
				for d in lastSegment.dataset:
					iSignal.append(d)
					atckI += 1
					if abs(d-dTesting[atckI]) < 0.1:
						brkDummy = True
						break
					if len(iSignal) >= len(dTesting):
						brkDummy = True
						break
				if brkDummy:
					iSignal += dTesting[atckI+1:]
					break		
				
				finalValue = lastSegment.dataset[-1]
				goal = dTesting[atckI]
				diff = abs(finalValue - goal)
			
		return (startSignal, iSignal)
		
		
		
# uniform attack
	def uniform_attack(self, goal, dTesting):
		iSignal = [dTesting[-1]]
		
		while iSignal[-1] < goal:
			iSignal.append(iSignal[-1]+0.015)
		
		return ([], iSignal)
		
		
		
		
		