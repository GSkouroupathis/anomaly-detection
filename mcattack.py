from attack import *
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
		
		for w in [5, 7, 8, 9, 10, 12, 15, 20, 30]:
			segment_list = numpy.array(self.segment_signal(dataset, w, 0))
			norm_segments = map(lambda x: x-numpy.mean(x), segment_list)
			(centroids, labels) = self.cluster(segment_list)
			# the sum of segments in each cluster
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
		for K in range(2,100):
			(centroids, labels) = self.k_means(segments, K)
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



	# 1 ##########################################################################	
	# Attacks sensor_id until goal
	def tree_attack(self, sensorID, goal, atckDelay, sensorsSegmentsReadingsDic, condProbTable):
				
		# find actual time of attack start
		firstDateTime = sensorsSegmentsReadingsDic[sensorID][0][-1][0][0] + ' ' + sensorsSegmentsReadingsDic[sensorID][0][-1][0][1]
		
		atkStartDateTime = datetime.datetime.strptime(firstDateTime, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(0,atckDelay)
		
		# constructs starting signal
		startSignal = []
		startSegment = None
		for (i, segmentInfo) in enumerate(sensorsSegmentsReadingsDic[sensorID]):
			firstSegmentDateTime = datetime.datetime.strptime(segmentInfo[0] + ' ' + segmentInfo[1], '%Y-%m-%d %H:%M:%S')

			# find if the segment will be added into the attack
			if firstSegmentDateTime < atkStartDateTime:
				startSignal += map(lambda x: x[-1], segmentInfo[-1])
			else:
				# we don't need segment i-1
				startSegment = sensorsSegmentsReadingsDic[sensorID][i-1]
				if i > 0:
					del sensorsSegmentsReadingsDic[sensorID][i-1]
				break
		
		iSignal = self.tree_attack_(sensorID, startSignal, startSegment, goal, sensorsSegmentsReadingsDic, condProbTable)
		return iSignal
		
	# 2 ##########################################################################	
	def tree_attack_(self, sensorID, startSignal, startSegment, goal, sensorsSegmentsReadingsDic, condProbTable):
		# first build attack tree & root node
		atckTree = AttackTree(sensorID, startSignal, 0.25, 29, condProbTable)
		rootNode = AttackNode(atckTree, sensorID, startSegment)
		atckTree.set_start_node(rootNode)
		
		# launch the attack
		return atckTree.tree_attack(sensorID, sensorsSegmentsReadingsDic)
		
		
		
		
		
		