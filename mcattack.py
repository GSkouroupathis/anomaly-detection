from attack import *
import dbOp
from scipy.cluster.vq import kmeans2
import numpy

class MCMimicry(Attack):
	# Constructor
	def __init__(self, dataset):
		self.dataset = dataset
		self.w = self.choose_window_size(dataset)

	# 1 ##########################################################################
	# Chooses the appropriate window size w
	def choose_window_size(self, dataset=None):
		if !dataset: dataset = self.dataset
		
		bestScore = -1
		
		for w in [5, 10, 15, 100]:
			segment_list = numpy.array(segment_signal(dataset, w, 0))
			(centroids, labels) = cluster(segment_list)
			# the sum of segments in each cluster
			clusterCount = [labels.count(i) for i in range(len(centroids))]
			condProbTable = create_cond_prob_table(centroids, labels, clusterCount)
			tableScore = eval_cond_prob_table(cond_prob_table, clusterCount)
			if tableScore > bestScore:
				bestW = w
				bestSegments = segment_list
				(bestCentroids, bestLabels) = (centroids, labels)
				bestCondProbTable = condProbTable
				bestScore = tableScore
				
		return (bestW, bestSegments, bestCentroids, bestLabels, bestCondProbTable)
			
	# 2 ##########################################################################
	# Divides signal into segments
	# a: signal
	# w: window size
	# h: hop size
	def segment_signal(z, w, h):
		segment_list = []
		for (i, r) in enumerate(z[::w]):
			segment = z[i*w:i*w+w]
			segment_list.append(segment)
		return segment_list

	# Clusters the segments into K clusters
	def cluster(segments):
		prevAvDist = 99999999
		prevCentroids = numpy.array([])
		prevLabels = []
		# tries to find K
		for K in range(2,100):
			(centroids, labels) = k_means(segments, K)
			avDist = get_average_distance(centroids, labels, segments)
			isKnee = find_knee(prevAvDist, avDist)
			# if correct K found
			if isKnee:
				return (prevCentroids, prevLabels)
			else:
				prevAvDist = avDist
				prevCentroids = centroids
				prevLabels = labels
		return (None, None)
			
	# Creates conditional probabilities table
	def create_cond_prob_table(centroids, labels, clusterCount):
		# the sum of segments in cluster given previous cluster
		countTable = [[0]*len(centroids) for i in range(len(centroids))]
		for (i, thisCluster) in enumerate(labels[1:]):
			lastCluster = labels[i]
			countTable[lastCluster][thisCluster] += 1
		
		condProbTable = dirichlet(countTable, clusterCount, K=len(centroids))
		return (condProbTable, clusterCount)
			
	# Evaluates conditional probabilities table
	def eval_cond_prob_table(condProbTable, clusterCount):
		negentropies = []
		for row in condProbTable:
			negentropies.append(negentropy(row))
		score = w_m_negentropy(negentropies, clusterCount):
		return score
		
	# 3 ##########################################################################
	# Performs K-means
	# K: number of clusters
	def k_means(segments, K):
		(centroids, labels) = kmeans2(segments, K)
		return (centroids, labels)
		
	# Gets average distance of segments to
	# their corresponding cluster centers
	def get_average_distance(centroids, labels, segments):
		totalDist = 0
		for (i, segment) in enumerate(segments):
			centroid = centroids[labels[i]]
			dist = numpy.linalg.norm(segment-centroid)
			totalDist += dist
		return totalDist / len(segments)
		
	# Finds knee on graph by computing the derivative
	def find_knee(prevAvDist, avDist):
		return prevAvDist - avDist > -0.3
	
	# 3 ##########################################################################
	# Applies Dirichlet distribution to create Conditional Probabilities Table
	def dirichlet(countTable, clusterCount, K):
		for (prevCluster, row) in enumerate(countTable):
			for (thisCluster, thisClusterCount) in enumerate(row):
				row[thisCluster] = (thisClusterCount + 1) / (clusterCount[prevCluster] + K)
		return countTable

	# 3 ##########################################################################	# Computes entropy of distribution
	def entropy(distribution):
		return -sum([p*math.log(p,10) for p in distribution])
		
	# Computes negentropy of distribution against uniform distribution
	def negentropy(distribution):
		uniform = [1.0/len(distribution) for i in distribution]
		return entropy(uniform) - entropy(distribution)
	
	# Weighted mean negentropy
	def w_m_negentropy(negentropies, clusterCount):
		totalClusterCount = sum(clusterCount)
		return sum(map(lambda x: x[0]*(x[1]*1.0/totalClusterCount), zip(negentropies, clusterCount)))

	# 1 ##########################################################################	
	# Attacks sensor_id until goal
	def attack(self, sensor_z, goal, starting_index, sensor_set_z):
		pass
		
	# 2 ##########################################################################	
	def build_attack_tree(segments, goal, starting_index):
		pass
		