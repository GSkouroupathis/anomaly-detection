from detector import *
import numpy

class EKFDetector(Detector):
	# Constructor
	def __init__(self, dataset, Q=None, delta=None):
		Detector.__init__(self, dataset)
		self.delta = delta
		if Q:
			self.Q = Q
		else:
			self.Q = 0.2809 # Q value for Intel Lab Data
		if delta:
			self.delta = delta
		else:
			self.delta = 3 * numpy.sqrt(self.Q) # Intel Lab delta

	# Sets dataset
	def set_dataset(self, dataset):
		Detector.set_dataset(self, dataset)

	# Sets delta
	def set_delta(self, delta):
		self.delta = delta
		
	# Creates x_priori_k list
	def comp_x_prioris(self):
		
		# Function relating x_k1 to x_k
		# assumes readings do not change
		# over time
		def f(x): return x
		#######################
		
		# calculate R, the variance
		# between z_k and z_k1
		def calculate_R():
			z_diff = [self.dataset[0]]
    
			for i in range(1, len(self.dataset)):
				z_diff.append(self.dataset[i] - self.dataset[i-1])

			R = numpy.var(z_diff)

			return R
		#######################
	
		Q = self.Q
		R = calculate_R()
		P_posteriori_k = 0
		P_priori_k = 0
		
		x_priori_k = self.dataset[0]
		x_prioris = [x_priori_k]
		for k in range(len(self.dataset)-1):
			datapoint_k = self.dataset[k] #datapoint before
			
			# Step 1
			K_k = P_priori_k / (P_priori_k +R)
			x_posteriori_k = x_priori_k + K_k*(datapoint_k - x_priori_k)
			
			# Step 2
			x_priori_k1 = f(x_posteriori_k)
		
			# Update values
			x_priori_k = x_priori_k1
			P_priori_k = P_posteriori_k + Q
			x_prioris.append(x_priori_k)
			
		return (x_prioris, R)
		
	# Detects anomalies
	def detect(self):
		
		self.alert = False
		ekfVals = [0]
		alertVals =[]
		x_prioris = self.comp_x_prioris()
		
		# Detect anomaly
		for i in range(len(self.dataset)-2):
			x_priori_k1 = x_prioris[i+1]
			datapoint_k1 = self.dataset[i+1]
			diff = abs(x_priori_k1 - datapoint_k1)
			
			if diff > self.delta:
				self.alert = True
				alertVals.append((i+1, diff))
			ekfVals.append(diff)

		return (ekfVals, alertVals, self.alert, self.delta)