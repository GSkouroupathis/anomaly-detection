from detector import *
from ekfdetector import *
import math, testing

class CUSUMDetector(Detector):
	# Constructor
	 # h: CUSUM threshold
	 # w: CUSUM window size
	 # EKFd: EKF Detector
	def __init__(self, dataset, h, w, EKFd):
		Detector.__init__(self, dataset)
		self.h = h
		self.w = w
		self.EKFd = EKFd

	# Sets dataset
	def set_dataset(self, dataset):
		Detectot.set_dataset(self, dataset)

	# Sets threshold
	def set_threshold(self, h):
		self.h =h

	# Sets window size
	def set_w(self, w):
		self.w = w
	
	# Detects anomalies
	def detect(self):
		s_ks_sum = 0
		self.alert = False
		y = []
		x_prioris = self.EKFd.comp_x_prioris()
		cusumVals = [0 for i in self.dataset]
		s_ks = [0 for i in self.dataset] #testing
		mu1 = 0
		w = self.w
		h = self.h
		sigma = math.sqrt(self.EKFd.Q)

		for k in range(len(self.dataset)):
			datapoint_k = self.dataset[k]
			try:
				x_priori_k = x_prioris[k]
			except:
				break
			y_k = datapoint_k - x_priori_k # Line 1 Alg 2
			y.append(y_k)
			
			if k >= w - 1:
				mu1 = (1.0/w) * sum(y[k-w+1:k+1]) # Line 2 Alg 2
			
			b = mu1 / sigma
			v = mu1
			s_k = y_k - v/2.0

			s_ks[k] = s_k
			s_ks_sum += s_k
			if k > 0:
				s_N = s_ks_sum * (b/sigma)
			else:
				s_N = 0

			if s_N > h:
				self.alert = True
			cusumVals[k] = s_N
			
		return (cusumVals, s_ks, self.alert, self.h, self.w)
			
	def cusum(self):

		z=self.dataset
		
		R=self.calculate_R()
		
    # h is the predefined threshold to determine anomaly
		h = 0.4

    # w is the window size used to estimate parameters
		w = 10
    
		k_size = len(z)
    
    # an array of all the x_plus(s) with arbitrary initial values = 0
    # index of array represents the time k in x_k_plus
		x_plus = [0] * k_size

    # an array of all the x_minus(s)
    # index represents the time k in x_k_minus
		x_minus = [0] * k_size
		x_minus[0] = z[0]

    # an array of all the P_plus(s)
    # index represents the time k in P_k_plus
		P_plus = [0] * k_size

    # an array of all the P_minus(s)
    # index represents the time k in P_k_minus
		P_minus = [0] * k_size

    # an array of all the K(s)
    # index represents the time k in K_k
		K = [0] * k_size

    # an array of all the y(s)
    # index represents the time k in y_k
		y = [0] * k_size

    # S_N_array keeps track of all the S_N in each iteration of k
		S_N_array = [0] * k_size

    # cus keeps track of the cumulative sum in each iteration of k
		cus = [0] * k_size
    

    # constants, Q
		Q = 0.2809  # estimated variance for Intel Dataset
		kokos = []
		for k in range(0, k_size):
        
        # updates x- with the (previous) value of x+
			if k > 0:
				x_minus[k] = x_plus[k-1]  # Eqn 3

    
      # computes y_k, the difference between the measurement
      # and the estimated state
			y[k] = z[k] - x_minus[k]

      # implementing line 2 of algorithm 2
      # miu_1 is the computation of line 2 of algorithm 2
			miu_one = 0

			if (k >= w - 1):
            # compute miu_1

				for i in range(k-w+1, k+1):
					miu_one += y[i]
            
				miu_one /= w

        # implementing line 3 of algorithm 2
        # S_N is the computation of line 3 of algorithm 2
			b_over_sigma = miu_one/Q

			s_k = (y[k] - miu_one/2.0)
			kokos.append(s_k)
			if (k == 0):
				cus[0] = s_k
			else:
            # k > 0
				cus[k] = cus[k-1] + s_k

			S_N_array[k] = cus[k] * b_over_sigma


        # implementing line 4 of algorithm 2
        #if (S_N > h):
            #print "Alert is raised on time k = " + str(k)

        # Start with P+ = P- = 0
			if k > 0:
				P_minus[k] = P_plus[k-1] + Q  # Eqn 4
			else:
				P_minus[k] = Q


        # computes K, the factor which multiplies the noise
        # K = P-/(P- + R)
			K[k] = P_minus[k]/(P_minus[k] + R)  # Eqn 5

       # update x+ with the measurement
       # x+ = x- + K * (y_k)
			x_plus[k] = x_minus[k] + K[k] * y[k]  # Eqn 6
        
        # update P+
			P_plus[k] = (1-K[k]) * P_minus[k]  # Eqn 7

    # multiply all values inS _N_array by b_over_sigma    
    #S_N_array = map(lambda x:x*b_over_sigma, S_N_array)

    # returns an array of predictions [x_0-, x_1-, x_2-, ...]
		return (x_minus, S_N_array, kokos)
			
	def calculate_R(self):
			z_diff = [self.dataset[0]]
    
			for i in range(1, len(self.dataset)):
				z_diff.append(self.dataset[i] - self.dataset[i-1])

			R = numpy.var(z_diff)

			return R
		