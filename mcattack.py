from attack import *
import dbOp

class MCMimicry(Attack):
	# Constructor
	def __init__(self, dataset):
		self.dataset = dataset
		self.w = self.choose_window_size(dataset)

	# Chooses the appropriate window size w
	def choose_window_size(self, dataset=None):
		if !dataset: dataset = self.dataset
		
	# Attacks sensor_id until goal
	def attack(self, sensor_id, goal, starting_index, sensor_set):
		dbOp.connectToDatabase("data/db")
		z =  map(lambda x: x[0], dbOp.selectReadingsFromNode(sensor_id))
		dbOp.closeConnectionToDatabase()
		