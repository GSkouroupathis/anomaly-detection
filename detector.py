# Detector Class
class Detector(object):

	def __init__(self, dataset=None):
		if dataset:
			self.dataset = dataset
		else:
			self.dataset = []
		self.alert = False

	def set_dataset(self, dataset):
		self.dataset = dataset

	def detect(self):
		raise NotImplementedError('detect() not implemented')