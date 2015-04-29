import matplotlib.pyplot as plt

class Plotter(object):
	def __init__(self, timeseries):
		self.ts = timeseries
		
	def plot(self):
		plt.plot(self.ts)
		plt.ylabel('temperature')
		plt.show()