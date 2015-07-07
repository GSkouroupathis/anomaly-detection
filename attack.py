# Attack class
class Attack(object):

	def __init__(self, dataset=None):
		pass

	def attack(self, sensor_z, goal, starting_index, sensor_set_z):
		raise NotImplementedError('attack() not implemented')