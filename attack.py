import math

def terence_mimicry():
	pass
	
def get_temperature_array_from_db():
	pass
	
def stitch():
	pass
	
def calculate_R():
	pass
	
def find_replay_values():
	pass

# given a dataset z, difference in temperature DT, and starting index
# adds a sin to dataset to increase and achieve the temperature difference from index
def add_sin(z, DT, starting_index, end_index):
	sin_z = z[:]
	
	for x in range(starting_index, end_index):
		sin_z[x] = sin_z[x] + ( (x-starting_index) * DT/(len(sin_z)-starting_index)) + 0.2*math.sin(0.2*math.radians(len(sin_z)-x))
		
	return sin_z

def stretch():
	pass