import math, dbOp
import numpy as np
from ekfdetector import *
from cusumdetector import *

#################################################################
#Sets up stuff###################################################
# sets the allowed limit temperature for sin wrapper to be called
DT_LIMIT = 2.0
# sets the allowed threshold for first order derivatives used during stitching
GRADIENT_THRESHOLD = 1.0
# sets the allowed threshold for S_N used during stitching
SN_THRESHOLD = 1.3
# sets the ratio limit before stretching of replay values
STRETCH_RATIO_THRESHOLD = 2.0
# sets the acceptable threshold difference used during comparison
TEMPERATURE_DIFF_THRESHOLD = 0.05
#################################################################
#################################################################

# given a particular sensor and a goal temperature,
# first finds the highest temperature and start from there
# finds replay values to try to reach goal
# if temperature is not found, uses sin to increase temperature
def terence_mimicry(sensor_id, goal, starting_index, sensor_set):
	dbOp.connectToDatabase("data/db")
	z =  map(lambda x: x[0], dbOp.selectReadingsFromNode(sensor_id))
	dbOp.closeConnectionToDatabase()
	
	if starting_index > len(z):
		print "Error: starting_id cannot be greater than length of data"
		return []
	
	final_attacked_values = z[0:starting_index+1]
	lowest_temp_index = -1
	lowest_temp = 99999
	
	for i in range(starting_index, min((starting_index+5000), len(z))):
		if z[i] < lowest_temp:
			lowest_temp = z[i]
			lowest_temp_index = i
			
		if (abs(z[i] - goal) <= 0.01):
			final_attacked_values += z[starting_index:i+1]
			return (final_attacked_values, lowest_temp_index)
			
	final_attacked_values += z[starting_index:(lowest_temp_index+1)]
	
	final_stitch = stitch(z[lowest_temp_index], goal, sensor_set, (len(z) - lowest_temp_index))
	final_attacked_values += final_stitch
	
	if (max(final_attacked_values) < goal):
		DT = goal - max(final_attacked_values)
		
		if (DT >= DT_LIMIT):
			return (final_attacked_values, lowest_temp_index)

		final_attacked_values = add_sin(final_attacked_values, DT, lowest_temp_index, len(final_attacked_values))
		
	while abs((len(z)-lowest_temp_index)*1.0 / (  len(final_attacked_values) - lowest_temp_index) ) >= STRETCH_RATIO_THRESHOLD:
		final_attacked_values[lowest_temp_index:] = stretch(final_attacked_values[lowest_temp_index:])
		
	return (final_attacked_values, lowest_temp_index)
	
# takes data from a set and stitch past measurements together
def stitch(start_temp, end_temp, sensor_set, length):
	stitched_value_index = 0
	final_stitch = []
	current_start_temp = start_temp
	
	for sensor in sensor_set:
		dbOp.connectToDatabase("data/db")
		z =  map(lambda x: x[0], dbOp.selectReadingsFromNode(sensor))
		dbOp.closeConnectionToDatabase()
	
		
		if len(z) > 0:
			z_array = np.array(z)
			first_order = np.gradient(z_array)
			
			EKFd = EKFDetector(z)
			CUSUMd = CUSUMDetector(z, h=0.4, w=10, EKFd=EKFd)
			detectionResults = CUSUMd.detect()
			
			(S_N_array, x_minus, R) = (detectionResults[0], detectionResults[2], detectionResults[3])
			
			start_index = 0
			
			while start_index < len(z):
				(current_replay, last_i) = find_replay_values(sensor, z, first_order, S_N_array, start_index, len(z), current_start_temp, end_temp)
				final_stitch = final_stitch + current_replay
				stitched_value_index += len(current_replay)
				
				if len(final_stitch) > 0:
					current_start_temp = final_stitch[-1]
					
				start_index = last_i + 1
				
				if (abs(final_stitch[-1] - end_temp) <= TEMPERATURE_DIFF_THRESHOLD):
					return final_stitch

	return final_stitch				
			
	
# given a set of data, its gradient, S_N, a desired start temperature and end temperature with a threshold,
# returns a set of past values that starts and ends with them respectively
# also considers the derivatives when choosing values
def find_replay_values(sensor_id, z, first_order, S_N_array, start_window, end_window, start_temp, end_temp):
	if z == []:
		return []
	if (start_window > end_window):
		return []
	if (end_window > len(z)):
		return []
	
	start_index = -1
	end_index = -1
	highest_index = -1    
	highest_temp = -1

	for i in range(start_window, end_window):
		if (abs(z[i] - start_temp) <= TEMPERATURE_DIFF_THRESHOLD and start_index == -1):
			start_index = i
			
		if (start_index != -1):
			if (abs(z[i] - end_temp) <= TEMPERATURE_DIFF_THRESHOLD):
				end_index = i
				break
				
		if (abs(first_order[i]) >= GRADIENT_THRESHOLD):
			break
			
		if (abs(S_N_array[min((len(first_order)-1) , (i+10))]) >= SN_THRESHOLD):
			break
		
		if (z[i] > highest_temp):
			highest_temp = z[i]
			highest_index = i
			
	if (end_index == -1 and start_index != -1):
		end_index = highest_index
		
	if (start_index == -1 or end_index == -1):
		return ([], i)
	
	return (z[start_index:end_index+1], i)
	
# given a dataset z, difference in temperature DT, and starting index
# adds a sin to dataset to increase and achieve the temperature difference from index
def add_sin(z, DT, starting_index, end_index):
	sin_z = z[:]
	
	for x in range(starting_index, end_index):
		sin_z[x] = sin_z[x] + ( (x-starting_index) * DT/(len(sin_z)-starting_index)) + 0.2*math.sin(0.2*math.radians(len(sin_z)-x))
		
	return sin_z

# Stretches the graph
def stretch(z):
	if z== []:
		return []
	if len(z) == 1:
		return [z]
		
	new_temp_list = []
	
	for i in range(0, len(z)-1):
		new_temp_list.append(z[i])
		new_temp_list.append( (z[i]+z[i+1])/2.0 )
	
	new_temp_list.append(z[-1])
	
	return new_temp_list
	
	
	
	
	
	
	
	