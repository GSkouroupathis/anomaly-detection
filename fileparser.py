def parse_file(path, maxTemp):
	with open(path, 'r') as input_file:
		fileLines = input_file.readlines()
		readings = {}
		for (i, line) in enumerate(fileLines):
			if i%500==0: print " -- processing record", i
			
			sanCheck = 1
			
			line = line.rstrip()
			entry_b = line.split(" ")
			
			# check for correct format
			if len(entry_b) != 8:
				continue
			entry_a = [-1, "generic_date", "generic_time", -9999]
			
			# node ID
			try:
				entry_a[0] = int(entry_b[3])
			except:
				sanCheck = 0

			# date
			entry_a[1] = entry_b[0]

			# time, also remove decimals
			entry_a[2] = entry_b[1].split('.')[0]

			# temperature
			try:
				temperature = float(entry_b[4])
				
				# sanitising
				if temperature > maxTemp or temperature < 0:
					sanCheck = 0
					print " ** Invalid temperature. Dropped record", i
					
				entry_a[3] = temperature
			except:
				
				sanCheck = 0
			
			# voltage
			voltage = entry_b[7]
			if voltage < 2.3:
				sanCheck = 0
				print " ** Invalid voltage. Dropped record", i
			
			if sanCheck:
				if entry_a[0] not in readings.keys():
					readings[entry_a[0]] = [entry_a]
				else:
					readings[entry_a[0]].append(entry_a)
			
		return readings