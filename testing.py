import numpy

# Testing functions

def read_array(array, pause):
	for (i, elem) in enumerate(array):
		if i % pause == 0:
			print elem
			
def arrays_info(arrays):
	for array in arrays:
		print "Information for array", id(array)
		print "-+-+-+-+-+-+-+-+-"
		print "Length:", len(array)
		print "Mean:", numpy.mean(array)
		print "Variance:", numpy.var(array)
		print
		
def files_info(files):
	for f in files:
		print "Information for file", f
		print "-+-+-+-+-+-+-+-+-"
		print "Lines:", len(f.readlines())
		print