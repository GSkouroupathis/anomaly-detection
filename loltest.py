import dbOp, data

dbOp.connectToDatabase("data/db")
r7 = dbOp.selectReadingsFromNode(7)
dbOp.closeConnectionToDatabase()
(d3Training, d3Testing, r3Tr, r3Te) = data.getTrainingTesting(r7)

import matplotlib.pyplot as plt
#plt.axis('equal')
plt.plot(d3Training, 'b')
plt.show()