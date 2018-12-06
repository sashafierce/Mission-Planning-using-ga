import sys
import numpy as np
file=sys.argv[1]
f=open(file,"r")
lines=f.readlines()
result=[]
result2=[]
for x in lines:
    result.append(float(x.split(' ')[1]))
    result2.append(float(x.split(' ')[2]))
print result

mean = np.mean(np.array(result))
mean2 = np.mean(np.array(result2))
sd = np.std(result)
sd2 = np.std(result2)
print "mean , cost and sd and sdcost = ", mean,  sd,mean2, sd2
f.close()
