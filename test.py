  
import numpy as np
import matplotlib  
import matplotlib.pyplot as plt  

x = np.random.randn(50,30)
#basic  
f1 = plt.figure(1)  
plt.subplot(211)  
plt.scatter(x[:,1],x[:,0]) 