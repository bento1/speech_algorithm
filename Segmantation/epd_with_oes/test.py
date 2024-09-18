import numpy as np
import matplotlib.pyplot as plt
from changepy import pelt
from changepy.costs import normal_mean
x=np.linspace(0,10-0.1,100)
y=np.sin(x*2*np.pi*2/10)
y_sigmoid=4/(1+np.exp(-(x-np.mean(x))))
plt.plot(y_sigmoid)
x=np.linspace(0,20-0.1,200)
y=np.concatenate([y,y_sigmoid])+np.random.normal(0.2, 0.4, 200)
plt.plot(x,y)

result=pelt(normal_mean(y, np.power(np.std(y),2)), len(y))

from changepy import peltWithCost
from changepy.costs import normal_mean,normal_meanvar
cp,seg_costs=peltWithCost(normal_mean(y,np.power(np.std(y),2)),len(y))

fig, ax1 = plt.subplots()
ax1.plot(x, y, color='green')
for lim in result:
    plt.axvline(x=x[lim])



plt.show()