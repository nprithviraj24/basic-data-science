import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
#plt.ion()

for i in range(25):
    y = np.random.random()
    plt.scatter(i, y)
    plt.plot(i,y, '-o')
    plt.pause(0.05)

while True:
    plt.pause(0.01)