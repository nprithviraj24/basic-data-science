import matplotlib.pyplot as plt 

#Sample data
a = [123, 45, 78, 150]
b = [2.0, 3.4, 6.2, 9.2]

#Basic plot function in MatPlotLib
plt.plot(a,b)

#Output 1
plt.show()

#Scatter data visualisation
plt.scatter(a,b)

#setting the label for X axis
plt.xscale('log')

#Output 2
plt.show()

#Clearing the buffer
plt.clf()





plt.plot()