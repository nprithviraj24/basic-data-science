#Import the numpy package as np.
#Use np.array() to create a numpy array from the list pop. Call this Numpy array np_pop.
#Double the values in np_pop by assigning np_pop * 2 to np_pop again. Because np_pop is a Numpy array, each array element will be doubled.
#Change the s argument inside plt.scatter() to be np_pop instead of pop.


# Import numpy as np
import numpy as np
import matplotlib.pyplot as plt


#This file just for study purpose only. We don't know the value of pop.

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)
#'pop' population numbers for each country expressed in millions.

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
xVal= [1000, 10000, 100000]
xRepresentation =['1k', '10k', '100k']
plt.xticks(xVal,xRepresentation)

# Display the plot
plt.show()
