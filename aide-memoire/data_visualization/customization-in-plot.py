import matplotlib.pyplot as plt

#year = [1950, 1951, 1952, ..., 2100]
pop = [2.538, 2.57, 2.62, ..., 10.85]

year = []
for i in range(1950, 2100):
    year.append(i)

#Adding more data
year = [1800, 1850, 1900] + year
pop = [1.0, 1.262, 1.650] + pop

#Normal plotting
plt.plot(year, pop)

#customization.
plt.xlabel('Year')  #Labels X axis
plt.ylabel('Population') #Labels Y axis
plt.title('World Population Projections') #Gives the title to the plot


'''
plt.yticks() changes the scale in y axis with numerical values specified, 
then they are replaced by their equivalent index vale strings in the following array. 
'''
plt.yticks( [0,2,4,6,8,10],
['0', '2B', '4B', '6B', '10B'])  # B = Billions


plt.show()