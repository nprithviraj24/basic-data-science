import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import random

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

#This function will be called again at each interval recursively. 
# So any changes at the change in file will be represented in the graph 

counter = 0

def animate(i):
    global counter
    counter = counter + 1
    #The epoch limit
    x = np.array([1, 20])
    
    #Writing a random number to a file
    #Important: Reset the file everytime before running it.
    WriteRandomNumber = open("sample-file.txt", "a+") 
    randomNumber = str(random.uniform(0.0, 5.0))
    out = "\n"+str(counter)+","+randomNumber
    WriteRandomNumber.write(out)
    WriteRandomNumber.close()
    
    #For debugging purpose only
    print(counter)    

    #reading the random numbers from the same file
    pullData = open('sample-file.txt', 'r').read()        
    dataArray = pullData.split( '\n')        
    xar = []
    yar = []  
    #Giving it a range, x.min() is the minimum range, and counter how many images it has iterated.
    # 3rd argument tells us about the scaling between each
    plt.xticks(np.arange(x.min(), counter, 0.1))
    
    #Everytime each image is read. 
    for eachLine in dataArray:
        if len(eachLine)>1:  #Not necessary
            x1,y1 = eachLine.split(',') #first is going to be x and second one is y            
            xar.append(float(x1)) #Including them into an array
            yar.append(float(y1)) 

    ax1.clear()    #Remove this if you want to see distict colour graph at each interval.
    ax1.set_xlim([0, counter]) #max and min value to X. Updates at each new instance.
    ax1.set_ylim([0, 5]) # max and min value value to Y. Constant throughout.
    ax1.plot(xar,yar)
      

for j in range(0, 20):
    #This function is triggered 20 times in this case, where 20 represents the number of epochs
    animate = animation.FuncAnimation(fig, animate, interval = 1000)
    plt.show()

#Interval signifies how often we want to refresh the graph, unit is in milliseconds


