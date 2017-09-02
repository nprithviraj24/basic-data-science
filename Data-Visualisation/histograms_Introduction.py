from random import randint
import matplotlib.pyplot as plt
# Histogram of life_exp, 15 bins

life_exp = []
life_exp1950 = []

print("\nPresent day\t1950\n")
for x in range(1,10):
	x = randint(0,100)
	y = randint(0,100)
	life_exp.append(x)
	life_exp1950.append(y)
	print (str(x) + "\t\t" + str(y))


plt.hist(life_exp, bins=15)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins

plt.hist(life_exp1950, bins=15)

# Show and clear plot again
plt.show()
plt.clf()