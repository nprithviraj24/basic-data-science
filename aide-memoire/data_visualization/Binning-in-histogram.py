import matplotlib.pyplot as plt

values = [121, 15.0, 164, 74.24, 455, 255, 45.02, 99.3647, 45.00, 678.12, 95.127]

plt.hist(values)  #without bins.

plt.show()
plt.clf()
# Build histogram with 5 bins
plt.hist(values, 5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(values, 20)

# Show and clean up again
plt.show()
plt.clf()