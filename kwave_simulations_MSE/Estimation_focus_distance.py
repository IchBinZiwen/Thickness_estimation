import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt


# parameters
distance = np.arange(0, 100e-3, 0.1e-3)
diameters = np.arange(5e-3, 20e-3, 0.1e-3)
f = 1e6
c = 1500

# plot focus over diameter
fig = plt.figure()
focus_distances = diameters**2 * f / (4*c)
plt.plot(diameters, focus_distances)
plt.grid()
plt.show()