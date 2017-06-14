import json
import numpy as np

dist = np.genfromtxt("schoolDistricts.csv", delimiter = ',')
dist = dist[1:,dist.shape[1] - 1]
print(dist)