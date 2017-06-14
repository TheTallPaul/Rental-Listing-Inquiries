import json
import numpy as np
from shapely.geometry import shape, Point
# depending on your version, use: from shapely.geometry import shape, Point

def euclid(P1x, P1y, P2x, P2y):
	return np.sqrt((P2x[0] - P1x[0]) * (P2x[0] - P1x[0]) + (P2y[1] - P1y[1]) * (P2y[1] - P1y[1]))

def district(lat, lon):
	p = Point(lon, lat)
	#load data for geometry
	with open('NYC_School_Districts.geojson') as f:
		sc = json.load(f)
		
	# check each polygon to see if it contains the point
	for feature in sc['features']:
		polygon = shape(feature['geometry'])
		if polygon.contains(P):
			return feature['id']
			print('Point found in school district', feature['id'])

#Load data for listing points
with open('train.json') as f:
    js = json.load(f)
f = open("train.csv", 'w')	
coordKey = list(js['latitude'].keys())
print(js.keys())


'''
coords = []
for i in range(len(coordKey)):
	coords.append(Point(js['latitude'][coordKey[i]], js['longitude'][coordKey[i]]))
'''

