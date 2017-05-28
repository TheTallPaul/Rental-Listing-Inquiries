import json
import numpy as np
from shapely.geometry import shape, Point

def district(lat, lon):
	P = Point(lon, lat)
	#load data for geometry
	with open('NYC_School_Districts.geojson') as f:
		sc = json.load(f)
		
	# check each polygon to see if it contains the point
	for feature in sc['features']:
		polygon = shape(feature['geometry'])
		if polygon.contains(P):
			#print('Point found in school district', feature['id'])
			return feature['id']