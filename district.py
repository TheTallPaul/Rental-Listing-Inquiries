import json
from shapely.geometry import shape, Point

def district(lat, lon):
    P = Point(lon, lat)
    # Load data for geometry
    with open('NYC_School_Districts.geojson') as f:
        sc = json.load(f)
    
    # Check each polygon to see if it contains the point
    for feature in sc['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(P):
            print(feature['id'])
            return feature['id']