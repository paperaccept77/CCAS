"""Geographic distance features."""
import math


def kilometers(location, poi):
    lat1, lon1, lat2, lon2 = map(math.radians, (*location, poi.latitude, poi.longitude))
    value = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371 * 2 * math.asin(min(1.0, math.sqrt(value)))
