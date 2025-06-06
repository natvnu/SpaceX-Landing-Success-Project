#Geospatial representation of Launch Sites
#TASK 1: Mark all launch sites on a map
#TASK 2: Mark the success/failed launches for each site on the map
#TASK 3: Calculate the distances between a launch site to its proximities

#1. import libraries
import piplite
await piplite.install(['folium'])
await piplite.install(['pandas'])
import folium
import pandas as pd

# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon
from js import fetch
import io

#2. load data
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
resp = await fetch(URL)
spacex_csv_file = io.BytesIO((await resp.arrayBuffer()).to_py())
spacex_df=pd.read_csv(spacex_csv_file)

#3.create US map (us coordinates) and add Launch Sites markers 
us_map=folium.Map(location=[37.09024,-95.712891],zoom_start=4)
us_map

#3.1. Write Launch Site names on the map
#3.1.1. prepare data for Launch Site markers
'''
#one way
#keep only the columns we need
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.drop_duplicates(subset=['Launch Site','Lat','Long'])#we will keep only 1 unique combination of the cols 
launch_sites_df=launch_sites_df[['Launch Site','Lat','Long']].reset_index()
'''
#the other way
#keep only the columns we need
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df

#3.1.2. create and add folium.Marker for each Launch Site on the site map
list_of_coordinates=launch_sites_df[['Lat','Long']].values.tolist()
list_of_names=launch_sites_df[['Launch Site']].values.tolist()

for coordinate,name in zip(list_of_coordinates,list_of_names):
    marker = folium.map.Marker(
    coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % name[0],
        )
    )
    us_map.add_child(marker) #orange name
    
#3.2. #create marker cluster and mark the success/failed launches for each site on the map
# 3.2.1.Create a new column in `spacex_df` dataframe called `marker_color` to store the marker colors based on the `class` value, if class = 1 then green, otherwise red
spacex_df.tail(10)
marker_color_list=[]
for index, row in spacex_df.iterrows():
    if row["class"]==0:
        marker_color_list.append('Red')
    else:
        marker_color_list.append('Green')
spacex_df['marker_color']=marker_color_list
# 3.2.2. add marker clusters with information about failure or success (red or green)
from folium import plugins
#create marker cluster
sites=plugins.MarkerCluster().add_to(us_map)
#add markers with information about failure or succes
for lat, long, name, color in zip(spacex_df['Lat'],spacex_df['Long'],spacex_df['Launch Site'], spacex_df['marker_color']):
    #print(lat, long, name)
    folium.Marker(location=[lat,long], icon=folium.Icon(color='white', icon_color=color), popup=name).add_to(sites)
us_map # display map with newly added elements

# 4: Calculate the distances between a launch site to its proximities/draw lines to the proximities (railway, road, coastline...)
# 4.1. Add Mouse Position to get the coordinate (Lat, Long) for a place where you move the mouse over on the map -  coordinates in the top rigth corner
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)
us_map.add_child(mouse_position)

# 4.2. define a method that calculates distance between the Launch Site and the closest coastline road
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
'''
# 4.3.1. Calculate the distance between CCAFS SLC-40 Launch Site and closest road 
# Find coordinates of the closest road - we can see in the right top corner
road_lat = 28.56264
road_lon = -80.57053
#calculate the distance between CCAFS SLC-40 Launch Site and closest road 'Samuel C Philips PKWY
distance_coastline = calculate_distance(28.563197, -80.576820, road_lat, road_lon)
#add a blue marker on CCAFS SLC-40 Launch Site that specifies the distance to this road
folium.Marker(location=[28.563197, -80.576820],popup=str(distance_coastline)+' km distance to closest road').add_to(us_map)
'''
#we can do the same for the rest of Launch Sites

# 4.3.2 Draw a line from CCAFS SLC-40 Launch Site to the closest coastline 
coastline_lat = 28.56316
coastline_lon = -80.5679
# Define coordinates we will use - CCAFS SLC-40 Launch Site and Coastline lat and long
coordinates=[[28.563197, -80.576820],[coastline_lat,coastline_lon]]
# Create a `folium.PolyLine` object using the coordinates we created
lines=folium.PolyLine(locations=coordinates, weight=1)
us_map.add_child(lines)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
us_map #display map
