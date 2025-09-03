#We will predict if the Falcon 9 first stage will land successfully. 
#SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; 
#other providers cost upward of 165 million dollars each, much of the savings is because SpaceX can reuse the first stage. 
#Therefore if we can determine if the first stage will land, we can determine the cost of a launch. 

#to collect data we will send a request to the SpaceX API and then clean the requested data


#1. import libraries
# Requests allows us to make HTTP requests which we will use to get data from an API
import requests
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Datetime is a library that allows us to represent dates
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)

#2. define methods for gathering additional data from APIs
# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
        
        
# Takes the dataset and uses the launchpad column to call the API and append the data to the list
def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])
        
# Takes the dataset and uses the payloads column to call the API and append the data to the lists
def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])
        
# Takes the dataset and uses the cores column to call the API and append the data to the lists
def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])
            
#3. request data from SpaceX API. Keep data as json file. Read json file into pandas dataframe
#request rocket launch data from SpaceX API with the following URL
spacex_url="https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
#decode the response content as a Json using .json() and turn it into a Pandas dataframe using .json_normalize()
jsondata = response.json()
data=pd.json_normalize(jsondata)
#print the df
data.head()

#4. clean the dataframe and gather additional data
#a lot of the data are IDs. For example the rocket column has no information about the rocket just an identification number.
#we will now use the API again to get information about the launches using the IDs given for each launch. 

#first we will clean the dataframe we have
#specifically we will be using columns rocket, payloads, launchpad, and cores.
# Lets take a subset of our dataframe keeping only the features we want and the flight number, and date_utc.
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]
# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]
# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])
# We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
data['date'] = pd.to_datetime(data['date_utc']).dt.date
# Using the date we will restrict the dates of the launches
data = data[data['date'] <= datetime.date(2020, 11, 13)]

#now we will gather the rest od the data from diferent APIs using df columns

#From the rocket we would like to learn the booster name.
#From the payload we would like to learn the mass of the payload and the orbit that it is going to.
#From the launchpad we would like to know the name of the launch site being used, the longitude, and the latitude.
#From cores we would like to learn the outcome of the landing, the type of the landing, number of flights with that core, 
#whether gridfins were used, whether the core is reused, whether legs were used, the landing pad used, 
#the block of the core which is a number used to seperate version of cores, 
#the number of times this specific core has been reused, and the serial of the core.

#The data from these requests will be stored in lists and will be used to create a new dataframe.
#Lists:
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

#use the previously defined method getBoosterVersion to discover booster name by calling API using the rocket column 
getBoosterVersion(data)

#use the previously defined method getLaunchSite to discover LaunchSite name, Longitude and Latitude by calling API using the launchpad column 
getLaunchSite(data)

#use the previously defined method getPayloadData to discover PayloadMass and Orbit by calling API using the payloads column 
getPayloadData(data)

#use the previously defined method getCoreData to discover Block, ReusedCount, Serial, Flights, GridFins, Reused, Legs and LandingPad by calling API using the cores column 
getCoreData(data)

#5. combine all extracted data in dict and then use it to create df and export this df to csv
launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}
df = pd.DataFrame(launch_dict)
#keep only Falcon 9 records
df=df[df['BoosterVersion']=='Falcon 9']

#address missing values
df.isnull().sum()

# Calculate the mean for PayloadMass column anduse it to replace null values
payload_mean = df['PayloadMass'].mean()
# Replace NaN values with the mean
df['PayloadMass'].fillna(payload_mean, inplace=True)

#We will ignore landingPad null values for now, and export df to csv
df.to_csv('dataset_part_1.csv', index=False)
