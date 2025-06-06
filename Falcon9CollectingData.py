#Falcon9CollectingData.ipynb - collect data and make sure it is received in the correct format from an API
#Beginning of code
#1 import libraries
import requests
import pandas as pd
import numpy as np
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)

# 2 predefined methods to help us extract data
# From the rocket column we would like to learn the booster name, so iterate through the 'rocket' col of data dataframe
# add the id of the rocket to the url and download the dictionary (.json()) named url+rocketid to variable response.
#without the .json() part response will not be dict with all the data, but simply response status (200 in this case).
# Once it is safely saved we extract the name from the json file and append it to BosterVersion list variable
# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
         
# From the launchpad we would like to know the name of the launch site being used, the logitude, and the latitude       
def getLaunchSite(data):
    for x in data['launchpad']:
        if x:
            response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
            LaunchSite.append(response['name'])
            Longitude.append(response['longitude'])
            Latitude.append(response['latitude'])    

#From the payload we would like to learn the mass of the payload and the orbit that it is going to
def getPayloadData(data):
    for x in data['payloads']:
       if x:
        response = requests.get('https://api.spacexdata.com/v4/payloads/'+str(x)).json()
        Orbit.append(response['orbit'])
        PayloadMass.append(response['mass_kg'])
            
#From cores we would like to learn the outcome of the landing, the type of the landing, number of flights with that core, 
#whether gridfins were used, whether the core is reused, whether legs were used, the landing pad used, the block of the core
#which is a number used to seperate version of cores, the number of times this specific core has been reused, 
#and the serial of the core.

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
            
#3 Load data
#spacex_url="https://api.spacexdata.com/v4/launches/past"
#response=requests.get(spacex_url)
#chech the response content
#data=pd.json_normalize(response.json())
#or we can use:
static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
response=requests.get(static_json_url)
response.status_code #200 means sucess
data=pd.json_normalize(response.json())
data
                            
# 4 wrangle data                                
# Lets take a subset of our dataframe keeping only the features we want:rocket, payloads, launchpad, and cores, flight number and date_utc.
data = data[['rocket','payloads','launchpad','cores','flight_number','date_utc']]
data

# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
#their way
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

'''
#another way
#create 2 new empty lists multipayloads and multicores to keep the info how many payloads/cores per each row
#then create 2 new columns ('multipayload' and 'multicores') and fill them with the two lists created above
#then keep only the rows where both cols have value 1
multipayloads=[]
for i in range(0,len(data['payloads'])):
    multipayloads.append(len(data['payloads'][i]))
data['multipayload']=multipayloads
filtered_data = data[data['multipayload'] ==1]#we need a df with a new name
filtered_data=filtered_data.reset_index() #index needs to be reset because we deleted some of the rows
filtered_data

multicores=[]
for i in range(0,len(filtered_data['cores'])):
    multicores.append(len(data['cores'][i]))
filtered_data['multicores']=multicores
data = filtered_data[filtered_data['multicores'] ==1] #we need a df with a new name, so we will use the name data again

#drop the cols we used as help, we won't need them anymore
data = data.drop(['multicores', 'multipayload', 'index'], axis=1)
'''

#Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

# Convert the date_utc to a datetime datatype and then keep only the date
data['date_utc'] = pd.to_datetime(data['date_utc']).dt.date
                                                     
# 5 gather additional data and prepare the final dataset
# Using the methods defined above, we will use columns: rocket, payloads, launchpad, cores to find out the following data
# Global variables 
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
#call methods to populate the lists
getBoosterVersion(data)    
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)

#now that we have all the data we can construct a new dataframe
launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date_utc']),
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

#create a Pandas data frame from the dictionary launch_dict
data=pd.DataFrame(launch_dict)
data.head()

#filter the dataframe to only include Falcon 9 launches
data_falcon9=data[data['BoosterVersion']=='Falcon 9']
data.head()

#that we have removed some values we should reset the FlgihtNumber column
data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
data_falcon9.head()

#data_falcon9 data wrangling
data_falcon9.isnull().sum() #PayloadMass has 6 null values, LandingPad 26

#replace missing values in PayloadMass with mean
data_falcon9['PayloadMass'].fillna((data_falcon9['PayloadMass'].mean()), inplace=True)
data_falcon9.isnull().sum() #PayloadMass now has 0 null values, LandingPad 26, we will leave it like that

#export to csv file
data_falcon9.to_csv('dataset_part_1.csv', index=False)
#end of code

