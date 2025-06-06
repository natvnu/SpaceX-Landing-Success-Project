#Perform exploratory Data Analysis and Feature Engineering using Pandas and Matplotlib
#Preparing Data Feature Engineering

#1. import libraries
import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2. load the dataset
from js import fetch
import io

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp = await fetch(URL)
dataset_part_2_csv = io.BytesIO((await resp.arrayBuffer()).to_py())
df=pd.read_csv(dataset_part_2_csv)
df.head(5)

#3. EDA
#First, let's try to see how the FlightNumber (indicating the continuous launch attempts.) and Payload variables would affect the launch outcome.
#We can plot out the FlightNumber vs. PayloadMass and overlay the outcome of the launch. We see that as the flight number increases, the first stage is more likely to land successfully. 
#The payload mass also appears to be a factor; even with more massive payloads, the first stage often returns successfully.
sns.catplot(x="FlightNumber",y="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show() #FlightNumber>78 and FlightNumber between 20 and 45 with PayloadMass<5000 have 100% sucessful landings (Class=1) 

#TASK 1: Visualize the relationship between Flight Number and Launch Site
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
plt.xticks(ticks=df['FlightNumber'])
plt.title('Relationship between Flight Number and Launch Site')
plt.show() #FlightNumber>78 have 100% sucessful landings (Class=1), independent of launch sites

#TASK 2: Visualize the relationship between Payload Mass and Launch Site
sns.catplot( x="PayloadMass",y="LaunchSite", hue="Class", data=df, aspect = 5)
plt.xlabel("Payload Mass",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
plt.title('Relationship between Payload Mass and Launch Site')
#plt.xticks(ticks=df['FlightNumber'])
plt.show() #KSC LC 39A site with payload mass <5500 have 100% sucessful landings (Class=1), 
#most of the launches from all sites with payload mass>7000 have successful landings
#for the VAFB-SLC launchsite there are no rockets launched for heavypayload mass(greater than 10000)

#TASK 3: Visualize the relationship between success rate of each orbit type
df_orbit= df.groupby('Orbit').mean('Class').reset_index()
plt.bar(df_orbit['Orbit'], df_orbit['Class'])
plt.xlabel('Orbit')
plt.ylabel('Class')
plt.title('Landing Success per Orbit')
plt.show() #SO had 0 landing success, ES-L1, GTO, HEO i SSO have 100% sucess, while the rest of orbits lies somewhere in between

#TASK 4: Visualize the relationship between FlightNumber and Orbit type
sns.catplot(x="FlightNumber", y="Orbit", hue="Class", data=df, aspect = 5)
plt.xlabel("FlightNumber",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.xticks(ticks=df['FlightNumber'])
plt.title('Relationship between Flight Number and Orbit Type')
plt.show() #LEO orbit for FLightNumbers>9 has 100% sucess in landing
#ISS orbit for FLightNumbers>60 has 100% sucess in landing
#all orbits have 100% sucess in landing for FlightNumbers>78
#SO had 0 landing success, ES-L1, GTO, HEO i SSO have 100% sucess, while the rest of orbits lies somewhere in between

#TASK 5: Visualize the relationship between Payload Mass and Orbit type
sns.catplot(x="PayloadMass", y="Orbit", hue="Class", data=df, aspect = 5)
plt.xlabel("PayloadMass",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.title('Relationship between Payload Mass and Orbit Type')
plt.show() #ES-L1, SSO, HEO orbits, light payload (<4000) have 100% sucess in landing, others do not have much correlation

#TASK 6: Visualize the launch success yearly trend
df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df_year=df.groupby('Year').mean('Class').reset_index() #ovo bi bilo bolje koristiti kao df za TASK 6, makes more sense
plt.plot(df_year['Year'],df_year['Class'])
plt.xlabel("Year",fontsize=20)
plt.ylabel("Class",fontsize=20)
plt.xticks(ticks=df_year['Year'])
plt.title('Landing Success - Yearly Trend')
plt.show()

#4. Feature engineering
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()
features.dtypes #Orbit, LaunchSite, LandingPad, Serial

#TASK 7: Create dummy variables to categorical columns
'''
#I used get_dummies() because it automatically assigns column names. Disclaimer, get_dummies returns boolean instead of int values.
#we could have used LabelEncoder, which is in my opinion better, because it keeps the same no of columns, but the task specified the use of OneHotEncoder (above)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
features['Orbit']=labelencoder.fit_transform(features['Orbit'])
features['LaunchSite'] ...
'''
dummy_vars=pd.get_dummies(features[['Orbit','LaunchSite','LandingPad','Serial']])
features_one_hot=pd.concat([features,dummy_vars],axis=1 )
features_one_hot=features_one_hot.drop(['Orbit','LaunchSite','LandingPad','Serial'],axis=1)
features_one_hot
