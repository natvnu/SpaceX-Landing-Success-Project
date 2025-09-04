#Exploratory Data Analysis and Feature Engineering
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
'''
#3.EDA
#3.1. FlightNumber (indicating the continuous launch attempts) and Payload variables vs the launch outcome
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show() #We see that as the flight number increases, the first stage is more likely to land successfully. 
#FlightNumber>78 and FlightNumber between 20 and 45 with PayloadMass<5000 have 100% sucessful landings (Class=1) 

# 3.2. FlightNumber (indicating the continuous launch attempts) and Launch site vs the launch outcome
# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show() ##FlightNumber>78 have 100% sucessful landings (Class=1), independent of launch sites. Launch site WAFB SCL 4E has only 2 failed attempts on landing

#3.3. Visualize the relationship between Payload Mass and Launch Site
sns.catplot( x="PayloadMass",y="LaunchSite", hue="Class", data=df, aspect = 5)
plt.xlabel("Payload Mass",fontsize=20)
plt.ylabel("LaunchSite",fontsize=20)
plt.title('Relationship between Payload Mass and Launch Site')
#plt.xticks(ticks=df['FlightNumber'])
plt.show() #KSC LC 39A site with payload mass <5500 have 100% sucessful landings (Class=1), 
#most of the launches from all sites with payload mass>7000 have successful landings
#for the VAFB-SLC launchsite there are no rockets launched for heavypayload mass(greater than 10000)

#3.4. Visualize the relationship between success rate of each orbit type
df_orbit= df.groupby('Orbit').mean('Class').reset_index()
plt.bar(df_orbit['Orbit'], df_orbit['Class'])
plt.xlabel('Orbit')
plt.ylabel('Class')
plt.title('Landing Success per Orbit')
plt.show() #SO had 0 landing success, ES-L1, GTO, HEO i SSO have 100% sucess, while the rest of orbits lies somewhere in between

#3.5. Visualize the relationship between FlightNumber and Orbit type
sns.catplot(x="FlightNumber", y="Orbit", hue="Class", data=df, aspect = 5)
plt.xlabel("FlightNumber",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.xticks(ticks=df['FlightNumber'])
plt.title('Relationship between Flight Number and Orbit Type')
plt.show() #LEO orbit for FLightNumbers>9 has 100% sucess in landing 
#ISS orbit for FLightNumbers>60 has 100% sucess in landing

#3.6. Visualize the relationship between Payload Mass and Orbit type
sns.catplot(x="PayloadMass", y="Orbit", hue="Class", data=df, aspect = 5)
plt.xlabel("PayloadMass",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.title('Relationship between Payload Mass and Orbit Type')
plt.show() #ES-L1, SSO, HEO orbits, light payload (<4000) have 100% sucess in landing, others do not have much correlation

#3.7. Visualize the launch success yearly trend
df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df_year=df.groupby('Year').mean('Class').reset_index() #ovo bi bilo bolje koristiti kao df za TASK 6, makes more sense
plt.plot(df_year['Year'],df_year['Class'])
plt.xlabel("Year",fontsize=20)
plt.ylabel("Success rate",fontsize=20)
plt.xticks(ticks=df_year['Year'])
# Manually set x-axis range and ticks to include 2011
plt.xticks(ticks=range(2010, 2021))  # Includes 2010 through 2020
plt.xlim(2009.5, 2020.5)  # Extend slightly beyond the range
plt.title('Landing Success - Yearly Trend')
plt.show()
'''

#4. feature engineering
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()
#Use the function get_dummies and features dataframe to apply OneHotEncoder to the column Orbits, LaunchSite, LandingPad, and Serial
# Assign the value to the variable features_one_hot, display the results using the method head.
#Your result dataframe must include all features including the encoded ones.
dummy_vars=pd.get_dummies(features[['Orbit','LaunchSite','LandingPad','Serial']])
features_one_hot=pd.concat([features,dummy_vars],axis=1 )
features_one_hot=features_one_hot.drop(['Orbit','LaunchSite','LandingPad','Serial'],axis=1)
features_one_hot

#Cast all numeric columns to float64 and export df to csv file
#Identify numeric columns
numeric_columns = features_one_hot.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns
# Convert all numeric columns to float64
features_one_hot[numeric_columns] = features_one_hot[numeric_columns].astype('float64')
features_one_hot.to_csv('0Falcon9EDA.csv', index=False) #90 rows, 80 cols

