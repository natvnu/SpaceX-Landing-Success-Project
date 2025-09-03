# Exploratory Data Analysis (EDA) 
# Determining target var label
'''
!pip install pandas
!pip install numpy
'''
import pandas as pd
import numpy as np

df=pd.read_csv("0falcon9API.csv")
df
#Identify and calculate the percentage of the missing values in each attribute
df.isnull().sum()/len(df)*100

#Calculate the number and occurrence of each orbit
df['Orbit'].value_counts() #GTO is a transfer orbit and not itself geostationary

#Calculate the number and occurence of mission outcomes
landing_outcomes = df['Outcome'].value_counts() 
landing_outcomes
#create a set of the bad outcomes
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes
#Using the Outcome, create a list where the element is zero if the corresponding row in Outcome is in the set bad_outcome
#otherwise, it's one. Then assign it to the variable landing_class
landing_class=[]
for value in df['Outcome']:
    if value in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)
df['Class']=landing_class
#drop 'Outcome' column
df=df.drop('Outcome',axis=1)

#export pdf to csv
df.to_csv("0Falcon9TargetVar.csv", index=False)
