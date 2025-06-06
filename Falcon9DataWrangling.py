#Convert outcomes into Training Labels with 1 means the booster successfully landed 0 means it was unsuccessful.

!pip install pandas
!pip install numpy
import pandas as pd 
import numpy as np

#import dataset
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)

df.isnull().sum()/len(df)*100 #percentage of missing values for each column
df.dtypes #Identify which columns are numerical and categorical

df['LaunchSite'].value_counts() #calc the number of launches on each site or df.groupby('LaunchSite').count() 

df['Orbit'].value_counts() #calc the occurence of different orbits

#df.groupby(['Orbit', 'Outcome']).count()# calc the number and occurence of mission outcome of the orbits
df.head()

df['Outcome'].value_counts() #from column Outcome to determine the number of landing outcomes and assign it to landing_outcomes var
landing_outcomes=df['Outcome'].value_counts()

#for i,outcome in enumerate(landing_outcomes.keys()):
#    print(i,outcome)

bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]]) #We create a set of outcomes where the second stage did not land successfully

#Using the Outcome, create a list where the element is zero if the corresponding row in Outcome is in the set bad_outcome; otherwise, it's one. 
#Then assign it to the variable landing_class:
NewColumnList=[]
for index, row in df.iterrows():
    if (row['Outcome'] in bad_outcomes):
        NewColumnList.append(0)
    else:
        NewColumnList.append(1)

df['Class']=NewColumnList
df["Class"].mean()
df.to_csv("dataset_part_2.csv", index=False)
