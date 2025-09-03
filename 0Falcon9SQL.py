# Understand the Spacex DataSet
# Load the dataset into the corresponding table in a Db2 database
# Execute SQL queries to answer assignment questions
'''
# 1. install and import libraries
!pip install ipython-sql
!pip install prettytable
!pip install pandas
'''
%load_ext sql  
# Load SQL extension first
import csv, sqlite3  # to establish db connection
import prettytable
import pandas as pd

# Create database connection
con = sqlite3.connect("my_data1.db")
cur = con.cursor()

# Set default database connection for SQL magic commands
%sql sqlite:///my_data1.db

# Load data from URL into pandas DataFrame
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")

# Convert DataFrame to SQL table
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")

# Query the correct table name (SPACEXTBL, not SPACEXTABLE)
#%sql SELECT * FROM SPACEXTBL
'''
#TASK 1: Display the names of the unique launch sites in the space mission, if the column names are in mixed case enclose it in double quotes
%sql select distinct Launch_Site from SPACEXTBL


#TASK 2: Display 5 records where launch sites begin with the string 'CCA'
%sql select * from SPACEXTBL where Launch_Site like 'CCA%'  limit 5


#TASK 3: Display the total payload mass carried by boosters launched by NASA (CRS)
#%sql select sum(PAYLOAD_MASS__KG_) from SPACEXTBL where Customer = 'NASA (CRS)'

#TASK 4: Display average payload mass carried by booster version F9 v1.1
%sql select avg(PAYLOAD_MASS__KG_) from SPACEXTBL where Booster_Version='F9 v1.1'

#TASK 5: List the date when the first succesful landing outcome in ground pad was acheived
%sql select Date from SPACEXTBL where Landing_Outcome='Success (ground pad)' order by Date asc limit 1

#TASK 6: List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
%sql select Booster_Version from SPACEXTBL where (Landing_Outcome='Success (drone ship)' and PAYLOAD_MASS__KG_ between 4000 and 6000)

#TASK 7: List the total number of successful and failure mission outcomes
%sql select count(*) as Failure from SPACEXTABLE where Landing_Outcome like 'Failure%' 
%sql select count(*) as Success from SPACEXTABLE where Landing_Outcome like 'Success%' 


#TASK 8: List all the booster_versions that have carried the maximum payload mass. Use a subquery.
%sql select Booster_Version from SPACEXTBL where PAYLOAD_MASS__KG_=(select max(PAYLOAD_MASS__KG_) as maxp from SPACEXTBL)

#TASK 9: List the records which will display the month names, failure landing_outcomes in drone ship, booster versions, launch_site for the months in year 2015.
%sql select substr(Date, 6,2), Landing_Outcome, Booster_Version, launch_site from SPACEXTBL where Date like '2015%' and  Landing_Outcome='Failure (drone ship)'
'''
#TASK 10: Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
%sql select count(*) as Count, Landing_Outcome from SPACEXTBL where Date between '2010-06-04' and '2017-03-20' group by Landing_Outcome order by Count
