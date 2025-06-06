#SpaceX DataSet
#load the dataset into the corresponding table in a Db2 database
#Execute SQL queries to answer assignment questions

!pip install sqlalchemy==1.3.9
!pip install ipython-sql
!pip install ipython-sql prettytable
!pip install -q pandas
%load_ext sql
import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

#connect to db
con = sqlite3.connect("my_data1.db")
cur = con.cursor()
%sql sqlite:///my_data1.db

#import csv to 
import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")

#DROP THE TABLE IF EXISTS
%sql DROP TABLE IF EXISTS SPACEXTABLE;

%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null

#TASK 1: Display the names of the unique launch sites in the space mission, if the column names are in mixed case enclose it in double quotes
%sql select distinct Launch_Site from SPACEXTABLE

#TASK 2: Display 5 records where launch sites begin with the string 'CCA'
%sql select * from SPACEXTABLE where Launch_Site like 'CCA%' limit 5;

#TASK 3: Display the total payload mass carried by boosters launched by NASA (CRS)
%sql select sum(PAYLOAD_MASS__KG_) from SPACEXTABLE where Customer = 'NASA (CRS)'

#TASK 4: Display average payload mass carried by booster version F9 v1.1
%sql select avg(PAYLOAD_MASS__KG_) from SPACEXTABLE where Booster_Version = 'F9 v1.1'

#TASK 5: List the date when the first succesful landing outcome in ground pad was acheived
%sql select Date from SPACEXTABLE where Landing_Outcome='Success (ground pad)' order by Date asc limit 1

#TASK 6: List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
%sql select Booster_Version from SPACEXTABLE where (Landing_Outcome='Success (drone ship)' and PAYLOAD_MASS__KG_ between 4000 and 6000)

#TASK 7: List the total number of successful and failure mission outcomes
%sql select Landing_Outcome, count(*) as Failure from SPACEXTABLE where Landing_Outcome like 'Failure%' 
%sql select Landing_Outcome, count(*) as Success from SPACEXTABLE where Landing_Outcome like 'Success%' 

#TASK 8: List all the booster_versions that have carried the maximum payload mass. Use a subquery.
%sql select Booster_Version from SPACEXTABLE where PAYLOAD_MASS__KG_=(select max(PAYLOAD_MASS__KG_) from SPACEXTABLE)

#TASK 9: List the records which will display the month names, failure landing_outcomes in drone ship, booster versions, launch_site for the months in year 2015.
#Note: SQLLite does not support monthnames. So you need to use substr(Date, 6,2) as month to get the months and substr(Date,0,5)='2015' for year.
%sql select substr(Date, 6,2) as Month, Booster_Version, Launch_Site, Landing_Outcome from SPACEXTABLE where substr(Date,0,5)='2015' and Landing_Outcome='Failure (drone ship)';

#TASK 10: Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
%sql select Landing_Outcome, count(*) as No_of_Landing_Outcomes from SPACEXTABLE group by(Landing_Outcome) order by No_of_Landing_Outcomes desc 
