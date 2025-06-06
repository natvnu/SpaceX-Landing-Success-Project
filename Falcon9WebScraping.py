#web scraping to collect Falcon 9 historical launch records from a Wikipedia page
#https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches

#1. install all libraries
!pip install html5lib==1.1 -y
import pandas as pd
import bs4 
from bs4 import BeautifulSoup 
import requests
import re
import unicodedata

'''
#the super easy way to extract df from the url
url = 'https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
#SCRAPE HTML TABLES DIRECTLY FROM URL TO DATAFRAME USING read_html
list_of_all_df_on_the_page=pd.read_html(url,flavor='bs4')
len(list_of_all_df_on_the_page)#there are 27 tables on this list
#3 tables that we need are on with indexes 0, 1 and 2
df2023=list_of_all_df_on_the_page[0]
df2024=list_of_all_df_on_the_page[1]
df2025=list_of_all_df_on_the_page[2]
'''
#2. define methods we will be using
def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass
  
def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    

static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response = requests.get(static_url)
soup = BeautifulSoup(response.text, 'html.parser')

#print(soup.title) #print title of the page to see if the page is correct

#find all tables on the wiki page
html_tables=soup.find_all('table')

# Let's print the third table and check its content
first_launch_table = html_tables[2]
#print(first_launch_table)

#Lets iterate through the <th> elements and apply the provided extract_column_from_header() to extract column name one by one
column_names = []
# Apply find_all() function with `th` element on first_launch_table
# Iterate each th element and apply the provided extract_column_from_header() to get a column name
# Append the Non-empty column name (if name is not None and len(name) > 0) into a list called column_names

for th in first_launch_table.find_all('th'):
    name=extract_column_from_header(th)
    if ((name is not None) and (len(name) > 0)):
        column_names.append(name)
column_names# print the list to see if it is ok         
column_names.remove('Date and time ( )') #remove this item from the list, we won't be using it

#We will create an empty dictionary with keys from the extracted column names 
#Later, this dictionary will be converted into a Pandas dataframe
launch_dict=dict.fromkeys(column_names) #method dict.fromkeys(column_names) initializes the keys in the dict from the list we are passing as an argument

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

rowno=0

for row in first_launch_table.find_all('tr'):
    rowno=rowno+1
    
    if row.th: # if th element
        if row.th.string:
            flight_number=row.th.string.strip() #keeps flight number
            flag=flight_number.isdigit() #flag is true if flight number is digit
                
        if flag:
                launch_dict['Flight No.'].append(flight_number)
               
    if (row.td and rowno in (2,4,6,8,11,13,15)): #We could have also used flag
        cols=row.find_all('td')
        datetime=date_time(cols[0])
        date=datetime[0].strip(',')
        #print(date)
        launch_dict['Date'].append(date)
        
        time=datetime[1]
        launch_dict['Time'].append(time)
        
        bv=booster_version(cols[1])
        if not(bv):
            bv=cols[1].a.string #instead we could use cols[1].text.strip(),but it returns lober string with more info
        launch_dict['Version Booster'].append(booster_version)
        
        launch_site = cols[2].a.string #instead we could use cols[1].text.strip()  ...
        launch_dict['Launch site'].append(launch_site)
        
        payload = cols[3].a.string
        launch_dict['Payload'].append(payload)
        
        payload_mass = get_mass(cols[4])
        launch_dict['Payload mass'].append(payload_mass)
        
        orbit = cols[5].a.string
        launch_dict['Orbit'].append(orbit)        
        
        customer = cols[6].a.string
        launch_dict['Customer'].append(customer)
        
        launch_outcome = list(cols[7].strings)[0]
        launch_dict['Launch outcome'].append(launch_outcome)
        
        booster_landing = landing_status(cols[8])
        launch_dict['Booster landing'].append(booster_landing)          
        
df= pd.DataFrame({ key:pd.Series(value) for key, value in launch_dict.items() })

df.to_csv('spacex_web_scraped.csv', index=False)
