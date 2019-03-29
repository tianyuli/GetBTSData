'''
Created on Mar 26, 2019

@author: Tianyu Li

Script for download data from BTS website and combine them into one file
Using pandas to merge them may not be the efficient way but it is safe

Tables are from https://www.transtats.bts.gov/tables.asp?DB_ID=120
Table ID of First Table is 237, second table is 236
'''
from post_data import POST_DATA
from calendar import month_name

import requests
import smart_open
import os
import glob
import zipfile
import pandas

START_YEAR = 1987
START_MONTH = 10
END_YEAR = 2018
END_MONTH = 12

MAX_YEAR = 2018
MIN_YEAR = 1978
PATH = "/Users/tl/Documents/BTSDatas"
DB_URL = "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=236&Has_Group=3&Is_Zipped=0"
REFERER = "https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236"
ORIGIN = "https://www.transtats.bts.gov"
CONTENT_TYPE="application/x-www-form-urlencoded"

# Download an zip file from remote into PATH with specified year and month
def download(year, month):
    # Validate input variables
    if (year > MAX_YEAR or year < MIN_YEAR):
        raise Exception(f"Year {year} is out of bound")
    if (month not in range(1, 13)):
        raise Exception(f"Month {month} is invalid")
    
    print(f"Start downloading data from {year}/{month}")
    post_data = POST_DATA.format(year = year, month_name = month_name[month], month = month, frequency = month)
    
    with requests.Session() as s:
        s.headers['Origin'] = ORIGIN
        s.headers['Referer'] = REFERER
        s.headers['Content-Type'] = CONTENT_TYPE
        
        # Get remote file
        response = s.post(DB_URL, data = post_data, allow_redirects = False)
        remote_file = response.headers['Location']
        response = s.get(remote_file)

        # Write the remote file to local disk
        file = os.path.join(PATH, f"{year}-{month}.zip")
        with smart_open.smart_open(file, "wb") as local_file:
            local_file.write(response.content)
            
        print(s.headers)
        print(response.headers)
    
if __name__ == '__main__':
    # Download data for specified range
    for year in range(START_YEAR, END_YEAR + 1):
        if (year == START_YEAR):
            for month in range(START_MONTH, 13):
                download(year, month)
        elif (year == END_YEAR):
            for month in range(1, END_MONTH + 1):
                download(year, month)
        else:
            for month in range(1, 13):
                download(year, month)
     
    # Extract all zip files
    for file in glob.glob(os.path.join(PATH, '*.zip')):
        with zipfile.ZipFile(file) as zip_file:
            print(f"extracting zipfile {file}")
            zip_file.extractall(PATH)
    
    # Combine all csv files into one file
    df_list = []
    for file in glob.glob(os.path.join(PATH, "*.csv")):
        print(f"reading {file}")
        df = pandas.read_csv(file, index_col = None, header = 0, encoding = 'latin-1')
        df_list.append(df)
 
    print("Concating")
    data = pandas.concat(df_list, ignore_index = True)
    print("Writing")
    data.to_csv((os.path.join(PATH, 'all.csv')), index = False)
    print("Finished")