## General Assembly Data Science, San Francisco
## github.com/ga-students/DAT_SF_10
##
## Julee Burdekin
## juburdekin@gmail.com
##
## 20141013
## HW1
## Analyze a new app with provided json file, in the same dir,
## containing date and time stamps for logins (ga_hw_logins.json)

## Step 1: Load the data from ga_hw_logins.json into your python environment
## http://stackoverflow.com/questions/12451431/loading-parsing-json-file-in-python

import json
from datetime import datetime
import sqlite3
import numpy as np
import pandas.io.sql as psql


## create a data list
data = []
dateList = []

## open the json file as variable f
with open('ga_hw_logins_temp.json') as f:
    ## load each line in the file
    data  = json.load(f)

## Step 2: Convert the strings into datetime objects and append them to a list
## http://stackoverflow.com/questions/466345/converting-string-into-datetime/466376#466376
## https://docs.python.org/2/library/datetime.html?highlight=datetime.strptime#strftime-and-strptime-behavior

for it in data:
    todt = datetime.strptime(it, "%Y-%m-%d %H:%M:%S")
    dateList.append(todt)

## for item in dateList:
##     print item

## Extra Credit: Using python, create a sqlite3 database and upload the data into the database.
## Then query the data to find the date and hour with the most logins.
## Import sqlite package to get python functions for working with sqlite databases
## https://docs.python.org/2/library/sqlite3.html
## Use the group by feature in SQL queries

conn = sqlite3.connect('hw1_datelist.db')

c = conn.cursor()

## for debugging: just to get rid of last table attempt
c.execute('''DROP TABLE justdts''')

## create table
c.execute('''CREATE TABLE justdts
             (date text)''')

for dts in dateList:
    ## Insert dateList
    c.execute("INSERT INTO justdts VALUES (?)", (dts,))

## save (commit) the changes
conn.commit()

## close the connection
conn.close()

## To get column name:
## c.execute("select * from justdts")
## names = list(map(lambda x: x[0], c.description))
## print names

## select duplicates
## for row in c.execute("SELECT date FROM justdts GROUP BY date HAVING COUNT(*) > 1"): 
##    print row

cnxn = sqlite3.connect('hw1_datelist.db')
cursor = cnxn.cursor()
## sql = "SELECT date FROM justdts GROUP BY date HAVING COUNT(*) > 1"
sql = "SELECT date FROM justdts GROUP BY date HAVING COUNT(*) >1 ORDER BY date"

## cursor.execute("SELECT COUNT(*) from result where server_state= %s AND name LIKE %s",[2,digest+"_"+charset+"_%"])
## (number_of_rows,)=cursor.fetchone()


df = psql.read_sql(sql, cnxn)
cnxn.close()

for index in df:
    print df

## convert to datetime


