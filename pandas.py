#Based on questions given by Kieth Galli

import pandas as pd
import os


pd.set_option('display.max_columns', 10)

#Question 1: Whhich month had the highest sales?


#Task 1: Merging 12 moonths of sales into 1 csv



df1 = pd.read_csv("./Sales Project/Sales_January_2019.csv")

files = [file for file in os.listdir('./Sales Project')]

all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv('./Sales Project/'+file)
    all_months_data = pd.concat([all_months_data, df])
    
all_months_data.to_csv("all_data.csv", index=False)

#%%
#Read in updated dataframe

all_data = pd.read_csv('all_data.csv')

#%%

#Clean up the data
    #Getting rid of na values
nana_df = all_data[all_data.isna().any(axis=1)]

all_data = all_data.dropna(how='all')

    #Getting rid of 'or' values
all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']

#Task 2: Add Month Column

all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')


#Task 3: Add a sales column
all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype('int32')
all_data['Price Each'] = all_data['Price Each'].astype('float') 

# above could also be all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])


all_data['Sales'] = all_data['Quantity Ordered']*all_data['Price Each']
all_data['Sales'] = all_data['Sales'].astype('float')

#%%
#Task 4: Calculating total sales per month

results = all_data.groupby('Month').sum()

import matplotlib.pyplot as plt

months = range(1,13)

plt.bar(months, results['Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month Number')





#%%

#Question 2: What city had the highest number of sales

#Task 1: Create new column for city

def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]
all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' (' + get_state(x) + ')')
#using f:strings ----> .apply(lambda x: f"{get_city(x)} ({get_state(x)})")


results_city = all_data.groupby('City').sum()

print(results_city)

#%%

#Task 2: Plotting city performance

cities = [city for city, df in all_data.groupby('City')]
    #alignes x axes with its correct y values

plt.bar(cities, results_city['Sales'])
plt.xticks(cities, rotation='vertical', size=8)
plt.ylabel('Sales in USD ($)')
plt.xlabel('City Name')

#%%
#Question 3: What time shoudl we display advertisements to maximize likelihood of customers buying product?

#Task 1: convert to date format
all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])

#%%
all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute

all_data['Count'] = 1

hours = [hour for hour, df in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby(['Hour']).count())
    #using count() is a alternative to doing the sum method
plt.xticks(hours)
plt.grid()
plt.ylabel('Number of Orders')
plt.xlabel('Hour (Military)')

#%%
#Question 4: What products are most often sold together?

#Notice that Order ID shows what items were sold together

df = all_data[all_data['Order ID'].duplicated(keep=False)]
    # keep is answering what we want, False keeps everything, first keeps first, last keeps last occurance

df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

df = df[['Order ID','Grouped']].drop_duplicates()

#%%

#Counting pairs

from itertools import combinations
from collections import Counter

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
    
for key, value in count.most_common(10):
    print(key, value)

#%%

product_group = all_data.groupby('Product')

quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product, df in product_group]

plt.bar(products, quantity_ordered)
plt.xticks(products, rotation='vertical', size=8)
plt.ylabel('Quantity Ordered')
plt.xlabel('Product')

#%%

#Overlaying price with product on a plot

prices = all_data.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=8)

#%%

print(all_data.head())
