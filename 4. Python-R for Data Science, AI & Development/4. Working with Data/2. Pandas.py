#!/usr/bin/env python
# coding: utf-8

# ## Pandas
# 
# __Pandas__ is a library provides efficient data structures to manage
# big quantities of information
# 
# pip3 install pandas
# 
# We need to import the library before using it

import pandas as pd
import json

# ### Reading files
# 
# - csv: pandas.__read_csv__(filepath)
# - excel: pandas.__read_excel__(filepath)

csv_file: pd.DataFrame = pd.read_csv("IMDB-Movie-Data.csv")
print(csv_file)

excel_file: pd.DataFrame = pd.read_excel("IMDB-Movie-Data.xlsx")
print(excel_file)

# - __JSON__: There is a way to turn __JSON files into dataframes__
# 
# pandas.__json_normalize__(json_object)

# Let's transform a JSON file into dataframe object

books_json = None
books_file = None

try:
    # Opening JSON file
    books_file = open('books.json')

    # returns JSON object as 
    # a dictionary
    books_json = json.load(books_file)
    print(books_json)
except FileNotFoundError as e:
    print(e)
finally:
    if books_file is not None:
        books_file.close()

# Now we can easily turn the JSON object to dataframe

books_df: pd.DataFrame = pd.json_normalize(books_json)
print(books_df)

# - __Copy__ dataframe:
# 
# new_var = df.copy()

copy_csv_file: pd.DataFrame = csv_file.copy()

# ### Basic structures:
# 
# - Series: One dimensional array-like structures
# - Dataframes: Two dimensional array-like structures
# 
# NOTE: Convert from dict to dataframe
# 
# pd.__DataFrame__(_dict_)

countries_dict: dict = {"name": ["Spain", "France", "UK"],
                        "population": [47502512, 68042591, 67820364],
                        "GDP": [1.397, 2.958, 3.131]}
dataframe_countries: pd.DataFrame = pd.DataFrame(countries_dict)
print(dataframe_countries)

# ### Basic descriptions of the data in the file
# 
# - .info() -> Returns information about the dataframe/series contents
# - .describe() -> Returns basic statistical measures of the information
# in dataframe/series
# - .head(n) -> Returns n first rows in the dataframe (if we don't set
# n, 5 lines will be returned)

# info
csv_file.info()

excel_file.info()

excel_file.info()

dataframe_countries.info()

# describe
csv_file.describe()

excel_file.describe()

dataframe_countries.describe()

# head
csv_file.head()

excel_file.head(7)

dataframe_countries.head(2)

# - To get the __columns__ in a dataframe
# 
# dataframe_var.__columns__

print(csv_file.columns)

print(csv_file.columns.values)

# - __dimensions__ of the dataframe: .__shape__
# 
# First dimension is number of rows
# 
# Second dimension is number of columns
# 
# Will return a tuple with as many dimension as the object has

# Our dataframe has 12 columns and 1000 rows
print(csv_file.shape)

# - Adding __labels to rows__, in order to do so, you need to set a
# column as __index__
# 
# Then, all the values of such column become the label of its
# corresponding row
# 
# - .__set_index__('column_name') -> Sets a column as the index and
# therefore its value become the label of its corresponding row ->
# Returns new variable with the change, it is not applied to the
# dataframe

file_indexed: pd.DataFrame = csv_file.set_index('Revenue (Millions)')
print(file_indexed)

# - You can also add a __list as index of the dataframe__
# (_labels of rows in dataframe_)
# 
# new_var = dataframe.index = list

indexes: [str] = [chr(i) for i in range(0, 1000)]

copy_csv_file.index = indexes
print(copy_csv_file)

# ### Subset of columns
# 
# To get only a subset of columns
# 
# - dataframe[[__'col1', 'col2'__]]

csv_title_runtime_revenue: pd.DataFrame = csv_file[
    ["Title", "Runtime (Minutes)", "Revenue (Millions)"]]
print(csv_title_runtime_revenue)

excel_title_runtime_revenue: pd.DataFrame = excel_file[
    ["Title", "Runtime (Minutes)", "Revenue (Millions)"]]
print(excel_title_runtime_revenue)

country_population: pd.DataFrame = dataframe_countries[["name", "population"]]
print(country_population)

# For series is like this:
spain_dict: dict = {"name": "Spain", "population": 47502512,
                    "GDP": 1.397}
spain_series = pd.Series(spain_dict)
print(spain_series)

spain_name_population_series: pd.Series = spain_series[['name', 'population']]
print(spain_name_population_series)

# - .ix[i, j]: Selects column and row by index -> It is deprecated
# and replaced by
# - .__iloc__[i, j]

try:
    csv_file.ix[3, 3]
except AttributeError as e:
    print("ix has been deprecated")
    print(e)

print(csv_file.iloc[3, 3])

# - .__loc__[row_index/'row_name', 'column_name']: Similar to iloc but
# instead of using the indexes, uses the names of the rows/columns
# (it can use the index of the row)

print(csv_file.loc[0, 'Year'])

# We can use also the index which are now labels of the rows
print(file_indexed)


# Year of the movie whose Revenue was 17.54 millions
print(file_indexed.loc[17.54, 'Year'])

# ### Working with data
# 
# - __unique__(): Returns only the unique values in a column

# In the dataframe csv_file that contains information about movies in
# IMDB, let´s get the years the movies included were released
csv_file['Year'].unique()

# - __Filtering__: dataframe_var['col'] >=/>/... var/value -> Will
# return a list of True/False if row in column meets criteria or not

# Following the same example, let's get the movies that were made in
# 2014 and later
print(csv_file['Year'] >= 2014)

# As we can see, this is just a list of true false. To really filter
# the dataframe we need to apply this to the dataframe
# 
# __dataframe_var__[dataframe_var['col'] >=/>/... var/value]


print(csv_file[csv_file['Year'] >= 2014])

# The result of this filter that can be assigned to a new variable

movies_2014_and_after = csv_file[csv_file['Year'] >= 2014]
print(movies_2014_and_after)

# - Also __loc__ can be used with filters:

# Movies released in 2015 or before
print(csv_file.loc[csv_file['Year'] <= 2015])

# #### Slicing

# - __Slicing rows__: df.iloc[i:j]

# Last 10 rows of this dataframe
print(csv_file[990:1000])

# - __Slicing rows and columns__: df.iloc[i:j, m:n]

# First 4 columns of the last 10 rows
print(csv_file.iloc[990: 1000, 0:4])

# - __Slicing using labels__
# 
# Rows: df.loc['start_row_label': 'end_row_label'] -> Includes last row

print(file_indexed.loc[333.13: 325.02])

# Rows and columns: df.loc['start_label_row': ´'end_label_row',
# 'start_label_column': 'end_label_column'] -> Includes both last row
# and column

print(file_indexed.loc[333.13: 325.02, "Title": "Director"])


# - .__transform__(func=): Applies a function to a dataframe

# Let's get the revenue in dollars
def from_millions_to_dollars(quantity: float):
    return quantity * 10 ** 6


print(csv_file['Revenue (Millions)'].transform(func=from_millions_to_dollars))
