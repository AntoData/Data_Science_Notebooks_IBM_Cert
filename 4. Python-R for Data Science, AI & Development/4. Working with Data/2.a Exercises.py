#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from typing import Set
from datetime import datetime

# ### Pandas
# 
# Some exercises to practice
# 
# We will use the following files:
# 
# - books.json: Contains information about books in format JSON
# - IMDB-Movie-Data.csv: Contains information about movies from IMDB.com
# in CSV format
# - IMDB-Movie-Data.xlsx: Contains information about movies from
# IMDB.com in excel format
# - weather_data.csv: Contains information the weather relating it to
# latitude and longitude in different cities around the world
# - NASA_confirmed_exoplanets.csv: Contains information about exoplanets
# whose existence has been confirmed by NASA

# - __Exercise 1__: Open the file IMDB-Movie-Data.xlsx, get the Title,
# description, year and revenue of all movies directed by Ridley Scott


# We open the file
movies_imdb: pd.DataFrame = pd.read_excel('IMDB-Movie-Data.xlsx')
# We filter by Director == 'Ridley Scott'
movies_imdb_ridley_scott: pd.DataFrame = movies_imdb[movies_imdb['Director']
                                                     == 'Ridley Scott']
# We get the subset of columns whose information we need
movies_imdb_ridley_scott = movies_imdb_ridley_scott[
    ['Title', 'Director', 'Year', 'Revenue (Millions)']]
print(movies_imdb_ridley_scott)

#     - Exercise 1.a: Now sort the dataframe by year the movie was
#     released


movies_imdb_ridley_scott_sorted_by_year: pd.DataFrame = \
    movies_imdb_ridley_scott.sort_values(by=['Year'], ascending=True)
print(movies_imdb_ridley_scott_sorted_by_year)

#     - Exercise 1.b: Now get the movie directed by Ridley Scott with
#     the biggest revenue


print(movies_imdb_ridley_scott.sort_values(by=['Revenue (Millions)'],
                                           ascending=False).iloc[0, 0:4])

# - __Exersice 2__: Open the file IMDB-Movie-Data.csv, get Title,
# Description, Year, Revenue of all the movies released after 2014


# We open the file
movies_imdb_csv: pd.DataFrame = pd.read_csv('IMDB-Movie-Data.csv')
# We filter by row whose column Year has a value of 2014 or greater
movies_imdb_after_2014: pd.DataFrame = \
    movies_imdb_csv[movies_imdb_csv['Year'] >= 2014]
# We get a subset of the columns
movies_imdb_after_2014 = movies_imdb_after_2014[['Title', 'Description',
                                                 'Year', 'Revenue (Millions)']]
print(movies_imdb_after_2014)

# - __Exersice 3__: Open the file IMDB-Movie-Data.csv, get Title, Genre,
# Description, Director of all the movies starring Emma Stone


# We open the file
movies_imdb: pd.DataFrame = pd.read_csv('IMDB-Movie-Data.csv')
# We create a filter by rows whose column Actors contains the string
# "Emma Stone"
filter_emma_stone = movies_imdb['Actors'].str.contains("Emma Stone")
# We apply the filter to the dataframe and get the subset of columns we
# are interested in
movies_imdb_emma_stone = movies_imdb.loc[filter_emma_stone, 'Title':'Director']
print(movies_imdb_emma_stone)

# - __Exercise 4__: Open the file IMDB-Movie-Data.xlsx, get how many
# films each director has directed (use what we have learned), then
# let's get the directors with more and less movies


# We open the file
imdb_movies: pd.DataFrame = pd.read_excel('IMDB-Movie-Data.xlsx')
# We get the unique values of the column Director which will give us the
# directors registered in the dataframe
imdb_directors: [str] = imdb_movies['Director'].unique()
# We need to create an empty dictionary to contain the director as key
# and the number of films as value
directors_vs_number_movies: dict = {}
# Now we use the array of directors and iterate item by item
for director in imdb_directors:
    # We filter by director and count the number of rows (the number of
    # films this director has directed)
    number_movies: int = \
        imdb_movies[imdb_movies['Director'] == director]['Director'].count()
    # We add this director as key to the dictionary and the number of
    # movies they have directed as value
    directors_vs_number_movies[director] = number_movies
# We use dictionary.items() to create a dataframe where the first column
# is the name of the director and the second the number of films
pd_directors_vs_number_movies: pd.DataFrame = \
    pd.DataFrame(directors_vs_number_movies.items())
# We set appropriate names to the columns
pd_directors_vs_number_movies.columns = ['Director', 'Number films']
# We sort the resulting data frame by number of films
pd_directors_vs_number_movies = \
    pd_directors_vs_number_movies.sort_values(by=['Number films'])
print("")
# As the dataframe is sorted, we get the first row (0) and the second
# column (1) to get the minimum number of films directed
# by one of the directors in the dataframe
min_number_films: int = pd_directors_vs_number_movies.iloc[0, 1]
# We create a filter to get the rows where the number of films is equal
# to the min number of films
filter_min_number_films = pd_directors_vs_number_movies['Number films'] \
                          == min_number_films
# We apply this filter so we get the directors who directed the least
# number of movies in this dataset
directors_least_movies = \
    pd_directors_vs_number_movies[filter_min_number_films]['Director']
print("Directors with the least amount of movies: {0} movie(s)".
      format(min_number_films))
print(directors_least_movies.values)
print("")
# We use tail to get the last row in the dataset that was sorted by
# number of films and get the column
# Number films so we are getting the greatest number of films directed
# by a single director in the dataset
max_number_films: int = pd_directors_vs_number_movies.tail(1).iloc[0, 1]
# We create a filter to obtain the rows whose column Number films is
# equal to the max number of films directed by the same director, so we
# will get the directors that have worked in the most films
filter_max_number_films = pd_directors_vs_number_movies['Number films'] \
                          == max_number_films
# We apply the filter to get the directors that have worked in the most
# films
directors_max_movies = \
    pd_directors_vs_number_movies[filter_max_number_films]['Director']
print("Directors with the biggest amount of movies: {0} movie(s)".
      format(max_number_films))
print(directors_max_movies.values)

# - __Exercise 5__: Find how many movies each actor in the column Actors
# of the file IMDB-Movie-Data.csv has starred in and the get actor that
# has starred in the most films (and the number of films)


# We open the file
imdb_movies_df: pd.DataFrame = pd.read_csv('IMDB-Movie-Data.csv')
# We create an empty set to contain the names of each actor (remember
# that in sets, elements can't be repeated)
actors: Set = set()
# We get the actors starring in each movie in the dataset
for row_actors in imdb_movies_df['Actors']:
    # We split that row, to get the actors one by one (actors are
    # separated by ', ')
    actors_list_row = row_actors.split(", ")
    # We add each actor to the set, if one of them was already there is
    # not added again
    actors.update(actors_list_row)
# We create now an empty dictionary to contain each actor as key and the
# number of films they have starred in as value
actor_vs_movies: dict = {}
# We iterate actor by actor in our set of actors
for actor in actors:
    # We create a filter where we will get the movies that actor has
    # starred in
    # We will get the rows in which the column Actors contains the name
    # of that actor
    filter_actor = imdb_movies_df['Actors'].str.contains(actor)
    # We apply the filter to get the movies that actor has starred in
    # and we count the rows
    # That way we get the number of movies that actor has starred in
    n_movies = imdb_movies_df[filter_actor]['Title'].count()
    # We add the actor as key and the number of movies as value
    actor_vs_movies[actor] = n_movies
# We use the dictionary's items to create a dataframe where the first
# column is the actor
# and the second column is the number of movies that actor has starred in
df_actors_vs_movies: pd.DataFrame = pd.DataFrame(actor_vs_movies.items())
# We apply appropriate names to the columns
df_actors_vs_movies.columns = ['Actor', 'Number movies']
# We sort by Number movies from most to least
df_actors_vs_movies = \
    df_actors_vs_movies.sort_values(by='Number movies', ascending=False)
# The first row will contain the actor that has starred in the most movies
print("The actor that has starred in the most movies is {0}".
      format(df_actors_vs_movies.iloc[0, 0]), end="")
print(" with {0} movies".format(df_actors_vs_movies.iloc[0, 1]))

# - __Exercise 6__: Open the file weather csv that contains information
# about the weather in different cities across the US in different
# dates and turn the columns related to temperature from Fahrenheit to
# Celsius.

weather_csv: pd.DataFrame = pd.read_csv('weather.csv')


# We will use an approach where we could use this code later for other
# datasets, using functions


def convert_fahrenheit_to_celsius(temp_fahrenheit: float) -> float:
    return (temp_fahrenheit - 32) * 5 / 9


def convert_temperature_columns_fahrenheit_to_celsius(dataset: pd.DataFrame) \
        -> None:
    dataset_columns: [str] = dataset.columns
    for column in dataset_columns:
        if "temperature" in column.lower():
            dataset[column] = dataset[column].transform(
                func=convert_fahrenheit_to_celsius)


convert_temperature_columns_fahrenheit_to_celsius(weather_csv)
print(weather_csv)
print(weather_csv.sort_values(by=['Date.Full'], ascending=False))


# - __Exercise 7__: Now convert the columns that contain information
# about speed from miles per hour to km per hour in a similar manner


def from_mph_to_kph(mph: float) -> float:
    return mph * 1.60934


def convert_mph_to_kph(dataset: pd.DataFrame) -> None:
    columns: [str] = dataset.columns
    for column in columns:
        if "speed" in column.lower():
            dataset[column] = dataset[column].transform(func=from_mph_to_kph)


convert_mph_to_kph(weather_csv)
print(weather_csv)


# - __Exercise 8__: Find the most recent observation(s) in that dataset


def str_to_datetime_by_row(dataset: pd.DataFrame, column_name,
                           format_column_str):
    column_to_update = dataset[column_name]
    converted_values: [datetime] = []
    for row in column_to_update:
        try:
            row = datetime.strptime(str(row), format_column_str)
            converted_values.append(row)
        except Exception as e:
            converted_values.append("NA")
    dataset[column_name] = pd.Series(converted_values)
    return dataset


weather_csv = str_to_datetime_by_row(weather_csv, 'Date.Full', '%Y-%m-%d')
weather_csv = weather_csv.sort_values(by=['Date.Full'], ascending=False)
most_recent_date = weather_csv.iloc[0, 1]
filtered_most_recent = \
    weather_csv[weather_csv['Date.Full'] == most_recent_date]
print(filtered_most_recent)
# str_to_datetime_by_row(weather_csv, 'Date.Full', '%Y-%m-%d')
# print(weather_csv)


# - __Exercise 9__: Search most recent data for city Indianapolis (use
# index or set_index and loc)

filtered_most_recent = filtered_most_recent.set_index('Station.City')
print(filtered_most_recent.loc['Indianapolis'])

# - __Exercise 10__: Open the file weather_data and provide basic
# information about the dataframe

weather: pd.DataFrame = pd.read_csv('weather_data.csv')
print(weather.info())

# - __Exercise 11__: Set the cities as index

weather = weather.set_index('City')
print(weather)

# - __Exercise 12__: Get the data of the South African city which is the
# northernmost and southernmost


# First we have to create a filter for cities in South Africa
za_cities_filter = weather['Country'] == 'ZA'
# Now we sort by Latitude (which determines north/south positions on Earth)
weather_za_latitude = \
    weather[za_cities_filter].sort_values(by=['Latitude'], ascending=False)
# The biggest value (smallest negative) is the northernmost point
northern_most_city_za = weather_za_latitude.iloc[0]
# The smallest value (biggest negative) is the southernmost point
southern_most_city_za = weather_za_latitude.iloc[-1]
print("Weather in the northernmost city in South Africa in this dataset")
print(northern_most_city_za)
print("Weather in the southernmost city in South Africa in this dataset")
print(southern_most_city_za)

# - __Exercise 13__: Find the data of the entry closest to one of the
# poles and closest to the Equator


# Let's create a new column where that will contain the abs value of the
# column Latitude (this column actually determines how close
# a point is to the north or South Pole
weather['abs.latitude'] = weather['Latitude'].transform(func=abs)

# Closest to one of the poles would be now the point with the greatest
# value for abs.latitude
print("Weather in the point closest to one of the poles in dataset")
print(weather.sort_values(by=['abs.latitude'], ascending=False).iloc[0])

# Closest to the Equator would be now the point with the smallest value
# for abs.latitude
print("\nWeather in the point closest to the Equator in dataset")
print(weather.sort_values(by=['abs.latitude'], ascending=False).iloc[-1])

# - __Exercise 14__: Find all the registers of cities in Australia,
# New Zealand and the Solomon Islands

aus_nz_sb: pd.DataFrame = weather[weather['Country'].isin(['AU', 'NZ', 'SB'])]
print(aus_nz_sb)

# - __Exercise 15__: Find all entries where the temperature is greater
# than 30 and humidity is greater than 70

hottest_weather: pd.DataFrame = weather[(weather['Temperature'] > 30) &
                                        (weather['Humidity'] > 70)]
print(hottest_weather)

# - __Exercise 16__: Find the hottest city in the northern and western
# hemisphere

# In[18]:


nw_hemisphere: pd.DataFrame = weather[(weather['Latitude'] > 0) &
                                      (weather['Longitude'] < 0)]. \
    sort_values(by=['Temperature'], ascending=False)
print(nw_hemisphere.iloc[0])

# - __Exercise 17__: Open the file _exoplanets_NASA_20231204.csv_ and
# find the exoplanet that was discovered most recently
# 
# NOTE: Fields are explained
# [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)

exoplanets: pd.DataFrame = pd.read_csv('exoplanets_NASA_20231217.csv')

print(exoplanets)

print(exoplanets.columns)

print(exoplanets.sort_values(by=['releasedate', 'pl_pubdate', 'rowupdate'],
                             ascending=False).iloc[0])

# - __Exercise 18__: Find the furthest planet from Earth in the dataset

print(exoplanets.sort_values(by=['sy_dist'], ascending=False).iloc[0]
      [['pl_name', 'hostname', 'sy_dist', 'discoverymethod']])

# - __Exercise 19__: Count how many planets have been discovered by
# every discovery method


discovery_methods_counts: pd.DataFrame = exoplanets[
    ['discoverymethod', 'pl_name']].groupby(['discoverymethod']).count()
discovery_methods_counts.columns = ['count']
print(discovery_methods_counts)

# - __Exercise 20__: Get the planet that is furthest from Earth that has
# been discovered by each method

further_exoplanet_method = exoplanets[['discoverymethod', 'sy_dist']].groupby(
    ['discoverymethod']).max()

print(exoplanets[exoplanets['sy_dist'].
      isin(further_exoplanet_method['sy_dist'])]
      [['pl_name', 'hostname', 'discoverymethod', 'sy_dist']])

# - __Exercise 21__: Get the average mass of a planet discovered by each
# method

exoplanets[['discoverymethod', 'pl_bmasse']].groupby(
    ['discoverymethod']).mean()

# - __Exercise 22__: Get the number of planets discovered each year by
# each method

planets_discovered_each_year_by_method: pd.DataFrame = exoplanets[
    ['pl_name', 'discoverymethod', 'disc_year']].groupby(
    ['discoverymethod', 'disc_year']).count()
planets_discovered_each_year_by_method.columns = ['count']
planets_discovered_each_year_by_method.sort_values(by=['disc_year'])

# - __Exercise 23__: For each host work out how many planets each has
#   - Then, workout which system has the most planets
#   - Average planets per system

planets_per_system: pd.DataFrame = exoplanets[['pl_name', 'hostname']].groupby(
    ['hostname']).count()

planets_per_system.columns = ['n_planets']
print(planets_per_system)

# System with the most planets
print(planets_per_system.sort_values('n_planets', ascending=False).iloc[0])

# Average Number of planets per system
planets_per_system['n_planets'].mean()

# - __Exercise 24__: Get the number of planets discovered in each system
# by each different discovery method

pd.set_option('display.max_rows', exoplanets.shape[0] + 1)
print(exoplanets.groupby(['hostname', 'discoverymethod']).count())

# - __Exercise 25__: Get the planet with the biggest mass in each
# planetary system

biggest_mass: pd.Series = \
    exoplanets.groupby(['hostname']).max('pl_masse')['pl_masse']
print(biggest_mass)

result = \
    pd.merge(exoplanets, biggest_mass,
             on=['pl_masse', 'hostname'], how='inner')[['pl_name',
                                                        'hostname',
                                                        'pl_masse']].dropna()
print(result)
