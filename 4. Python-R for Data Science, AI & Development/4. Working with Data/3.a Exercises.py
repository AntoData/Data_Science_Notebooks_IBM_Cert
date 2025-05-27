#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ### Numpy
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
# - EU GDP.xlsx: Contains the GDP of every country in the EU from 1995 to 2022
# - EU GDP per capita.xlsx: Contains the GDP per capita of every country
# in the EU from 1995 to 2022
# - EU population.xlsx: Contains the population of every country in the EU

# - __Exercise 1__: Open the file IMDB-Movie-Data.xlsx, get the revenue
# and use it to create a numpy array

imdb_movies: pd.DataFrame = pd.read_csv('IMDB-Movie-Data.csv').dropna()
revenue_narray: np.array = np.array(imdb_movies['Revenue (Millions)'])
print(revenue_narray)

# - __Exercise 2__: First, get the last of the revenues and after that
# get the first 10

print(revenue_narray[len(revenue_narray) - 1])

print(revenue_narray[0:10])


# - __Exercise 3__: Create a numpy array with the revenue for the first
# 10 movies, but instead of millions of dollars, convert it to dollars
# using slicing

revenue_dollars: np.array = np.array(revenue_narray[0:10])
revenue_dollars[0:10] = revenue_narray[0] * 10**6, \
    revenue_narray[1] * 10**6, revenue_narray[2] * 10**6, \
    revenue_narray[3] * 10**6, revenue_narray[4] * 10**6, \
    revenue_narray[5] * 10**6, revenue_narray[6] * 10**6, \
    revenue_narray[7] * 10**6, revenue_narray[8] * 10**6, \
    revenue_narray[9] * 10**6
 
print(revenue_dollars)

# - __Exercise 4__: Get the revenue of the movies in the odd positions

numpy_len: int = len(revenue_narray)
odd_positions: [int] = [x for x in range(1, numpy_len, 2)]
odd_revenue: np.array = revenue_narray[odd_positions]
print(odd_revenue)


# - __Exercise 5__: Imagine all EU countries where German is an official
# language (Germany, Austria, Belgium and Luxembourg) unite, find out
# what the GDP of this new country would have been

eu_gdp: pd.DataFrame = pd.read_excel('EU GDP.xlsx')
print(eu_gdp)

germany_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'Germany'])
austria_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'Austria'])
luxembourg_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'Luxembourg'])
belgium_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'Belgium'])

germany_gdp = germany_gdp[0, 1:]
austria_gdp = austria_gdp[0, 1:]
luxembourg_gdp = luxembourg_gdp[0, 1:]
belgium_gdp = belgium_gdp[0, 1:]

new_country_gdp: np.array = germany_gdp + austria_gdp + luxembourg_gdp + \
                            belgium_gdp

print(new_country_gdp)


# - __Exercise 6__: Get the difference between the GDP of France and
# Spain through the years

france_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'France'])
spain_gdp: np.array = np.array(eu_gdp[eu_gdp['TIME'] == 'Spain'])

france_gdp = france_gdp[0, 1:]
spain_gdp = spain_gdp[0, 1:]

diff_gdp_fr_es: np.array = france_gdp - spain_gdp
print(diff_gdp_fr_es)


# - __Exercise 7__: Turn the revenue of movies from millions of dollars
# to dollars

revenue_millions_narray: np.array = np.array(imdb_movies['Revenue (Millions)'])
revenue_dollars_narray: np.array = revenue_millions_narray * 10 ** 6
print(revenue_dollars_narray)


# - __Exercise 8__: Take the GDP per capita of every country in the EU
# in 2022 and the population in 2022 and get the total GDP of every
# country in 2022 (you need to multiply both array)

eu_population: pd.DataFrame = pd.read_excel('EU population.xlsx')
eu_population_2022: pd.DataFrame = eu_population['2022']
eu_population_2022_np = np.array(eu_population_2022)
print(eu_population_2022_np)

eu_gdp_cap: pd.DataFrame = pd.read_excel('EU GDP per capita.xlsx')
eu_gdp_cap_2022: pd.DataFrame = eu_gdp_cap['2022']
eu_gdp_cap_2022_np: np.array = np.array(eu_gdp_cap_2022)
print(eu_gdp_cap_2022_np)

eu_population_2022_np * eu_gdp_cap_2022_np

# - __Exercise 9__: Now instead using both arrays, work out the GDP of
# the whole European Union (use the dot product):

np.dot(eu_gdp_cap_2022_np, eu_population_2022_np)


# - __Exercise 10__: Work out the revenue per minute of the movies in
# IMDB-Movie-Data.csv

imdb_movies: pd.DataFrame = pd.read_csv('IMDB-Movie-Data.csv').dropna()
revenue_narray: np.array = np.array(imdb_movies['Revenue (Millions)'])
minutes_narray: np.array = np.array(imdb_movies['Runtime (Minutes)'])

print(revenue_narray / minutes_narray)

# - __Exercise 11__: Work out the average revenue of the movies

print(revenue_narray.mean())

# - __Exercise 12__: Work out the max revenue of the movies registered

print(np.max(revenue_narray))

# - __Exercise 13__: Work out the min revenue of the movies registered

print(np.min(revenue_narray))


# - __Exercise 14__: Work out the standard deviation of the movies
# registered

print(revenue_narray.std())


# - __Exercise 15__: Plot the function y = sin(x) from  -2π < x < 2π.
# Use 100 points that are equally distanced between them

x_values: np.array = np.linspace(-2 * np.pi, 2 * np.pi, 100)

y_values: np.array = np.array([np.sin(x) for x in x_values])

plt.plot(x_values, y_values)

