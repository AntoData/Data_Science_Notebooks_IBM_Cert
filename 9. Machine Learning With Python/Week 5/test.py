import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    root_mean_squared_error, r2_score


def keep_country_and_years_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the columns in datasets downloaded from eurostat so it only
    includes the column TIME (and renames it to country) and the columns
    of years and return points and other characters from numeric columns

    :param df: Dataset downloaded from eurostat
    :type df: pd.DataFrame
    :return: DataFrame that only contains the columns with the countries
    and data per year
    :rtype pd.DataFrame
    """
    # This array will contain the columns with numeric value, therefore
    # years
    column_year: [int] = []
    # Now we go column by column and try to convert the title to int
    for column in df.columns:
        try:
            year = int(column)
            # If an exception is not triggered we added it to the array
            column_year.append(year)
        except ValueError:
            # Otherwise, we just move on
            pass

    # Now we only keep the column TIME and the columns in the array
    # column_year
    df = df[["TIME"] + [str(x) for x in range(min(column_year),
                                              max(column_year))]]
    # We remove any possible . in the columns that contain numbers
    df = df.astype(str).replace(r'\.', '', regex=True)
    # If there are , that means we need to convert them to .
    df = df.astype(str).replace(r'\,', '.', regex=True)
    # We convert any column that contains numeric data in the year
    # series to integer
    try:
        df[[str(x) for x in range(min(column_year), max(column_year))]] = df[
            [str(x) for x in range(min(column_year), max(column_year))]]. \
            astype("int")
    except ValueError:
        df[[str(x) for x in range(min(column_year), max(column_year))]] = df[
            [str(x) for x in range(min(column_year), max(column_year))]]. \
            astype("float")
    # We fill in any NA value in the numeric data in the yearly series
    # with the mean of those columns
    df[[str(x) for x in range(min(column_year), max(column_year))]] = \
        df[[str(x) for x in range(min(column_year),
                                  max(column_year))]].fillna(
            df[[str(x) for x in range(min(column_year),
                                      max(column_year))]].mean())
    # We rename column TIME to Country
    df.rename(columns={"TIME": "Country"}, inplace=True)
    return df


"""
1. DATA CLEANING
"""
# First we open the dataset
print("1. Opening GDP in the EU dataset: ")
df_gdp_eu: pd.DataFrame = pd.read_csv('GDP EU.xlsx - Sheet 1.csv')
# The first row is empty so we remove it
df_gdp_eu = df_gdp_eu.loc[1:]
# We clean the data
print("2. Preprocessing")
print("2.1 Cleaning this dataset")
df_gdp_eu = keep_country_and_years_columns(df_gdp_eu)
print(df_gdp_eu)
print(" ")

# We open the second dataset
print("2.2 Opening CO2 emissions in the EU dataset")
co2_emissions: pd.DataFrame = \
    pd.read_csv('Greenhouse gas emissions EU - Sheet 1.csv')
# We clean data
print("2.3 We clean this dataset")
co2_emissions = keep_country_and_years_columns(co2_emissions)
print(co2_emissions)
print("")

# Now we need to keep only the subsets of years and countries
print("2.4 Working out common columns and rows")
common_columns: [str] = \
    list(set(df_gdp_eu.columns).intersection(set(co2_emissions)))
print("Common columns: {0}".format(common_columns))
common_countries: [str] = list(set(df_gdp_eu["Country"]).intersection(
    set(co2_emissions["Country"])))
print("Common countries: {0}".format(common_countries))
print("")

# Now we filter the datasets to get only common countries and year
print("2.5 Leaving only common columns and rows in datasets")
df_gdp_eu = df_gdp_eu[common_columns]
df_gdp_eu = df_gdp_eu[df_gdp_eu["Country"].isin(common_countries)]
df_gdp_eu.set_index("Country", inplace=True)
print("GDP in the EU completely filtered and cleaned")
print(df_gdp_eu)
co2_emissions = co2_emissions[common_columns]
co2_emissions = co2_emissions[co2_emissions["Country"].isin(common_countries)]
co2_emissions.set_index("Country", inplace=True)
print("CO2 emission in the EU completely filtered and cleaned")
print(co2_emissions)
print(" ")

# We sort both by column Country
print("2.6 Sorting both datasets by country alphabetically")
df_gdp_eu.sort_values(by="Country", inplace=True)
df_gdp_eu_t: pd.DataFrame = df_gdp_eu.transpose().rename_axis("Years").sort_values(by="Years")
var_x: np.ndarray = df_gdp_eu_t.to_numpy().flatten()
print("GDP in the EU after being sorted: ")
print(df_gdp_eu)
co2_emissions.sort_values(by="Country", inplace=True)
co2_emissions_t: pd.DataFrame = co2_emissions.transpose().rename_axis("Years").sort_values(by="Years")
var_y: np.ndarray = co2_emissions_t.to_numpy().flatten()
print("CO2 emissions in the EU after being sorted: ")
print(co2_emissions)
print(df_gdp_eu.shape)
print(co2_emissions.shape)

# Now we convert every dataset to a single variable
# x independent variable is GPD
# y dependent variable is co2 emissions
x_gdp: [int] = []
y_co2: [float] = []
for col1, col2 in zip(df_gdp_eu, co2_emissions):
    assert col1 == col2
    x_r = df_gdp_eu[col1].tolist()
    y_r = co2_emissions[col2].tolist()
    x_gdp.extend(df_gdp_eu[col1].tolist())
    y_co2.extend(co2_emissions[col2].tolist())
print("As both datasets have the same number of rows and columns and "
      "same order, we can pair them and we created a dataset with all"
      " related records from both datasets")
dataset_x_y = \
    pd.DataFrame({"GDP": df_gdp_eu, "CO2E": co2_emissions})
print(dataset_x_y)