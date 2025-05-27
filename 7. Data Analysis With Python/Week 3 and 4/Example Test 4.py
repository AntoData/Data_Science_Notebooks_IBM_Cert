"""
prc_hpi_a_page_spreadsheet (1).xlsx: House price index
Source: https://ec.europa.eu/eurostat/databrowser/view/prc_hpi_a/
default/table?lang=en&category=prc.prc_hpi.prc_hpi_inx

tps00005_page_spreadsheet (1).xlsx: Population as a percentage of EU
Source: https://ec.europa.eu/eurostat/databrowser/view/tps00005/
default/table?lang=en&category=t_demo.t_demo_ind
"""
from decimal import Decimal, getcontext
import pandas
from scipy import stats
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def find_the_highest_correlation_in_corr_matrix(df_formatted: pd.DataFrame) \
        -> (int, int, np.float64, np.float64):
    """
    Returns the two columns where the correlation coefficient is highest
    but not a variable with itself (which means both columns are the
    same column in different axis and therefore always 1) and given
    that p-value shows a strong certainty of the correlation

    :param df_formatted: Dataframe ready to execute .corr()
    :type df_formatted: pd.Dataframe
    :return: index of column and row where the correlation is highest
    and its value and its p-value
    """
    getcontext().prec = 320
    # We execute the function .corr to get the correlation matrix
    df_corr: pd.DataFrame = df_formatted.corr()
    # By default our correlation value is 0, and i=j=0
    max_value: np.float64 = np.float64(0.0)
    final_p_value: np.float64 = np.float64(0.0)
    i_max: int = 0
    j_max: int = 0

    # Now we start iterating through the matrix element by element
    for i in range(0, df_corr.shape[0]):
        for j in range(0, df_corr.shape[1]):
            curr_corr_coef: float = df_corr.iloc[i, j]

            # If the current correlation coefficient is higher than
            # the one we found as highest and it is not in the diagonal
            # (i != j which means it is not a variable with itself)
            if abs(max_value) < abs(curr_corr_coef) < 1 and (i != j):
                curr_pearson_coef, curr_p_value = stats.pearsonr(
                    df_formatted[df_formatted.columns[i]],
                    df_formatted[df_formatted.columns[j]])
                if curr_p_value <= np.float64(0.001):
                    # Current correlation coefficient is the highest so far
                    max_value = np.float64(curr_corr_coef)
                    # And p-value is valid, shows high certainty
                    final_p_value = np.float64(curr_p_value)
                    # The columns that will get the highest correlation
                    # are our current columns
                    i_max = i
                    j_max = j
    return i_max, j_max, Decimal(max_value), Decimal(final_p_value)


def find_highest_correlation_same_columns(df_formatted1: pd.DataFrame,
                                          df_formatted2: pd.DataFrame) \
        -> (str, float, float):
    """
    Returns the column where the correlation coefficient is the highest
    taking its data from both dataframes

    :param df_formatted1: First dataframe we will take the variable from
    :param df_formatted2: Second dataframe we will take the variable from
    :return: Variable that produces the highest correlation coefficient,
    correlation coefficient and p-value
    """

    max_coef: float = 0
    col_max = None
    res_p_value = None
    for col in df_formatted1.columns:
        if col in df_formatted2.columns:
            common_years_list = set(df_formatted1[col].index).intersection(
                df_formatted2[col].index)
            min_common_year: int = min(common_years_list)
            max_common_year: int = max(common_years_list)
            pearson_coef, p_value = stats.pearsonr(
                df_formatted1[col].loc[min_common_year: max_common_year],
                df_formatted2[col].loc[min_common_year: max_common_year])
            if abs(pearson_coef) > abs(max_coef):
                max_coef = pearson_coef
                col_max = col
                res_p_value = p_value
    return col_max, max_coef, res_p_value


# First, let's open the file about population in the European Union
# by percentages
df_eu_population_as_percentage: pd.DataFrame = \
    pd.read_excel("./tps00005_page_spreadsheet (1).xlsx")
# Now, we need to format it for use, we make TIME the index and we
# transpose the dataframe so time is the rows index and contries are
# the columns
df_eu_population_as_percentage_format: pd.DataFrame = \
    df_eu_population_as_percentage.set_index(keys=["TIME"]).transpose()

# Let's rename Spain, España in its original language
df_eu_population_as_percentage_format.rename(
    columns={"Spain": "España"}, inplace=True)

print("Columns")
print(df_eu_population_as_percentage_format.columns)
print(df_eu_population_as_percentage_format.dtypes)
print("")

print("European population as percentage of the EU")
print(df_eu_population_as_percentage_format)
print("")
# Let´s replace any NAN with the mean of every column
# We check first if there is any NaN/Null value
print("Replacing NaN Values")
if True in df_eu_population_as_percentage_format.isnull():
    df_eu_population_as_percentage_format.replace(
        np.NaN, df_eu_population_as_percentage_format.mean(), inplace=True)

# Let's get some quick stats
print("")
print("Describe dataframe")
print(df_eu_population_as_percentage_format.describe(include='all'))
print("Info dataframe")
print(df_eu_population_as_percentage_format.info())
print("")

print("Let's pick 2024 and divide countries on how much population they"
      " contribute to the EU in 4 groups: \n Very Large \n Large \n Medium "
      "\n Small")
# First we need to get the values for year 2024
df_eu_population_as_percentage_2024: pd.Series = \
    df_eu_population_as_percentage_format.loc["2024"]
# Then we divide the interval of values in 4 bins
bins = np.linspace(
    df_eu_population_as_percentage_2024.min(),
    df_eu_population_as_percentage_2024.max(), 5)

df_eu_population_descriptive_2024: pd.Series = \
    pd.cut(df_eu_population_as_percentage_2024, bins,
           labels=["Small", "Medium", "Large", "Very Large"])
print(df_eu_population_descriptive_2024)
print("One Hot Encoding")
print(pandas.get_dummies(df_eu_population_descriptive_2024))
print("Let's count how many of each type we have")
print(df_eu_population_descriptive_2024.value_counts())
print("")
print("Let's find out which is the country that provided the biggest "
      "percentage "
      "of population in record and the year")
print("First we get the year the population percentage was max for each "
      "country")
print(df_eu_population_as_percentage_format.idxmax())
print("Now we build a series with the percentage for each tuple country-year ")
max_percentages_index_order: pd.Series = pd.Series([
    df_eu_population_as_percentage_format.loc[
        df_eu_population_as_percentage_format.idxmax().iloc[i],
        df_eu_population_as_percentage_format.idxmax().index[i]]
    for i in range(0, len(df_eu_population_as_percentage_format.idxmax()))])
print(max_percentages_index_order)
print("Now we get the index of the max value, this gives us the index of the "
      "index country and column population in percentage")
max_percentages_index = max_percentages_index_order.idxmax()
print("Country and year are both in position {0} in dataframe.idxmax()".
      format(max_percentages_index))
year_max_percentage_population: int = \
    df_eu_population_as_percentage_format.idxmax().iloc[max_percentages_index]
country_max_percentage_population: str = \
    df_eu_population_as_percentage_format.idxmax().index[max_percentages_index]
max_percentage_in_record: float = \
    max_percentages_index_order.iloc[max_percentages_index]
print(
    "The country that contributed the highest percentage to the EU "
    "historically was {0} in the year {1} with {2}%".format(
        country_max_percentage_population, year_max_percentage_population,
        max_percentage_in_record))

# To classify countries in 4 groups in the overall dataframe
# We create the bins
bins = np.linspace(
    np.min(df_eu_population_as_percentage_format),
    np.max(df_eu_population_as_percentage_format), 5)

# We create an empty dataframe
df_population_descriptive: pd.DataFrame = pd.DataFrame()

# We go now column by column, classify the data and add it to the empty
# dataframe
for column_ in df_eu_population_as_percentage_format:
    df_eu_population_descriptive_year: pd.DataFrame = \
        pd.cut(df_eu_population_as_percentage_format[column_],
               bins, labels=["Small", "Medium", "Large", "Very Large"])
    df_population_descriptive = pd.concat([df_population_descriptive,
                                           df_eu_population_descriptive_year],
                                          axis=1)

# Line plot with the population evolution of Germany

plt.plot(df_eu_population_as_percentage_format.index,
         df_eu_population_as_percentage_format["Germany"])
plt.show()

# Scatter plot between independent variable population in Germany
# and dependent variable population in Austria
# Later we will see they have a high correlation coefficient with
# very high certainty
plt.scatter(df_eu_population_as_percentage_format["Germany"],
            df_eu_population_as_percentage_format["Austria"])
plt.show()

# Histogram of the population in the EU by country and by size category
plt.hist(df_eu_population_descriptive_2024.astype("str"), 4)
plt.show()

# Bar plot of the EU population in 2024
plt.bar(x=df_eu_population_as_percentage_2024.index,
        height=df_eu_population_as_percentage_2024)
plt.show()

# Pseudo color plot with EU population (all data)
plt.pcolor(df_eu_population_as_percentage_format)
plt.show()

# Regression plot between the population in Germany and Austria
sns.regplot(x='Germany', y='Austria',
            data=df_eu_population_as_percentage_format)
plt.show()

# Box plot of EU population in 2024
# Shows that most of the countries have a population between 0,x and 2.x%
sns.boxplot(x="2024", data=df_eu_population_as_percentage_format.transpose())
plt.show()

# Residual plot between populations in Germany and Austria
sns.residplot(x="Germany", y="Austria",
              data=df_eu_population_as_percentage_format)
plt.show()

# KDE population in 2024
sns.kdeplot(x="2024", data=df_eu_population_as_percentage_format.transpose())
plt.show()

# Distribution plot population in 2024
sns.displot(x='2024', data=df_eu_population_as_percentage_format.transpose(),
            kde=True)
plt.show()

### CORRELATION
# We have 2 ways to get correlation
# Correlation matrix, returns the correlation coefficient
# between all columns in a matrix
# Problem is, one column with itself will always have correlation 1
# So the diagonal in the matrix is always 1
print(df_eu_population_as_percentage_format.corr())

# PEARSON COEFFICIENT
# We also have a specific method to get the correlation coefficient
# plus p-value given two variables
# correlation coefficient: Tells us if there is correlation between the
# variables and how it is:
# -1: Very large negative correlation
# 0: No correlation
# 1: Very large positive correlation

# p-value: Tells us how accurate is the correlation coefficient
# < 0.001 -> Strong certainty
# < 0.05 -> Moderate certainty
# < 0.01 -> Weak certainty
# > 0.01 -> No certainty

pearson_coef, p_value = stats.pearsonr(
    df_eu_population_as_percentage_format["Germany"],
    df_eu_population_as_percentage_format["Austria"])
print("Correlation study Germany-Austria")
print("Correlation coefficient = {0}".format(pearson_coef))
print("p-value={0}".format(p_value))

# Housing prices as index in the EU
df_house_price_index: pd.DataFrame = \
    pd.read_excel("./prc_hpi_a_page_spreadsheet (1).xlsx")
df_house_price_index_format: pd.DataFrame = \
    df_house_price_index.set_index(keys=["TIME"]).transpose()

# Let's rename Spain, España in its original language
df_house_price_index_format.rename(
    columns={"Spain": "España"}, inplace=True)

# We developed a method to get the highest correlation coefficient
# between two variables

# Highest correlation between two countries regarding population as
# percentage in the EU
i_res, j_res, corr_coef, p_value = \
    find_the_highest_correlation_in_corr_matrix(
        df_eu_population_as_percentage_format)
print(df_eu_population_as_percentage_format.columns[i_res])
print(df_eu_population_as_percentage_format.columns[j_res])
print(np.float64(corr_coef))
print(np.float64(p_value))

# Linear Regression model
# Country 1 with highest correlation coefficient
country_1: str = df_eu_population_as_percentage_format.columns[i_res]
country_2: str = df_eu_population_as_percentage_format.columns[j_res]
lm = LinearRegression()
lm.fit(pd.DataFrame(df_eu_population_as_percentage_format[country_1]),
       df_eu_population_as_percentage_format[country_2])
prediction = lm.predict(np.array([10, 20]).reshape(-1, 1))
print(prediction)
print(df_eu_population_as_percentage_format["Austria"])

# Highest correlation between two countries regarding housing price as
# index in the EU
i_res, j_res, corr_coef, p_value = \
    find_the_highest_correlation_in_corr_matrix(df_house_price_index_format)
print(df_house_price_index_format.columns[i_res])
print(df_house_price_index_format.columns[j_res])
print(corr_coef)
print(p_value)

lm_housing_index: LinearRegression = LinearRegression()
lm_housing_index.fit(pd.DataFrame(df_house_price_index_format["Czechia"]),
                     df_house_price_index_format["Netherlands"])
print(lm_housing_index.coef_)

prediction = lm.predict(np.array([13.5, 17.9]).reshape(-1, 1))
print(prediction)

# Poland
col_max, max_coef, res_p_value = \
    find_highest_correlation_same_columns(
        df_eu_population_as_percentage_format, df_house_price_index_format)
print(max_coef)
print(col_max)
print(res_p_value)

lm_poland_population_housing: LinearRegression = LinearRegression()
lm_poland_population_housing.fit(
    pd.DataFrame(df_eu_population_as_percentage_format[col_max].loc['2014':
                                                                    '2023']),
    df_house_price_index_format[col_max].loc['2014': '2023'])
prediction = lm_poland_population_housing.predict(np.array(5).reshape(-1, 1))
print(prediction)
print(lm_poland_population_housing.coef_)
print(lm_poland_population_housing.intercept_)

# Multiple Linear Regression
# predict = x*b1 + y*b2 + z*b3 + ... + b0

mlr: LinearRegression = LinearRegression()
mlr.fit(df_house_price_index_format[["Czechia", "France", "Belgium"]],
            df_house_price_index_format["Netherlands"])
prediction = mlr.predict(np.array([1.3, 1.3, 1.5]).reshape(1, -1))
print(prediction)
print(mlr.coef_)

print(df_house_price_index_format.corr()["Netherlands"])
