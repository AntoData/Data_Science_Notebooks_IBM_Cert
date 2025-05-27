import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# First dataset from
# https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price

# Second dataset from
# https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price

# Let's open the file Mobile phone price.csv and print it
df_mobile_prices: pd.DataFrame = pd.read_csv('Mobile phone price.csv')
print("Dataframe Sample")
print(df_mobile_prices.head())
print("")

print("")
# Let's rename some columns
print("Column names before")
print(df_mobile_prices.columns)
df_mobile_prices.rename(columns={"Screen Size (inches)": "Screen_Size_in",
                                 "Camera (MP)": "Camera_mp",
                                 "Battery Capacity (mAh)":
                                     "Battery_Capacity_mAh",
                                 "Price ($)": "Price_$",
                                 'Storage ': 'Storage',
                                 'RAM ': 'RAM'}, inplace=True)
print("Column names after")
print(df_mobile_prices.columns)

# Now let's get some information about its contents
print("Dataframe columns")
print(df_mobile_prices.columns)
print("")

# Let's open the other file that contains data about mobile phone prices
df_mobile_prices_more: pd.DataFrame = pd.read_csv('ndtv_data_final.csv')

# Let's check the types of this new dataset
print("Dataframe2 columns")
print(df_mobile_prices_more.dtypes)
print("")

print("Changing the names of the columns in dataframe2")
df_mobile_prices_more.rename(columns={"Screen size (inches)": "Screen_Size_in",
                                      "Front camera": "Camera_mp",
                                      "Battery capacity (mAh)":
                                          "Battery_Capacity_mAh",
                                      "Price": "Price_$",
                                      'Internal storage (GB)': 'Storage',
                                      'RAM (MB)': 'RAM'}, inplace=True)
print("Transforming prices in dataframe2 from INR to USD")
df_mobile_prices_more["Price_$"] = df_mobile_prices_more["Price_$"] * 0.012
print("Transforming RAM in dataframe2 from MB to GB")
df_mobile_prices_more["RAM"] = df_mobile_prices_more["RAM"] / 1024
print("Getting only the same columns than in dataframe1 for dataframe2")
df_mobile_prices_more = df_mobile_prices_more[df_mobile_prices.columns]
print("Concatting dataframes")
df_mobile_prices: pd.DataFrame = \
    pd.concat([df_mobile_prices, df_mobile_prices_more], axis=0)
print(df_mobile_prices.columns)
print(df_mobile_prices)
print("")

# Let's see if there are null values
# With this we will identify them
print("Scanning dataframe for null values")
for column in df_mobile_prices.isnull().columns:
    for i in range(0, len(df_mobile_prices.isnull()[column])):
        if df_mobile_prices.isnull()[column].iloc[i] is True:
            print("NULL VALUE WAS IDENTIFIED")
            break
else:
    print("No null value in the dataframe")

# Let's see if there are na values
# With this we will identify them
print("Scanning dataframe for NaN values")
for column in df_mobile_prices.isna().columns:
    for i in range(0, len(df_mobile_prices.isna()[column])):
        if df_mobile_prices.isna()[column].iloc[i] is True:
            print("NA VALUE WAS IDENTIFIED")
            df_mobile_prices.dropna(inplace=True)
            break
else:
    print("No na value in the dataframe")

# Let's see how the types of each column
print("Types of each column")
print(df_mobile_prices.dtypes)
print("")

# We see there are mistakes in these types as except for Brand, Model
# and screen size, all of them should be numeric (int64, float64)
print("Transforming data")
for column in df_mobile_prices.columns:
    print(column)
    if column in ['Screen_Size_in']:
        print("Setting Screen_Size_in to numeric")
        df_mobile_prices[column] = pd.to_numeric(
            df_mobile_prices[column], errors='coerce')
        print("All values that could not be cast as numeric will be "
              "replaced by the mean value of the column")
        df_mobile_prices[column] = \
            df_mobile_prices[column].fillna(df_mobile_prices[column].mean())
    elif column in ['Storage', 'RAM']:
        print("Setting {0} to integer".format(column))
        print("Removing GB from columns")
        df_mobile_prices[column] = \
            df_mobile_prices[column].str.replace("GB", '')
        df_mobile_prices.dropna(subset=column, inplace=True)
        df_mobile_prices[column] = df_mobile_prices[column].astype('int64')
    elif column in ['Price_$']:
        print("Setting Prices to integer")
        print("Removing the $ sign from the column")
        df_mobile_prices[column] = \
            df_mobile_prices[column].str.replace("$", '')
        df_mobile_prices[column] = \
            df_mobile_prices[column].str.replace(",", '')
        df_mobile_prices[column] = df_mobile_prices[column].astype('int64')
print("")
print("Column types after transformation")
print(df_mobile_prices.dtypes)
print("")

# Now are going to add new columns for Price, Storage and RAM but
# normalized
print("Normalizing fields Price, Storage and RAM")
# First let's normalize price using simple feature scaling
print("Simple Feature Scaling for Price_$")
df_mobile_prices['Price_$_scaled'] = \
    df_mobile_prices['Price_$'] / df_mobile_prices['Price_$'].max()
print(df_mobile_prices['Price_$_scaled'])
print("")

# Now, let's normalize Storage using the method mix max
print("min Max Scaling for Storage")
df_mobile_prices['Storage_mMn'] = \
    (df_mobile_prices['Storage'] - df_mobile_prices['Storage'].min()) / \
    (df_mobile_prices['Storage'].max() - df_mobile_prices['Storage'].min())
print(df_mobile_prices['Storage_mMn'])
print("")

# Let's normalize RAM using the method Z-score
print("Z-score for RAM")
df_mobile_prices['RAM_zscore'] = \
    (df_mobile_prices['RAM'] - df_mobile_prices['RAM'].mean()) / \
    df_mobile_prices['RAM'].std()
print(df_mobile_prices['RAM_zscore'])
print("")

# Let's create also a category for prices using pandas.cut
print("Let's create now a categorical column for Price_$")
df_mobile_prices['Price_categorical'] = \
    pd.cut(df_mobile_prices['Price_$'], 5,
           labels=['Lowest', 'Low', 'Medium', 'High', 'Expensive'])
print(df_mobile_prices['Price_categorical'])
print("")
print("We will create a unitary matrix for the brands")
df_brand_dummy: pd.DataFrame = pd.get_dummies(df_mobile_prices['Brand'])
print(df_brand_dummy)
print("")

# Let's get the most expensive phone and its information
i_most_expensive: int = df_mobile_prices['Price_$'].idxmax()
print("Most expensive phone: row={0}".format(i_most_expensive))
print("Phone details: ")
print(df_mobile_prices.iloc[i_most_expensive])
print("")

# Let's get now the average prices using group by of the following data
print("Average mobile phone prices by price category")
df_mean_prices_cat: pd.DataFrame = \
    df_mobile_prices[['Price_categorical', 'Price_$']].groupby(
        by="Price_categorical", observed=False).mean()
print(df_mean_prices_cat)
print("")

print("Average mobile phone prices by brand")

df_mean_prices_brand: pd.DataFrame = \
    df_mobile_prices[['Brand', 'Price_$']].groupby(by='Brand',
                                                   observed=False).mean()
print(df_mean_prices_brand)
print("")

print("Average mobile phone prices by brand and price category")
df_mean_prices_brand_cat: pd.DataFrame = \
    df_mobile_prices[['Brand', 'Price_categorical', 'Price_$']].groupby(
        by=["Brand", "Price_categorical"], observed=False).mean()
print(df_mean_prices_brand_cat)
print("")

# With this latest dataset, let's create a matrix where x is the brands
# and y the price categories and the value of the cells is the average
# price in by that brand in that price range
print("Price matrix by brand vs price category")
df_mean_prices_brand_cat.reset_index(inplace=True)
df_price_matrix: pd.DataFrame = \
    df_mean_prices_brand_cat.pivot(index='Brand', columns='Price_categorical')
print(df_price_matrix)
print("")

# Now, let's display a heatmap using this matrix
plt.pcolor(df_price_matrix)
plt.show()

sns.heatmap(df_price_matrix)
plt.show()

# Let's display in a line plot in x = RAM and y= mean price both scaled
df_mobile_ram_vs_price: pd.DataFrame = \
    df_mobile_prices[['RAM_zscore', 'Price_$_scaled']].groupby(
        by='RAM_zscore').mean()

df_mobile_ram_vs_price.reset_index(inplace=True)

plt.plot(df_mobile_ram_vs_price['RAM_zscore'],
         df_mobile_ram_vs_price["Price_$_scaled"])
plt.xlabel("RAM scaled")
plt.ylabel("Price scaled")
plt.title("RAM vs mean Price")
plt.show()

# Let's create a scatter plot of RAM vs prices
plt.scatter(df_mobile_prices['RAM'], df_mobile_prices['Price_$'])
plt.title("RAM vs Price")
plt.xlabel("RAM in GB")
plt.ylabel("Price in $")
plt.show()

# Let's create a scatter plot of RAM vs mean prices scaled
plt.scatter(df_mobile_ram_vs_price['RAM_zscore'],
            df_mobile_ram_vs_price['Price_$_scaled'])
plt.title("RAM vs Price mean scaled")
plt.xlabel("RAM scaled")
plt.ylabel("Mean price scaled in $")
plt.show()

# Let's create a histogram where we display the number of phones per
# price ranges
plt.hist(df_mobile_prices['Price_$'], 10)
plt.title("Distribution of phones per price range")
plt.xlabel("Prices in $")
plt.ylabel("Number of phones")
plt.show()

df_mobile_models_by_brand: pd.Series = \
    df_mobile_prices[['Brand']].groupby(
        by='Brand', observed=False).value_counts()
df_mobile_models_by_brand: pd.DataFrame = df_mobile_models_by_brand.to_frame()
df_mobile_models_by_brand.reset_index(inplace=True)
print("Number of model per brand:")
print(df_mobile_models_by_brand)
print("")

# Let's create a bar plot with the number of phones per brand
plt.bar(x=df_mobile_models_by_brand['Brand'],
        height=df_mobile_models_by_brand['count'])
plt.title("Number of phones per brand")
plt.xlabel("Brands")
plt.ylabel("Number of phones")
plt.show()

# Let's make a regression plot of the RAM memory vs Price
sns.regplot(x='RAM', y='Price_$', data=df_mobile_prices)
plt.xlabel("RAM in GB")
plt.ylabel("Prices in $")
plt.title("RAM (GB) vs Price ($)")
plt.show()

# Let's make a regression plot of the scaled RAM and mean scaled price
sns.regplot(x='RAM_zscore', y='Price_$_scaled', data=df_mobile_ram_vs_price)
plt.title("RAM scaled vs mean price scaled")
plt.xlabel("RAM scaled")
plt.ylabel("Mean price scaled")
plt.show()

# Let's create a boxplot that displays the price ranges per brand
sns.boxplot(x='Brand', y='Price_$', data=df_mobile_prices)
plt.xlabel("Brand")
plt.ylabel("Prices in $")
plt.title("Prices in $ by brand")
plt.show()

# Let's display now a residual plot of RAM vs Price
sns.residplot(x='RAM', y='Price_$', data=df_mobile_prices)
plt.xlabel("RAM in GB")
plt.ylabel("Price in $")
plt.title("RAM in GB vs Price in $")
plt.show()

# Let's display now a residual plot of scaled RAM vs scaled mean price
sns.residplot(x='RAM_zscore', y='Price_$_scaled', data=df_mobile_ram_vs_price)
plt.title("RAM vs Price mean scaled")
plt.xlabel("RAM scaled")
plt.ylabel("Mean price scaled in $")
plt.show()

# Let's display the KDE of RAM
sns.kdeplot(x='RAM', data=df_mobile_prices)
plt.xlabel("RAM in GB")
plt.show()

# Let's display the KDE of Prices
sns.kdeplot(x='Price_$', data=df_mobile_prices)
plt.xlabel("Price in $")
plt.show()

# Let's display a distribution plot of Prices
sns.displot(x='Price_$', data=df_mobile_prices, kde=True)
plt.xlabel("Prices in $")
plt.show()

print("Correlation coefficient of all variables")
df_correlation_matrix: pd.DataFrame = \
    df_mobile_prices[[column for column in df_mobile_prices.columns
                      if column not in ['Brand', 'Model', 'Camera_mp',
                                        'Price_categorical', 'Price_$',
                                        'Storage', 'RAM']]].corr()
print("Correlation matrix")
print(df_correlation_matrix)

print("Let's fill the diagonal which value is always 1 because if it "
      "the correlation of a field with itself")
np.fill_diagonal(df_correlation_matrix.values, np.NaN)
print(df_correlation_matrix)
print("Now let's convert the matrix to absolute values, so we focus on "
      "the biggest correlation coefficient")
df_correlation_matrix_abs: pd.DataFrame = df_correlation_matrix.abs()
print(df_correlation_matrix_abs)
df_correlation_matrix_abs_unstacked: pd.DataFrame \
    = df_correlation_matrix_abs.unstack()
sorted_correlation = \
    df_correlation_matrix_abs_unstacked.sort_values(kind="quicksort",
                                                    ascending=False)
print("The two fields with the highest correlation are: ")
print(sorted_correlation.idxmax())
print("")
print("Let's work out the correlation coefficient again and the p-value")
pearson_corr, p_value = pearsonr(
    df_mobile_prices[sorted_correlation.idxmax()[0]],
    df_mobile_prices[sorted_correlation.idxmax()[1]])
print("Correlation coeff = {0}".format(pearson_corr))
print("p value = {0}".format(p_value))
if pearson_corr >= 0.75:
    print("There is strong positive correlation")
elif 0.75 > pearson_corr >= 0.5:
    print("There is some positive correlation")
elif 0.5 > pearson_corr > 0:
    print("There is very light positive correlation")
elif pearson_corr == 0:
    print("There is no correlation")
elif 0 > pearson_corr > -0.5:
    print("There is very light negative correlation")
elif -0.5 > pearson_corr >= -0.75:
    print("There is very some negative correlation")
elif -0.75 >= pearson_corr >= -1:
    print("There is very strong negative correlation")

if p_value <= 0.001:
    print("There is strong certainty of correlation")
elif 0.001 < p_value <= 0.05:
    print("There is moderate certainty of correlation")
elif 0.05 < p_value <= 0.1:
    print("There is weak certainty of correlation")
elif p_value > 0.1:
    print("There is no certainty of correlation")