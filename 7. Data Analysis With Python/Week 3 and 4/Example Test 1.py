import numpy as np
import pandas as pd

print(np.linspace(100, 300, 4))


df = pd.DataFrame({"price": [1000, 400, 500, 2000, 3000],
                   "type": ["Laptop", "Computer", "Tablet", "Computer",
                            "Laptop"]})


categories_thresholds: [int] = np.linspace(df["price"].min(), df["price"].max(), 4)
df["price-dummy"] = pd.cut(df["price"], categories_thresholds,
                           labels=["low", "medium", "high"],
                           include_lowest=True)
print(df)

df_dummies: pd.DataFrame = pd.get_dummies(df["type"])

for column_name in df_dummies.columns:
    df[column_name] = df_dummies[column_name]

print(df)

import seaborn as sns
import matplotlib.pyplot as plt


sns.boxplot(x='type', y='price', data=df)
plt.show()

df["hz"] = [300,200,200,250,100]


plt.scatter(x=df["price"], y=df["hz"])
plt.title("Price vs Hz")
plt.xlabel("Price")
plt.ylabel("Hz")
plt.show()

df_test = df[["price", "type", "hz"]]
df_grouped: pd.DataFrame = df_test.groupby(['type'], as_index=False).mean()
print(df_grouped)

