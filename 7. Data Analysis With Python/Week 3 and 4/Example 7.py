import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df_usedcars: pd.DataFrame = pd.read_csv("./usedcars.csv")
print(df_usedcars['make'].value_counts())
df_usedcars_corr_matrix: pd.DataFrame = df_usedcars.corr(numeric_only=True)
df_usedcars
corr_max: float = 0
i_res: int = -1
j_res: int = -1

for i in range(0, len(df_usedcars_corr_matrix)):
    for j in range(0, len(df_usedcars_corr_matrix.columns)):
        if i != j:
            corr_aux = df_usedcars_corr_matrix.iloc[i,j]
            if corr_aux > corr_max:
                corr_max = corr_aux
                i_res = i
                j_res = j
print(corr_max)
print(i_res)
print(j_res)

