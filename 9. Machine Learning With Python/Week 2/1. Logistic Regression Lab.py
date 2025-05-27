import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

print("1. Opening our dataset")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain." \
      "cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/" \
      "Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
print(churn_df)
print(" ")

print("2. Data Preprocessing")
print("2.1 Filtering the dataset")
churn_df = churn_df[['tenure', 'age', 'address', 'ed', 'equip', 'churn']]
print("2.2 Setting y as integer")
churn_df['churn'] = churn_df['churn'].astype('int')

print("2.3 Getting only the variable x")
x_var = np.asarray(churn_df[['tenure', 'age', 'address', 'ed', 'equip']])

print("2.4 Getting the variable y")
y_var = np.asarray(churn_df['churn'])

print("2.5 Applying standard scaler to x")
x_norm = StandardScaler().fit(x_var).transform(x_var)

print("")

print("3. Splitting the dataset")
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_var,
                                                    test_size=0.2,
                                                    random_state=4)
print("")

print("4. Training Logistic Model")
log_reg_model: LogisticRegression = LogisticRegression().fit(x_train, y_train)

print("4.1 Predicting test values")
y_pred = log_reg_model.predict(x_test)

print("4.2 Predicting probabilities for test values of x")
y_pred_prob = log_reg_model.predict_proba(x_test)

coefficients = pd.Series(log_reg_model.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

print("5. Evalutating the model using log_loss")
pref_ev: float = log_loss(y_test, y_pred_prob)
print(pref_ev)
