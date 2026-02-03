print("A Shalini-24BAD409")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(
    r"C:\Users\SHALINI A\Downloads\archive (7).zip"
)
np.random.seed(42)
df["study_hours"] = np.random.randint(1, 8, len(df))
df["attendance"] = np.random.randint(60, 100, len(df))
df["sleep_hours"] = np.random.randint(4, 9, len(df))
le = LabelEncoder()
df["parental_education"] = le.fit_transform(df["parental level of education"])
df["test_prep"] = le.fit_transform(df["test preparation course"])
df["final_score"] = df[["math score","reading score","writing score"]].mean(axis=1)
X = df[["study_hours","attendance","parental_education","test_prep","sleep_hours"]]
y = df["final_score"]
X = SimpleImputer().fit_transform(X)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))
features = ["Study Hours","Attendance","Parental Education","Test Prep","Sleep Hours"]
for f, c in zip(features, model.coef_):
    print(f, ":", c)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()
plt.bar(features, model.coef_)
plt.title("Coefficient Comparison")
plt.show()
plt.hist(y_test - y_pred, bins=30)
plt.title("Residual Distribution")
plt.show()
print("Ridge R²:",
      r2_score(y_test, Ridge(alpha=1).fit(X_train,y_train).predict(X_test)))
print("Lasso R²:",
      r2_score(y_test, Lasso(alpha=0.1).fit(X_train,y_train).predict(X_test)))

