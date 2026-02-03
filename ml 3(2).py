print("A Shalini-24BAD409")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\SHALINI A\Downloads\archive (8).zip")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
X = df[["horsepower"]]
y = df["mpg"]
X = SimpleImputer().fit_transform(X)
for d in [2, 3, 4]:
    X_poly = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    X_poly = StandardScaler().fit_transform(X_poly)
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nDegree {d}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²:", r2_score(y_test, y_pred))
X_poly = PolynomialFeatures(4, include_bias=False).fit_transform(X)
X_poly = StandardScaler().fit_transform(X_poly)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
ridge = Ridge(alpha=1).fit(X_train, y_train)
print("\nRidge R²:", r2_score(y_test, ridge.predict(X_test)))
poly = PolynomialFeatures(3, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly = StandardScaler().fit_transform(X_poly)
model = LinearRegression().fit(X_poly, y)
idx = X[:, 0].argsort()
plt.scatter(X, y)
plt.plot(X[idx], model.predict(X_poly[idx]))
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression (Degree 3)")
plt.show()
