import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")
print(df.head())


X = df[['speed_kmh']]
y = df['fuel_consumption_l_per_100km']

print("Features:\n", X)
print("Target:\n", y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


degrees = range(1, 10)
best_degree = None
best_mse = float('inf')
best_model = None
best_poly = None

print("\nОцінка моделей різного степеня:")

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Степінь {d}: MSE={mse:.4f}, MAE={mae:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_degree = d
        best_model = model
        best_poly = poly

print(f"\nНайкращий степінь полінома: {best_degree}")
print(f"Найменший MSE: {best_mse:.4f}")


plt.scatter(X, y, label="Дані")


X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_range_poly = best_poly.transform(X_range)
y_range_pred = best_model.predict(X_range_poly)

plt.plot(X_range, y_range_pred, color='red',
         label=f"Polynomial Regression (degree={best_degree})")

plt.xlabel("Швидкість (км/год)")
plt.ylabel("Витрати пального (л/100 км)")
plt.title("Залежність витрат пального від швидкості")
plt.legend()
plt.show()


speeds_to_predict = np.array([[35], [95], [140]])
speeds_poly = best_poly.transform(speeds_to_predict)
predictions = best_model.predict(speeds_poly)

print("\nПрогнози витрат пального:")
for s, p in zip([35, 95, 140], predictions):
    print(f"{s} км/год → {p:.2f} л/100 км")
