import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("./assets/electricity.csv")
print(df.head())

X = df[['temperature', 'humidity', 'hour', 'is_weekend']]  
y = df['consumption']                                      

print("Features: ", X)
print("Target:", y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


sample_input = pd.DataFrame([{
    'temperature': 15,     
    'humidity': 55,        
    'hour': 19,           
    'is_weekend': 0       
}])

predicted = model.predict(sample_input)
print(f"Прогнозоване споживання електроенергії: {predicted[0]:.2f} кВт·год")


y_pred = model.predict(X_test)


mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")


plt.scatter(y_test, y_pred)
plt.xlabel("Справжнє споживання (кВт·год)")
plt.ylabel("Прогнозоване споживання (кВт·год)")
plt.title("Справжнє vs Прогнозоване електроспоживання")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
