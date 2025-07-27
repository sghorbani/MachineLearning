
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns

#, r2_score3

columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin", "car_name"]

df = pd.read_csv('auto-mpg.data',
                 delim_whitespace=True,
                 names=columns,
                 na_values='?',
                 quotechar='"')

print("******Missing values:")
print(df.isnull().sum())

print("******datafarame info:")
print(df.info())

  

print("******horsepower missed value before:")
print(df['horsepower'].isnull().sum())   

median_hp = df['horsepower'].median() 
df.fillna({'horsepower': median_hp}, inplace=True)

print("******horsepower missed value after:")
print(df['horsepower'].isnull().sum())  # باید صفر شود

print("******df.shape:")
print(f"rows:{df.shape[0]} columns: {df.shape[1]}")

print("******df.head:")
print(df.head())
 
df.to_csv('auto-mpg-filled.csv', index=False)



# One-Hot Encoding برای origin
df = pd.get_dummies(df, columns=['origin'], prefix='origin')

# ساده‌سازی car_name: استخراج برند و One-Hot
df['brand'] = df['car_name'].str.split().str[0]
popular_brands = df['brand'].value_counts().nlargest(10).index
df['brand'] = df['brand'].apply(lambda x: x if x in popular_brands else 'other')
df = pd.get_dummies(df, columns=['brand'], prefix='brand')

# اگر cylinders رو بخوای one-hot کنی:
df = pd.get_dummies(df, columns=['cylinders'], prefix='cyl')

# حذف car name اصلی
df = df.drop('car_name', axis=1)
 
# X = df.drop(['mpg'], axis=1)  # تمام ستون‌ها به‌جز mpg
y = df['mpg']
X = df.drop('mpg', axis=1)
 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 score: {r2:.2f}")

residuals = y_test - y_pred

# رسم نمودار پسماندها
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='dashed')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs. Predicted MPG')
plt.show()


plt.figure(figsize=(8, 4))
sns.kdeplot(residuals, fill=True, color='green', alpha=0.8)
plt.xlabel('Residual')
plt.ylabel('Density')
plt.title('Distribution of Residuals (Kernel Density Plot)')
plt.show()
