import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('iris.csv')

# Konwertujemy 'variety' na kategorię
df['variety'] = pd.Categorical(df['variety'])

# Oryginalne dane
plt.figure(figsize=(6, 6))
plt.scatter(df['sepal.length'], df['sepal.width'], c=df['variety'].cat.codes)
plt.title('Oryginalne dane')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')
plt.show()

# Min-max
scaler = MinMaxScaler()
df[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])
plt.figure(figsize=(6, 6))
plt.scatter(df['sepal.length'], df['sepal.width'], c=df['variety'].cat.codes)
plt.title('Znormalizowane dane (min-max)')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')
plt.show()

# Z-score
scaler = StandardScaler()
df[['sepal.length', 'sepal.width']] = scaler.fit_transform(df[['sepal.length', 'sepal.width']])
plt.figure(figsize=(6, 6))
plt.scatter(df['sepal.length'], df['sepal.width'], c=df['variety'].cat.codes)
plt.title('Zeskalowane dane (z-score)')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')
plt.show()