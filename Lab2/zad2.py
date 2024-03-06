import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj dane
df = pd.read_csv("iris.csv")

# Wyodrębnij cechy numeryczne
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
x = df.loc[:, features].values

# Wykonaj PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# Stwórz DataFrame dla principal components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Dodaj kolumnę z gatunkami
finalDf = pd.concat([principalDf, df[['variety']]], axis = 1)

# wariacja
print('Explained variance ratio:', pca.explained_variance_ratio_)

# wykres
plt.figure(figsize = (8,8))
sns.scatterplot(x="principal component 1", y="principal component 2", hue="variety", data=finalDf)
plt.show()

