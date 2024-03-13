from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
import numpy as np

# Wczytaj dane
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Zainicjuj drzewo decyzyjne
clf = DecisionTreeClassifier()

# Wytrenuj drzewo decyzyjne
clf.fit(X_train, y_train)

# Dokonaj ewaluacji klasyfikatora
y_pred = clf.predict(X_test)

# Wyświetl dokładność klasyfikatora
print("Accuracy:", accuracy_score(y_test, y_pred))

# Wyświetl macierz błędów
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#wyszlo tak samo bo jakos odgadlem wartosci w poprzednim 