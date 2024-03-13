from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
import numpy as np

# Wczytaj dane
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Zainicjuj klasyfikatory
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn11 = KNeighborsClassifier(n_neighbors=11)
gnb = GaussianNB()

# Wytrenuj klasyfikatory
knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn11.fit(X_train, y_train)
gnb.fit(X_train, y_train)

# Dokonaj ewaluacji klasyfikatorów
classifiers = [knn3, knn5, knn11, gnb]
names = ["3-NN", "5-NN", "11-NN", "Naive Bayes"]

for clf, name in zip(classifiers, names):
    y_pred = clf.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))