import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("iris.csv")

# Podział na zbiór treningowy i testowy
train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=278782)

# Wyświetlenie  zbiorów testowych
print(test_set)
print(test_set.shape[0]) #45



# Podział na inputy i klasy
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# Funkcja klasyfikująca
def classify_iris(sl, sw, pl, pw):
    if pl <= 2.0:
        return "Setosa"
    elif pw > 1.7:
        return "Virginica"
    else:
        return "Versicolor"

# Testowanie klasyfikatora
good_predictions = 0
len = test_set.shape[0]
for i in range(len):
    if classify_iris(*test_inputs[i]) == test_classes[i]: # *test_inputs[i] - rozpakowanie listy
        good_predictions = good_predictions + 1


train_set_df = pd.DataFrame(train_set) #do ulepszenia classify_iris
print(train_set_df.describe())
print(good_predictions) # 1:( dla stergo classify_iris
print(good_predictions/len*100, "%") # 2.2222222222222223 % dla stergo classify_iris

# dla nowego classify_errors 43 oraz 95.55555555555556 %