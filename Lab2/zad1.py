import pandas as pd
import numpy as np

df = pd.read_csv("iris_with_errors.csv")

# a)
# Używamy metody isnull().sum() do policzenia brakujących wartości w każdej kolumnie
missing_data = df.isnull().sum()
print("Brakujace dane:\n", missing_data)

# Wyświetl statystyki bazy danych z błędami
# Metoda describe() daje podstawowe statystyki dla wszystkich kolumn numerycznych
print("\nDataframe statistics:\n", df.describe())

# b)
# Wybieramy kolumny numeryczne, a następnie dla każdej kolumny sprawdzamy, czy wartości są w zakresie (0, 15)
# Jeśli wartość jest poza tym zakresem, zastępujemy ją średnią z danej kolumny
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    df[column] = np.where((df[column] < 0) | (df[column] > 15), df[column].mean(), df[column])

# c)
# Definiujemy poprawne nazwy gatunków, a następnie dla każdej wartości w kolumnie 'variety' sprawdzamy, czy jest ona jednym z poprawnych gatunków
# Jeśli nie, zastępujemy ją wartością 'Unknown'
species = ['Setosa', 'Versicolor', 'Virginica']
df['variety'] = df['variety'].apply(lambda x: 'Unknown' if x not in species else x)

# Poprawione dane
print("\nCorrected data:\n", df)import pandas as pd
import numpy as np

df = pd.read_csv("iris_with_errors.csv")

# a)
# Używamy metody isnull().sum() do policzenia brakujących wartości w każdej kolumnie
missing_data = df.isnull().sum()
print("Brakujace dane:\n", missing_data)

# Wyświetl statystyki bazy danych z błędami
# Metoda describe() daje podstawowe statystyki dla wszystkich kolumn numerycznych
print("\nDataframe statistics:\n", df.describe())

# b)
# Wybieramy kolumny numeryczne, a następnie dla każdej kolumny sprawdzamy, czy wartości są w zakresie (0, 15)
# Jeśli wartość jest poza tym zakresem, zastępujemy ją średnią z danej kolumny
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    df[column] = np.where((df[column] < 0) | (df[column] > 15), df[column].mean(), df[column])

# c)
# Definiujemy poprawne nazwy gatunków, a następnie dla każdej wartości w kolumnie 'variety' sprawdzamy, czy jest ona jednym z poprawnych gatunków
# Jeśli nie, zastępujemy ją wartością 'Unknown'
species = ['Setosa', 'Versicolor', 'Virginica']
df['variety'] = df['variety'].apply(lambda x: 'Unknown' if x not in species else x)

# Poprawione dane
print("\nCorrected data:\n", df)
