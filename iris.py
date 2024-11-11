import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import time

# Medir el tiempo de procesamiento total
start_time = time.time()

# Cargar el dataset proporcionado
file_path = 'Iris.csv'
dataset = pd.read_csv(file_path)

# Mostrar las primeras filas para entender mejor el dataset
print(dataset.head())

# Verificar información del dataset
print(dataset.info())

# Verificar si hay valores nulos
print(dataset.isnull().sum())

# Preparar los datos para el modelo
# Species es la variable objetivo, mientras que los demás campos son características
X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = dataset['Species']

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión
modelo = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Visualizar el árbol de decisión
plt.figure(figsize=(20, 10))
tree.plot_tree(modelo, filled=True, feature_names=X.columns, class_names=y.unique())
plt.show()

# Evaluar el modelo
accuracy = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# Medir el tiempo de procesamiento total
end_time = time.time()
processing_time = end_time - start_time
print(f"Tiempo total de procesamiento: {processing_time:.2f} segundos")
