import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#Crear modelo de Regresión lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

#predecir el conjunto de test
y_pred = regression.predict(x_test)

#visualizar los resultados de entrenamiento
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,regression.predict(x_train),color = "blue")
plt.title("Sueldo vs años de experiencia")
plt.xlabel("años de trabajo")
plt.ylabel("sueldo")
plt.show()

#visualizar los resultados de test
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_train,regression.predict(x_train),color = "blue")
plt.title("Sueldo vs años de experiencia")
plt.xlabel("años de trabajo")
plt.ylabel("sueldo")
plt.show()