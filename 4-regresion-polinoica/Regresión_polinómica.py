import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)

x_grid_poly = poly_reg.fit_transform(x_grid)

plt.scatter(x,y,color = "red")
plt.plot(x,lin_reg.predict(x), color = "blue")
plt.plot(x_grid,lin_reg_2.predict(x_grid_poly),color = "green")
plt.title("Modelo de Regresion Lineal y Polinomica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()

#Prediccion de nuestros modelos
lin_reg.predict([[6.5]])
arreglo = poly_reg.fit_transform([[6.5]])
lin_reg_2.predict(arreglo)