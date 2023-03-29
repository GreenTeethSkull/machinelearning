import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))


from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(x,y)

seis_cinco = np.array([[6.5]]) 

seis = sc_x.transform(seis_cinco)

y_pred = regression.predict(seis)
y_pred = sc_y.inverse_transform(y_pred)

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
y_grid = regression.predict(x_grid)

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color = "red")
plt.plot(sc_x.inverse_transform(x_grid),sc_y.inverse_transform(y_grid),color = "blue")
plt.title("Modelo de regresion (SVR)")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en ($)")
plt.show()