import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import make_column_transformer
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)

#Evitar la trampa de las variables ficticias
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Ajustar el modelo de regresión lineal multiple 
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

#prediccion de resultados en el conjunto de testing
y_pred = regression.predict(x_test)

#construir el modelo optimo de RLM utilizando eliminacion hacia atrás
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values =  x, axis = 1)
SL = 0.05
x_opt = x[:,[0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regression_OLS.summary()

x_opt = x[:,[0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regression_OLS.summary()

x_opt = x[:,[0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regression_OLS.summary()

x_opt = x[:,[0,3,5]].tolist()
regression_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regression_OLS.summary()

