
#Como importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar el dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#tratamiento de los NAN
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import make_column_transformer
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#dividir el dataset en entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)