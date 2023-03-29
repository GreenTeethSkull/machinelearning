#XGBoost
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Churn_Modelling.csv")

#Pre procesado de datos
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import make_column_transformer
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
onehotencoder = make_column_transformer((OneHotEncoder(), [1]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#Ajustr el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

#Evaluar el modelo y calcular predicciones finales
y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)