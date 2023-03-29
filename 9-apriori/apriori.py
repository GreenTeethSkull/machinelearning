import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([dataset.values[i, j] for j in range(0,20)])

#Entrenar el algoritmo de Apryoro
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualizacion de los resultados
results = list(rules)