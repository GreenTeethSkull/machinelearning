import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#utilizar el dendrograma
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()

#Ajustar el clustering jerarquico
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean",linkage = "ward")
y_pred = hc.fit_predict(x)

plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c="red",label="Cluster 1")
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c="blue",label="Cluster 2")
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c="green",label="Cluster 3")
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100,c="brown",label="Cluster 4")
plt.scatter(x[y_pred == 4,0],x[y_pred == 4,1],s=100,c="cyan",label="Cluster 5")
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150,c="yellow",label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("ingresos anuales")
plt.ylabel("Puantuacion de gasto")
plt.legend()
plt.show