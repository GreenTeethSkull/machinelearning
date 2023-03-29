import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#metodo del codo
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init = 10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

#aplicar el metodo de kmeans para segmentar la data
kmeans = KMeans(n_clusters = 5,init="k-means++",max_iter=300,n_init = 10, random_state=0)
y_pred = kmeans.fit_predict(x)

#visualizacion de los clusters
plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c="red",label="Cluster 1")
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c="blue",label="Cluster 2")
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c="green",label="Cluster 3")
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100,c="brown",label="Cluster 4")
plt.scatter(x[y_pred == 4,0],x[y_pred == 4,1],s=100,c="cyan",label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150,c="yellow",label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("ingresos anuales")
plt.ylabel("Puantuacion de gasto")
plt.legend()
plt.show