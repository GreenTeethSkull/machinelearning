dataset = read.csv("Mall_Customers.csv")
x = dataset[,4:5]

#Utilizar el dendrograma
dendrogram = hclust(dist(x, method = "euclidean"),method = "ward.D")
plot(dendrogram,
     main = "Dendrograma",
     xlab = "Clientes del centro comercial",
     ylab = "Distancia Euclidea")

#Ajustar el clustering jerarquico
hc = hclust(dist(x, method = "euclidean"),method = "ward.D")
y_hc=cutree(hc, k = 5)

#Visualizar los clusters
library(cluster)
clusplot(x,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 1,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "ingresos anuales",
         ylab = "Puntuacion")