dataset = read.csv("Mall_Customers.csv")
x = dataset[,4:5]

#metodo del codo
set.seed(6)
wcss = vector()
for (i in 1:10) {
  wcss[i] <- sum(kmeans(x,i)$withinss)
}
plot(1:10,wcss,type = 'b', main = "Metodo del codo",
     xlab="Numero de clusters",
     ylab="WCSS")

#Aplicar el algoritmo de k-means con k optimo
set.seed(29)
classifier = kmeans(x,5,iter.max = 300,nstart = 10)

ver = classifier$cluster
#visualizacion de los clusters
install.packages("cluster")
library(cluster)
clusplot(x,
         ver,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 1,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "ingresos anuales",
         ylab = "Puntuacion")
