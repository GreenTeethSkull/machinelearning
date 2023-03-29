dataset = read.csv("Market_Basket_Optimisation.csv",header = FALSE)
install.packages("arules")

library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",sep =",",rm.duplicates = TRUE)
itemFrequencyPlot(dataset, topN = 10)

#Entrenar algoritmo apriori
rules = apriori(data = dataset, 
                parameter = list(support = 0.004, confidence = 0.2))

#Visualizacion de los resultados
inspect(sort(rules,by = 'lift')[1:10])

