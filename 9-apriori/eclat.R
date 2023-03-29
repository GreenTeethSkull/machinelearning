dataset = read.csv("Market_Basket_Optimisation.csv",header = FALSE)

library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",sep =",",rm.duplicates = TRUE)
itemFrequencyPlot(dataset, topN = 10)

#Entrenar algoritmo Eclat con el dataset
rules = eclat(data = dataset,parameter = list(support = 0.004,minlen = 2))

#Visualizar los resultados
inspect(sort(rules,by = 'support')[1:10])