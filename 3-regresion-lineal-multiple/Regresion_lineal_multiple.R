dataset = read.csv("50_Startups.csv")

dataset$State = factor(dataset$State,levels=c("New York","California","Florida"),labels = c(1,2,3))

#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
testing_set = subset(dataset,split == FALSE)

#Ajustar el modelo de regresion lineal multiple
regression = lm(formula = Profit ~ .,data = training_set)

#predecir los resultados con el conjunto de testing
y_pred = predict(regression,newdata = testing_set)

#construir un modelo optimo con eliminacion hacia atrás
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,data = dataset)
summary(regression)
