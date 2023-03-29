#Regresion Lineal simple

dataset = read.csv("Salary_Data.csv")

install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset,split == TRUE)
testing_set = subset(dataset,split == FALSE)

# Ajustar el modelo de regresion lineal simple
regressor = lm(formula = Salary ~ YearsExperience,data = training_set)

#Predecir resultados con el conjunto de test
y_pred = predict(regressor,newdata = testing_set)

#visualizacion de los resultados 
install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,y = training_set$Salary),colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = "blue") +
  ggtitle("Sueldo vs años de experiencia") +
  xlab("Años de experiencia") +
  ylab("Sueldo")

ggplot() +
  geom_point(aes(x = testing_set$YearsExperience,y = testing_set$Salary),colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = "blue") +
  ggtitle("Sueldo vs años de experiencia") +
  xlab("Años de experiencia") +
  ylab("Sueldo")