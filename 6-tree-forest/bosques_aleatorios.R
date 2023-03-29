dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x = dataset[1], y = dataset$Salary, ntree = 300)

y_pred = predict(regression, newdata = data.frame(Level = 6.5))

library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x =  dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid,y = predict(regression, newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Prediccion del sueldo en funcion del nivel") +
  xlab("Nivel del empleado") +
  ylab("Sueldo en $")
