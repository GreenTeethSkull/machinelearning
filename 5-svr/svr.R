dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

install.packages("e1071")
library(e1071)

regression = svm(formula = Salary ~ .,
                 data = dataset,
                 type = "eps-regression",
                 kernel = "radial")

y_pred = predict(regression, newdata = data.frame(Level = 6.5))


library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level), 0.1)
dataset2 = data.frame(Level = x_grid,Level2 = x_grid^2,Level3 = x_grid^3)
ggplot() +
  geom_point(aes(x =  dataset$Level, y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level,y = predict(regression, newdata = dataset)),
            color = "blue") +
  ggtitle("Prediccion del sueldo en funcion del nivel") +
  xlab("Nivel del empleado") +
  ylab("Sueldo en $")