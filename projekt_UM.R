install.packages("ggplot2")
install.packages("ggeffects")
install.packages("caTools")
install.packages("caret")
install.packages("magrittr")
install.packages("adabag")
install.packages("randomForest")
install.packages("traineR")
install.packages("DALEX")
library(ggplot2)
library(ggeffects)
library(caTools)
library(caret)
library(magrittr)
library(adabag)
library(randomForest)
library(traineR)
library(DALEX)

data <- Employee

summary(data)

edukacja <- table(data$Education)
edukacja
rok_zaczęcia <- table(data$JoiningYear)
rok_zaczęcia
miasto <- table(data$City)
miasto
poziom_płatności <- table(data$PaymentTier)
poziom_płatności
płeć <- table(data$Gender)
płeć
bez_przypisanej_pracy <- table(data$EverBenched)
bez_przypisanej_pracy
doświadczenie <- table(data$ExperienceInCurrentDomain)
doświadczenie
czy_zostawić <- table(data$LeaveOrNot)
czy_zostawić

#modelu regresji liniowej
model <- lm(LeaveOrNot ~., data = data)
summary(model)
#wykres efektów dla jednej zmiennej
efekty <- ggpredict(model, terms = c("ExperienceInCurrentDomain"))
plot(efekty)

#czy są braki danych
sum(is.na(data))

boxplot(data$Age)

data$LeaveOrNot <- as.numeric(data$LeaveOrNot)


set.seed(123)
prop <- 0.7
n <- nrow(data)
train_indices <- sample(1:n, prop * n)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

table(train_data$LeaveOrNot)
table(test_data$LeaveOrNot)

tuneGrid = expand.grid(C = seq(0, 2, length = 20)

c_num <- seq(0, 2, length = 20)
sensitivity_values <- numeric(length(c_num))


#model SVM
for(i in seq_along(c_num)){
model <- train(
  LeaveOrNot ~., data = train_data, method = "svmLinear",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(C = c_num),
  preProcess = c("center","scale")
  )
test_pred1 <- predict(knn_fit, newdata = test_data)
conf_matrix1 <- confusionMatrix(test_pred1, test_data$LeaveOrNot)
sensitivity_values[i] <- conf_matrix1[["byClass"]]["Specificity"]
}

# Wykres zależności czułości od C
plot(c_num, sensitivity_values, type = "b", 
     xlab = "Wartość C", ylab = "Specyficzność",
     main = "Zależność specyficzności od wartości C dla SVM liniowego")

plot(model)

predicted.classes <- model %>% predict(test_data)

mean(predicted.classes == test_data$LeaveOrNot)

model$bestTune

conf_matrix <- confusionMatrix(predicted.classes, test_data$LeaveOrNot)

#RANDOM FOREST

ntree_num <- seq(20, 500, by = 20)
sensitivity_values <- numeric(length(ntree_num))

for (i in seq_along(ntree_num)){
model_rf <- randomForest(LeaveOrNot ~ ., data = train_data, ntree = ntree)  
predictions <- predict(model_rf, newdata = test_data)
# Obliczanie metryk jakości
conf_matrix <- confusionMatrix(predictions, test_data$LeaveOrNot)
sensitivity_values[i] <- conf_matrix[["byClass"]]["Specificity"]
}

# Wykres zależności czułości od C
plot(ntree_num, sensitivity_values, type = "b", 
     xlab = "Ilośc drzew", ylab = "Specyficzność",
     main = "Zależność specyficzności od ilości drzew dla lasów losowych")


#METODA KNN

k_num <- seq(1,20, by = 2)
sensitivity_values <- numeric(length(k_num))

for(i in seq_along(k_num)){
  model_knn <- train(
    LeaveOrNot ~ ., 
    data = train_data, 
    method = "knn", 
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = expand.grid(k = k_num)
)
test_pred <- predict(knn_fit, newdata = test_data)
conf_matrix <- confusionMatrix(test_pred, test_data$LeaveOrNot)
sensitivity_values[i] <- conf_matrix[["byClass"]]["Balanced Accuracy"]
}
# Wykres zależności czułości od C
plot(k_num, sensitivity_values, type = "b", 
     xlab = "Wartość k", ylab = "Dokładność",
     main = "Zależność dokładności od wartości k dla metody KNN")


#INTERPRETOWALNOŚĆ

model_knn <- train(
  LeaveOrNot ~ ., 
  data = train_data, 
  method = "knn", 
  trControl = trainControl(method = "cv", number = 5)
)

train_data$LeaveOrNot <- as.numeric(train_data$LeaveOrNot)


explain_glm <- explain(model = model_knn,
                       data = train_data[,-9],
                       y = train_data$LeaveOrNot,
                       type = "classification",
                       label = "Logistic regression")

obs <- train_data[47,]
obs

pcp <- predict_profile(explainer = explain_glm,
                       new_observation = obs)

plot(pcp, variables = c("Age"))

plot(pcp, variables = c("Education"), 
     variable_type = "categorical", categorical_type = "bars")

plot(pcp, variables = c("JoiningYear"))

plot(pcp, variables = c("City"), 
     variable_type = "categorical", categorical_type = "bars")

plot(pcp, variables = c("PaymentTier"))

plot(pcp, variables = c("Gender"), 
     variable_type = "categorical", categorical_type = "bars")

plot(pcp, variables = c("EverBenched"), 
     variable_type = "categorical", categorical_type = "bars")

plot(pcp, variables = c("ExperienceInCurrentDomain"))

pdp <- model_profile(explainer = explain_glm, variables = "JoiningYear")
plot(pdp)

plot(pdp, geom = "profiles") + 
  ggtitle("PCP and PDP for JoiningYear")

pdp <- model_profile(explainer = explain_glm, variables = "JoiningYear", groups="ExperienceInCurrentDomain")
plot(pdp, geom = "profiles") + 
  ggtitle("PCP and PDP for JoiningYear") 

obs = train_data[29,]
obs

predict(model_knn, obs, type="prob")

bd1 <- predict_parts(explainer = explain_glm,
                     new_observation = obs,
                     type = "break_down_interactions", 
                     order = c("Gender", "PaymentTier", "JoiningYear", "City", "EverBenched", "ExperienceInCurrentDomain", "Education", "Age"))
p1 <- plot(bd1)
bd2 <- predict_parts(explainer = explain_glm,
                     new_observation = obs,
                     type = "break_down_interactions", 
                     order = c("City", "EverBenched", "PaymentTier", "ExperienceInCurrentDomain", "Gender", "Education", "JoiningYear", "Age"))
p2 <- plot(bd2)
bd3 <- predict_parts(explainer = explain_glm,
                     new_observation = obs,
                     type = "break_down_interactions", 
                     order = c("Age", "Education", "PaymentTier", "ExperienceInCurrentDomain", "EverBenched", "Gender", "City", "JoiningYear"))
p3 <- plot(bd3)
library(gridExtra)
grid.arrange(p1, p2, p3, nrow = 2)

shap <- predict_parts(explainer = explain_glm, 
                      new_observation = obs, 
                      type = "shap")
p1 <- plot(shap)
p2 <- plot(shap, show_boxplots = FALSE) 
grid.arrange(p1, p2, nrow = 1)
