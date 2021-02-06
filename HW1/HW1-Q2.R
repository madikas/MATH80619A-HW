library(glmnet)
library(caret)
library(MASS)
#Read datasets
trainset <- read.csv("C://HEC/Advanced StatLearning/HW/MATH80619A-HW/HW1/data/music_origin_lat_train_set.csv", header=TRUE) 
testset <- read.csv("C://HEC/Advanced StatLearning/HW/MATH80619A-HW/HW1/data/music_origin_lat_test_set.csv", header=TRUE)
#Subsets of x
xtrain <- trainset[,1:68]
xtest <- testset[,1:68]

#OLS with all variables as benchmark
lm.fit <- lm(y~., data=trainset)

lm.predict <- predict(lm.fit, newdata=testset)
#Aggregated data frame of different models' performance
results <- data.frame("OLS", mean((lm.predict-testset$y)^2), 
                      mean(abs(lm.predict-testset$y)))
names(results) <- c("Model", "MSE", "MAE")

#Stepwise, forward, backward based on AIC
set.seed(123)
fullmodel <- lm(y~., data=trainset)
emptymodel <- lm(y~1, data=trainset)
backward <- stepAIC(fullmodel,direction="backward", k=2)
forward <- stepAIC(emptymodel,direction="forward",scope=list(upper=fullmodel,lower=emptymodel), k=2)
stepwise <- stepAIC(emptymodel,direction="both",scope=list(upper=fullmodel,lower=emptymodel), k=2)

backward.predict <- predict(backward, newdata = testset)
forward.predict <- predict(forward, newdata = testset)
stepwise.predict <- predict(stepwise, newdata = testset)

results[nrow(results) + 1,] = list(Model="backward", MSE=mean((backward.predict-testset$y)^2), 
                                   MAE=mean(abs(backward.predict-testset$y)))
results[nrow(results) + 1,] = list(Model="forward", MSE=mean((forward.predict-testset$y)^2), 
                                   MAE=mean(abs(forward.predict-testset$y)))
results[nrow(results) + 1,] = list(Model="stepwise", MSE=mean((stepwise.predict-testset$y)^2), 
                                   MAE=mean(abs(stepwise.predict-testset$y)))

#Ridge Regression
ridge.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0)
plot(ridge.fit)

predridge <- predict(ridge.fit, new=as.matrix(xtest), s="lambda.min")

results[nrow(results) + 1,] = list(Model="Ridge Regression", MSE=mean((predridge-testset$y)^2), 
                                   MAE=mean(abs(predridge-testset$y)))

#Lasso Regression
lasso.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 1)
plot(lasso.fit)

predlasso <- predict(lasso.fit, new=as.matrix(xtest), s="lambda.min")

results[nrow(results) + 1,] = list(Model="Lasso Regression", MSE=mean((predlasso-testset$y)^2), 
                                   MAE=mean(abs(predlasso-testset$y)))

#Elastic net with alpha=0.5, 0.2 and 0.8
elnet0.2.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.2)
elnet0.5.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.5)
elnet0.8.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.8)

predelnet0.2 <- predict(elnet0.2.fit, new=as.matrix(xtest), s="lambda.min")
predelnet0.5 <- predict(elnet0.5.fit, new=as.matrix(xtest), s="lambda.min")
predelnet0.8 <- predict(elnet0.8.fit, new=as.matrix(xtest), s="lambda.min")


results[nrow(results) + 1,] = list(Model="Elastic Net Alpha=0.2", MSE=mean((predelnet0.2-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.2-testset$y)))
results[nrow(results) + 1,] = list(Model="Elastic Net Alpha=0.5", MSE=mean((predelnet0.5-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.5-testset$y)))
results[nrow(results) + 1,] = list(Model="Elastic Net Alpha=0.8", MSE=mean((predelnet0.8-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.8-testset$y)))

#Elastic net with optimal alpha and lambda using caret
elnet.fit <- train(
  y ~., data = trainset, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)

optimal.alpha  <- elnet.fit$bestTune$alpha 
optimal.lambda <- elnet.fit$bestTune$lambda

optimal.elnet.fit <- glmnet(as.matrix(xtrain), trainset$y, alpha = optimal.alpha, lambda = optimal.lambda)

predoptimalelnet <- predict(optimal.elnet.fit, newx=as.matrix(xtest))

results[nrow(results) + 1,] = list(Model="Optimal Elastic Net", MSE=mean((predoptimalelnet-testset$y)^2), 
                                   MAE=mean(abs(predoptimalelnet-testset$y)))


#Relax Lasso Regression
relax.lasso.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 1, relax=TRUE)

predrelaxlasso <- predict(relax.lasso.fit, new=as.matrix(xtest), s="lambda.min", gamma="gamma.min")

results[nrow(results) + 1,] = list(Model="Relaxed Lasso", MSE=mean((predrelaxlasso-testset$y)^2), 
                                   MAE=mean(abs(predrelaxlasso-testset$y)))

results