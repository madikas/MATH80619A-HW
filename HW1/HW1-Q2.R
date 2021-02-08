library(glmnet)
library(caret)
library(MASS)
#Read datasets
trainset <- read.csv("data/music_origin_lat_train_set.csv", header=TRUE) 
testset <- read.csv("data/music_origin_lat_test_set.csv", header=TRUE)
#Subsets of x
xtrain <- trainset[,1:68]
xtest <- testset[,1:68]
musicset <- rbind(trainset, testset)
xmusicset <- musicset[,1:68]
#OLS with all variables as benchmark
lm.fit <- lm(y~., data=trainset)

lm.predict <- predict(lm.fit, newdata=testset)
#Aggregated data frame of different models' performance
Q2results <- data.frame("OLS", mean((lm.predict-testset$y)^2), 
                      mean(abs(lm.predict-testset$y)))
names(Q2results) <- c("Model", "MSE", "MAE")

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

Q2results[nrow(Q2results) + 1,] = list(Model="backward", MSE=mean((backward.predict-testset$y)^2), 
                                   MAE=mean(abs(backward.predict-testset$y)))
Q2results[nrow(Q2results) + 1,] = list(Model="forward", MSE=mean((forward.predict-testset$y)^2), 
                                   MAE=mean(abs(forward.predict-testset$y)))
Q2results[nrow(Q2results) + 1,] = list(Model="stepwise", MSE=mean((stepwise.predict-testset$y)^2), 
                                   MAE=mean(abs(stepwise.predict-testset$y)))

#Ridge Regression
ridge.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0)
plot(ridge.fit)

predridge <- predict(ridge.fit, new=as.matrix(xtest), s="lambda.min")

Q2results[nrow(Q2results) + 1,] = list(Model="Ridge Regression", MSE=mean((predridge-testset$y)^2), 
                                   MAE=mean(abs(predridge-testset$y)))

#Lasso Regression
lasso.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 1)
plot(lasso.fit)

predlasso <- predict(lasso.fit, new=as.matrix(xtest), s="lambda.min")

Q2results[nrow(Q2results) + 1,] = list(Model="Lasso Regression", MSE=mean((predlasso-testset$y)^2), 
                                   MAE=mean(abs(predlasso-testset$y)))

#Elastic net with alpha=0.5, 0.2 and 0.8
elnet0.2.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.2)
elnet0.5.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.5)
elnet0.8.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 0.8)

predelnet0.2 <- predict(elnet0.2.fit, new=as.matrix(xtest), s="lambda.min")
predelnet0.5 <- predict(elnet0.5.fit, new=as.matrix(xtest), s="lambda.min")
predelnet0.8 <- predict(elnet0.8.fit, new=as.matrix(xtest), s="lambda.min")


Q2results[nrow(Q2results) + 1,] = list(Model="Elastic Net Alpha=0.2", MSE=mean((predelnet0.2-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.2-testset$y)))
Q2results[nrow(Q2results) + 1,] = list(Model="Elastic Net Alpha=0.5", MSE=mean((predelnet0.5-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.5-testset$y)))
Q2results[nrow(Q2results) + 1,] = list(Model="Elastic Net Alpha=0.8", MSE=mean((predelnet0.8-testset$y)^2), 
                                   MAE=mean(abs(predelnet0.8-testset$y)))

#Elastic net with optimal alpha and lambda using caret
alpha.grid <- seq(0, 1, 0.01)
srchGrd = expand.grid(.alpha = alpha.grid, .lambda = "all")

elnet.fit <- train(
  y ~., data = trainset, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10, tuneGrid = srchGrd
)

optimal.alpha  <- elnet.fit$bestTune$alpha 
optimal.lambda <- elnet.fit$bestTune$lambda

optimal.elnet.fit <- glmnet(as.matrix(xtrain), trainset$y, alpha = optimal.alpha, lambda = optimal.lambda)

predoptimalelnet <- predict(optimal.elnet.fit, newx=as.matrix(xtest))

Q2results[nrow(Q2results) + 1,] = list(Model="Optimal Elastic Net Caret", MSE=mean((predoptimalelnet-testset$y)^2), 
                                   MAE=mean(abs(predoptimalelnet-testset$y)))

#Elastic net with own function 
cv.optimalElasticNet <- function(seed, x, y, fold, alpha, lambda, xtrain, ytrain, xtest, ytest) {
  
  set.seed(seed)
  #Create equally sized folds
  folds <- cut(seq(1,nrow(x)),breaks=fold,labels=FALSE)
  #alpha lambda grid search
  searchgrid <- expand.grid(alpha, lambda)
  grid <- nrow(searchgrid)
  #Store intermediate results for MSE and MAE for alpha-lambda combinations
  tuneResults <- cbind(searchgrid, data.frame(matrix(0, ncol = fold , nrow=grid)))
  #Perform cross validation to tune alpha and lambda based on MSE like caret does
  for (i in 1:grid) {
    for(j in 1:fold){
      #Segment data by fold using the which() function 
      testIndexes <- which(folds==j,arr.ind=TRUE)
      testx <- x[testIndexes, ]
      testy <- y[testIndexes]
      trainx <- x[-testIndexes, ]
      trainy <- y[-testIndexes]
      #Fit and test the alpha-lambda combination for elastic net
      model.fit <- glmnet(as.matrix(trainx) , trainy , alpha = tuneResults[i,1] , lambda = tuneResults[i,2])
      model.pred <- predict(model.fit , as.matrix(testx) , s=model.fit$lambda)
      meanSquaredError <- mean((model.pred - testy)^2)
      
      tuneResults[i, j+2] <- meanSquaredError
    }
  }
  #Calculate average MSE and use that alpha-lambda for optimal model
  tuneResults$MSE_AVG <- rowMeans(tuneResults[,3:(3+fold-1)])
  optimalAlpha <- tuneResults[which(tuneResults$MSE_AVG==min(tuneResults$MSE_AVG)), 1]
  optimalLambda <- tuneResults[which(tuneResults$MSE_AVG==min(tuneResults$MSE_AVG)), 2]
  
  optimal.model.fit <- glmnet(as.matrix(xtrain) , ytrain , alpha = optimalAlpha , lambda = optimalLambda)
  optimal.model.pred <- predict(optimal.model.fit , as.matrix(xtest) , s=model.fit$lambda)
  
  optimalMSE <- mean((optimal.model.pred - ytest)^2)
  optimalMAE <- mean(abs(optimal.model.pred - ytest))
  
  #optimalResult <- list("Optimal Elastic Net No Caret", optimalMSE, optimalMAE)
  optimalResult <- data.frame("Optimal Elastic Net No Caret", optimalMSE, 
                              optimalMAE)
  names(optimalResult) <- c("Model", "MSE", "MAE")
  return(optimalResult)
}

alpha.grid <- seq(0, 1, 0.01)
lambda.grid <- seq(1,4, 0.1)
ElasticNetNoCaret <- cv.optimalElasticNet(123456, xmusicset, musicset$y, 10, alpha.grid, lambda.grid, xtrain, trainset$y, xtest, testset$y)
Q2results[nrow(Q2results) + 1,] = list(Model=ElasticNetNoCaret$Model,MSE=ElasticNetNoCaret$MSE, MAE=ElasticNetNoCaret$MAE)


#Relax Lasso Regression
relax.lasso.fit <- cv.glmnet(as.matrix(xtrain), trainset$y, alpha = 1, relax=TRUE)

predrelaxlasso <- predict(relax.lasso.fit, new=as.matrix(xtest), s="lambda.min", gamma="gamma.min")

Q2results[nrow(Q2results) + 1,] = list(Model="Relaxed Lasso", MSE=mean((predrelaxlasso-testset$y)^2), 
                                   MAE=mean(abs(predrelaxlasso-testset$y)))

