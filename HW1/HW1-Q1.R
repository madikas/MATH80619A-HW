library(glmnet)
library(caret)
source("ames_preprocess.R")
#a.Use the R library caret to simultaneously optimize the a and l parameters by cross
#validation in elastic net regression. Compute the MSE and MAE of the Ames data
#using elastic net regression with optimal parameters.
set.seed(123456)
alpha.grid <- seq(0, 1, 0.01)
srchGrd <- expand.grid(.alpha = alpha.grid, .lambda = "all")
elnet.fit <- train(
  Sale_Price ~., data = amesdumtrain, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10, tuneGrid = srchGrd
)

optimal.alpha  <- elnet.fit$bestTune$alpha 
optimal.lambda <- elnet.fit$bestTune$lambda

optimal.elnet.fit <- glmnet(xdumtrain, amesdumtrain$Sale_Price, alpha = optimal.alpha, lambda = optimal.lambda)

predoptimalelnet <- predict(optimal.elnet.fit, newx=xdumtest)

Q1results <- data.frame("Optimal Elastic Net Caret", mean((predoptimalelnet-amesdumtest$Sale_Price)^2), 
                      mean(abs(predoptimalelnet-amesdumtest$Sale_Price)), optimal.alpha, optimal.lambda)
names(Q1results) <- c("Model", "MSE", "MAE", "Alpha", "Lambda")

#b.Write your own code in order to simultaneously optimize the a and l parameters by
#cross validation. Compute the MAE and MSE on the Ames data and compare your
#answer with a).

cv.optimalElasticNet <- function(seed, x, y, fold, alpha, lambda, xtrain, ytrain, xtest, ytest) {
  
  set.seed(seed)
  
  #Create equally sized folds
  folds <- cut(seq(1,nrow(x)),breaks=fold,labels=FALSE)
  #alpha lambda grid search
  searchgrid <- expand.grid(alpha, lambda)
  grid <- nrow(searchgrid)
  #Store intermediate results for MSE and MAE for alpha-lambda combinations
  tuneResults <- cbind(searchgrid, data.frame(matrix(0, ncol = fold, nrow=grid)))
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
      model.fit <- glmnet(trainx , trainy , alpha = tuneResults[i,1] , lambda = tuneResults[i,2])
      model.pred <- predict(model.fit , testx , s=model.fit$lambda)
      meanSquaredError <- mean((model.pred - testy)^2)
      
      tuneResults[i, j+2] <- meanSquaredError
    }
  }
  #Calculate average MSE and use that alpha-lambda for optimal model
  tuneResults$MSE_AVG <- rowMeans(tuneResults[,3:(3+fold-1)])
  optimalAlpha <- tuneResults[which(tuneResults$MSE_AVG==min(tuneResults$MSE_AVG)), 1]
  optimalLambda <- tuneResults[which(tuneResults$MSE_AVG==min(tuneResults$MSE_AVG)), 2]
  
  optimal.model.fit <- glmnet(xtrain , ytrain , alpha = optimalAlpha , lambda = optimalLambda)
  optimal.model.pred <- predict(optimal.model.fit , xtest , s=model.fit$lambda)
  
  optimalMSE <- mean((optimal.model.pred - ytest)^2)
  optimalMAE <- mean(abs(optimal.model.pred - ytest))
  
  optimalResult <- data.frame("Optimal Elastic Net No Caret", optimalMSE, 
                              optimalMAE, optimalAlpha, optimalLambda)
  names(optimalResult) <- c("Model", "MSE", "MAE", "Alpha", "Lambda")
  return(optimalResult)
}

#Run 10fold CV Elastc Net

lambda.grid <- seq(1,4, 0.1)
ElasticNetNoCaret <- cv.optimalElasticNet(123456, xdum, amesdum$Sale_Price, 10, alpha.grid, lambda.grid, xdumtrain, amesdumtrain$Sale_Price, xdumtest, amesdumtest$Sale_Price)
Q1results[nrow(Q1results) + 1,] = list(Model=ElasticNetNoCaret$Model, MSE=ElasticNetNoCaret$MSE, 
                                       MAE=ElasticNetNoCaret$MAE, Alpha=ElasticNetNoCaret$Alpha, Lambda=ElasticNetNoCaret$Lambda)
