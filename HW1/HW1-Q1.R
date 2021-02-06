library(glmnet)
library(caret)
source("C://HEC/Advanced StatLearning/HW/MATH80619A-HW/HW1/ames_preprocess.R")
#a.Use the R library caret to simultaneously optimize the a and l parameters by cross
#validation in elastic net regression. Compute the MSE and MAE of the Ames data
#using elastic net regression with optimal parameters.
set.seed(123456)
alpha.grid <- seq(0, 1, 0.01)
srchGrd = expand.grid(.alpha = alpha.grid, .lambda = "all")
elnet.fit <- train(
  Sale_Price ~., data = amesdumtrain, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10, tuneGrid = srchGrd
)

optimal.alpha  <- elnet.fit$bestTune$alpha 
optimal.lambda <- elnet.fit$bestTune$lambda

optimal.elnet.fit <- glmnet(xdumtrain, amesdumtrain$Sale_Price, alpha = optimal.alpha, lambda = optimal.lambda)

predoptimalelnet <- predict(optimal.elnet.fit, newx=xdumtest)

results <- data.frame("Optimal Elastic Net", mean((predoptimalelnet-amesdumtest$Sale_Price)^2), 
                      mean(abs(predoptimalelnet-amesdumtest$Sale_Price)))
names(results) <- c("Model", "MSE", "MAE")

results

#b.Write your own code in order to simultaneously optimize the a and l parameters by
#cross validation. Compute the MAE and MSE on the Ames data and compare your
#answer with a).
#Randomly shuffle the data
amesdata<-ames[sample(nrow(ames)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(amesdata)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segment data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- amesdata[testIndexes, ]
  trainData <- amesdata[-testIndexes, ]
  #Use the test and train data partitions however you desire...
}

#c.Compare the results in a) and b) with the other methods (see slide 83)