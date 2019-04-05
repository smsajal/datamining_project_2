#library(tree)
library(randomForest)
#library(gbm)
library(adabag)
library(pROC)
#library(ROCR)
#library(car)
library(dummies)

rm(list=ls())

setwd("/Users/rxh655/Documents/Spring2019/STAT557/Project2/projectCode/datamining_project_2/data")

cmDatatraining =read.csv("training.csv")
cmDatatest = read.csv("testing.csv")
classification.test = cmDatatest$class

#### TO SEPARATE TRAINING INTO TEST AND TRAINING
set.seed(1)
train=sample(1:nrow(cmDatatraining), .75*nrow(cmDatatraining))
cmDatatraining = cmDatatraining[train,]
cmDatatest=cmDatatraining[-train,]
classification.test = cmDatatest$class
########### Read data End #############

######### Random Forest #############

rf(cmDatatraining, cmDatatest)

rf <- function(cmDatatraining, cmDatatest)
{
  classification.test = cmDatatest$class
  
  set.seed(1)
  s1 = Sys.time()
  bag.sm = randomForest(cmDatatraining$class~.-cmDatatraining$class, data=cmDatatraining,  importance=TRUE)
  s2 = Sys.time()
  trainTime = s2-s1
  print(paste("Train time ", trainTime))
  
  s1 = Sys.time()
  predict.class = predict(bag.sm,newdata=cmDatatest)
  s2 = Sys.time()
  classifyTime = s2-s1
  print(paste("Classification time", classifyTime))
  #yhat.bag
  #table(yhat.bag, classification.test)
  print(paste('Accuracy', 1-mean(as.character(predict.class) != as.character(classification.test))))
  print(paste('Error rate: ', mean(as.character(predict.class) != as.character(classification.test))))
  
  attach(mtcars)
  par(mfrow=c(4,3))
  
  
  #ROC for train
  pred_prob = predict(bag.sm, newdata=cmDatatraining, type = "prob")
  
  pred = pred_prob[,1]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classfarm
  roc1 = roc(y, pred)
  print(paste('AUC: ', auc(roc1)))
  plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class (Train data)")
  
  pred = pred_prob[,2]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classforest
  roc2 = roc(y, pred)
  print(paste('AUC: ', auc(roc2)))
  plot(roc2, col = 1, lty = 2, main = "ROC for Forest Class (Train data)")
  
  pred = pred_prob[,3]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classgrass
  roc3 = roc(y, pred)
  print(paste('AUC: ', auc(roc3)))
  plot(roc3, col = 1, lty = 2, main = "ROC for Grass Class (Train data)")
  
  pred = pred_prob[,4]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classimpervious
  roc4 = roc(y, pred)
  print(paste('AUC: ', auc(roc4)))
  plot(roc4, col = 1, lty = 2, main = "ROC for Impervious Class (Train data)")
  
  pred = pred_prob[,5]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classorchard
  roc5 = roc(y, pred)
  print(paste('AUC: ', auc(roc5)))
  plot(roc5, col = 1, lty = 2, main = "ROC for Orchard Class (Train data)")
  
  pred = pred_prob[,6]
  dummytest <- dummy.data.frame(cmDatatraining)
  y = dummytest$classwater
  roc6 = roc(y, pred)
  print(paste('AUC: ', auc(roc6)))
  plot(roc6, col = 1, lty = 2, main = "ROC for Water Class (Train data)")
  
  print(paste("Average AUC ", mean(auc(roc1), auc(roc2), auc(roc3), auc(roc4), auc(roc5), auc(roc6))))
  
  #ROC for test
  pred_prob = predict(bag.sm, newdata=cmDatatest, type = "prob")
  
  pred = pred_prob[,1]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classfarm
  roc1 = roc(y, pred)
  print(paste('AUC: ', auc(roc1)))
  plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class (Test Data)")
  
  pred = pred_prob[,2]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classforest
  roc2 = roc(y, pred)
  print(paste('AUC: ', auc(roc2)))
  plot(roc2, col = 1, lty = 2, main = "ROC for Forest Class (Test Data)")

  pred = pred_prob[,3]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classgrass
  roc3 = roc(y, pred)
  print(paste('AUC: ', auc(roc3)))
  plot(roc3, col = 1, lty = 2, main = "ROC for Grass Class (Test Data)")
  
  pred = pred_prob[,4]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classimpervious
  roc4 = roc(y, pred)
  print(paste('AUC: ', auc(roc4)))
  plot(roc4, col = 1, lty = 2, main = "ROC for Impervious Class (Test Data)")
  
  pred = pred_prob[,5]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classorchard
  roc5 = roc(y, pred)
  print(paste('AUC: ', auc(roc5)))
  plot(roc5, col = 1, lty = 2, main = "ROC for Orchard Class (Test Data)")
  
  pred = pred_prob[,6]
  dummytest <- dummy.data.frame(cmDatatest)
  y = dummytest$classwater
  roc6 = roc(y, pred)
  print(paste('AUC: ', auc(roc6)))
  plot(roc6, col = 1, lty = 2, main = "ROC for Water Class (Test Data)")
  
  print(paste("Average AUC ", mean(auc(roc1), auc(roc2), auc(roc3), auc(roc4), auc(roc5), auc(roc6))))
  
  
}

######### Random Forest End #############

############# Boosting ##################

drawROCboost(trainingData = cmDatatraining, cmDatatest)

drawROCboost <- function(trainingData, testData)
{
  classification.test = cmDatatest$class
  
  s1 = Sys.time()
  adaboost = boosting(class~., data = trainingData, mfinal = 500)
  s2 = Sys.time()
  trainTime = s2-s1
  trainTime
  print(paste("Train time ", trainTime))
  
  s1 = Sys.time()
  pred.boost = predict.boosting(adaboost, newdata = testData, type = "prob")
  s2 = Sys.time()
  classifyTime = s2-s1
  classifyTime
  print(paste("Classification time", classifyTime))
  
  print(paste('Accuracy',1-mean(as.character(pred.boost$class) != as.character(classification.test))))
  print(paste("Error rate: ", mean(as.character(pred.boost$class) != as.character(classification.test))))
  
  
  attach(mtcars)
  par(mfrow=c(4,3))
  
  #ROC train
  pred.boost = predict.boosting(adaboost, newdata = trainingData, type = "prob")
  
  pred = (pred.boost$prob)[,1]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classfarm
  roc1 = roc(y, pred)
  print(paste('AUC: ', auc(roc1)))
  plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class (Train data)")
  
  pred = (pred.boost$prob)[,2]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classforest
  roc2 = roc(y, pred)
  print(paste('AUC: ', auc(roc2)))
  plot(roc2, col = 1, lty = 2, main = "ROC for Forest Class (Train data)")
  
  pred = (pred.boost$prob)[,3]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classgrass
  roc3 = roc(y, pred)
  print(paste('AUC: ', auc(roc3)))
  plot(roc3, col = 1, lty = 2, main = "ROC for Grass Class (Train data)")
  
  pred = (pred.boost$prob)[,4]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classimpervious
  roc4 = roc(y, pred)
  print(paste('AUC: ', auc(roc4)))
  plot(roc4, col = 1, lty = 2, main = "ROC for Impervious Class (Train data)")
  
  pred = (pred.boost$prob)[,5]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classorchard
  roc5 = roc(y, pred)
  print(paste('AUC: ', auc(roc5)))
  plot(roc5, col = 1, lty = 2, main = "ROC for Orchard Class (Train data)")
  
  pred = (pred.boost$prob)[,6]
  dummytest <- dummy.data.frame(trainingData)
  y = dummytest$classwater
  roc6 = roc(y, pred)
  print(paste('AUC: ', auc(roc6)))
  plot(roc6, col = 1, lty = 2, main = "ROC for Water Class (Train data)")
  
  print(paste("Average AUC ", mean(auc(roc1), auc(roc2), auc(roc3), auc(roc4), auc(roc5), auc(roc6))))
  
  #ROC test
  
  pred.boost = predict.boosting(adaboost, newdata = testData, type = "prob")
  
  pred = (pred.boost$prob)[,1]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classfarm
  roc1 = roc(y, pred)
  print(paste('AUC: ', auc(roc1)))
  plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class (Test data)")
  
  pred = (pred.boost$prob)[,2]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classforest
  roc2 = roc(y, pred)
  print(paste('AUC: ', auc(roc2)))
  plot(roc2, col = 1, lty = 2, main = "ROC for Forest Class (Test data)")
  
  pred = (pred.boost$prob)[,3]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classgrass
  roc3 = roc(y, pred)
  print(paste('AUC: ', auc(roc3)))
  plot(roc3, col = 1, lty = 2, main = "ROC for Grass Class (Test data)")
  
  pred = (pred.boost$prob)[,4]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classimpervious
  roc4 = roc(y, pred)
  print(paste('AUC: ', auc(roc4)))
  plot(roc4, col = 1, lty = 2, main = "ROC for Impervious Class (Test data)")
  
  pred = (pred.boost$prob)[,5]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classorchard
  roc5 = roc(y, pred)
  print(paste('AUC: ', auc(roc5)))
  plot(roc5, col = 1, lty = 2, main = "ROC for Orchard Class (Test data)")
  
  pred = (pred.boost$prob)[,6]
  dummytest <- dummy.data.frame(testData)
  y = dummytest$classwater
  roc6 = roc(y, pred)
  print(paste('AUC: ', auc(roc6)))
  plot(roc6, col = 1, lty = 2, main = "ROC for Water Class (Test data)")
  
  print(paste("Average AUC ", mean(auc(roc1), auc(roc2), auc(roc3), auc(roc4), auc(roc5), auc(roc6))))
}

?boosting
