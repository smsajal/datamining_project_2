library("e1071")
library("dummies")
library(pROC)
require(ROCR)

rm(list=ls())
setwd("/Users/Sherlock/Box Sync/PSU Spr19/STAT 557/datamining_project_2/data/")

#### READ DATA
#data.test=read.csv("testing.csv")
data.train=read.csv("training.csv")
data=read.csv("training.csv")

#### SEPARATE DATA INTO TEST AND TRAINING
set.seed(1)
train = sample(1:nrow(data), size = 0.75*nrow(data))
data.train = data[train,]
data.test = data[-train,]

### FOR ONE-AGAINST-ALL ROC CURVE
dummytrain <- dummy.data.frame(data.train)
data.train$class = dummytrain$classwater
print(data.train)

dummytest <- dummy.data.frame(data.test)
data.test$class = dummytest$classwater
print(data.test)

x <- subset(data.test, select=-class)
y <- data.test$class

xtrain <- subset(data.train, select=-class)
ytrain <- data.train$class

#svm_model <- svm(class ~ ., data=data.train, type="C-classification", probability=T)
svm_model <- svm(class ~ ., data=data.train)
summary(svm_model)

#system.time(svm_tune <- tune(svm, train.x=x, train.y=y, 
#                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2))))

# svm_tune <- tune(svm, train.x=x, train.y=y, 
#                  kernel="radial", ranges=list(cost= 10^(-3:1), gamma=10^(-5:-1)))
#print(svm_tune)

# best parameters found are:
#cost 10
#gamma 0.5

system.time(svm_model_after_tune <- svm(class ~ ., data=data.train, kernel="radial", cost=10, gamma=0.5))
summary(svm_model_after_tune)

model = svm_model_after_tune

library(ROCR)
system.time(pred <- predict(model, x,  decision.values = TRUE))
svm.roc <- prediction(attributes(pred)$decision.values, y)
svm.auc <- performance(svm.roc, 'tpr', 'fpr') 
aucsvm <- performance(svm.roc, 'auc') 
print(aucsvm)
plot(svm.auc, col = 2, main="ROC for Class water")

### ERROR Rate for multiclass
# d = table(pred,y)
# error = 1-sum(diag(d))/sum(d) #incorrect classification
# print(paste('Error rate', error))
# print(paste('Accuracy', 1-error))

### RESULTS FOR ONE-VS-ALL SVM
## proc roc
system.time(pred <- predict(model, x))
roc1 = roc(y, pred)
print(auc(roc1))
plot(roc1, col = 1, lty = 2, main = "ROC for water Class")

# fitted.results <- predict(model,newdata=data.test,type='probabilities')
# fitted.results <- ifelse(fitted.results > 0.5,1,0)
# misClasificError <- mean(fitted.results != (data.test$class))
# print(paste('Accuracy',1-misClasificError))
# print(paste('Error Rate ',misClasificError))

