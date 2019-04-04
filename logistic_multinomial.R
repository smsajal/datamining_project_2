library(dummies)
library(car)
library(pROC)
library(nnet)

rm(list=ls())
setwd("/Users/Sherlock/Box Sync/PSU Spr19/STAT 557/datamining_project_2/data/")

data.test=read.csv("testing.csv")
data.train=read.csv("training.csv")
data = read.csv("training.csv")

#### TO SEPARATE TRAINING INTO TEST AND TRAINING
set.seed(1)
train = sample(1:nrow(data), size = 0.75*nrow(data))
data.train = data[train,]
data.test = data[-train,]
print(dim(data.train))


### VARIABLE SELECTION
#multinomModel = step(multinom(class ~ ., data=data.train), direction = "backward")
multinomModel <- multinom(class ~ ., data=data.train) # multinom Model
summary (multinomModel) # model summary
print(summary(multinomModel)$coefficients)


predicted_scores <- predict (multinomModel, data.test, "probs") # predict on new data
predicted_class <- predict (multinomModel, data.test)
table(predicted_class, data.test$class)
print(paste('Accuracy',1-mean(as.character(predicted_class) != as.character(data.test$class))))

########

#########

#farm forest grass impervious orchard water

attach(mtcars)
par(mfrow=c(3,3))
### ROC Curve
pred = (predicted_scores)[,1]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classfarm
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class")

pred = (predicted_scores)[,2]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classforest
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Forest Class")

pred = (predicted_scores)[,3]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classgrass
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Grass Class")

pred = (predicted_scores)[,4]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classimpervious
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Impervious Class")

pred = (predicted_scores)[,5]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classorchard
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Orchard Class")

pred = (predicted_scores)[,6]
dummytest <- dummy.data.frame(data.test)
y = dummytest$classwater
roc1 = roc(y, pred)
print(paste('AUC: ', auc(roc1)))
plot(roc1, col = 1, lty = 2, main = "ROC for Water Class")
