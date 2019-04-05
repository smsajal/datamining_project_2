library(tree)
library(randomForest)
library(gbm)
library(adabag)
library(pROC)
library(ROCR)
library(car)
library(caret)
setwd("/Users/rxh655/Documents/Spring2019/STAT557/Project2/projectCode/datamining_project_2/data")

cmDatatraining =read.csv("training.csv")
cmDatatest = read.csv("testing.csv")
classification.test = cmDatatest$class

set.seed(1)
train=sample(1:nrow(cmDatatraining), .75*nrow(cmDatatraining))
test=cmDatatraining[-train,]
test.class=test$class
########### Read data End #############

#system.time(tree(cmDatatraining$class~.-cmDatatraining$class,cmDatatraining))
s1 <- Sys.time()
tree.cm = tree(cmDatatraining$class~.-cmDatatraining$class,cmDatatraining)
s2 <- Sys.time()
s2-s1
tree.pred=predict(tree.cm, cmDatatest ,type ="class")
table(tree.pred , classification.test)

(36+71+7+30+7)/(301)

######### Random Forest #############
set.seed(1)
s1 = Sys.time()
bag.sm = randomForest(cmDatatraining$class~.-cmDatatraining$class,data=cmDatatraining,  importance=TRUE)
s2 = Sys.time()
trainTime = s2-s1
trainTime
bag.sm

s1 = Sys.time()
yhat.bag = predict(bag.sm,newdata=cmDatatest)
s2 = Sys.time()
classifyTime = s2-s1
yhat.bag
table(yhat.bag, classification.test)
classifyTime

p = predict(bag.sm, newdata = cmDatatest, type = 'prob')
write.csv(p, file = "predictedVal.csv")
write.csv(classification.test, file = "labels.csv")

ROC_rf = multiclass.roc(classification.test, predict(bag.sm, newdata = cmDatatest, type = 'prob'))
ROC_rf$rocs$`farm/forest`
ROC_rf

?randomForest
attach(mtcars)
par(mfrow=c(4,2))
plot(ROC_rf$rocs$`farm/forest`[[1]], main = "farm/forest")
plot(ROC_rf$rocs$`farm/forest`[[2]], main = "farm/forest")
plot(ROC_rf$rocs$`farm/grass`[[1]], main = "farm/grass")
plot(ROC_rf$rocs$`farm/grass`[[2]], main = "farm/grass")
plot(ROC_rf$rocs$`farm/impervious`[[1]], main = "farm/impervious")
plot(ROC_rf$rocs$`farm/impervious`[[2]], main = "farm/impervious")
plot(ROC_rf$rocs$`farm/orchard`[[1]], main = "farm/orchard")
plot(ROC_rf$rocs$`farm/orchard`[[2]], main = "farm/orchard")

plot(ROC_rf$rocs$`farm/water`[[1]], main = "farm/water")
plot(ROC_rf$rocs$`farm/water`[[2]], main = "farm/water")
plot(ROC_rf$rocs$`forest/grass`[[1]], main = "forest/grass")
plot(ROC_rf$rocs$`forest/grass`[[2]], main = "forest/grass")
plot(ROC_rf$rocs$`forest/impervious`[[1]], main = "forest/impervious")
plot(ROC_rf$rocs$`forest/impervious`[[2]], main = "forest/impervious")
plot(ROC_rf$rocs$`forest/orchard`[[1]], main = "forest/orchard")
plot(ROC_rf$rocs$`forest/orchard`[[2]], main = "forest/orchard")

plot(ROC_rf$rocs$`forest/water`[[1]], main = "forest/water")
plot(ROC_rf$rocs$`forest/water`[[2]], main = "forest/water")
plot(ROC_rf$rocs$`grass/impervious`[[1]], main = "grass/impervious")
plot(ROC_rf$rocs$`grass/impervious`[[2]], main = "grass/impervious")
plot(ROC_rf$rocs$`grass/orchard`[[1]], main = "grass/orchard")
plot(ROC_rf$rocs$`grass/orchard`[[2]], main = "grass/orchard")
plot(ROC_rf$rocs$`grass/water`[[1]], main = "grass/water")
plot(ROC_rf$rocs$`grass/water`[[2]], main = "grass/water")

plot(ROC_rf$rocs$`impervious/orchard`[[1]], main = "impervious/orchard")
plot(ROC_rf$rocs$`impervious/orchard`[[2]], main = "impervious/orchard")
plot(ROC_rf$rocs$`impervious/water`[[1]], main = "impervious/water")
plot(ROC_rf$rocs$`impervious/water`[[2]], main = "impervious/water")
plot(ROC_rf$rocs$`orchard/water`[[1]], main = "orchard/water")
plot(ROC_rf$rocs$`orchard/water`[[2]], main = "orchard/water")


(301-(40+70+7+38+37))/301

### Using 75-25 on training data #####
s1 = Sys.time()
bag.sm = randomForest(cmDatatraining$class~.-cmDatatraining$class,data=cmDatatraining, subset = train, importance=TRUE)
s2 = Sys.time()
trainTime = s2-s1
trainTime

s1 = Sys.time()
yhat.bag = predict(bag.sm, newdata = test)
s2 = Sys.time()
classifyTime = s2-s1
classifyTime
table(yhat.bag, test.class)

yhat.bag = predict(bag.sm, newdata = test, type = "prob")
yhat.bag
dim(yhat.bag)
length(test.class)

ROC_rf = multiclass.roc(test.class, predict(bag.sm, newdata = test, type = 'prob'))
a = predict(bag.sm, newdata = test, type = 'prob')
a
ROC_rf$rocs$`farm/forest`
plot(ROC_rf$rocs$`farm/forest`[[2]])


########### Random Forest End ###########

(40+67+5+36+23)/301
(44+65+7+35+22)/301
(288+1842+90+215+1+37)/(.25*nrow(cmDatatraining))

set.seed(1)
boost.cm = gbm(cmDatatraining$class~., data = cmDatatraining, distribution = "bernoulli",n.trees=500,interaction.depth=4)
yhat.boost = predict(boost.cm,newdata=cmDatatest,n.trees=5000)
table(yhat.boost, classification.test)
summary(yhat.boost)

#### Using 75-25 on training data #####
set.seed(1)
boost.cm = gbm(cmDatatraining[train,]$class~., data = cmDatatraining[train,], distribution = "multinomial",interaction.depth=4)
yhat.boost = predict(boost.cm,newdata=test,n.trees=5000, type = "prob")

yhat.boost
table(yhat.boost, test.class)
summary(yhat.boost)

?predict

adaboost = boosting(class~., data = cmDatatraining[train,], mfinal = 200)
pred.boost = predict.boosting(adaboost,newdata=cmDatatraining[-train, ], type = "prob")
is.matrix(pred.boost$prob)
dim(pred.boost$prob)
length(test.class)


pr = predict(adaboost, newdata = cmDatatraining[-train,], type = "prob")
pr
pr = rbind(c("farm", "forest", "grass", "impervious", "orchard", "water"), pr$prob)
ROC_rf = multiclass.roc(test.class, pr)
####boost roc works ##########
pred = (pred.boost$prob)[,1]
dummytest <- dummy.data.frame(cmDatatraining[-train,])
y = dummytest$classfarm
roc1 = roc(y, pred)
plot(roc1, col = 1, lty = 2, main = "ROC for Farm Class")


(45+72+22+37+42+49)/301
(39+57+7+37+9+21)/301

(287+1853+91+217+38)/(.25*nrow(cmDatatraining))


train.gbm <- train(as.factor(cmDatatraining[train,]$class)~., data=data.train ,method="gbm",verbose=F)
?head
