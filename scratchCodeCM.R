library(tree)
library(randomForest)
library(gbm)
setwd("/Users/rxh655/Documents/Spring2019/STAT557/Project2")

cmDatatraining =read.csv("CrowdsourcedMapping/training.csv")
cmDatatest = read.csv("CrowdsourcedMapping/testing.csv")

tree.cm = tree(cmDatatraining$class, cmDatatraining)
tree.pred

classification.test = cmDatatest$class


tree.cm = tree(cmDatatraining$class~.-cmDatatraining$class,cmDatatraining)
tree.pred=predict(tree.cm, cmDatatest ,type ="class")
table(tree.pred , classification.test)

(36+71+7+30+7)/(301)

set.seed(1)
bag.sm = randomForest(cmDatatraining$class~.-cmDatatraining$class,data=cmDatatraining,mtry=28,importance=TRUE)
bag.sm

yhat.bag = predict(bag.sm,newdata=cmDatatest)

yhat.bag
table(yhat.bag, classification.test)
(40+67+5+36+23)/301

set.seed(1)
boost.cm = gbm(cmDatatraining$class~., data = cmDatatraining, distribution = "gaussian",n.trees=5000,interaction.depth=4)
yhat.boost = predict(boost.cm,newdata=cmDatatest,n.trees=5000)
table(yhat.boost, classification.test)
summary(yhat.boost)
