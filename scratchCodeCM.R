library(tree)
library(randomForest)
library(gbm)
setwd("/Users/rxh655/Documents/Spring2019/STAT557/Project2")

mushroom =read.csv("CrowdsourcedMapping/training.csv")

tree.mushroom = tree(mushroomData$classification~.-mushroomData$classification,mushroomData, subset = train)
tree.pred=predict(tree.mushroom, mushroomData.test ,type ="class")
table(tree.pred , classification.test)