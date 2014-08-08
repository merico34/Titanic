setwd("C:/Users/HomeUser/Documents/Intro To Data Science (UW)/Projects/Titanic - Machine Learning from Disaster")

train <- read.csv("train.csv",stringsAsFactors=F)
str(train)

test <- read.csv("test.csv",stringsAsFactors=F)

names(train)

## SOME EXPLORATORY ANALYSIS

library(ggplot2)
qplot(Pclass,data=train,fill=as.factor(Survived))
# barplot(table(train$Sex),col="green")
qplot(Sex,data=train,fill=as.factor(Survived))
# hist(train$Age,col="green")
# rug(train$Age)
qplot(Age,data=train,fill=as.factor(Survived))
# hist(train$SibSp,col="green")
# hist(train$Parch,col="green")
# hist(train$Fare,col="green")
# barplot(table(train$Embarked),col="green")
qplot(SibSp,data=train,fill=as.factor(Survived))
qplot(Parch,data=train,fill=as.factor(Survived))
qplot(Fare,data=train,fill=as.factor(Survived))
qplot(Embarked,data=train,fill=as.factor(Survived))

qplot(Sex,data=train,facets=Pclass~.,fill=as.factor(Survived))
qplot(Age,data=train,facets=.~Pclass+Sex+Parch+SibSp,fill=as.factor(Survived))

# pairs(~.-PassengerId-Name-Cabin-Ticket,data=train,col=train$Survived)
pairs(~Pclass+Sex+Age+Fare+Embarked,data=train,col=1+train$Survived)

# Spinning 3d Scatterplot
library(rgl)
with(train,plot3d(Sex,Age,Pclass,col=1+Survived))

# par(mfrow = c(1,1))
# with(train,qplot(Age,Sex,col=Survived))

## SOME ROUGH INFORMATION GAIN CALCULATION (decision tree algorithms should be better...)
# library(entropy)
library(FSelector)
weights <- information.gain(Survived~., train)
print(weights)

#selection des >0 mais dont length(unique(var))<0.5length(train) pour des var non continuous??
variete <- sapply(names(train), function (x) {length(unique(train[,x]))})
variete_df <- data.frame(var = names(variete),uniqueness = variete)

classes <- sapply(names(train), function (x) {class(train[,x])})
classes_df <- data.frame(var = names(classes),class = classes)

weights_df = data.frame(var=rownames(weights),infogain=as.numeric(weights$attr_importance))
weights_df = merge(weights_df,variete_df)
weights_df = merge(weights_df,classes_df)

weights_df_sel = weights_df[(weights_df$infogain>0.0001
                             & (weights_df$uniqueness/nrow(train)<3/4 | weights_df$class == "numeric")),]
weights_df_sel[order(weights_df_sel$infogain),]

#faire un scatterplot sur les variables "rejetées" par acquis de conscience...?

# data(iris)
# weights <- information.gain(Species~., iris)
# print(weights)
# subset <- cutoff.k(weights, 2)
# f <- as.simple.formula(subset, "Species")
# print(f)

## MACHINE LEARNING
library(caret)

# See variables with:
# 1) many missing values (NAs or "") -> drop
# 2) few missing values -> replacement strategy?

train <- read.csv("train.csv",stringsAsFactors=T) # todo: chargement non redondant
summary(train) 
# -> keep as is: Survived+Pclass+Sex+SibSp+Parch+Fare
# -> set Embarked = "U" for "" values
# -> set Age = mean(train$Age, na.rm = T) for NA's values (too simple?)
train <- read.csv("train.csv",stringsAsFactors=F) # todo: chargement non redondant

test <- read.csv("test.csv",stringsAsFactors=T) # todo: chargement non redondant
summary(test)
# -> keep as is: Survived+Pclass+Sex+SibSp+Parch
# -> set NA's Fare = mean(test$Fare, na.rm = T) (1 occurence only)
# -> set Embarked = "U" for "" values
# -> set Age = mean(test$Age, na.rm = T) for NA's values (too simple?)
test <- read.csv("test.csv",stringsAsFactors=F) # todo: chargement non redondant

##NAs
NAs <- apply(train,2,function(x) {sum(is.na(x))}) #count NAs for each variable
NAs

### DATA PREPARATION

rawtrain <- train
rawtrain[rawtrain$Embarked == "",]$Embarked <- "U"
rawtrain[is.na(rawtrain$Age),]$Age <- mean(train$Age, na.rm = T)
rawtrain[is.na(rawtrain$Fare),]$Fare <- mean(train$Fare, na.rm = T)

rawtrain$Survived <- as.factor(rawtrain$Survived)
rawtrain$Sex <- as.factor(rawtrain$Sex)
rawtrain$Pclass <- as.factor(rawtrain$Pclass)
rawtrain$Embarked <- as.factor(rawtrain$Embarked)

str(rawtrain)
summary(rawtrain)

# rawtrain$Cabin <- NULL
# rawtrain <- rawtrain[complete.cases(rawtrain),]

# rawtest <- test[complete.cases(test),]
# natest <- test[-complete.cases(test),]

rawtest <- test
rawtest[rawtest$Embarked == "",]$Embarked <- "U"
rawtest[is.na(rawtest$Age),]$Age <- mean(test$Age, na.rm = T)
rawtest[is.na(rawtest$Fare),]$Fare <- mean(test$Fare, na.rm = T)

rawtest$Sex <- as.factor(rawtest$Sex)
rawtest$Pclass <- as.factor(rawtest$Pclass)
rawtest$Embarked <- as.factor(rawtest$Embarked)

str(rawtest)
summary(rawtest)

### TRAINING

intrain <- createDataPartition(y=rawtrain$Survived,p=0.7,list=F)
training <- rawtrain[intrain,]
testing <- rawtrain[-intrain,]
modFit <- train(Survived ~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, method = "rf", data = training)
modFit$finalModel
summary(modFit)
modFit

library(rattle)
fancyRpartPlot(modFit$finalModel)

pred <- predict(modFit,newdata=testing)
length(pred)
confusionMatrix(pred,testing$Survived)

# SUBMISSION

predTest <- predict(modFit,newdata=rawtest)
length(predTest)

kaggle.sub <- cbind(test$PassengerId,predTest)
colnames(kaggle.sub) <- c("PassengerId", "Survived")
write.csv(kaggle.sub, file = "kaggle.csv", row.names = FALSE)

