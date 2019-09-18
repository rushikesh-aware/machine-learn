library(ggplot2) # Data visualization
library("e1071") # SVM library 
traindata <- read.csv("C:/Users/Asus/Desktop/train.csv",header=T)
testdata <- read.csv("C:/Users/Asus/Desktop/test.csv",header=T)
traindata
testdata
data<-rbind(traindata,testdata)
data
nameVec <- make.names(names(data),unique=TRUE)
names(data) <- nameVec
traindata<-data[1:7352,]
testdata<-data[-c(1:7352),]
dim(data)
pc <- prcomp(traindata[,-563], center=TRUE, scale=TRUE)
pc.var <- pc$sdev^2
pc.pvar <- pc.var/sum(pc.var)
plot(cumsum(pc.pvar),xlab="Principal component", ylab="Cumulative Proportion of variance explained",type='b',main="Principal Components proportions",col="red")
abline(h=0.95)
abline(v=100)
train.data<-data.frame(activity=traindata$Activity,pc$x)
train.data<-train.data[,1:100]


svm_model <- svm(activity ~ ., data=train.data)

test.data<-predict(pc,newdata=testdata)
test.data<-as.data.frame(test.data)
test.data<-test.data[,1:100]
svm_model <- svm(activity ~ ., data=train.data)
#Preparing testing data for modelling with PCA(Principal Component Analysis)
test.data<-predict(pc,newdata=testdata)
test.data<-as.data.frame(test.data)
test.data<-test.data[,1:100]
#Predicting testing data with train SVM model
result<-predict(svm_model,test.data,type="class")
#Generating Confusion Matrix
test.data$Activity=testdata$Activity
references<-test.data$Activity
k<-table(references,result)
k
Accuracy <- (k[1,1]+k[2,2]+k[3,3]+k[4,4]+k[5,5]+k[6,6])/sum(t)
AccuracyRate <- Accuracy*100
c("Accuracy",AccuracyRate)
