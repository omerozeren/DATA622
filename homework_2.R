
library(class)
library(pROC)
library(ROCR)
library(caret)
library(C50)
library(e1071)
library(naivebayes)
library(MASS)
library(mltools)
library(knitr)

# PART A

#STEP 0: Pick any two classifiers of (SVM,Logistic,DecisionTree,NaiveBayes). Pick heart or ecoli dataset. Heart is simpler and ecoli compounds the problem as it is NOT a balanced dataset. From a grading perspective both carry the same weight.

# STEP 1 For each classifier, Set a seed (43)

#STEP#2 Do a 80/20 split and determine the Accuracy, AUC and as many metrics as returned by the Caret package (confusionMatrix) Call this the base_metric. Note down as best as you can development (engineering) cost as well as computing cost(elapsed time).
#Start with the original dataset and set a seed (43). Then run a cross validation of 5 and 10 of the model on the training set. Determine the same set of metrics and compare the cv_metrics with the base_metric. Note down as best as you can development (engineering) cost as well as computing cost(elapsed time).
#Start with the original dataset and set a seed (43) Then run a bootstrap of 200 resamples and compute the same set of metrics and for each of the two classifiers build a three column table for each experiment (base, bootstrap, cross-validated). Note down as best as you can development (engineering) cost as well as computing cost(elapsed time).


data <- read.csv("https://raw.githubusercontent.com/omerozeren/DATA622/master/heart.csv",head=T,sep=',',stringsAsFactors=F, fileEncoding = "UTF-8-BOM")



## Split Data into Train(80%) and Test data(20%)  


set.seed(43)
split_df <- createDataPartition(data$target, p = .80, list = FALSE)
data_train <- data[split_df,]
data_test <- data[-split_df,]


## Model Performance Estimater

estimate_model_performance <- function(y_true, y_pred, model_name){
  cm <- confusionMatrix(table(y_true, y_pred))
  cm_table <- cm$table
  tpr <- cm_table[[1]] / (cm_table[[1]] + cm_table[[4]])
  fnr <- 1 - tpr
  fpr <- cm_table[[3]] / (cm_table[[3]] + cm_table[[4]])
  tnr <- 1 - fpr
  accuracy <- cm$overall[[1]]
  for_auc <- prediction(c(y_pred), y_true)
  auc <- performance(for_auc, "auc")
  auc <- auc@y.values[[1]]
  return(data.frame(Algo = model_name, AUC = auc, ACCURACY = accuracy, TPR = tpr, FPR = fpr, TNR = tnr, FNR = fnr))
}


## NaiveBayes Model


start_tm <- proc.time()
nb_model<-naiveBayes(data_train$target~.,data=data_train)
object.size(nb_model) 
nb_testpred<-predict(nb_model,data_test,type='raw')
nb_testclass<-unlist(apply(round(nb_testpred),1,which.max))-1
nb_table<-table(data_test$target, nb_testclass)
base_metric_nb<-caret::confusionMatrix(nb_table)
base_metric_nb
end_tm<-proc.time() 
print(paste("time taken to run NaiveBayes Model",(end_tm-start_tm),sep=":"))


### Estimate NB model test data () performance


base_metric_nb_table_standalone<-estimate_model_performance(data_test$target,nb_testclass,'NB')
base_metric_nb_table_standalone


### NaiveBayes with Cross Validation folds = 5  


set.seed(43)
start_tm <- proc.time()
df     <- data[sample(nrow(data)),]
folds  <- cut(seq(1,nrow(data)),breaks=5,labels=FALSE)
nb_pred <- list()
nb_testclass <- list()
nb_testclass<-list()
nb_table <- list()
base_metric_nb <- list()
base_metric_nb_table_cv_5 <- list()
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData    <- df[testIndexes, ]
  trainData   <- df[-testIndexes, ]
  nb_model       <- naiveBayes(trainData$target ~ .,data=trainData) # naiveBayes(data_train$target~.,data=data_train)
  nb_pred[[i]]<-predict(nb_model,testData,type='raw')
  nb_testclass[[i]]<-unlist(apply(round(nb_pred[[i]]),1,which.max))-1
  nb_table[[i]]<-table(testData$target, nb_testclass[[i]])
  base_metric_nb[[i]]<-caret::confusionMatrix(nb_table[[i]])
  base_metric_nb_table_cv_5[[i]]<-estimate_model_performance(testData$target,nb_testclass[[i]],paste('NB fold',i,sep =":" ))
}

end_tm<-proc.time() 

print(paste("time taken to run NaiveBayes Model with CV with 5 Folds",(end_tm-start_tm),sep=":"))


### Base Metric for NaiveBayes with Cross Validation folds = 5 

base_metric_nb


### The Mean of NaiveBayes with Cross Validation folds = 5 

rst<-do.call(rbind.data.frame, base_metric_nb_table_cv_5)
base_metric_nb_table_cv_5_mean<-data.frame(cbind(Algo='NB_CV_5',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
base_metric_nb_table_cv_5_mean


### NaiveBayes with Cross Validation folds = 10  


set.seed(43)
df     <- data[sample(nrow(data)),]
folds  <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)
nb_pred <- list()
nb_testclass <- list()
nb_testclass<-list()
nb_table <- list()
base_metric_nb <- list()
base_metric_nb_table_cv_10 <- list()
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData    <- df[testIndexes, ]
  trainData   <- df[-testIndexes, ]
  nb_model       <- naiveBayes(trainData$target ~ .,data=trainData) # naiveBayes(data_train$target~.,data=data_train)
  nb_pred[[i]]<-predict(nb_model,testData,type='raw')
  nb_testclass[[i]]<-unlist(apply(round(nb_pred[[i]]),1,which.max))-1
  nb_table[[i]]<-table(testData$target, nb_testclass[[i]])
  base_metric_nb[[i]]<-caret::confusionMatrix(nb_table[[i]])
  base_metric_nb_table_cv_10[[i]]<-estimate_model_performance(testData$target,nb_testclass[[i]],paste('NB fold',i,sep =":" ))
}



### Base Metric for NaiveBayes with Cross Validation folds = 10

base_metric_nb


### The Mean of NaiveBayes with Cross Validation folds = 5 

rst<-do.call(rbind.data.frame, base_metric_nb_table_cv_10)
base_metric_nb_table_cv_10_mean<-data.frame(cbind(Algo='NB_CV_10',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
base_metric_nb_table_cv_10_mean


## Logistic Regression

start_tm <- proc.time()
lr_model <- glm(target ~ ., data=data_train,family = "binomial")
object.size(lr_model) 
lr_testpred = predict(lr_model, newdata=data_test,type="response")
lr_pred <- prediction(as.numeric(lr_testpred > 0.5),data_test$target)
lr_testclass <- lr_pred@predictions[[1]]
lr_table<-table(data_test$target, lr_testclass)
base_metric_lr<-caret::confusionMatrix(lr_table)
base_metric_lr
end_tm<-proc.time() 
print(paste("time taken to run Logistic Regression Model",(end_tm-start_tm),sep=":"))


### Estimate Logistic Regression model test data () performance


base_metric_lr_table_standalone<-estimate_model_performance(data_test$target,lr_testclass,'LR')
base_metric_lr_table_standalone


### Logistic Regression with Cross Validation folds = 5  


set.seed(43)
start_tm <- proc.time()
df     <- data[sample(nrow(data)),]
folds  <- cut(seq(1,nrow(data)),breaks=5,labels=FALSE)
lr_pred <- list()
lr_testclass <- list()
lr_table <- list()
base_metric_lr <- list()
base_metric_lr_table_cv_5 <- list()
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData    <- df[testIndexes, ]
  trainData   <- df[-testIndexes, ]
  lr_model       <- glm(target ~ .,family="binomial",data=trainData)
  lr_pred[[i]] <- prediction(as.numeric(predict(lr_model, newdata=testData,type="response") > 0.5),testData$target)
  lr_testclass[[i]] <- lr_pred[[i]]@predictions[[1]]
  lr_table[[i]]<-table(testData$target, lr_testclass[[i]])
  base_metric_lr[[i]]<-caret::confusionMatrix(lr_table[[i]])
  base_metric_lr_table_cv_5[[i]]<-estimate_model_performance(testData$target,lr_testclass[[i]],paste('LR fold',i,sep =":" ))
}

end_tm<-proc.time() 

print(paste("time taken to run Logistic Regression Model with CV with 5 Folds",(end_tm-start_tm),sep=":"))


### Base Metric for Logistic Regression with Cross Validation folds = 5 

base_metric_lr


### The Mean of Logistic Regression with Cross Validation folds = 5 

rst<-do.call(rbind.data.frame, base_metric_lr_table_cv_5)
base_metric_lr_table_cv_5_mean<-data.frame(cbind(Algo='LR_CV_5',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
base_metric_lr_table_cv_5_mean


### Logistic Regression with Cross Validation folds = 10  


set.seed(43)
df     <- data[sample(nrow(data)),]
folds  <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)
lr_pred <- list()
lr_testclass <- list()
lr_table <- list()
base_metric_lr <- list()
base_metric_lr_table_cv_10 <- list()
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData    <- df[testIndexes, ]
  trainData   <- df[-testIndexes, ]
  lr_model       <- glm(target ~ .,family="binomial",data=trainData)
  lr_pred[[i]] <- prediction(as.numeric(predict(lr_model, newdata=testData,type="response") > 0.5),testData$target)
  lr_testclass[[i]] <- lr_pred[[i]]@predictions[[1]]
  lr_table[[i]]<-table(testData$target, lr_testclass[[i]])
  base_metric_lr[[i]]<-caret::confusionMatrix(lr_table[[i]])
  base_metric_lr_table_cv_10[[i]]<-estimate_model_performance(testData$target,lr_testclass[[i]],paste('LR fold',i,sep =":" ))
  
}


### Base Metric for Logistic Regression with Cross Validation folds = 10 

base_metric_lr


### The Mean of Logistic Regression with Cross Validation folds = 10 

rst<-do.call(rbind.data.frame, base_metric_lr_table_cv_10)
base_metric_lr_table_cv_10_mean<-data.frame(cbind(Algo='LR_CV_10',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
base_metric_lr_table_cv_10_mean


## Compate Metrics


print(paste('NaiveBayes:'))
base_metric_nb_table_standalone
print(paste('NB with cv fold=5:'))
base_metric_nb_table_cv_5_mean
print(paste('NB with cv fold=10:'))
base_metric_nb_table_cv_10_mean
print(paste('Logistic Regression:'))
base_metric_lr_table_standalone
print(paste('LR with cv fold=5:'))
base_metric_lr_table_cv_5_mean
print(paste('LR with cv fold=10:'))
base_metric_lr_table_cv_10_mean



## Bootstrap Methodology - NaiveBayes Model

I'm going to create a function for boostrap purposes first.I'm going to run NaiveBayes model  200 times and store the performance metrics for each data boostrap.


set.seed(43)
apply_bootstrap_data <- function(data, proportion = 0.8, sample_with_replacement = TRUE){
  observation <- round(nrow(data) * proportion, 0)
  return(data[sample(nrow(data), observation, replace = sample_with_replacement),])
}



start <- proc.time()
n_times <- 200
for (i in 1:n_times){
  sample <- apply_bootstrap_data(data_train)
  nb_model <- naiveBayes(sample$target ~ ., data = sample)
  y_pred <- predict(nb_model, data_test,type='raw') # probability
  y_pred_class<-unlist(apply(round(y_pred),1,which.max))-1 # class
  performance <- estimate_model_performance(data_test$target, y_pred_class, paste("NB Bootstrap ", i))
  if(exists("performance_table_nb")){
    performance_table_nb <- rbind(performance_table_nb, performance)
  } else {
    performance_table_nb <- performance
  }
}
elapsed_time <- (proc.time() - start)[[3]]
elapsed_time
 

### NB Boostrap Results Table


performance_table_nb


### The Mean of Boostrap NB model 


rst<-performance_table_nb
performance_table_nbboostrap_mean<-data.frame(cbind(Algo='NB_Bosstrap',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
performance_table_nbboostrap_mean


## Bootstrap Methodology - Logistic Regression Model

I'm going to create a function for boostrap purposes first.I'm going to run Logistic Regression model  200 times and store the performance metrics for each data boostrap.


set.seed(43)
apply_bootstrap_data <- function(data, proportion = 0.8, sample_with_replacement = TRUE){
  observation <- round(nrow(data) * proportion, 0)
  return(data[sample(nrow(data), observation, replace = sample_with_replacement),])
}



start <- proc.time()
n_times <- 200
for (i in 1:n_times){
  sample <- apply_bootstrap_data(data_train)
  lr_model <- glm(target ~ ., data=sample,family = "binomial")
  lr_testpred <- predict(lr_model, data_test,type='response') # probability
  lr_pred <- prediction(as.numeric(lr_testpred > 0.5,1,0),data_test$target)
  y_pred_class<-lr_pred@predictions[[1]] # class
  performance <- estimate_model_performance(data_test$target, y_pred_class, paste("LR Bootstrap", i))
  if(exists("performance_table_lr")){
    performance_table_lr <- rbind(performance_table_lr, performance)
  } else {
    performance_table_lr <- performance
  }
}
elapsed_time <- (proc.time() - start)[[3]]
elapsed_time
 

### LR Boostrap Results Table


performance_table_lr


### The Mean of Boostrap LR model 


rst<-performance_table_lr
performance_table_lrboostrap_mean<-data.frame(cbind(Algo='LR_Boostrap',AUC=mean(rst$ACCURACY),ACCURACY=mean(rst$ACCURACY),TPR=mean(rst$TPR),FPR=mean(rst$FPR),TNR=mean(rst$TNR),FNR=mean(rst$FNR)))
# Average
performance_table_lrboostrap_mean



## Summary Performance Results

#putting results in dataFrame
data.frame(rbind(base_metric_nb_table_standalone,base_metric_nb_table_cv_5_mean,base_metric_nb_table_cv_10_mean,performance_table_nbboostrap_mean,base_metric_lr_table_standalone,base_metric_lr_table_cv_5_mean,base_metric_lr_table_cv_10_mean,performance_table_lrboostrap_mean))


# PART B
For the same dataset, set seed (43) split 80/20.
Using randomForest grow three different forests varuing the number of trees atleast three times. Start with seeding and fresh split for each forest. Note down as best as you can development (engineering) cost as well as computing cost(elapsed time) for each run. And compare these results with the experiment in Part A. Submit a pdf and executable script in python or R.


data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
data$ca <- as.factor(data$ca)
data$sex <- as.factor(data$sex)
data$restecg <- as.factor(data$restecg)
data$thal <- as.factor(data$thal)
data$target <- as.factor(data$target)
# do a 80/20 split 
set.seed(43)
split_df <- sample(seq_len(nrow(data)), size = floor(0.8 * nrow(data)))
train_heart <- data[ split_df,]
test_heart  <- data[-split_df,]


### Random Forest - 10 Trees

start <- proc.time()
rf_10_trees <- train(form = target ~ .,
                     data = train_heart,
                     method = 'rf',
                     trControl = trainControl(),
                     ntree = 10)
rf_10_trees
elapsed_time <- (proc.time() - start)[[3]]
elapsed_time


### Random Forest - 10 Trees Performance 

pred<-predict(rf_10_trees, subset(test_heart, select = -c(target)))
rst_class<-as.factor(pred)
model_cm <-confusionMatrix(rst_class,test_heart$target)
rst_rf_10<-estimate_model_performance(rst_class,test_heart$target,'Random Forest - 10 Trees')
rst_rf_10


### Random Forest - 30 Trees

start <- proc.time()
rf_30_trees <- train(form = target ~ .,
                     data = train_heart,
                     method = 'rf',
                     trControl = trainControl(),
                     ntree = 30)
rf_30_trees
elapsed_time <- (proc.time() - start)[[3]]
elapsed_time


### Random Forest - 30 Trees Performance 

pred<-predict(rf_30_trees, subset(test_heart, select = -c(target)))
rst_class<-as.factor(pred)
model_cm <-confusionMatrix(rst_class,test_heart$target)
rst_rf_30<-estimate_model_performance(rst_class,test_heart$target,'Random Forest - 30 Trees')
rst_rf_30




### Random Forest - 90 Trees

start <- proc.time()
rf_90_trees <- train(form = target ~ .,
                     data = train_heart,
                     method = 'rf',
                     trControl = trainControl(),
                     ntree = 90)
rf_90_trees
elapsed_time <- (proc.time() - start)[[3]]
elapsed_time


### Random Forest - 90 Trees Performance 

pred<-predict(rf_90_trees, subset(test_heart, select = -c(target)))
rst_class<-as.factor(pred)
model_cm <-confusionMatrix(rst_class,test_heart$target)
rst_rf_90<-estimate_model_performance(rst_class,test_heart$target,'Random Forest - 90 Trees')
rst_rf_90


### Combine Random Forest Results

data.frame(rbind(rst_rf_10, rst_rf_30, rst_rf_90))


## Part C
#Include a summary of your findings. Which of the two methods bootstrap vs cv do you recommend to your customer? And why? Be elaborate. Including computing costs, engineering costs and model performance. Did you incorporate Pareto's maxim or the Razor and how did these two heuristics influence your decision?

#**Answer:**
#I would use cross validation methodlogies over bootstrapp methods, I can see that it was less computationally expensive and cross-validation resulted in better Accuracy than boostrapping methods.
#All four Logistic models created  high accuracy, and AUC. However we dont see huge differences in Accuracy results between CV=5 and CV=10.The Logistic Regression with 10-fold CV model does not add much accuracy or stability to 5-fold CV model.The Occam's razor suggests that the simpler model (the 5-fold CV) should be used for Logistic Regression Models.
#NaiveBayes models also performed well compare to average results.We do see decrease in accuracy changing cross-validation from 5-folds to 10 folds.The Occam's razor suggests that the simpler model (the 5-fold CV) should be used since there is no additional increase in accuracy.
#Random Forest model, increasing ntrees from 30 to 90 actually incread the accuracy. I would use randomforest with 90 trees.


final_results <- data.frame(rbind(base_metric_nb_table_standalone,base_metric_nb_table_cv_5_mean,base_metric_nb_table_cv_10_mean,performance_table_nbboostrap_mean,base_metric_lr_table_standalone,base_metric_lr_table_cv_5_mean,base_metric_lr_table_cv_10_mean,performance_table_lrboostrap_mean,rst_rf_10,rst_rf_30,rst_rf_90))
final_results
