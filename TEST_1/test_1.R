library(class)
library(pROC)
library(ROCR)
library(caret)
library(e1071)
library(naivebayes)
library(MASS)
library(mltools)
library(knitr)
library(dplyr)
library(tidyr)


## A)

#Run Bagging (ipred package)   

# sample with replacement

# estimate metrics for a model

# repeat as many times as specied and report the average



### Load the Data


df <- read.table("~/GitHub/DATA622/data.txt",header = T,sep=',')
df$label <- ifelse(df$label =="BLACK",1,0)
df$y <- as.numeric(df$y)
df$X <- as.factor(df$X)


### Split Data into Train (70%) and Test data(30%)  

set.seed(42)
split_df <- createDataPartition(df$label, p = .70, list = FALSE)
df_train <- df[split_df,]
df_test <- df[-split_df,]


### Model Performance Estimator

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



### NB Model Building - Standalone


nb_model<-naiveBayes(df_train$label~.,data=df_train)
nb_testpred<-predict(nb_model,df_test,type='raw')
nb_testclass<-unlist(apply(round(nb_testpred),1,which.max))-1
nb_table<-table(df_test$label, nb_testclass)
nb_cm<-caret::confusionMatrix(nb_table)
nb_cm


### Estimate NB model test data () performance


rst_nb<-estimate_model_performance(df_test$label,nb_testclass,'NB')
rst_nb



### Bagging Methodology - NB Model

#I'm going to create a function for boostrap purposes first.I'm going to run NB model  50 times and store the performance metrics for each data boostrap.


apply_bootstrap_data <- function(data, proportion = 0.7, sample_with_replacement = TRUE){
  observation <- round(nrow(data) * proportion, 0)
  return(data[sample(nrow(data), observation, replace = sample_with_replacement),])
}



for (i in 1:50){
  sample <- apply_bootstrap_data(df_train)
  nb_model <- naiveBayes(sample$label ~ ., data = sample)
  y_pred <- predict(nb_model, df_test,type='raw') # probability
  y_pred_class<-unlist(apply(round(y_pred),1,which.max))-1 # class
  performance <- estimate_model_performance(df_test$label, y_pred_class, paste("NB Bootstrap ", i))
  if(exists("performance_table_nb")){
    performance_table_nb <- rbind(performance_table_nb, performance)
  } else {
    performance_table_nb <- performance
  }
}



### NB Boostrap Results Table


performance_table_nb


### The Mean of Boostrap NB model 


mean(performance_table_nb$ACCURACY)


### The Variance of Boostrap NB model 


var(performance_table_nb$ACCURACY)


#Now, I'm going to try KNN stand alone and boostrap methodology.For the KNN model, I will use K =3.

### KNN Model Building - Standalone


knn_y_true<- knn(df_train[1:2],df_test[1:2], cl = df_train$label, k = 5)
knn_testclass<-knn_y_true
knn_table<-table(df_test$label, knn_testclass)
knn_cm<-caret::confusionMatrix(knn_table)
knn_cm


### Estimate KNN model test data () performance


rst_knn<-estimate_model_performance(df_test$label,knn_testclass,'KNN')
rst_knn

### Bagging Methodology - KNN Model

#I'm going to create a function for boostrap purposes first.I'm going to run KNN model 50 times and store the performance metrics for each data boostrap.


apply_bootstrap_data <- function(data, proportion = 0.7, sample_with_replacement = TRUE){
observation <- round(nrow(data) * proportion, 0)
return(data[sample(nrow(data), observation, replace = sample_with_replacement),])
}



for (i in 1:50){
sample <- apply_bootstrap_data(df_train)
y_pred <- knn(sample[1:2],df_test[1:2], cl = sample$label, k = 3)
y_pred_class<-y_pred
performance <- estimate_model_performance(df_test$label, y_pred_class, paste("KNN Bootstrap ", i))
if(exists("performance_table_knn")){
performance_table_knn <- rbind(performance_table_knn, performance)
} else {
performance_table_knn <- performance
}
}


### KNN Boostrap Results Table


performance_table_knn


### The Mean of Boostrap KNN model 


mean(performance_table_knn$ACCURACY)


### The Variance of Boostrap KNN model 


var(performance_table_knn$ACCURACY)


## B)

#Run LOOCV (jacknife) for the same dataset

# iterate over all points

# keep one observation as test

# train using the rest of the observations

# determine test metrics

# aggregate the test metrics

#end of loop

#find the average of the test metric(s)

#Compare (A), (B) above with the results you obtained in HW-1  and write 3 sentences explaining the observed difference.

### Jacknife: Leave One Out (LOO) Cross Validation -KNN Model

#For each observation train with aLL other observations  predict that one observation.


y_pred_train_loocv_knn <- c()
for (i in 1:nrow(df_train)){
loocv_test <- df_train[i,]
loocv_train_df <- df_train[-c(i),]
y_pred_train_loocv_knn <- c(y_pred_train_loocv_knn, knn(loocv_train_df[1:2], loocv_test[1:2], loocv_train_df$label, k = 3))
y_pred_test_loocv_knn <- knn(loocv_train_df[1:2], df_test[1:2], loocv_train_df$label, k = 3)
performance <- estimate_model_performance(df_test$label, y_pred_test_loocv_knn, paste("KNN - LOOCV", i))
if(exists("performance_table_knn_loocv")){
performance_table_knn_loocv <- rbind(performance_table_knn_loocv, performance)
} else {
performance_table_knn_loocv <- performance
}
}


### KNN LOOCV Results

performance_table_knn_loocv


### The Mean of LOOCV KNN model 


mean(performance_table_knn_loocv$ACCURACY)


### The Variance of LOOCV KNN mode 


var(performance_table_knn_loocv$ACCURACY)


### Jacknife: Leave One Out (LOO) Cross Validation -NB Model


y_pred_train_loocv_nb <- c()
for (i in 1:nrow(df_train)){
loocv_test <- df_train[i,]
loocv_train_df <- df_train[-c(i),]
nb_model <- naiveBayes(loocv_train_df$label ~ ., data = loocv_train_df)
y_pred_train_loocv_nb <- predict(nb_model, loocv_test[1:2],type='raw') # probability
y_pred_train_loocv_class_nb<-unlist(apply(round(y_pred_train_loocv_nb),1,which.max))-1 # class
y_pred_train_loocv_nb <- c(y_pred_train_loocv_nb,y_pred_train_loocv_class_nb)  
y_pred_test_loocv_nb <- predict(nb_model, df_test[1:2],type='raw') # probability
y_pred_test_loocv_class_nb<-unlist(apply(round(y_pred_test_loocv_nb),1,which.max))-1 # class
performance <- estimate_model_performance(df_test$label, y_pred_test_loocv_class_nb, paste("NB - LOOCV", i))  
if(exists("performance_table_nb_loocv")){
performance_table_nb_loocv <- rbind(performance_table_nb_loocv, performance)
} else {
performance_table_nb_loocv <- performance
}
}
 

### NB LOOCV Results


performance_table_nb_loocv


### The Mean of LOOCV NB model 

mean(performance_table_nb_loocv$ACCURACY)


### The Variance of LOOCV NB model 


var(performance_table_nb_loocv$ACCURACY)


### Compate Metrics


print(paste('NB:',rst_nb$ACCURACY))
print(paste('Bagging -NB :',mean(performance_table_nb$ACCURACY)))
print(paste('KNN:',rst_knn$ACCURACY))
print(paste('Bagging -KNN :',mean(performance_table_knn$ACCURACY)))
print(paste('LOO-CV/KNN:',mean(performance_table_knn_loocv$ACCURACY)))
print(paste('LOO-CV/NB:',mean(performance_table_nb_loocv$ACCURACY)))




## Summary

#The above display shows results of each methods model performance.Initially, HMW1 KNN stand alone model(0.8) results somewhere better than both bagging methodologies for KNN(0.72) and NB(0.7).The KNN stand alone model performs better than the KNN with bagging method.In addition to that, stand alone NB model performs better than bagging methodology with NB. This is one of the drawnback boostrap methodology that differences due to randomly assign samples in boostrap method.In the boostrap methodoloy ,I used 50 iterations and for each iteration model perform vary due to sampling data set for training randomly.The stand alone KNN and NB models perform almost the the same results with LOOCV.


