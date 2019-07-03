# File: qpe.r
# Description:
#    Functions to perfrom QPE from radar data
#    
#=========================================================================================
cmToMetrics <- function(cm, positive=1){
  # Decode the matrix
  if(positive==0){
    TP <- cm[1,1]/sum(cm)
    TN <- cm[2,2]/sum(cm)
    FP <- cm[1,2]/sum(cm)
    FN <- cm[2,1]/sum(cm)
  } else {
    TP <- cm[2,2]/sum(cm)
    TN <- cm[1,1]/sum(cm)
    FP <- cm[2,1]/sum(cm)
    FN <- cm[1,2]/sum(cm)
  }
  # Derive Metrics
  sensitivity <- TP/(TP+FN)
  specificity <- TN/(FP+TN)
  prevalence <- TP+FN/(FN+TP+FP+TN)
  ppv <- TP/(TP+FP)
  npv <- TN/(TN+FN)
  fpr <- FP/(FP+TN)
  fnr <- FN/(FN+TP)
  fdr <- FP/(FP+TP)
  FOR <- FN/(TN+FN)
  accuracy <- (TP+TN)/(FN+TP+FP+TN)
  F1 <- 2*TP/(2*TP+FP+FN)
  MCC <- (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  informedness <- sensitivity + specificity - 1
  markedness <- ppv + npv -1
  # Output
  output <- c("Accuracy"=accuracy,
              "True.Positive"=TP,
              "False.Negative"=FN,
              "False.Positive"=FP,
              "True.Negative"=TN,
              "Sensitivity"=sensitivity, 
              "Specificity"=specificity,
              "Prevalence"=prevalence, 
              "Positive.Predictive.Value"=ppv,
              "Negative.Predictive.Value"=npv,
              "False.Positive.Rate"=fpr,
              "False.Discovery.Rate"=fdr,
              "False.Negative.Rate"=fnr,
              "False.Omission.Rate"=FOR,
              "F1.Score"=F1,
              "Matthews.correlation.coefficient"=MCC,
              "Informedness"=informedness,
              "Markedness"=markedness)
  return(output)
}
#=========================================================================================
# Load input/output data
load("input.1316.ae.RData")
load("../workspace/output.1316.RData")
rseed = 1234543
# Load library
require(caret)
require(kernlab)
require(parallel)
# Set up multi-core
library(doParallel)
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
# Collect results
results.glm <- data.frame(NULL)
results.svm <- data.frame(NULL)
ys <- NULL
mod.glm <- NULL
mod.svm <- NULL
# Run through each station
nstation <- length(y.1316)
for(i in 1:nstation){
  # 
  print(paste("Creating IO data for",names(y.1316)[i]))
  # Combine IO
  tmp <- data.frame('date'=y.1316[[i]]$date,'y'=(y.1316[[i]]$t1hr>=40.)*1) # 1:>=40mm/hr
  iodata <- merge(tmp, input.1316, by.x="date", by.y=0)
  # Clean up NA and move date to row.names
  print("Cleaning up data...")
  row.names(iodata) <- iodata$date
  iodata <- iodata[,-1]
  iodata <- iodata[complete.cases(iodata),]
  #iodata <- iodata[1:1000,]
  print(paste("    Number of valid records:", nrow(iodata)))
  # Setup train-control
  print("Creating folds for cross validation...")
  set.seed(rseed)
  cvOut <- createFolds(iodata$y, k=10)
  cvIn <- cvOut
  for(i in 1:10){
    cvIn[[i]] <- (1:length(iodata$y))[-cvOut[[i]]]
  }
  trctrl <- trainControl("cv", index=cvIn, indexOut=cvOut, allowParallel=T, savePred=T)
  # Fit model
  print("Training and cross validating...")
  # GLM
  fit.glm <- train(y~., data=iodata, method="glm", family="binomial", preProcess="scale", trControl=trctrl)
  rec <- fit.glm$results
  yhat.glm <- fit.glm$finalModel$fitted.values
  rec$RMSE.insample <- RMSE(exp(yhat.glm)-1, iodata$y)
  rec$CORR.log <- cor(yhat.glm, log(iodata$y+1))
  rec$CORR.mm <- cor(exp(yhat.glm)-1, iodata$y)
  print("GLM")
  print(rec)
  results.glm <- rbind(results.glm, rec)
  #coef.glm <- c(coef.glm, list(coef(summary(fit.glm))))
  # SVM
  fit.svmr <- train(y~., data=iodata, method="svmRadial", preProcess="scale", trControl=trctrl)
  rec <- fit.svmr$results[1,]
  yhat.svm <- fit.svmr$finalModel@fitted
  rec$RMSE.insample <- RMSE(exp(yhat.svm)-1, iodata$y)
  rec$CORR.log <- cor(yhat.svm, log(iodata$y+1))
  rec$CORR.mm <- cor(exp(yhat.svm)-1, iodata$y)
  print("SVR")
  print(rec)
  results.svm <- rbind(results.svm, rec)
  # Collection predictions
  #y <- data.frame(NULL)
  y <- data.frame(iodata$y)
  y$y.glm <- yhat.glm
  y$y.svm <- yhat.svm
  ys <- c(ys, list(y))
  # Save model
  mod.glm <- c(mod.glm, list(fit.glm$finalModel))
  mod.svm <- c(mod.svm, list(fit.svmr$finalModel))
}
#names(coef.glm) <- names(y.1316)
names(ys) <- names(y.1316)
names(mod.glm) <- names(y.1316)
names(mod.svm) <- names(y.1316)
# Clean up
rm(i, iodata, cvOut, cvIn, trctrl, fit.glm, fit.svmr)
# Stop parallel
stopCluster(cluster)
registerDoSEQ()
# Save
save(results.glm, results.svm, ys, file="qpe1316.ae.RData")
#save(mod.glm, mod.svm, ys, file="qpe1316.mod.RData")






