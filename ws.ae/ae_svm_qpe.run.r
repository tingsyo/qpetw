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
#library(doParallel)
#cluster <- makeCluster(12, outfile='aesvm.par.log') # convention to leave 8 core for OS
#registerDoParallel(cluster)
# Collect results
cmsByModels <- NULL
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
  iodata$y <- factor(iodata$y)
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
  # SVM
  fit.svm <- train(y~., data=iodata, method="svmRadial", preProcess="scale", trControl=trctrl)
  # Collection predictions
  cms <- list("glm"=confusionMatrix.train(fit.glm)$table,
              "svm"=confusionMatrix.train(fit.svm)$table)
  # Save model
  mod.glm <- c(mod.glm, list(fit.glm$finalModel))
  mod.svm <- c(mod.svm, list(fit.svm$finalModel))
  cmsByModels <- c(cmsByModels, list(cms))
}
names(cmsByModels) <- names(y.1316)
names(mod.glm) <- names(y.1316)
names(mod.svm) <- names(y.1316)
# Clean up
rm(i, iodata, cvOut, cvIn, trctrl, fit.glm, fit.svm)
# Stop parallel
#stopCluster(cluster)
#registerDoSEQ()
# Save
save(cmsByModels, file="qpe1316_ae_svm.results.RData")
save(mod.svm, file="qpe1316_ae_svm.mod.RData")






