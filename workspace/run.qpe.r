# File: qpe.r
# Description:
#    Functions to perfrom QPE from radar data
# Load input/output data
load("io.tpe2016.pc20.RData")
rseed = 1234543
# Load library
require(caret)
require(kernlab)

# Collect results
results.glm <- data.frame(NULL)
#results.svm <- data.frame(NULL)
#coef.glm <- NULL
ys <- NULL
# Run through each station
nstation <- length(ys.tpe2016)
for(i in 1:nstation){
  # 
  print(paste("Creating IO data for",names(ys.tpe2016)[i]))
  # Combine IO
  iodata <- cbind("y"=ys.tpe2016[[i]]$t1hr, input.2016.pc20)
  # Clean up NA and move date to row.names
  print("Cleaning up data...")
  row.names(iodata) <- ys.tpe2016[[i]]$date
  iodata <- iodata[complete.cases(iodata),]
  print(paste("    Number of valid records:", nrow(iodata)))
  # Setup train-control
  print("Creating folds for cross validation...")
  set.seed(rseed)
  cvOut <- createFolds(iodata$y, k=10)
  cvIn <- cvOut
  for(i in 1:10){
    cvIn[[i]] <- (1:length(iodata$y))[-cvOut[[i]]]
  }
  trctrl <- trainControl("cv", index=cvIn, indexOut=cvOut, savePred=T)
  # Fit model
  print("Training and cross validating...")
  # GLM
  fit.glm <- train(log(y+1)~., data=iodata, method="glm", preProcess="scale", trControl=trctrl)
  rec <- fit.glm$results
  yhat <- fit.glm$finalModel$fitted.values
  rec$RMSE.insample <- RMSE(exp(yhat)-1, iodata$y)
  rec$CORR.log <- cor(yhat, log(iodata$y+1))
  rec$CORR.mm <- cor(exp(yhat)-1, iodata$y)
  print("GLM")
  print(rec)
  results.glm <- rbind(results.glm, rec)
  #coef.glm <- c(coef.glm, list(coef(summary(fit.glm))))
  # SVM
  #fit.svmr <- train(log(y+1)~., data=iodata, method="svmRadial", preProcess="scale", trControl=trctrl)
  #print("SVR")
  #print(fit.svmr$results)
  #list.svm <- c(list.svm, list(fit.svmr))
  #results.svm <- rbind(results.svm, fit.svmr$results)
  # Collection predictions
  y$y <- iodata$y
  y$y.glm <- fit.glm$finalModel$fitted.values
  #y$y.svm <- fit.svmr$pred$pred
  ys <- c(ys, list(y))
}
#names(list.glm) <- names(ys.tpe2016)
#names(list.svm) <- names(ys.tpe2016)
#names(coef.glm) <- names(ys.tpe2016)
names(ys) <- names(ys.tpe2016)
# Clean up
rm(i, iodata, cvOut, cvIn, trctrl, fit.glm, fit.svmr)
# Save
save.image("qpe2016.RData")






