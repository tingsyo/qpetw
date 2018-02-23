# File: qpe.r
# Description:
#    Functions to perfrom QPE from radar data
# Load input/output data
source("utils.r")
load("io.tpe2016.pc20.RData")
rseed = 1234543
# Load library
require(caret)
require(kernlab)

# Collect results
#list.glm <- NULL
#list.svm <- NULL

# Run through Each output
results <- data.frame(NULL)
nstation <- length(ys.tpe2016)
for(i in 1:nstation){
  # 
  print(paste("Creating IO data for",names(ys.tpe2016)[i]))
<<<<<<< HEAD
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
  fit.glm <- train(y~., data=iodata, method="glm", trControl=trctrl)
  print("GLM")
  print(fit.glm$results)
  #list.glm <- c(list.glm, list(fit.glm))
  results.glm <- rbind(results.glm, fit.glm$results)
  coef.glm <- c(coef.glm, list(coef(summary(fit.glm))))
  # SVM
  fit.svmr <- train(y~., data=iodata, method="svmRadial", trControl=trctrl)
  print("SVR")
  print(fit.svmr$results)
  #list.svm <- c(list.svm, list(fit.svmr))
  results.svm <- rbind(results.svm, fit.svmr$results[1,])
  # Collection predictions
  y <- iodata$y
  y.glm <- fit.glm$finalModel$fitted.values
  y.svm <- fit.svmr$pred$pred
  ys <- c(ys, list(data.frame("y"=y, "y.glm"=y.glm, "y.svm"=y.svm)))
=======
  # Extract Y
  y <- ys.tpe2016[[i]]
  # Evaluate QPF
  print(paste("QPE for station:",names(ys.tpe2016)[i]))
  tmp <- test.qpf(y, input.2016.pc20)
  # Collect results
  res <- cbind("station"=names(ys.tpe2016)[i], tmp)
  results <- rbind(results, res)
>>>>>>> b27bf269671f3235c1abe7a28f47ad5b2cd899eb
}
# Clean up
rm(i)
# Save
save.image("qpf2016.RData")






