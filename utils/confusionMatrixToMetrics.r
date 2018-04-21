# Derive statistics from confusion matrix
# [ref] https://en.wikipedia.org/wiki/Confusion_matrix
generateCM <- function(pred, ref){
  return(table(pred, ref))
}

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