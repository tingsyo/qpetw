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
  # Extract Y
  y <- ys.tpe2016[[i]]
  # Evaluate QPF
  print(paste("QPE for station:",names(ys.tpe2016)[i]))
  tmp <- test.qpf(y, input.2016.pc20)
  # Collect results
  res <- cbind("station"=names(ys.tpe2016)[i], tmp)
  results <- rbind(results, res)
}
# Clean up
rm(i)
# Save
save.image("qpf2016.RData")






