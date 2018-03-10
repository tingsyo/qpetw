# File: utils.r
# Description:
#    Functions to read and process QPE data: radar and station precipitation tag

# Read in radar data: fixed width format
read.radar <- function(f){
  tmp <- read.fwf(fname, c(8,8,8), header=F, col.names=c("lat","lon","dbz"))
  return(tmp)
}

# Plot radar data with ggmap: remove all 0s and interpolate on the map
plot.radar <- function(dbz){
  require(ggmap)
  map <- get_map(location='Taiwan', zoom=7)
  dbz <- dbz[which(dbz$dbz>0),]
  g <- ggmap(map) + stat_density2d(aes(x=lon, y=lat, fill=..level.., alpha=..level..), 
                                   size=2, bins=4, data=dbz, geom="polygon")
  g
  return(g)
}

# Read in quantitative precipitations of CWB stations in Taipei area
# - return:
#   - dataframe of
#     - date: strings represent YYYYMMDDHH
#     - tXhr: accumulative precipitation in X hours (HH-1 ~ HH+X-1)
read.tpeqp <- function(fname, filter.year=NULL){
  # Read file
  tmp <- read.csv(fname, stringsAsFactors=F, na.strings=c("NA","-999", " "), fileEncoding="BIG-5")
  # Select columns
  tmp <- tmp[,c(3,4,9,14,19,24,29,34,39)]
  names(tmp) <- c("date","t1hr","t3hr","t6hr","t12hr","t24hr","t48hr","t72hr","t120hr")
  # 
  tmp$date <- as.character(tmp$date)
  if(!is.null(filter.year)){
    tmp <- tmp[which(substr(tmp$date,1,4)==filter.year),]
  }
  row.names(tmp) <- NULL
  return(tmp)
}

# Aggregate output data object: a list of dataframes, named by the station
create.output <- function(datadir, filter.year=NULL){
  # Get all stations
  flist <- list.files(datadir)
  # Loop through all stations
  outputs <- NULL
  onames <- NULL
  for(f in flist){
    fname <- paste(datadir,f,sep="/")
    print(paste("Processing file:",fname))
    tmp <- read.tpeqp(fname, filter.year=filter.year)
    print(dim(tmp))
    if(nrow(tmp) %in% c(365*24, 366*24) ){
      outputs <- c(outputs, list(tmp))
      onames <- c(onames, f)
    } else {
      print(paste("Number of records incorrect:", nrow(tmp)))
    }
  }
  names(outputs) <- substr(onames,1,6)
  return(outputs)
}



# 
dbz.flist <- function(srcdir){
  output <- NULL
  # Get all days
  daylist <- list.files(srcdir)
  # Loop through days
  for(d in daylist){
    ddir <- paste0(srcdir,d,"/")
    print(ddir)
    fltmp <- list.files(ddir)
    ftab <- data.frame(do.call(rbind, lapply(fltmp, function(x){unlist(strsplit(x,'.', fixed=T))})))
    tmp <- cbind("day"=rep(d, length(fltmp)), "fname"=fltmp, ftab)
    output <- rbind(output, tmp)
  }
  return(output)
}

# Read in preprocessed DBZ data: 
read.dbzpcs <- function(fname, npc=20){
  tmp <- read.csv("dbz.ipca50.csv", stringsAsFactors = F, header=F, colClasses = c("character","character",rep("numeric", npc)))
  names(tmp) <- c("date","hhmm",paste("pc", 1:npc, sep="_"))
  return(tmp[,1:(npc+2)])
}

# Create input data for QPE:
# - Use given Y(precipitation) and it's time stamp(YYYYMMDDHH)
# - For the same YYYYMMDD, each HH of precipitation is corresponding to: 
#   - 6 HHMM of DBZ data records (with 10min interval)
#   - HH-60min, HH-50min, HH-40min, HH-30min, HH-20min, and HH-10min
#   - impute NA for missing DBZ records
create.input.qpe <- function(y, x, filter.year=NULL){
  # Table for hour-minute correspondence
  hhlist <- c("00","01","02","03","04","05","06","07","08","09","10","11",
              "12","13","14","15","16","17","18","19","20","21","22","23")
  mmlist <- c("00","10","20","30","40","50")
  hhmm <- NULL
  for(h in hhlist){hhmm <- c(hhmm, paste0(h, mmlist))}
  hhmm <- matrix(hhmm, ncol=6, byrow=T)
  # Target hour
  thh <- c("01","02","03","04","05","06","07","08","09","10","11","12",
           "13","14","15","16","17","18","19","20","21","22","23","24")
  # target day-hour
  ydays <- sort(unique(substr(y$date,1,8)))
  newx <- NULL
  for(i in 1:nrow(y)){
    # Process date-hour string
    ydh <- y$date[i]                # YYYYMMDDHH string
    yday <- substr(ydh,1,8)         # YYYYMMDD
    yhh <- substr(ydh,9,10)         # HH
    idx.day <- which(ydays==yday)   # Index of YYYYMMDD
    idx.h <- which(thh==yhh)        # Index of HH
    # Find corresponding X
    ridx.x <- which((x[,1]==yday) & (x[,2] %in% hhmm[idx.h,]))
    # Concatenate all x as the new x
    print(ydh)
    #print(length(ridx.x))
    #print(x[ridx.x, 1:2])
    if(length(ridx.x)==6){
      rec <- as.vector(t(x[ridx.x, -(1:2)]))
      #print(length(rec))
    } else {
      print("missing data")
      rec <- rep(NA, 6*(ncol(x)-2))
    }
    newx <- rbind(newx, rec)
  }
  # Create colnames
  newx <- data.frame(newx)
  pcn <- names(x)[-(1:2)]
  ts <- paste0("m", seq(60,10,-10))
  names(newx) <- apply(expand.grid(pcn, ts), 1, paste, collapse="_")
  #
  return(cbind(y,newx))
}

# Run a loop to test QPF
test.qpf <- function(y, x, rseed=12345){
  # Number of tests of prediction
  ntest <- ncol(y) - 1
  ynames <- names(y)
  # Loop through 
  results <- NULL
  for(i in 1:ntest){
    print(paste("Evaluating QPF:", ynames[i+1]))
    # Combine input/output data
    iodata <- cbind("y"=y[,i+1], x)
    row.names(iodata) <- y$date
    # Shift y for forecast
    iodata$y <- c(iodata$y[2:nrow(iodata)], NA)
    iodata <- iodata[complete.cases(iodata),]
    print(paste("    Number of valid records:", nrow(iodata)))
    # Setup train-control
    print("    Creating folds for cross validation...")
    set.seed(rseed)
    cvOut <- createFolds(iodata$y, k=10)
    cvIn <- cvOut
    for(i in 1:10){
      cvIn[[i]] <- (1:length(iodata$y))[-cvOut[[i]]]
    }
    trctrl <- trainControl("cv", index=cvIn, indexOut=cvOut, savePred=T)
    # Fit model
    print("    Training and cross validating...")
    fit.glm <- train(log(y+1)~., data=na.omit(iodata), method="glm", preProcess="scale", trControl=trctrl)
    rec <- fit.glm$results
    yhat <- fit.glm$finalModel$fitted.values
    rec$RMSE.insample <- RMSE(exp(yhat)-1, iodata$y)
    rec$CORR.log <- cor(yhat, log(iodata$y+1))
    rec$CORR.mm <- cor(exp(yhat)-1, iodata$y)
    results <- rbind(results, rec)
  }
  results <- cbind("test"=ynames[-1], data.frame(results))
  return(results)
}

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