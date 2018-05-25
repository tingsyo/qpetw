#=======================================================================
# File: qpetw_predict.r
# Purpose:
#   Perform QPE and 1hr-QPF with transformed DBZ data. 
#=======================================================================
# Parameters
#=======================================================================
# Functions
#=======================================================================
# Read parameters from configuration file
readConfiguration <- function(fname){
  # Read text file
  l <- readLines(fname)
  # Skip empty lines and comment
  l <- l[-(which(substr(l,1,1) %in% c("#","")))]
  # Parse keys and values
  params <- do.call(rbind, strsplit(gsub("\"","",l), "="))
  e <- new.env()
  apply(params,1, function(x){assign(x[1],x[2], env=e)})
  return(e)
}

# Create input data for QPE:
# - Use given time stamp(YYYYMMDDHH), ts
# - For the same YYYYMMDD, each HH of precipitation is corresponding to: 
#   - 6 HHMM of DBZ data records (with 10min interval)
#   - HH-60min, HH-50min, HH-40min, HH-30min, HH-20min, and HH-10min
#   - impute NA for missing DBZ records
create.input.qpe <- function(ts, x){
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
  newx <- NULL
  # Process date-hour string
  ydh <- ts                       # YYYYMMDDHH string
  yday <- substr(ydh,1,8)         # YYYYMMDD
  yhh <- substr(ydh,9,10)         # HH
  idx.h <- which(thh==yhh)        # Index of HH
  # Find corresponding X
  ridx.x <- which((x[,1]==yday) & (x[,2] %in% hhmm[idx.h,]))
  # Concatenate all x as the new x
  print(ydh)
  #print(x[ridx.x, 1:2])
  if(length(ridx.x)==6){
    rec <- as.vector(t(x[ridx.x, -(1:2)]))
    #print(length(rec))
  } else {
    print("missing data")
    rec <- rep(NA, 6*(ncol(x)-2))
  }
  newx <- rbind(newx, rec)
  # Create colnames
  newx <- data.frame(newx)
  pcn <- names(x)[-(1:2)]
  ts <- paste0("m", seq(60,10,-10))
  names(newx) <- apply(expand.grid(pcn, ts), 1, paste, collapse="_")
  #
  return(newx)
}

#=======================================================================
# Script commands
#=======================================================================
require(caret)
require(kernlab)
# Read and parse configuration file
params <- readConfiguration("qpetw.cfg")
path_mod_qpe <- params$"MOD_PATH_QPE"
path_mod_qpf <- params$"MOD_PATH_QPF"
flist.mod_qpe <- list.files(path_mod_qpe)
flist.mod_qpf <- list.files(path_mod_qpf)
sids <- substr(flist.mod_qpe,1,6)
# Read transformed data and create input data
dbz.pc20 <- read.csv(params$DBZ_INPUT, stringsAsFactors=F, fileEncoding="UTF-8-BOM")
attach(dbz.pc20)
dbz.pc20 <- dbz.pc20[order(date, hhmm),]
dbz.input <- create.input.qpe("201601311200", dbz.pc20)
# QPE
for(fqpe in flist.mod_qpe){
  load(paste0(params$"MOD_PATH_QPE","/",fqpe))
  qpe <- predict(mod, newdata=dbz.input)
  print(paste(fqpe, qpe))
}

# 1hr-QPF

