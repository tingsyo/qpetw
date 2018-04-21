# File: tpwpProcessing.r
# Description:
#    Functions to read and process QPE data: radar and station precipitation tag

# Read in quantitative precipitations of CWB stations in Taipei area
# - return:
#   - dataframe of
#     - date: strings represent YYYYMMDDHH
#     - tXhr: accumulative precipitation in X hours (HH-1 ~ HH+X-1)
read.tpwp <- function(fname, filter.year=NULL){
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

