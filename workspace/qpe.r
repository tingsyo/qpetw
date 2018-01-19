# File: qpe.r
# Description:
#    Functions to perfrom QPE from radar data





create.output <- function(fname, filter.year=NULL){
  tmp <- read.csv(fname, stringsAsFactors=F, na.strings=c("NA","-999", " "), encoding="big5")
  tmp <- tmp[,c("date","X1.hr")]
  tmp$date <- as.character(tmp$date)
  #tmp$year <- substr(tmp$date,1,4)
  #tmp$month <- substr(tmp$date,5,6)
  #tmp$day <- substr(tmp$date,7,8)
  #tmp$hour <- substr(tmp$date,9,10)
  if(!is.null(filter.year)){
    tmp <- tmp[which(substr(tmp$date,1,4)==filter.year),]
  }
  row.names(tmp) <- NULL
  return(tmp)
}

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

read.radar <- function(f){
  tmp <- read.fwf(fname, c(8,8,8), header=F, col.names=c("lat","lon","dbz"))
  return(tmp)
}

create.input <- function(srcdir, filter.year=NULL){
  # Table for hour-minute correspondence
  hhlist <- c("00","01","02","03","04","05","06","07","08","09","10","11",
              "12","13","14","15","16","17","18","19","20","21","22","23")
  mmlist <- c("00","10","20","30","40","50")
  hhmm <- NULL
  for(h in hhlist){hhmm <- c(hhmm, paste0(h, mmlist))}
  hhmm <- matrix(hhmm, ncol=6, byrow=T)
  # Get all files parsed
  flist <- paste(srcdir,list.files(srcdir, recursive=T), sep="/")
  ftab <- do.call(rbind, lapply(flist, function(x){unlist(strsplit(x,'.', fixed=T))}))
  ftab <- data.frame(ftab, stringsAsFactors=FALSE, row.names=c("prefix","date","hhmm","ext"))
  # Generate desired files and check the file existence
  days <- sort(unique(ftab$date))
  
  #
  return(tmp)
}
