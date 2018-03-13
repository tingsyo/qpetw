# File: dbzProcessing.r
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
