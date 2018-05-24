#=======================================================================
# File: qpetw_predict.r
# Purpose:
#   Perform QPE and 1hr-QPF with transformed DBZ data. 
#=======================================================================
# Parameters
NEWDIR <- "../workspace/new/"
NEW_REDUCED_DATA <- "new_input.csv"
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
  return(params)
}


#=======================================================================
# Script commands
#=======================================================================
require(caret)
require(kernlab)
# Read and parse configuration file
params <- readConfiguration("qpetw.cfg")
path_mod_qpe <- params[which(params[,1]=="MOD_PATH_QPE"),2]
path_mod_qpf <- params[which(params[,1]=="MOD_PATH_QPF"),2]
flist.mod_qpe <- list.files(path_mod_qpe)
flist.mod_qpf <- list.files(path_mod_qpf)
sids <- substr(flist.mod_qpe,1,6)
# Read input data


# QPE


# 1hr-QPF

