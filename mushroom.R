
# source of data
# https://www.kaggle.com/uciml/mushroom-classification

# set this to your working direcotry
# setwd("C:/Users/joony/Documents/CS/project")

# read in the original csv data file
dat <- read.csv("mushrooms.csv", header=T, sep=",")

# check if any of the data are missing
table(is.na(dat))

# in the following loop, we will see for each column, 
# how many of certain values are present 
# for all the values in that column

# iterate through all of the columns
for (i in names(dat)) {
  # print out the column name
  print(i)
  # print out all of the values present in the data
  # and the number of the values also
  print(table(dat[i]))
}

# "stalk.root" and "veil.type" columns are removed for following reasons.
# "stalk.root" has ? as a value and therefore removed. 
# "veil.type" only has one value type and therefore removed. 

# get a list of booleans that indicate which columns are the ones we are removing
colInd <- names(dat) %in% c("stalk.root", "veil.type")
# only select the columns we are not removing
refinedDat <- dat[,!colInd]

# write the new data in a file called "mushrooms_fixed.csv"
write.csv(refinedDat, "mushrooms_fixed.csv", quote=F, row.names=F)

