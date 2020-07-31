setwd("C:/Users/joony/Documents/CS/project")

dat <- read.csv("mushrooms.csv", header=T, sep=",")

table(is.na(dat))

for (i in names(dat)) {
  print(i)
  print(table(dat[i]))
  
}

#  "stalk.root" has ? as a value
# "veil.type" only has one type 
colInd <- names(dat) %in% c("stalk.root", "veil.type")
refinedDat <- dat[,!colInd]

write.csv(refinedDat, "mushrooms_fixed.csv", quote=F, row.names=F)


# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
