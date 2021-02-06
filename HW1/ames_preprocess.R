library(AmesHousing)
# function to combine levels, with less than a specified number of observations, of a factor variables

comblev=function(x,nmin)
{
  # x = variable (must be a factor, if not the original variable is returned)
  # nmin = levels with less than nmin observations will be combined
  # output
  #  	same variable with a new level called "othcomb" replacing the combined levels 
  #	(NA's are not affected)
  
  if(!is.factor(x)) {return(x)}
  library(rockchalk)
  ta=table(x)
  combineLevels(x,levs = names(table(x))[table(x)<nmin], newLabel = c("othcomb") )
}

ames=make_ordinal_ames()
# remove an observation with a missing
ames=ames[!is.na(ames$Electrical),]

# remove the variable "Utilities" because it is almost constant 
# with frequencies (1,1,2909).  
ames$Utilities=NULL

# converts the target variable (1 = 1000)
ames$Sale_Price=ames$Sale_Price/1000

# get the names of the ordinal variables
ord_vars=vapply(ames, is.ordered, logical(1))
namored=names(ord_vars)[ord_vars]
# converts the ordered factors to numeric (this preserves the ordering of the factor)
ames[,namored]=data.frame(lapply(ames[,namored],as.numeric))

# get the names of the factor variables
fac_vars=vapply(ames, is.factor, logical(1))
namfac=names(fac_vars)[fac_vars]

# group together levels with less than 30 observations
ames=data.frame(lapply(ames,comblev,nmin=30))

# remove the space in the values (string) of some variables to prevent problems later
ames[,"Exterior_1st"]=as.factor(gsub(" ","",ames[,"Exterior_1st"]))
ames[,"Exterior_2nd"]=as.factor(gsub(" ","",ames[,"Exterior_2nd"])) 

num_vars=vapply(ames, is.numeric, logical(1))
# names of the numeric variables
namnum=names(num_vars)[num_vars]

##################################
# Create dummy variables for the factors

library(dummies)

amesdum=dummy.data.frame(ames)
# to remove the last level as a reference level
amesdum=amesdum[,-c(unlist(lapply(attributes(amesdum)$dummies,max)))]

# Now all variables are numeric.
# There are 160 covariates and 1 target "Sale_Price".



##################################
# Splitting the data into a training (ntrain=1000) and a 
#  test (ntest=1929) set

set.seed(364565)
ntrain=1000
ntest=nrow(ames)-ntrain
indtrain=sample(1:nrow(ames),ntrain,replace=FALSE)

xdum=amesdum
xdum$Sale_Price=NULL
xdum=as.matrix(xdum)

amestrain=ames[indtrain,]
amestest=ames[-indtrain,]
amesdumtrain=amesdum[indtrain,]
amesdumtest=amesdum[-indtrain,]
xdumtrain=xdum[indtrain,]
xdumtest=xdum[-indtrain,]