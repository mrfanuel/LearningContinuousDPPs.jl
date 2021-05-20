library(spatstat)
library(stats)

#For parallelization
library(foreach)
library(doSNOW)
library(parallel)

#source("MLE_DPP.R") #Load the DPP function, make sure it is in the working directory.

###############################Exemples of using the function MLEDPP#################################
#####On a rectangular window#####
#Gauss
rho0 = 100
alpha0 = 0.05
S_length = 1
S = simulate(dppGauss(lambda=rho0, alpha=alpha0, d=2), 1, W=owin(c(0,S_length), c(0,S_length)))
plot(S)

S.x