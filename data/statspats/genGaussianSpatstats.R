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
rho0 = 50; # 100 originally
alpha0 = 0.05
bound = 1

for (x in 1:10) {
    S = simulate(dppGauss(lambda=rho0, alpha=alpha0, d=2), 1, W=owin(c(0,bound), c(0,bound)))
    #plot(S)
    GaussDPPsample = data.frame(S)
    write.csv(GaussDPPsample,paste('data/statspats/samples/GaussDPPsample_alpha0_00p',toString(alpha0*100),'_rho0_',toString(rho0),'_nb_',toString(x),'.csv', sep = ""))
}

 