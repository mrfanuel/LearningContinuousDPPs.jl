# Nonparametric estimation of continuous DPP
## code of the paper Fanuel, M. and Bardenet, R., Nonparametric estimation of continuous DPPs with kernel methods [arxiv:2106.14210](https://arxiv.org/pdf/2106.14210.pdf)

### Dependencies
 KernelFunctions, Plots, CSV, DelimitedFiles, DataFrames, LinearAlgebra, KernelFunctions, Distributions
### example: to reproduce the results of the paper
####  Figure 1  top row

    include("demos/estimation_Gaussian.jl");
    s = 1;
    n = 1000;
    sigma = 0.1;
    lambda = 0.1;
    tol = 1e-5;
    intensity = 50;
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity);
####  Figure 1  bottom row

    include("demos/estimation_Gaussian.jl");
    s = 1;
    n = 1000;
    sigma = 0.1;
    lambda = 0.1;
    tol = 1e-5;
    intensity = 100;
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity);