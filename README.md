# code for "Nonparametric estimation of continuous DPPs with kernel methods" (arxiv:2106.14210)


## example: to reproduce the results of Figure 1 (bottom row)

    include("demos/estimation_Gaussian.jl");
    s = 1;
    n = 1000;
    sigma = 0.1;
    lambda = 0.1;
    tol = 1e-5;
    intensity = 100;
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity);