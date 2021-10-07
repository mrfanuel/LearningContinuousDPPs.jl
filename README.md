# Nonparametric estimation of continuous DPPs with kernel methods

This repository is the official implementation of Fanuel, M. and Bardenet, R., [Nonparametric estimation of continuous DPPs with kernel methods](https://arxiv.org/pdf/2106.14210.pdf) to appear at NeurIPS 2021.

## Dependencies
Please install the following packages:

KernelFunctions, Plots, CSV, DelimitedFiles, DataFrames, LinearAlgebra, KernelFunctions, Distributions, JLD

~~~julia
julia> using Pkg; Pkg.install("PackageName");
~~~

## Estimation 
The first figure of the paper can be reproduced by running the following code.
###  Figure 1 top row
~~~julia
julia> include("demos/estimation_Gaussian.jl");
julia> s = 1; n = 1000; sigma = 0.1; lambda = 0.1; tol = 1e-2; intensity = 50;
julia> estimate_Gaussian(s,n,sigma,lambda,tol,intensity);
~~~
###  Figure 1 bottom row
~~~julia
julia> include("demos/estimation_Gaussian.jl");
julia> s = 1; n = 1000; sigma = 0.1; lambda = 0.1; tol = 1e-2; intensity = 100;
julia> estimate_Gaussian(s,n,sigma,lambda,tol,intensity);
~~~
The other figures are obtained by adapting the corresponding parameters.

## Convergence of the optimization algorithm
An illustration of the convergence of Regularization Picard Algorithm on a toy model is obtained by executing:
~~~julia
julia> include("demos/convergence_exact_solution.jl");
julia> convergence_exact_solution();
~~~
