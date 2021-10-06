# Nonparametric estimation of continuous DPPs
#### Fanuel, M. and Bardenet, R., <em>Nonparametric estimation of continuous DPPs with kernel methods</em>  <br />
[arxiv:2106.14210](https://arxiv.org/pdf/2106.14210.pdf) to appear at NeurIPS 2021.


## Examples

### Estimation 

####  Figure 1 top row
~~~julia
julia> include("demos/estimation_Gaussian.jl");
julia> s = 1; n = 1000; sigma = 0.1; lambda = 0.1; tol = 1e-2; intensity = 50;
julia> estimate_Gaussian(s,n,sigma,lambda,tol,intensity);
~~~julia

####  Figure 1 bottom row

    include("demos/estimation_Gaussian.jl");
    s = 1;
    n = 1000;
    sigma = 0.1;
    lambda = 0.1;
    tol = 1e-2;
    intensity = 100;
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity);


### Convergence  

    include("demos/convergence_exact_solution.jl");
    convergence_exact_solution();

## Dependencies

KernelFunctions, Plots, CSV, DelimitedFiles, DataFrames, LinearAlgebra, KernelFunctions, Distributions, JLD
