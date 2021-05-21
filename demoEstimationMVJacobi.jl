
include("algo/estimateMVJacobiB.jl")
# s : number of DPP samples

# number of uniform samples for Fredholm
n = 50;

# number of uniform samples for correlation kernel
p = 500;

# type of kernel and parameters

#s = 2; kernel = "MaternKernel"; sigma = 1. ;lambda = .01; nu = 5/2.;
s = 1; kernel = "SqExponentialKernel"; sigma = 1.; lambda = .1; nu = 0.;

# regularizer for K positive definite
epsilon = 1e-6; 

# max number of iteration
it_max = 5000;

# relative objective tolerance
tol = 1e-6;

# Plotting number of grid points along x-axis
n_step_plot = 100; 

GramMatrix = estimateMVJacobiB(s,n,p,kernel,nu,sigma,lambda,epsilon,it_max,tol,n_step_plot);

# heatmap(GramMatrix)




