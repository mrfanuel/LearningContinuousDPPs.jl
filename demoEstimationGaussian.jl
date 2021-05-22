include("algo/estimateGaussianB.jl")
# Here we use the square SqExponentialKernel

# number of DPP samples
s = 10;

# number of uniform samples for Fredholm
n = 100 # 100 is better;

# number of uniform samples for correlation kernel
p = 2000;

# kernel bw (Float64)
sigma = 0.1;

# regularization (Float64)
lambda =  .01

# regularizer for K positive definite
epsilon = 1e-8; 

# max number of iteration
it_max = 5000;

# relative objective tolerance
tol = 1e-6;

# Plotting number of grid points along x-axis
n_step_plot = 100; 


GramMatrix = estimateGaussianB(s,n,p,sigma,lambda,epsilon,it_max,tol,n_step_plot)
