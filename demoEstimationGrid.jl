include("algo/estimateGridB.jl")
# Here we use the square SqExponentialKernel

# number of DPP samples
s = 1;

# number of uniform samples for Fredholm
n = 300 # 100 is better;

# number of uniform samples for correlation kernel
p = 200;

# kernel bw (Float64)
sigma = 0.1 #.5;

# regularization (Float64)
lambda =  1e-5;#.1

# regularizer for K positive definite
epsilon = 1e-8; 

# max number of iteration
it_max = 5000;

# relative objective tolerance
tol = 1e-3;

# Plotting number of grid points along x-axis
n_step_plot = 100; 


GramMatrix,B = estimateGridB(s,n,p,sigma,lambda,epsilon,it_max,tol,n_step_plot)
